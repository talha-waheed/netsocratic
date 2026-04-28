"""
Generator Agent
===============
Takes a clarified network intent and a knowledge base, then produces N candidate
rule sets by running the rules extractor at varied temperatures.  Each rule set is
then compiled into a deterministic OSPF configuration at a fixed temperature.

Workflow per candidate
----------------------
1. Rules LLM (temperature varies per candidate) receives the clarified intent and
   outputs a structured ground-truth rules JSON (reachability / waypoint /
   loadbalancing).  This is the primary unit of analysis — candidates differ here.
2. Config LLM (fixed temperature) receives the rules JSON + topology + KB and
   outputs only the router configs that need to change.
3. All remaining routers are copied verbatim from `topo_dir`.
4. Each candidate is saved to results_dir/candidates/candidate_N/ containing:
     rules.json          ← the rules that drove this candidate
     decision_summary.txt
     configs/<Router>.cfg  ← full router set (LLM-modified + base copies)

Single source of truth
-----------------------
The network topology is parsed directly from the `topo_dir` .cfg files at runtime
and injected into the LLM prompt as "## Network Topology". There is no static
topology.md — the topology section always reflects the actual base configs.
Other knowledge-base files (ospf_config_format.md, routing_policy.md) are still
loaded from `kb_dir` and injected as-is.
"""

import json
import os
import re

from llm.base import BaseLLMClient, Message


# ── Rules-extraction prompt ────────────────────────────────────────────────────

RULES_EXTRACTOR_SYSTEM = """\
You are a network policy analyst. Convert the clarified network routing intent
into a structured ground-truth rules object using the format below. Output ONLY
valid JSON — no prose, no markdown fences.

Format:
{
  "reachability": {
    "<source_router>": ["<prefix>", ...]
  },
  "waypoint": {
    "(<source_router>,<prefix>)": ["<waypoint_router>", ...]
  },
  "loadbalancing": {
    "(<source_router>,<prefix>)": <integer path count>
  }
}

Rules:
- Router names are lowercase (e.g. "athens", "london").
- Prefixes use CIDR notation (e.g. "100.0.29.0/24").
- Include a reachability entry for every source-prefix pair mentioned.
- Include a waypoint entry only when a mandatory intermediate router is required.
- Include a loadbalancing entry only when a specific path count is required.
- Omit sections that have no entries (use empty objects {}).
- Output nothing except the JSON object.
"""


# ── Configuration-generation prompt ───────────────────────────────────────────

GENERATOR_SYSTEM = """\
You are Network Configuration Generator, an expert in Cisco IOS OSPF configuration.

## Your job
Given structured routing rules (reachability / waypoint / loadbalancing) and
supporting topology/policy context, produce the minimal set of modified Cisco IOS
router configs that correctly implement those rules. Unchanged routers will be
copied from the base topology automatically — do NOT output them.

## Strict rules
1. Output ONLY the routers whose `ip ospf cost` values must change.
   Every other router stays at its base configuration (do not emit it).
2. Use OSPF process ID 100.
3. Always include both of these in `router ospf 100`:
     redistribute connected subnets
     maximum-paths 32
4. Do NOT add `ip ospf` commands on the stub interface (Fa0/0).
5. Keep interface names, IP addresses, masks, descriptions, and all other lines
   IDENTICAL to the base config shown in Network Topology. Only `ip ospf cost` may differ.
6. Implement routing policy via cost values only:
   - Cost 1    = preferred path, participates in ECMP
   - Cost 1000 = effectively blocked (mandatory waypointing: block all bypass paths)
7. To achieve exactly N equal-cost ECMP paths: ensure exactly N routes share the
   minimum total metric; all other routes must be strictly higher.
8. Think step-by-step: enumerate candidate paths, calculate total costs, verify
   that every constraint in the ## Routing Rules section is satisfied before
   writing configs.

## Output format — follow exactly
### <RouterName>
```
<full Cisco IOS config, identical to base except for modified ip ospf cost lines>
```

### decision_summary.txt
```
Routers modified: <list>

Reasoning:
<step-by-step path enumeration, cost calculation, and verification that every
rule (reachability, waypoint, loadbalancing) is satisfied>

Limitations (if any):
<topology constraints that prevented full intent satisfaction, or "None">
```
"""


# ── Per-candidate strategy hints ──────────────────────────────────────────────
#
# Each entry has two keys:
#   "rules_hint"  : appended to the rules-extraction user message
#   "config_hint" : appended to the config-generation user message
#
# Strategies are the primary diversity driver across candidates; temperature
# variation is a secondary stochastic lever within each strategy's space.

CANDIDATE_STRATEGIES: list[dict[str, str]] = [
    {
        # Candidate 1 — Conservative
        "rules_hint": """\
## Extraction Strategy: Conservative
- Include only explicitly stated reachability pairs; do not infer extras.
- If a waypoint is ambiguous (e.g. "prefer", "should"), treat it as PREFERRED —
  omit it from the "waypoint" section (waypoint entries mean mandatory traversal).
- If a load-balancing count is not stated as an exact integer, omit it.
- When in doubt, produce fewer constraints.""",
        "config_hint": """\
## Implementation Strategy: Conservative
- Make the fewest possible cost changes from the base topology.
- Any waypoint listed is preferred, not mandatory — do NOT set any link to cost 1000.
- Achieve load-balancing with the minimum number of modified routers.
- Leave all non-essential routers at base cost 1.""",
    },
    {
        # Candidate 2 — Standard
        "rules_hint": """\
## Extraction Strategy: Standard
- Include every reachability pair that is explicitly or clearly implicitly stated.
- Treat every waypoint as MANDATORY (include in "waypoint") unless the intent
  explicitly says "prefer" or "if possible".
- Use the exact integer for load balancing as stated.
- Produce the most straightforward structured representation of the intent.""",
        "config_hint": """\
## Implementation Strategy: Standard
- For mandatory waypoints: cost 1000 on all bypass paths, cost 1 on waypoint paths.
- For ECMP: ensure exactly the specified number of equal-cost paths by adjusting
  the source router's outgoing link costs first before touching intermediate routers.
- Document every cost decision in decision_summary.txt.""",
    },
    {
        # Candidate 3 — Exploratory
        "rules_hint": """\
## Extraction Strategy: Exploratory
- Include any reachability pairs that are strongly implied (e.g. bidirectional).
- If a waypoint could be either mandatory or preferred, treat it as MANDATORY.
- If load-balancing is ambiguous, round up to the next power of two, capped at
  the number of distinct paths available in the topology.
- Prefer more constraints over fewer when ambiguity exists.""",
        "config_hint": """\
## Implementation Strategy: Exploratory
- Use a graduated cost scale: preferred=1, acceptable alternatives=10, blocked=1000.
  This creates a tiered fallback rather than binary on/off.
- Distribute cost changes across intermediate routers (not only the source router)
  for a more topology-wide implementation.
- Note in decision_summary.txt how this graduated scale differs from the standard
  binary approach and what operational benefit it provides.""",
    },
]


# ── Knowledge base loader ──────────────────────────────────────────────────────

def load_knowledge_base(kb_dir: str) -> dict[str, str]:
    """Load all .md and .txt files from kb_dir, excluding topology.md (generated at runtime)."""
    kb: dict[str, str] = {}
    if not os.path.isdir(kb_dir):
        return kb
    for fname in sorted(os.listdir(kb_dir)):
        if fname.endswith((".md", ".txt")) and fname != "topology.md":
            with open(os.path.join(kb_dir, fname), encoding="utf-8") as f:
                kb[fname] = f.read()
    return kb


def load_topo_configs(topo_dir: str) -> dict[str, str]:
    """Load all .cfg files from topo_dir. Returns {RouterName: config_text}."""
    configs: dict[str, str] = {}
    if not os.path.isdir(topo_dir):
        return configs
    for fname in os.listdir(topo_dir):
        if fname.endswith(".cfg"):
            router_name = fname[:-4]  # strip .cfg
            with open(os.path.join(topo_dir, fname), encoding="utf-8") as f:
                configs[router_name] = f.read()
    return configs


def generate_topology_doc(topo_configs: dict[str, str]) -> str:
    """
    Parse the loaded topo .cfg files and produce a topology markdown document
    equivalent to the old static topology.md — but always in sync with topo_dir.
    """
    routers: dict[str, dict] = {}
    for router_name, content in topo_configs.items():
        stub_ip = stub_subnet = None
        backbone = []
        for block in re.split(r"\n(?=interface )", content):
            if not block.strip().startswith("interface"):
                continue
            name_m = re.search(r"interface (\S+)", block)
            ip_m   = re.search(r"ip address (\S+) \S+", block)
            desc_m = re.search(r'description "([^"]+)"', block)
            cost_m = re.search(r"ip ospf cost (\d+)", block)
            if not ip_m or not name_m:
                continue
            ip    = ip_m.group(1)
            iface = name_m.group(1)
            nbr   = desc_m.group(1).replace("To ", "") if desc_m else ""
            cost  = int(cost_m.group(1)) if cost_m else 1
            if ip.startswith("100."):
                stub_ip, stub_subnet = ip, nbr
            else:
                backbone.append((iface, ip, cost, nbr))
        routers[router_name] = {"stub_ip": stub_ip, "stub_subnet": stub_subnet, "backbone": backbone}

    lines = [
        "## Network Topology",
        "",
        "Parsed from `topo/` at runtime — always in sync with the base configs.",
        "All interfaces run OSPF area 0, process ID 100. Base cost for every link is **1**.",
        "The Generator may only change `ip ospf cost` values; all other fields are fixed.",
        "",
        "### Router Summary",
        "",
        "| Router | Stub Subnet | Stub IP | Backbone Neighbours |",
        "|--------|-------------|---------|---------------------|",
    ]
    for r, info in sorted(routers.items()):
        nbrs = ", ".join(n for _, _, _, n in info["backbone"]) or "—"
        lines.append(f"| {r} | {info['stub_subnet']} | {info['stub_ip']} | {nbrs} |")

    lines += ["", "### Per-Router Interface Tables", "",
              "Modify **only** `ip ospf cost`; keep every other field identical to the base.", ""]
    for r, info in sorted(routers.items()):
        lines.append(f"#### {r}")
        lines.append(f"- **Stub** `Fa0/0`: `{info['stub_ip']} 255.255.255.0` → `{info['stub_subnet']}`")
        for iface, ip, cost, nbr in info["backbone"]:
            lines.append(f"- **`{iface}`**: `{ip} 255.255.255.254`  cost `{cost}`  → `{nbr}`")
        lines.append("")

    return "\n".join(lines)


# ── Agent ─────────────────────────────────────────────────────────────────────

class GeneratorAgent:
    """
    Generates N candidate rule sets from a clarified intent, then compiles each
    into a deterministic OSPF configuration.

    Parameters
    ----------
    llm                  : LLM client (any BaseLLMClient implementation)
    kb_dir               : knowledge-base directory (markdown/text files)
    topo_dir             : base topology directory containing one .cfg per router
    num_candidates       : number of rule-set candidates to generate
    rules_temperatures   : LLM temperature per rules-extraction run
                           (length must equal num_candidates)
    config_temperature   : fixed LLM temperature for OSPF config generation
    use_strategies       : if True (default), inject CANDIDATE_STRATEGIES hints into
                           each candidate's prompts for semantic diversity; if False,
                           rely on temperature variation alone (simpler, faster)
    dry_run              : skip LLM calls and return canned stubs
    """

    DEFAULT_RULES_TEMPERATURES = [0.2, 0.7, 1.0]
    DEFAULT_CONFIG_TEMPERATURE = 0.0

    def __init__(
        self,
        llm: BaseLLMClient,
        kb_dir: str = "agents/knowledge-base",
        topo_dir: str = "",
        num_candidates: int = 3,
        rules_temperatures: list[float] | None = None,
        config_temperature: float | None = None,
        use_strategies: bool = False,
        dry_run: bool = False,
    ) -> None:
        self._llm = llm
        self._kb_dir = kb_dir
        self._topo_dir = topo_dir
        self._num_candidates = num_candidates
        self._rules_temperatures = rules_temperatures or self.DEFAULT_RULES_TEMPERATURES[:num_candidates]
        self._config_temperature = config_temperature if config_temperature is not None \
            else self.DEFAULT_CONFIG_TEMPERATURE
        self._use_strategies = use_strategies
        self._dry_run = dry_run

        if len(self._rules_temperatures) != self._num_candidates:
            raise ValueError(
                f"rules_temperatures length ({len(self._rules_temperatures)}) "
                f"must equal num_candidates ({self._num_candidates})"
            )

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        clarified_intent: str,
        results_dir: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Generate N candidate rule sets and compile each into OSPF configs.

        Returns
        -------
        list[dict[str, str]]
            One dict per candidate:
              "__rules__"          : rules JSON string
              "decision_summary.txt": LLM reasoning text
              <RouterName>         : full Cisco IOS config text (all routers)
        """
        kb = load_knowledge_base(self._kb_dir)
        base_topo = load_topo_configs(self._topo_dir) if self._topo_dir else {}
        topo_doc = generate_topology_doc(base_topo) if base_topo else ""

        candidates: list[dict[str, str]] = []
        for i, temp in enumerate(self._rules_temperatures, start=1):
            strategy = (
                CANDIDATE_STRATEGIES[i - 1]
                if self._use_strategies and i - 1 < len(CANDIDATE_STRATEGIES)
                else None
            )
            strategy_name = (
                strategy["rules_hint"].split("\n")[0].replace("## Extraction Strategy:", "").strip()
                if strategy else "temperature-only"
            )

            print(f"\n[Generator] Candidate {i}/{self._num_candidates} — "
                  f"extracting rules (temperature={temp}, strategy={strategy_name})…")

            rules_json = self._extract_rules(clarified_intent, temperature=temp, strategy=strategy)
            print(f"[Generator] Rules:\n{rules_json}")

            print(f"[Generator] Candidate {i}/{self._num_candidates} — "
                  f"generating OSPF configs (temperature={self._config_temperature}, "
                  f"strategy={strategy_name})…")

            llm_output = self._generate_configs(
                rules_json, clarified_intent, kb, topo_doc, strategy=strategy
            )

            candidate = dict(base_topo)
            candidate.update(llm_output)
            candidate["__rules__"] = rules_json

            candidates.append(candidate)

            if results_dir:
                self._save_candidate(candidate, llm_output, rules_json, results_dir, i)

        return candidates

    # ── Rules extractor ───────────────────────────────────────────────────────

    def _extract_rules(
        self,
        clarified_intent: str,
        temperature: float,
        strategy: dict[str, str] | None = None,
    ) -> str:
        """Convert clarified intent to structured ground-truth rules JSON string."""
        if self._dry_run:
            # Vary the dry-run output slightly so candidates differ
            lb = {} if temperature < 0.5 else {"(athens,100.0.29.0/24)": 2}
            return json.dumps({
                "reachability": {"athens": ["100.0.29.0/24"]},
                "waypoint": {"(athens,100.0.29.0/24)": ["london"]},
                "loadbalancing": lb,
            }, indent=2)

        if clarified_intent.startswith("MORE_QUESTIONS"):
            raise ValueError(
                "Cannot extract rules: the clarified intent was not fully resolved "
                "(it still contains unanswered clarification questions). "
                "Increase --max-rounds or provide a more specific intent."
            )

        user_content = clarified_intent
        if strategy and strategy.get("rules_hint"):
            user_content = clarified_intent + "\n\n" + strategy["rules_hint"].strip()

        messages = [
            Message(role="system", content=RULES_EXTRACTOR_SYSTEM),
            Message(role="user", content=user_content),
        ]
        raw = self._llm.complete(messages, temperature=temperature, max_tokens=1024)

        # Strip any accidental markdown fences the model may add
        raw = re.sub(r"^```[^\n]*\n", "", raw.strip())
        raw = re.sub(r"\n```$", "", raw.strip())

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Rules extractor returned invalid JSON: {exc}\n---\n{raw}"
            ) from exc

        # Detect unfilled template placeholders (e.g. "<destination_subnet>")
        raw_str = json.dumps(parsed)
        if re.search(r"<[^>]+>", raw_str):
            raise ValueError(
                "Rules extractor returned a template with unfilled placeholders. "
                "The clarified intent is likely still ambiguous.\n"
                f"Partial rules:\n{raw}"
            )

        return raw

    # ── Config generator ──────────────────────────────────────────────────────

    def _generate_configs(
        self,
        rules_json: str,
        clarified_intent: str,
        kb: dict[str, str],
        topo_doc: str,
        strategy: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Compile rules JSON + topology into OSPF router configs."""
        if self._dry_run:
            return self._parse_response(self._dry_run_response())

        user_message = self._build_config_prompt(
            rules_json, clarified_intent, kb, topo_doc, strategy=strategy
        )
        messages = [
            Message(role="system", content=GENERATOR_SYSTEM),
            Message(role="user", content=user_message),
        ]
        response = self._llm.complete(
            messages, temperature=self._config_temperature, max_tokens=4096
        )
        return self._parse_response(response)

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_config_prompt(
        self,
        rules_json: str,
        clarified_intent: str,
        kb: dict[str, str],
        topo_doc: str,
        strategy: dict[str, str] | None = None,
    ) -> str:
        parts = []
        if topo_doc:
            parts.append(topo_doc)
            parts.append("")
        if kb:
            parts.append("## Knowledge Base\n")
            for fname, content in kb.items():
                parts.append(f"### {fname}\n\n{content}\n")
        parts.append("---\n")
        parts.append("## Routing Rules\n")
        parts.append("```json")
        parts.append(rules_json)
        parts.append("```\n")
        parts.append(f"## Clarified Intent\n\n{clarified_intent}")
        if strategy and strategy.get("config_hint"):
            parts.append("")
            parts.append(strategy["config_hint"].strip())
        return "\n".join(parts)

    # ── Response parser ───────────────────────────────────────────────────────

    def _parse_response(self, response: str) -> dict[str, str]:
        """Parse ### RouterName\\n```...``` blocks. Returns {name: content}."""
        result: dict[str, str] = {}
        pattern = re.compile(
            r"^###\s+([^\n]+?)\s*\n```[^\n]*\n(.*?)```",
            re.MULTILINE | re.DOTALL,
        )
        for match in pattern.finditer(response):
            name   = match.group(1).strip()
            config = match.group(2).strip()
            result[name] = config
        return result

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _save_candidate(
        self,
        candidate: dict[str, str],
        llm_output: dict[str, str],
        rules_json: str,
        results_dir: str,
        index: int,
    ) -> None:
        candidate_dir = os.path.join(results_dir, "candidates", f"candidate_{index}")
        configs_dir   = os.path.join(candidate_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)

        _meta = {"decision_summary.txt", "__rules__"}
        modified_routers = [k for k in llm_output if k not in _meta]

        with open(os.path.join(candidate_dir, "rules.json"), "w", encoding="utf-8") as f:
            f.write(rules_json + "\n")

        for name, content in candidate.items():
            if name == "__rules__":
                continue
            if name == "decision_summary.txt":
                path = os.path.join(candidate_dir, name)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content + "\n")
            else:
                path = os.path.join(configs_dir, f"{name}.cfg")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                    if not content.endswith("\n"):
                        f.write("\n")

        print(f"[Generator] Candidate {index} saved → {candidate_dir}/")
        if modified_routers:
            print(f"            Modified routers: {', '.join(sorted(modified_routers))}")
        else:
            print("            (no routers modified — base topology only)")

    # ── Dry-run stub ──────────────────────────────────────────────────────────

    def _dry_run_response(self) -> str:
        return """\
### Athens
```
!
version 15.2
!
service timestamps debug datetime msec
!
service timestamps log datetime msec
!
boot-start-marker
!
boot-end-marker
!
no aaa new-model
!
ip cef
!
no ipv6 cef
!
multilink bundle-name authenticated
!
ip forward-protocol nd
!
no ip http server
!
no ip http secure-server
!
ip bgp-community new-format
!
hostname Athens
!
!
interface Fa0/0
 ip address 100.0.5.1 255.255.255.0
 description "To 100.0.5.0/24"
 speed auto
 duplex auto
!
interface Fa1/0
 ip address 10.0.0.26 255.255.255.254
 ip ospf 100 area 0
 ip ospf cost 1
 description "To Sofia"
 speed auto
 duplex auto
!
interface Fa0/1
 ip address 10.0.0.13 255.255.255.254
 ip ospf 100 area 0
 ip ospf cost 1
 description "To Istanbul"
 speed auto
 duplex auto
!
!
router ospf 100
 redistribute connected subnets
 maximum-paths 32
!
!
control-plane
!
!
line con 0
 stopbits 1
line aux 0
 stopbits 1
line vty 0 4
 login
!
end
```

### decision_summary.txt
```
[dry-run]
Routers modified: Athens

Reasoning:
Athens connects to Sofia (10.0.0.26) and Istanbul (10.0.0.13).
Both links kept at cost 1 to allow ECMP through both neighbours.
London owns 100.0.29.0/24 — traffic from Athens must ultimately reach London.

Limitations: None (dry-run placeholder).
```
"""
