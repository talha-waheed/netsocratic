"""
Generator Agent
===============
Takes a clarified network intent and a knowledge base, then produces N candidate
OSPF router configurations by running an LLM multiple times (varied temperatures).

Workflow per candidate
----------------------
1. LLM receives the full knowledge base + clarified intent and outputs ONLY the
   router configs that need to change (to save tokens).
2. All remaining routers are copied verbatim from `topo_dir` (the base topology).
3. The full set (changed + unchanged) is saved to
   results_dir/candidates/candidate_N/configs/<Router>.cfg
   alongside a decision_summary.txt at results_dir/candidates/candidate_N/

Single source of truth
-----------------------
The network topology is parsed directly from the `topo_dir` .cfg files at runtime
and injected into the LLM prompt as "## Network Topology". There is no static
topology.md — the topology section always reflects the actual base configs.
Other knowledge-base files (ospf_config_format.md, routing_policy.md) are still
loaded from `kb_dir` and injected as-is.
"""

import os
import re

from llm.base import BaseLLMClient, Message


# ── System prompt ──────────────────────────────────────────────────────────────

GENERATOR_SYSTEM = """\
You are Network Configuration Generator, an expert in Cisco IOS OSPF configuration.

## Your job
Given a clarified network routing intent and supporting topology/policy context,
produce the minimal set of modified Cisco IOS router configs that correctly implement
the intent. Unchanged routers will be copied from the base topology automatically —
do NOT output them.

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
   exactly N paths tie at the minimum before writing configs.

## Output format — follow exactly
### <RouterName>
```
<full Cisco IOS config, identical to base except for modified ip ospf cost lines>
```

### decision_summary.txt
```
Routers modified: <list>

Reasoning:
<step-by-step path enumeration, cost calculation, and verification that the intent
is satisfied — waypoint enforced, correct ECMP count, no unintended paths>

Limitations (if any):
<topology constraints that prevented full intent satisfaction, or "None">
```
"""


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
    Generates N candidate OSPF configurations from a clarified intent.

    Parameters
    ----------
    llm            : LLM client (any BaseLLMClient implementation)
    kb_dir         : knowledge-base directory (markdown/text files)
    topo_dir       : base topology directory containing one .cfg per router
    num_candidates : number of candidates to generate
    temperatures   : LLM temperature per run (length must equal num_candidates)
    dry_run        : skip LLM calls and return a canned stub
    """

    DEFAULT_TEMPERATURES = [0.2, 0.7, 1.0]

    def __init__(
        self,
        llm: BaseLLMClient,
        kb_dir: str = "agents/knowledge-base",
        topo_dir: str = "",
        num_candidates: int = 3,
        temperatures: list[float] | None = None,
        dry_run: bool = False,
    ) -> None:
        self._llm = llm
        self._kb_dir = kb_dir
        self._topo_dir = topo_dir
        self._num_candidates = num_candidates
        self._temperatures = temperatures or self.DEFAULT_TEMPERATURES[:num_candidates]
        self._dry_run = dry_run

        if len(self._temperatures) != self._num_candidates:
            raise ValueError(
                f"temperatures length ({len(self._temperatures)}) "
                f"must equal num_candidates ({self._num_candidates})"
            )

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        clarified_intent: str,
        results_dir: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Generate candidate configurations.

        Returns
        -------
        list[dict[str, str]]
            One dict per candidate: {RouterName: config_text, "decision_summary.txt": text}
            Includes ALL routers (LLM-modified + base copies).
        """
        kb = load_knowledge_base(self._kb_dir)
        base_topo = load_topo_configs(self._topo_dir) if self._topo_dir else {}
        topo_doc = generate_topology_doc(base_topo) if base_topo else ""
        user_message = self._build_user_message(clarified_intent, kb, topo_doc)

        candidates: list[dict[str, str]] = []
        for i, temp in enumerate(self._temperatures, start=1):
            print(f"\n[Generator] Generating candidate {i}/{self._num_candidates} "
                  f"(temperature={temp})…")

            if self._dry_run:
                response = self._dry_run_response(i)
            else:
                messages = [
                    Message(role="system", content=GENERATOR_SYSTEM),
                    Message(role="user", content=user_message),
                ]
                response = self._llm.complete(messages, temperature=temp, max_tokens=4096)

            # Parse LLM output (only changed routers + decision_summary)
            llm_output = self._parse_response(response)

            # Merge: base topology + LLM overrides
            candidate = dict(base_topo)           # start with all base configs
            candidate.update(llm_output)          # LLM-generated ones overwrite

            candidates.append(candidate)

            if results_dir:
                self._save_candidate(candidate, llm_output, results_dir, i)

        return candidates

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_user_message(
        self, clarified_intent: str, kb: dict[str, str], topo_doc: str
    ) -> str:
        parts = []
        # Topology first — generated from topo_dir, always up to date
        if topo_doc:
            parts.append(topo_doc)
            parts.append("")
        # Remaining KB files (ospf_config_format.md, routing_policy.md, …)
        if kb:
            parts.append("## Knowledge Base\n")
            for fname, content in kb.items():
                parts.append(f"### {fname}\n\n{content}\n")
        parts.append("---\n")
        parts.append(f"## Clarified Intent\n\n{clarified_intent}")
        return "\n".join(parts)

    # ── Response parser ───────────────────────────────────────────────────────

    def _parse_response(self, response: str) -> dict[str, str]:
        """Parse ### RouterName\\n```...``` blocks. Returns {name: content}."""
        result: dict[str, str] = {}
        pattern = re.compile(
            r"^###\s+(\S.*?)\s*\n```[^\n]*\n(.*?)```",
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
        results_dir: str,
        index: int,
    ) -> None:
        candidate_dir = os.path.join(results_dir, "candidates", f"candidate_{index}")
        configs_dir   = os.path.join(candidate_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)

        modified_routers = [k for k in llm_output if k != "decision_summary.txt"]

        for name, content in candidate.items():
            if name == "decision_summary.txt":
                # Saved at candidate level, not inside configs/
                path = os.path.join(candidate_dir, name)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content + "\n")
            else:
                path = os.path.join(configs_dir, f"{name}.cfg")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                    if not content.endswith("\n"):
                        f.write("\n")

        print(f"[Generator] Candidate {index} → {configs_dir}/")
        if modified_routers:
            print(f"            Modified: {', '.join(sorted(modified_routers))}")
        else:
            print("            (no routers modified — base topology only)")

    # ── Dry-run stub ──────────────────────────────────────────────────────────

    def _dry_run_response(self, candidate_index: int) -> str:
        return f"""\
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
[dry-run candidate {candidate_index}]
Routers modified: Athens

Reasoning:
Athens connects to Sofia (10.0.0.26) and Istanbul (10.0.0.13).
Both links kept at cost 1 to allow ECMP through both neighbours.
London owns 100.0.29.0/24 — traffic from Athens must ultimately reach London.

Limitations: None (dry-run placeholder).
```
"""
