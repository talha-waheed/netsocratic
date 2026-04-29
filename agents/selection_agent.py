"""
Selection Agent
===============
Uses Batfish differences between N candidate configurations to ask targeted
follow-up questions, then synthesises a new Network Specification from the
operator's answers.

Sub-steps
---------
1. EC-based pruning (Batfish, deterministic)
   All pairwise candidate comparisons are run via the existing batfish scripts:
     - diff_analysis.py   → reachability differences
     - diff_advanced.py   → waypointing + load-balancing differences
   Candidates with identical behaviour across all three EC dimensions are
   collapsed to one representative, following the same approach as run_all_diffs.py.

2. Follow-up Q&A (LLM + operator)
   For each surviving pair, the LLM converts the Batfish-detected behavioural
   difference into a targeted plain-English question.  Answers are accumulated
   as additional constraints instead of forcing selection of one existing
   candidate.

3. Synthesis
   The clarified intent plus follow-up Q&A are merged into a refined intent.
   The Generator Agent then produces one fresh candidate from that refined
   intent, allowing the final configuration to combine dimensions that were
   split across the original candidates.

4. Recovery
   Two conditions trigger recovery (pass runtime context back to the
   Clarification Agent):
     a. User rejection  — operator says "try again", "none of these", etc.
     b. Max rounds      — Q&A loop exhausts max_rounds with >1 candidate alive.

All intermediate artefacts are written to results_dir/selection/.
"""

import os
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations

from agents.generator_agent import GeneratorAgent
from interaction.terminal import TerminalInteractor
from llm.base import BaseLLMClient, Message


# ── Prompts ───────────────────────────────────────────────────────────────────

DISTINGUISHING_QUESTION_SYSTEM = """\
You are a network policy expert helping a network operator clarify what their original
network intent meant. Two candidate configurations have been generated from that intent
because the intent was genuinely ambiguous — both candidates are valid interpretations
of what was written. Given the clarified intent and a concrete behavioural difference
detected by Batfish, write exactly ONE plain-English question that clarifies what the
original intent meant, using the detected behavioural difference to identify the specific
point of ambiguity to resolve.

Rules:
1. Ask about observable routing behaviour only — which routers traffic passes through,
   how many paths exist, whether traffic reaches its destination.
2. One sentence only — no preamble, no explanation.
3. Frame as a question about what the original intent required, not about what the
   operator would like to change or add. For example: "Did your original intent require
   traffic to always pass through London, or only to prefer London when available?" is
   better than "Would you prefer traffic to always pass through London?"
4. Use the router names and subnets from the intent; never mention OSPF costs or
   interface names.
"""

FURTHER_CLARIFY_SYSTEM = """\
You are a network intent synthesiser. You will be given:
1. A clarified network intent (already resolved from an initial vague intent).
2. Selection Q&A — questions asked to distinguish between candidate configurations,
   and the operator's answers, which reveal additional preferences.

Your task: produce a single, precise, updated clarified intent that incorporates all
information from both the original clarified intent and the selection Q&A.

Rules:
- Preserve ALL constraints already in the clarified intent — never drop or weaken them.
- From selection Q&A: incorporate ONLY concrete routing constraints such as specific
  router names, exact ECMP path counts, or mandatory/preferred waypoint requirements.
- If an answer says a constraint was "not required", "not needed", or should not be
  required, omit that constraint entirely. Do NOT convert absence of a requirement
  into a negative policy such as "must not be reachable".
- Do NOT turn a waypoint router into a new traffic source. A waypoint answer for
  "traffic from A to P via W" only constrains A→P; it does not create W→P reachability
  or load-balancing requirements.
- Rewrite the result as explicit per-flow policy statements. For every source→prefix
  pair, name the source router and destination prefix directly in each reachability,
  waypoint, and load-balancing sentence.
- Do NOT use inherited or vague phrases such as "same requirements", "other subnets",
  "redundancy", "reliability", "if possible", "consider", "includes", or "available
  paths" in the final output. Expand them into concrete per-flow constraints or omit
  them if the Q&A did not resolve them.
- For waypoint constraints, use the exact wording:
  "Traffic from <source> to <prefix> must pass through <waypoint>."
- For load-balancing constraints, use the exact wording:
  "Traffic from <source> to <prefix> must be load-balanced across exactly <N>
  equal-cost paths."
- If a source→prefix pair has no resolved waypoint or no resolved exact path count,
  do not invent one.
- Do NOT incorporate meta-level evaluation criteria from selection questions, such as
  "permissive vs strict", "unintended traffic", "all traffic reaches destination",
  "strict routing", or similar framing — these describe evaluation dimensions, not
  routing policy.
- Write concise, imperative sentences.
- Output ONLY the updated intent text — no preamble, no explanation.
"""


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RuntimeContext:
    """Packages everything from a failed selection pass for the Clarification Agent."""
    prior_clarification_qa: str
    candidate_summaries: list[str]
    batfish_findings: str
    selection_qa: list[tuple[str, str]] = field(default_factory=list)
    rejection_reason: str | None = None

    def render(self) -> str:
        lines = [
            "=== RUNTIME CONTEXT (previous attempt) ===",
            "",
            "Prior clarification Q&A:",
            self.prior_clarification_qa or "(none)",
            "",
            "Candidates that were generated:",
        ]
        for i, summary in enumerate(self.candidate_summaries, 1):
            lines.append(f"Candidate {i}:\n{summary}")
        lines += [
            "",
            "Batfish behavioural analysis:",
            self.batfish_findings or "(Batfish not available or no differences found)",
            "",
        ]
        if self.selection_qa:
            lines.append("Selection Q&A (distinguishing questions asked and answered):")
            for q, a in self.selection_qa:
                lines.append(f"  Q: {q}")
                lines.append(f"  A: {a}")
            lines.append("")
        reason = self.rejection_reason or "Max rounds reached without converging"
        lines.append(f"Reason for recovery: {reason}")
        lines.append("==========================================")
        return "\n".join(lines)


# ── Agent ─────────────────────────────────────────────────────────────────────

class SelectionAgent:
    """
    Produces a corrected Network Specification from candidate differences.

    Parameters
    ----------
    llm               : LLM client for distinguishing-question generation
    interactor        : shared TerminalInteractor for operator Q&A
    batfish_script_dir: directory containing diff_analysis.py / diff_advanced.py
    kb_dir            : knowledge-base directory passed to the synthesis generator
    topo_dir          : base topology directory passed to the synthesis generator
    max_rounds        : max distinguishing-question rounds before triggering recovery
    auto_start_batfish: run "docker start <container>" once if Batfish is down
    batfish_container : Docker container name for the Batfish service
    dry_run           : skip Batfish + LLM; return first candidate immediately
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        interactor: TerminalInteractor,
        batfish_script_dir: str = "batfish",
        kb_dir: str = "agents/knowledge-base",
        topo_dir: str = "",
        max_rounds: int = 10,
        auto_start_batfish: bool = True,
        batfish_container: str = "batfish",
        dry_run: bool = False,
    ) -> None:
        self._llm = llm
        self._interactor = interactor
        self._script_dir = batfish_script_dir
        self._kb_dir = kb_dir
        self._topo_dir = topo_dir
        self._max_rounds = max_rounds
        self._auto_start_batfish = auto_start_batfish
        self._batfish_container = batfish_container
        self._dry_run = dry_run
        self._batfish_start_attempted = False

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        candidates: list[dict[str, str]],
        clarified_intent: str,
        results_dir: str,
        prior_clarification_qa: str = "",
    ) -> tuple[dict[str, str], str] | tuple[None, None]:
        """
        Synthesize the configuration matching the operator's intent.

        Parameters
        ----------
        candidates             : list of {RouterName: config_text, ...} dicts
        clarified_intent       : output of the Clarification Agent
        results_dir            : run directory (e.g. results/2026-04-22_03-36-57)
        prior_clarification_qa : clarification Q&A text from this run (for runtime context)

        Returns
        -------
        (winner_dict, further_clarified_intent) on success, or (None, None) to
        trigger a recovery loop.  winner_dict is a freshly generated config when
        Batfish follow-up Q&A revealed additional constraints.
        """
        sel_dir = os.path.join(results_dir, "selection")
        bat_dir = os.path.join(sel_dir, "batfish")
        os.makedirs(bat_dir, exist_ok=True)

        self._interactor.display_banner("NetSocratic — Selection Agent")

        n = len(candidates)
        cand_names = [f"candidate_{i + 1}" for i in range(n)]
        cands_dir = os.path.join(results_dir, "candidates")
        selection_qa: list[tuple[str, str]] = []
        log_lines: list[str] = []

        # ── Dry run: skip everything, return first candidate ──────────────────
        if self._dry_run:
            self._interactor.display("[Selection] dry-run mode — returning candidate 1.")
            self._save_winner(candidates[0], sel_dir)
            return candidates[0], clarified_intent

        # ── Step 1: run all pairwise Batfish diffs ────────────────────────────
        self._interactor.display(f"\n[Selection] Running Batfish on {n} candidate(s)…")
        pairwise = self._run_all_diffs(cands_dir, cand_names, bat_dir, log_lines)
        candidate_rules = [_parse_rules(c.get("__rules__", "")) for c in candidates]

        # ── Step 2: EC-based pruning ──────────────────────────────────────────
        survivors = self._ec_prune(n, pairwise, log_lines)
        self._interactor.display(
            f"[Selection] After EC pruning: {n} → {len(survivors)} unique candidate(s)."
        )
        log_lines.append(f"EC pruning: {n} candidates → {len(survivors)} survivors: "
                         f"{[cand_names[i] for i in survivors]}")

        if len(survivors) == 0:
            self._interactor.display("[Selection] No valid candidates found.")
            return self._do_recovery(
                candidates, cand_names, pairwise, prior_clarification_qa,
                selection_qa, "No reachable candidates after EC pruning", sel_dir, log_lines,
            ), None

        if len(survivors) == 1:
            winner = candidates[survivors[0]]
            self._save_winner(winner, sel_dir)
            self._save_log(log_lines, sel_dir)
            return winner, clarified_intent

        # ── Step 3: follow-up Q&A loop ───────────────────────────────────────
        # We ask about Batfish-observed differences without pruning candidates.
        # This lets the final generated config combine constraints that were
        # correct in different original candidates.
        asked_diffs: set[str] = set()

        for round_num in range(1, self._max_rounds + 1):
            self._interactor.display_section(
                f"Follow-up Round {round_num}",
                f"({len(survivors)} behaviourally unique candidate(s) being compared)",
            )

            diff_info = self._find_best_pair(
                survivors, cand_names, pairwise, candidate_rules, skip_diffs=asked_diffs,
            )

            if diff_info is None:
                log_lines.append("Remaining candidates are behaviourally indistinguishable "
                                 "or all distinguishable pairs already asked.")
                self._interactor.display(
                    "[Selection] No more distinguishable Batfish differences to ask about."
                )
                break

            question = self._generate_question(clarified_intent, diff_info)
            answer = self._interactor.ask(question)
            selection_qa.append((question, answer))
            log_lines.append(f"Round {round_num}  Q: {question}")
            log_lines.append(f"Round {round_num}  A: {answer}")

            # Check for explicit rejection
            if _is_rejection(answer):
                self._interactor.display(
                    "\n[Selection] Operator rejected all options — triggering recovery."
                )
                return self._do_recovery(
                    candidates, cand_names, pairwise, prior_clarification_qa,
                    selection_qa, f"User rejected all options: '{answer}'", sel_dir, log_lines,
                ), None

            asked_diffs.add(diff_info["detail_id"])

        else:
            # Loop exhausted without converging
            log_lines.append(f"Reached max {self._max_rounds} follow-up rounds.")

        further_clarified = self._synthesise_further_clarified(
            clarified_intent, selection_qa, sel_dir,
        )
        winner = self._synthesise_winner(further_clarified, log_lines)
        self._save_winner(winner, sel_dir)
        self._save_log(log_lines, sel_dir)
        self._interactor.display_section(
            "Synthesized Configuration",
            "Generated a new configuration from the follow-up answers.",
        )
        if further_clarified != clarified_intent:
            self._interactor.display_section("Further Clarified Intent", further_clarified)
        return winner, further_clarified

    # ── Batfish orchestration ─────────────────────────────────────────────────

    def _run_all_diffs(
        self,
        cands_dir: str,
        cand_names: list[str],
        bat_dir: str,
        log_lines: list[str],
    ) -> dict[tuple[int, int], dict]:
        """
        Run diff_analysis.py and diff_advanced.py for every candidate pair.
        Saves raw stdout to bat_dir.  Returns pairwise result dict keyed by (i, j).
        """
        pairwise: dict[tuple[int, int], dict] = {}

        for i, j in combinations(range(len(cand_names)), 2):
            ci, cj = cand_names[i], cand_names[j]
            print(f"[Selection]   Batfish: {ci} vs {cj}…")

            reach_out = self._run_script_with_batfish_retry(
                "diff_analysis.py", cands_dir, ci, cj, log_lines,
            )
            adv_out = self._run_script_with_batfish_retry(
                "diff_advanced.py", cands_dir, ci, cj, log_lines,
            )

            # Save raw output to files
            _write(os.path.join(bat_dir, f"{ci}_vs_{cj}_reachability.txt"), reach_out)
            _write(os.path.join(bat_dir, f"{ci}_vs_{cj}_advanced.txt"),     adv_out)

            reach_diff = _parse_reachability(reach_out)
            adv_diff   = _parse_advanced(adv_out)

            pairwise[(i, j)] = {
                "reach_diff": reach_diff,
                "adv_diff":   adv_diff,
            }

            log_lines.append(
                f"{ci} vs {cj}: reach_diff={reach_diff}, "
                f"waypoint_diff={adv_diff['waypoint_diff']}, lb_diff={adv_diff['lb_diff']}"
            )

        return pairwise

    def _run_script_with_batfish_retry(
        self,
        script_name: str,
        folder: str,
        c1: str,
        c2: str,
        log_lines: list[str],
    ) -> str:
        output = self._run_script(script_name, folder, c1, c2)
        if not _batfish_script_failed(output):
            return output

        if self._auto_start_batfish and _looks_like_batfish_down(output):
            if self._start_batfish_container(log_lines):
                output = self._run_script(script_name, folder, c1, c2)
                if not _batfish_script_failed(output):
                    return output

        raise RuntimeError(
            f"Batfish diff failed for {script_name} ({c1} vs {c2}). "
            f"Last output:\n{output[-2000:]}"
        )

    def _run_script(self, script_name: str, folder: str, c1: str, c2: str) -> str:
        """Call a batfish diff script as a subprocess; return its captured stdout+stderr."""
        cmd = [
            sys.executable,
            os.path.join(self._script_dir, script_name),
            "--folder", folder,
            "--c1", c1,
            "--c2", c2,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            output = result.stdout
            if result.stderr:
                output += f"\nERRORS:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[Script exited with status {result.returncode}]"
            return output
        except Exception as e:
            return f"[Script execution failed: {e}]"

    def _start_batfish_container(self, log_lines: list[str]) -> bool:
        if self._batfish_start_attempted:
            return False
        self._batfish_start_attempted = True

        cmd = ["docker", "start", self._batfish_container]
        log_lines.append(f"Batfish unavailable; attempting: {' '.join(cmd)}")
        self._interactor.display(
            f"[Selection] Batfish appears unavailable; starting Docker container "
            f"'{self._batfish_container}'..."
        )
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception as exc:
            log_lines.append(f"Failed to start Batfish container: {exc}")
            return False

        output = ((result.stdout or "") + (result.stderr or "")).strip()
        if result.returncode != 0:
            log_lines.append(
                f"docker start {self._batfish_container} failed "
                f"with status {result.returncode}: {output}"
            )
            return False

        log_lines.append(f"Started Batfish container: {output}")
        time.sleep(3)
        return True

    # ── EC count (verification) ───────────────────────────────────────────────

    def count_ecs(self, n: int, cands_dir: str, results_dir: str) -> int:
        """
        Run Batfish pairwise diffs on N pre-saved candidates and return the
        number of behaviourally distinct equivalence classes.

        A return value of 1 means all candidates are identical — the intent
        was fully disambiguated.  Used for verification after re-generating
        from a further-clarified intent.

        Parameters
        ----------
        n          : number of candidates (candidate_1 … candidate_N)
        cands_dir  : directory containing candidate_1/, candidate_2/, … sub-dirs
        results_dir: where to write batfish output and the EC log
        """
        if self._dry_run:
            return n  # conservative placeholder: assume all different

        if n <= 1:
            return n

        cand_names = [f"candidate_{i + 1}" for i in range(n)]
        bat_dir = os.path.join(results_dir, "batfish")
        os.makedirs(bat_dir, exist_ok=True)
        log_lines: list[str] = []

        pairwise = self._run_all_diffs(cands_dir, cand_names, bat_dir, log_lines)
        survivors = self._ec_prune(n, pairwise, log_lines)

        log_lines.append(f"Verification EC count: {len(survivors)}")
        self._save_log(log_lines, results_dir)
        return len(survivors)

    # ── EC pruning ────────────────────────────────────────────────────────────

    def _ec_prune(
        self,
        n: int,
        pairwise: dict[tuple[int, int], dict],
        log_lines: list[str],
    ) -> list[int]:
        """
        Build equivalence classes via union-find: candidates are equivalent iff
        both diff scripts report no difference.  Return one representative per class.
        """
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for (i, j), data in pairwise.items():
            if not data["reach_diff"] and not data["adv_diff"]["has_diff"]:
                union(i, j)
                log_lines.append(f"candidate_{i+1} and candidate_{j+1} are EC-equivalent → merged")

        # Keep the lowest-index representative of each class
        seen: dict[int, int] = {}
        for idx in range(n):
            root = find(idx)
            if root not in seen:
                seen[root] = idx

        return sorted(seen.values())

    # ── Distinguishing pair selection ─────────────────────────────────────────

    def _find_best_pair(
        self,
        survivors: list[int],
        cand_names: list[str],
        pairwise: dict[tuple[int, int], dict],
        candidate_rules: list[dict] | None = None,
        skip_diffs: set[str] | None = None,
    ) -> dict | None:
        """
        Scan survivor pairs for the most salient behavioural difference.
        Priority: reachability > waypointing > load-balancing.
        Entries in skip_diffs are ignored after that concrete rule/trace difference
        has been asked.
        Returns a diff_info dict or None if all are indistinguishable.
        """
        skip_diffs = skip_diffs or set()
        best: dict | None = None
        best_priority = -1

        for i, j in combinations(survivors, 2):
            key = (min(i, j), max(i, j))
            data = pairwise.get(key, {})
            adv  = data.get("adv_diff", {})
            rule_options = _rule_diff_options(
                i, j, cand_names, candidate_rules or [], data, adv,
            )

            if rule_options:
                batch_option = _batch_rule_diff_option(rule_options)
                options = ([batch_option] if batch_option else []) + rule_options
            else:
                only_i = adv.get("nodes_only_in_c1", set()) if i < j else adv.get("nodes_only_in_c2", set())
                only_j = adv.get("nodes_only_in_c2", set()) if i < j else adv.get("nodes_only_in_c1", set())
                has_concrete_waypoint_diff = adv.get("waypoint_diff") and (only_i or only_j)
                options: list[dict] = []
                if has_concrete_waypoint_diff:
                    # Waypoint enforcement causes reachability diffs in Batfish; ask
                    # about the waypoint directly.
                    options.append({
                        "priority": 3,
                        "dimension": "waypointing",
                        "detail_id": "waypointing:batfish-trace",
                        "c1_desc": f"{cand_names[i]} routes through {only_i or '(direct)'}",
                        "c2_desc": f"{cand_names[j]} routes through {only_j or '(direct)'}",
                    })
                if data.get("reach_diff"):
                    options.append({
                        "priority": 2,
                        "dimension": "reachability",
                        "detail_id": "reachability:batfish-trace",
                        "c1_desc": f"{cand_names[i]} permits some traffic that {cand_names[j]} denies",
                        "c2_desc": f"{cand_names[j]} permits some traffic that {cand_names[i]} denies",
                    })
                elif adv.get("waypoint_diff"):
                    # waypoint_diff detected but no concrete node names to distinguish;
                    # still ask, letting the LLM frame it from the clarified intent.
                    options.append({
                        "priority": 2,
                        "dimension": "waypointing",
                        "detail_id": "waypointing:batfish-trace",
                        "c1_desc": f"{cand_names[i]} routes through {only_i or '(direct)'}",
                        "c2_desc": f"{cand_names[j]} routes through {only_j or '(direct)'}",
                    })
                if adv.get("lb_diff"):
                    p_i = adv.get("paths_c1") if i < j else adv.get("paths_c2")
                    p_j = adv.get("paths_c2") if i < j else adv.get("paths_c1")
                    options.append({
                        "priority": 1,
                        "dimension": "load_balancing",
                        "detail_id": "load_balancing:batfish-trace",
                        "c1_desc": f"{cand_names[i]} provides {p_i} equal-cost path(s)",
                        "c2_desc": f"{cand_names[j]} provides {p_j} equal-cost path(s)",
                    })

            for option in options:
                if option["detail_id"] in skip_diffs:
                    continue
                if option["priority"] > best_priority:
                    best_priority = option["priority"]
                    best = {
                        "c1_orig_idx": i,
                        "c2_orig_idx": j,
                        "c1_name": cand_names[i],
                        "c2_name": cand_names[j],
                        "dimension": option["dimension"],
                        "detail_id": option["detail_id"],
                        "c1_desc": option["c1_desc"],
                        "c2_desc": option["c2_desc"],
                        "adv": adv,
                    }
                    if "question" in option:
                        best["question"] = option["question"]

        return best

    # ── Further clarified intent synthesis ────────────────────────────────────

    def _synthesise_further_clarified(
        self,
        clarified_intent: str,
        selection_qa: list[tuple[str, str]],
        sel_dir: str,
    ) -> str:
        """
        Merge the original clarified intent with selection Q&A preferences into
        a refined intent.  Saves the result to selection/further_clarified_intent.txt.
        If there is no selection Q&A, returns the clarified intent unchanged.
        """
        if not selection_qa:
            _write(
                os.path.join(sel_dir, "further_clarified_intent.txt"),
                clarified_intent + "\n",
            )
            return clarified_intent

        filtered_qa = _filter_selection_qa_for_synthesis(clarified_intent, selection_qa)
        if len(filtered_qa) != len(selection_qa):
            dropped = [qa for qa in selection_qa if qa not in filtered_qa]
            _write(
                os.path.join(sel_dir, "selection_qa_filtered.txt"),
                "\n\n".join(f"Q: {q}\nA: {a}" for q, a in filtered_qa) + "\n",
            )
            _write(
                os.path.join(sel_dir, "selection_qa_dropped.txt"),
                "\n\n".join(f"Q: {q}\nA: {a}" for q, a in dropped) + "\n",
            )
        if not filtered_qa:
            _write(
                os.path.join(sel_dir, "further_clarified_intent.txt"),
                clarified_intent + "\n",
            )
            return clarified_intent

        qa_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in filtered_qa)
        user_content = (
            f"Clarified intent:\n{clarified_intent}\n\n"
            f"Selection Q&A:\n{qa_text}"
        )
        messages = [
            Message(role="system", content=FURTHER_CLARIFY_SYSTEM),
            Message(role="user", content=user_content),
        ]
        further = self._llm.complete(messages, temperature=0.1).strip()
        _write(
            os.path.join(sel_dir, "further_clarified_intent.txt"),
            further + "\n",
        )
        return further

    # ── Distinguishing question generation ────────────────────────────────────

    def _generate_question(self, clarified_intent: str, diff_info: dict) -> str:
        """Use LLM to convert a Batfish diff into a plain-English operator question."""
        if diff_info.get("question"):
            return diff_info["question"]

        diff_desc = (
            f"Option A ({diff_info['c1_name']}): {diff_info['c1_desc']}.\n"
            f"Option B ({diff_info['c2_name']}): {diff_info['c2_desc']}."
        )
        user_msg = (
            f"Clarified intent:\n{clarified_intent}\n\n"
            f"Behavioural difference ({diff_info['dimension']}):\n{diff_desc}"
        )
        messages = [
            Message(role="system", content=DISTINGUISHING_QUESTION_SYSTEM),
            Message(role="user",   content=user_msg),
        ]
        return self._llm.complete(messages, temperature=0.3, max_tokens=128).strip()

    # ── Corrected configuration synthesis ────────────────────────────────────

    def _synthesise_winner(
        self,
        further_clarified: str,
        log_lines: list[str],
    ) -> dict[str, str]:
        """Generate one fresh candidate from the refined intent."""
        self._interactor.display(
            "[Selection] Generating corrected configuration from follow-up answers…"
        )
        gen_agent = GeneratorAgent(
            llm=self._llm,
            kb_dir=self._kb_dir,
            topo_dir=self._topo_dir,
            num_candidates=1,
            rules_temperatures=[0.0],
            use_strategies=False,
            dry_run=self._dry_run,
        )
        generated = gen_agent.run(further_clarified, results_dir=None)[0]
        log_lines.append("Synthesized corrected candidate from further clarified intent.")
        return generated

    # ── Answer classification + pruning ───────────────────────────────────────

    def _prune(
        self,
        survivors: list[int],
        rejected_idx: int,
        pairwise: dict[tuple[int, int], dict],
        log_lines: list[str],
        cand_names: list[str],
    ) -> list[int]:
        """
        Remove every survivor that is behaviourally equivalent to rejected_idx
        (i.e. has no diff with rejected_idx on any Batfish dimension).
        """
        result = []
        for s in survivors:
            if s == rejected_idx:
                log_lines.append(f"Pruned {cand_names[s]} (directly rejected)")
                continue
            key  = (min(s, rejected_idx), max(s, rejected_idx))
            data = pairwise.get(key, {})
            adv  = data.get("adv_diff", {})
            same_behavior = (not data.get("reach_diff") and not adv.get("has_diff"))
            if same_behavior:
                log_lines.append(
                    f"Pruned {cand_names[s]} (same behaviour as rejected {cand_names[rejected_idx]})"
                )
            else:
                result.append(s)
        return result

    # ── Recovery ──────────────────────────────────────────────────────────────

    def _do_recovery(
        self,
        candidates: list[dict[str, str]],
        cand_names: list[str],
        pairwise: dict[tuple[int, int], dict],
        prior_clarification_qa: str,
        selection_qa: list[tuple[str, str]],
        rejection_reason: str,
        sel_dir: str,
        log_lines: list[str],
    ) -> None:
        """Build and save the RuntimeContext, then return None to trigger recovery."""
        # Collect candidate decision summaries
        summaries = []
        for name, cand in zip(cand_names, candidates):
            summaries.append(f"[{name}]\n{cand.get('decision_summary.txt', '(no summary)')}")

        # Build Batfish findings summary
        findings_lines = []
        for (i, j), data in pairwise.items():
            adv = data.get("adv_diff", {})
            line = (
                f"{cand_names[i]} vs {cand_names[j]}: "
                f"reach_diff={data.get('reach_diff', False)}, "
                f"waypoint_diff={adv.get('waypoint_diff', False)}, "
                f"lb_diff={adv.get('lb_diff', False)}"
            )
            findings_lines.append(line)

        ctx = RuntimeContext(
            prior_clarification_qa=prior_clarification_qa,
            candidate_summaries=summaries,
            batfish_findings="\n".join(findings_lines),
            selection_qa=selection_qa,
            rejection_reason=rejection_reason,
        )

        ctx_path = os.path.join(sel_dir, "runtime_context.txt")
        _write(ctx_path, ctx.render())
        log_lines.append(f"Recovery triggered: {rejection_reason}")
        log_lines.append(f"RuntimeContext written to {ctx_path}")
        self._save_log(log_lines, sel_dir)

        self._interactor.display(
            f"\n[Selection] Runtime context written to: {ctx_path}"
        )
        return None

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _save_winner(self, winner: dict[str, str], sel_dir: str) -> None:
        winner_dir  = os.path.join(sel_dir, "winner")
        configs_dir = os.path.join(winner_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)

        for name, content in winner.items():
            if name == "__rules__":
                path = os.path.join(winner_dir, "rules.json")
            elif name == "decision_summary.txt":
                path = os.path.join(winner_dir, name)
            else:
                path = os.path.join(configs_dir, f"{name}.cfg")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
                if not content.endswith("\n"):
                    f.write("\n")

        print(f"[Selection] Winner saved to: {winner_dir}/")

    def _save_log(self, log_lines: list[str], sel_dir: str) -> None:
        _write(os.path.join(sel_dir, "selection_log.txt"), "\n".join(log_lines) + "\n")


# ── Module-level helpers ──────────────────────────────────────────────────────

def _parse_reachability(stdout: str) -> bool:
    """Returns True if diff_analysis.py detected a reachability difference."""
    return "[DIFFERENCE DETECTED]" in stdout


def _parse_advanced(stdout: str) -> dict:
    """
    Parse diff_advanced.py stdout.
    Returns dict with waypoint_diff, lb_diff, node sets, path counts.
    """
    has_div   = "[DIVERGENCE DISCOVERED]" in stdout
    wp_diff   = "Waypointing Difference" in stdout
    lb_diff   = "Load Balancing Difference" in stdout

    nodes_c1: set[str] = set()
    nodes_c2: set[str] = set()
    paths_c1: int | None = None
    paths_c2: int | None = None

    for line in stdout.splitlines():
        m = re.search(r"Candidate 1 uniquely transits:\s*\{([^}]*)\}", line)
        if m:
            nodes_c1 = {n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()}
        m = re.search(r"Candidate 2 uniquely transits:\s*\{([^}]*)\}", line)
        if m:
            nodes_c2 = {n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()}
        m = re.search(r"Candidate 1 has (\d+) paths", line)
        if m:
            paths_c1 = int(m.group(1))
        m = re.search(r"Candidate 2 has (\d+) paths", line)
        if m:
            paths_c2 = int(m.group(1))

    return {
        "has_diff":      has_div,
        "waypoint_diff": wp_diff,
        "lb_diff":       lb_diff,
        "nodes_only_in_c1": nodes_c1,
        "nodes_only_in_c2": nodes_c2,
        "paths_c1": paths_c1,
        "paths_c2": paths_c2,
    }


def _parse_rules(raw: str) -> dict:
    """Parse a candidate rules JSON string. Invalid/missing rules become empty."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _rule_diff_options(
    i: int,
    j: int,
    cand_names: list[str],
    candidate_rules: list[dict],
    data: dict,
    adv: dict,
) -> list[dict]:
    """
    Build concrete follow-up options from structured rule differences for a
    Batfish-distinguished pair. Batfish tells us the candidates differ in
    behavior; rules tell us which source/prefix constraint likely caused it.
    """
    if i >= len(candidate_rules) or j >= len(candidate_rules):
        return []
    if not data.get("reach_diff") and not adv.get("has_diff"):
        return []

    ri = candidate_rules[i] or {}
    rj = candidate_rules[j] or {}
    options: list[dict] = []

    waypoint_keys = sorted(
        set((ri.get("waypoint") or {}).keys()) |
        set((rj.get("waypoint") or {}).keys())
    )
    for key in waypoint_keys:
        vi = _normalise_rule_value((ri.get("waypoint") or {}).get(key))
        vj = _normalise_rule_value((rj.get("waypoint") or {}).get(key))
        if vi == vj:
            continue
        src, prefix = _split_rule_key(key)
        options.append({
            "priority": 4,
            "dimension": "waypointing",
            "detail_id": f"waypointing:{key}",
            "c1_desc": _waypoint_desc(cand_names[i], src, prefix, vi),
            "c2_desc": _waypoint_desc(cand_names[j], src, prefix, vj),
            "question": _waypoint_question(src, prefix, vi, vj),
        })

    lb_keys = sorted(
        set((ri.get("loadbalancing") or {}).keys()) |
        set((rj.get("loadbalancing") or {}).keys())
    )
    for key in lb_keys:
        vi = (ri.get("loadbalancing") or {}).get(key)
        vj = (rj.get("loadbalancing") or {}).get(key)
        if vi == vj:
            continue
        src, prefix = _split_rule_key(key)
        options.append({
            "priority": 5,
            "dimension": "load_balancing",
            "detail_id": f"load_balancing:{key}",
            "c1_desc": _lb_desc(cand_names[i], src, prefix, vi),
            "c2_desc": _lb_desc(cand_names[j], src, prefix, vj),
            "question": _lb_question(src, prefix, vi, vj),
        })

    reach_pairs_i = _reachability_pairs(ri)
    reach_pairs_j = _reachability_pairs(rj)
    for src, prefix in sorted(reach_pairs_i ^ reach_pairs_j):
        in_i = (src, prefix) in reach_pairs_i
        options.append({
            "priority": 2,
            "dimension": "reachability",
            "detail_id": f"reachability:({src},{prefix})",
            "c1_desc": _reach_desc(cand_names[i], src, prefix, in_i),
            "c2_desc": _reach_desc(cand_names[j], src, prefix, not in_i),
            "question": (
                f"Did your original intent require traffic from {src} to {prefix} "
                "to be reachable, or should that reachability not be required?"
            ),
        })

    return options


def _batch_rule_diff_option(options: list[dict]) -> dict | None:
    """Ask several same-dimension rule questions at once for dense intents."""
    if len(options) < 3:
        return None

    lb_options = [o for o in options if o.get("dimension") == "load_balancing"]
    if len(lb_options) >= 3:
        selected = lb_options[:8]
        flows = []
        for option in selected:
            match = re.search(r"load_balancing:\(([^,]+),([^)]+)\)", option["detail_id"])
            if match:
                flows.append(f"{match.group(1)} to {match.group(2)}")
        if flows:
            return {
                "priority": 6,
                "dimension": "load_balancing",
                "detail_id": "load_balancing:batch:" + "|".join(flows),
                "c1_desc": "candidate rules disagree on several exact ECMP counts",
                "c2_desc": "candidate rules disagree on several exact ECMP counts",
                "question": (
                    "For each of these flows, did your original intent require an "
                    "exact load-balanced path count, and if so what count: "
                    + "; ".join(flows)
                    + "? Answer as source to prefix equals count, or no exact count."
                ),
            }

    waypoint_options = [o for o in options if o.get("dimension") == "waypointing"]
    if len(waypoint_options) >= 3:
        selected = waypoint_options[:8]
        flows = []
        for option in selected:
            match = re.search(r"waypointing:\(([^,]+),([^)]+)\)", option["detail_id"])
            if match:
                flows.append(f"{match.group(1)} to {match.group(2)}")
        if flows:
            return {
                "priority": 5,
                "dimension": "waypointing",
                "detail_id": "waypointing:batch:" + "|".join(flows),
                "c1_desc": "candidate rules disagree on several waypoint requirements",
                "c2_desc": "candidate rules disagree on several waypoint requirements",
                "question": (
                    "For each of these flows, did your original intent require a "
                    "specific waypoint, and if so which router: "
                    + "; ".join(flows)
                    + "? Answer as source to prefix equals waypoint, or no waypoint."
                ),
            }

    return None


def _normalise_rule_value(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(v).lower() for v in value)
    return (str(value).lower(),)


def _split_rule_key(key: str) -> tuple[str, str]:
    m = re.match(r"\(?\s*([^,]+?)\s*,\s*([^)]+?)\s*\)?$", key)
    if not m:
        return key, "the destination prefix"
    return m.group(1).strip().lower(), m.group(2).strip()


def _reachability_pairs(rules: dict) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    reach = rules.get("reachability") or {}
    for src, prefixes in reach.items():
        if isinstance(prefixes, list):
            for prefix in prefixes:
                pairs.add((str(src).lower(), str(prefix)))
    return pairs


def _waypoint_desc(candidate_name: str, src: str, prefix: str, waypoints: tuple[str, ...]) -> str:
    if waypoints:
        return f"{candidate_name} requires {src} to {prefix} to pass through {', '.join(waypoints)}"
    return f"{candidate_name} has no waypoint requirement for {src} to {prefix}"


def _waypoint_question(src: str, prefix: str, vi: tuple[str, ...], vj: tuple[str, ...]) -> str:
    if vi and vj:
        return (
            f"Did your original intent require traffic from {src} to {prefix} "
            f"to pass through {', '.join(vi)}, through {', '.join(vj)}, "
            "or through a different waypoint?"
        )
    waypoints = vi or vj
    return (
        f"Did your original intent require traffic from {src} to {prefix} "
        f"to pass through {', '.join(waypoints)}, was no waypoint required, "
        "or was a different waypoint required?"
    )


def _lb_desc(candidate_name: str, src: str, prefix: str, count) -> str:
    if count is None:
        return f"{candidate_name} has no exact ECMP count for {src} to {prefix}"
    return f"{candidate_name} requires {count} equal-cost path(s) for {src} to {prefix}"


def _lb_question(src: str, prefix: str, vi, vj) -> str:
    if vi is not None and vj is not None:
        return (
            f"Did your original intent require traffic from {src} to {prefix} "
            f"to be load-balanced across exactly {vi} paths, exactly {vj} paths, "
            "or a different exact path count?"
        )
    count = vi if vi is not None else vj
    return (
        f"Did your original intent require traffic from {src} to {prefix} "
        f"to be load-balanced across exactly {count} paths, was no exact path count required, "
        "or was a different exact path count required?"
    )


def _reach_desc(candidate_name: str, src: str, prefix: str, required: bool) -> str:
    if required:
        return f"{candidate_name} requires reachability from {src} to {prefix}"
    return f"{candidate_name} does not require reachability from {src} to {prefix}"


def _is_rejection(answer: str) -> bool:
    """Return True if the operator's answer signals rejection of all options."""
    lower = answer.lower()
    signals = [
        "none of these", "neither", "none of the above", "start over",
        "try again", "restart", "wrong", "i don't like", "not right",
        "none of them", "go back", "redo", "incorrect", "all wrong",
    ]
    return any(s in lower for s in signals)


def _batfish_script_failed(output: str) -> bool:
    failure_markers = [
        "[Script execution failed:",
        "[Script exited with status",
        "Traceback (most recent call last):",
        "ConnectionRefusedError",
        "MaxRetryError",
        "HTTPConnectionPool(host='localhost', port=9996)",
        "An error occurred during analysis:",
    ]
    return any(marker in output for marker in failure_markers)


def _looks_like_batfish_down(output: str) -> bool:
    down_markers = [
        "ConnectionRefusedError",
        "MaxRetryError",
        "Failed to establish a new connection",
        "HTTPConnectionPool(host='localhost', port=9996)",
    ]
    return any(marker in output for marker in down_markers)


def _filter_selection_qa_for_synthesis(
    clarified_intent: str,
    selection_qa: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Keep only Q&A that can add positive constraints for source-prefix pairs in
    the clarified intent. This prevents selection artifacts like waypoint-owner
    reachability from becoming new policy.
    """
    intent_pairs = _extract_intent_pairs(clarified_intent)
    filtered: list[tuple[str, str]] = []

    for question, answer in selection_qa:
        if _answer_declines_requirement(answer):
            continue

        pair = _extract_question_pair(question)
        if pair is not None and intent_pairs and pair not in intent_pairs:
            continue

        filtered.append((question, answer))

    return filtered


def _answer_declines_requirement(answer: str) -> bool:
    lower = answer.lower()
    if re.search(r"\b\d+\s*(?:paths?|equal-cost|$)", lower):
        return False
    if re.search(r"\b(?:equals|=|is|via|through|waypoint)\s+[a-z][a-z0-9_-]*\b", lower):
        return False
    patterns = [
        r"\b(no|not|none)\s+(exact\s+)?(path\s+count|waypoint|requirement)\b",
        r"\b(no|not)\s+(required|needed|specified)\b",
        r"\bdid\s+not\s+require\b",
        r"\bdoes\s+not\s+require\b",
        r"\bshould\s+not\s+be\s+required\b",
        r"\bnot\s+be\s+reachable\b",
        r"\bshould\s+not\s+.*reachable\b",
    ]
    return any(re.search(pattern, lower) for pattern in patterns)


def _extract_question_pair(question: str) -> tuple[str, str] | None:
    lower = question.lower()
    prefix = _first_prefix(lower)
    if not prefix:
        return None

    source_patterns = [
        r"traffic\s+from\s+([a-z][a-z0-9_-]*)\s+to\s+\d+\.\d+\.\d+\.\d+/\d+",
        r"from\s+([a-z][a-z0-9_-]*)\s+to\s+\d+\.\d+\.\d+\.\d+/\d+",
    ]
    for pattern in source_patterns:
        match = re.search(pattern, lower)
        if match:
            return match.group(1), prefix
    return None


def _extract_intent_pairs(intent: str) -> set[tuple[str, str]]:
    lower = intent.lower()
    pairs: set[tuple[str, str]] = set()

    patterns = [
        r"traffic\s+(?:originating\s+)?from\s+([a-z][a-z0-9_-]*)\s+(?:can\s+)?(?:reach|to)\s+(?:the\s+)?(?:subnet\s+|target\s+subnet\s+|main\s+subnet\s+)?(\d+\.\d+\.\d+\.\d+/\d+)",
        r"traffic\s+from\s+([a-z][a-z0-9_-]*)\s+to\s+(\d+\.\d+\.\d+\.\d+/\d+)\s+must\s+pass\s+through",
        r"traffic\s+from\s+([a-z][a-z0-9_-]*)\s+to\s+(\d+\.\d+\.\d+\.\d+/\d+)\s+must\s+be\s+load-balanced",
        r"routing\s+traffic\s+between\s+([a-z][a-z0-9_-]*)\s+and\s+(?:the\s+)?(?:subnet\s+)?(\d+\.\d+\.\d+\.\d+/\d+)",
        r"connect(?:ivity\s+from)?\s+([a-z][a-z0-9_-]*)\s+(?:to\s+)?(?:the\s+)?(?:subnet\s+|target\s+subnet\s+)?(\d+\.\d+\.\d+\.\d+/\d+)",
        r"([a-z][a-z0-9_-]*)\s+(?:needs?|should|can)\s+(?:to\s+)?(?:reach|connect\s+to|access)\s+(?:the\s+)?(?:subnet\s+|target\s+subnet\s+)?(\d+\.\d+\.\d+\.\d+/\d+)",
        r"([a-z][a-z0-9_-]*)\s+and\s+(?:the\s+)?(?:subnet\s+)?(\d+\.\d+\.\d+\.\d+/\d+)\s+should\s+be\s+connected",
        r"([a-z][a-z0-9_-]*)\s+and\s+(?:the\s+)?(?:subnet\s+)?(\d+\.\d+\.\d+\.\d+/\d+)\s+(?:need|should)\s+connectivity",
        r"([a-z][a-z0-9_-]*)\s*\(\s*(\d+\.\d+\.\d+\.\d+/\d+)\s*\)\s+is\s+accessible",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            pairs.add((match.group(1), match.group(2)))

    return pairs


def _first_prefix(text: str) -> str | None:
    match = re.search(r"\d+\.\d+\.\d+\.\d+/\d+", text)
    return match.group(0) if match else None


def _classify_answer(
    answer: str,
    diff_info: dict,
) -> tuple[int | None, int | None]:
    """
    Heuristic: determine which side the operator chose from their answer.
    Returns (chosen_orig_idx, rejected_orig_idx) or (None, None) if unclear.
    """
    lower = answer.lower()
    c1_idx = diff_info["c1_orig_idx"]
    c2_idx = diff_info["c2_orig_idx"]
    dim    = diff_info["dimension"]

    if dim == "waypointing":
        adv = diff_info.get("adv", {})
        c1_has_waypoint = bool(adv.get("nodes_only_in_c1"))
        want_through = bool(re.search(r"\b(always|mandatory|must|yes|through|via|require|enforce)\b", lower))
        want_bypass  = bool(re.search(r"\b(bypass|skip|no|optional|flexible|avoid|either)\b", lower))

        # Negation: "does not require", "not always", "not through", etc. cancel want_through.
        # This handles accidental-path nodes (e.g. Batfish reports c1 transits 'brussels' due
        # to default OSPF, but the operator says "not brussels, the waypoint is X").
        if re.search(r"\b(not|no|never)\s+(always|mandatory|must|required|through|via|require)\b", lower):
            want_through = False
            want_bypass = True

        if want_through and not want_bypass:
            chosen = c1_idx if c1_has_waypoint else c2_idx
            return chosen, (c2_idx if chosen == c1_idx else c1_idx)
        if want_bypass and not want_through:
            chosen = c1_idx if not c1_has_waypoint else c2_idx
            return chosen, (c2_idx if chosen == c1_idx else c1_idx)

    elif dim == "load_balancing":
        adv  = diff_info.get("adv", {})
        p_c1 = adv.get("paths_c1")
        p_c2 = adv.get("paths_c2")
        for m in re.finditer(r"\b(\d+)\b", lower):
            n = int(m.group(1))
            if p_c1 is not None and n == p_c1:
                return c1_idx, c2_idx
            if p_c2 is not None and n == p_c2:
                return c2_idx, c1_idx

    elif dim == "reachability":
        if re.search(r"\b(yes|reachable|need|connect|require|a)\b", lower):
            return c1_idx, c2_idx  # prefer the one with more reachability (c1 in diff_analysis)

    # Fallback: "option a/b", "first/second" — bare digits omitted to avoid false
    # matches against subnet addresses like 100.0.1.0/24.
    if re.search(r"\b(option a|first|candidate 1)\b", lower):
        return c1_idx, c2_idx
    if re.search(r"\b(option b|second|candidate 2)\b", lower):
        return c2_idx, c1_idx

    return None, None


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
