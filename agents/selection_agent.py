"""
Selection Agent
===============
Narrows N candidate configurations to a single Network Specification.

Sub-steps
---------
1. EC-based pruning (Batfish, deterministic)
   All pairwise candidate comparisons are run via the existing batfish scripts:
     - diff_analysis.py   → reachability differences
     - diff_advanced.py   → waypointing + load-balancing differences
   Candidates with identical behaviour across all three EC dimensions are
   collapsed to one representative, following the same approach as run_all_diffs.py.

2. Distinguishing Q&A (LLM + operator)
   For each surviving pair, the LLM converts the Batfish-detected behavioural
   difference into a targeted plain-English question.  The operator's answer
   prunes candidates whose behaviour matches the rejected side.

3. Recovery
   Two conditions trigger recovery (pass runtime context back to the
   Clarification Agent):
     a. User rejection  — operator says "try again", "none of these", etc.
     b. Max rounds      — Q&A loop exhausts max_rounds with >1 candidate alive.

All intermediate artefacts are written to results_dir/selection/.
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from itertools import combinations

from interaction.terminal import TerminalInteractor
from llm.base import BaseLLMClient, Message


# ── Prompts ───────────────────────────────────────────────────────────────────

DISTINGUISHING_QUESTION_SYSTEM = """\
You are a network policy expert helping a network operator choose between two configurations.
Given their clarified intent and a concrete behavioural difference detected by Batfish,
write exactly ONE plain-English question that determines which behaviour they prefer.

Rules:
1. Ask about observable routing behaviour only — which routers traffic passes through,
   how many paths exist, whether traffic reaches its destination.
2. One sentence only — no preamble, no explanation.
3. Frame as a binary choice or clear preference question.
4. Use the router names and subnets from the intent; never mention OSPF costs or interface names.
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
    Prunes candidate configurations to a single Network Specification.

    Parameters
    ----------
    llm               : LLM client for distinguishing-question generation
    interactor        : shared TerminalInteractor for operator Q&A
    batfish_script_dir: directory containing diff_analysis.py / diff_advanced.py
    max_rounds        : max distinguishing-question rounds before triggering recovery
    dry_run           : skip Batfish + LLM; return first candidate immediately
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        interactor: TerminalInteractor,
        batfish_script_dir: str = "batfish",
        max_rounds: int = 10,
        dry_run: bool = False,
    ) -> None:
        self._llm = llm
        self._interactor = interactor
        self._script_dir = batfish_script_dir
        self._max_rounds = max_rounds
        self._dry_run = dry_run

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        candidates: list[dict[str, str]],
        clarified_intent: str,
        results_dir: str,
        prior_clarification_qa: str = "",
    ) -> dict[str, str] | None:
        """
        Select the single configuration matching the operator's intent.

        Parameters
        ----------
        candidates             : list of {RouterName: config_text, ...} dicts
        clarified_intent       : output of the Clarification Agent
        results_dir            : run directory (e.g. results/2026-04-22_03-36-57)
        prior_clarification_qa : clarification Q&A text from this run (for runtime context)

        Returns
        -------
        dict[str, str] | None
            The winning candidate dict, or None to trigger a recovery loop.
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
            return candidates[0]

        # ── Step 1: run all pairwise Batfish diffs ────────────────────────────
        self._interactor.display(f"\n[Selection] Running Batfish on {n} candidate(s)…")
        pairwise = self._run_all_diffs(cands_dir, cand_names, bat_dir, log_lines)

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
            )

        if len(survivors) == 1:
            winner = candidates[survivors[0]]
            self._save_winner(winner, sel_dir)
            self._save_log(log_lines, sel_dir)
            return winner

        # ── Step 3: distinguishing Q&A loop ──────────────────────────────────
        for round_num in range(1, self._max_rounds + 1):
            if len(survivors) == 1:
                break

            self._interactor.display_section(
                f"Selection Round {round_num}",
                f"({len(survivors)} candidates still in contention)",
            )

            diff_info = self._find_best_pair(survivors, cand_names, pairwise)

            if diff_info is None:
                log_lines.append("Remaining candidates are behaviourally indistinguishable.")
                self._interactor.display(
                    "[Selection] Remaining candidates are indistinguishable — selecting first."
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
                )

            # Prune based on answer
            chosen_idx, rejected_idx = _classify_answer(answer, diff_info)
            if rejected_idx is not None:
                survivors = self._prune(survivors, rejected_idx, pairwise, log_lines, cand_names)
                self._interactor.display(
                    f"[Selection] {len(survivors)} candidate(s) remain after round {round_num}."
                )
            else:
                self._interactor.display(
                    "[Selection] Answer unclear — keeping all candidates and trying again."
                )

            if not survivors:
                return self._do_recovery(
                    candidates, cand_names, pairwise, prior_clarification_qa,
                    selection_qa, "All candidates eliminated by pruning", sel_dir, log_lines,
                )

        else:
            # Loop exhausted without converging
            self._interactor.display(
                f"\n[Selection] Reached max {self._max_rounds} rounds — triggering recovery."
            )
            return self._do_recovery(
                candidates, cand_names, pairwise, prior_clarification_qa,
                selection_qa, "Max rounds reached", sel_dir, log_lines,
            )

        winner = candidates[survivors[0]]
        self._save_winner(winner, sel_dir)
        self._save_log(log_lines, sel_dir)
        self._interactor.display_section(
            "Selected Configuration",
            f"Winner: {cand_names[survivors[0]]}",
        )
        return winner

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

            reach_out = self._run_script("diff_analysis.py", cands_dir, ci, cj)
            adv_out   = self._run_script("diff_advanced.py",  cands_dir, ci, cj)

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
            return output
        except Exception as e:
            return f"[Script execution failed: {e}]"

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
    ) -> dict | None:
        """
        Scan survivor pairs for the most salient behavioural difference.
        Priority: reachability > waypointing > load-balancing.
        Returns a diff_info dict or None if all are indistinguishable.
        """
        best: dict | None = None
        best_priority = -1

        for i, j in combinations(survivors, 2):
            key = (min(i, j), max(i, j))
            data = pairwise.get(key, {})
            adv  = data.get("adv_diff", {})

            if data.get("reach_diff"):
                priority = 3
                dimension = "reachability"
                c1_desc = f"{cand_names[i]} permits some traffic that {cand_names[j]} denies"
                c2_desc = f"{cand_names[j]} permits some traffic that {cand_names[i]} denies"
            elif adv.get("waypoint_diff"):
                priority = 2
                dimension = "waypointing"
                only_i = adv.get("nodes_only_in_c1", set()) if i < j else adv.get("nodes_only_in_c2", set())
                only_j = adv.get("nodes_only_in_c2", set()) if i < j else adv.get("nodes_only_in_c1", set())
                c1_desc = f"{cand_names[i]} routes through {only_i or '(direct)'}"
                c2_desc = f"{cand_names[j]} routes through {only_j or '(direct)'}"
            elif adv.get("lb_diff"):
                priority = 1
                dimension = "load_balancing"
                p_i = adv.get("paths_c1") if i < j else adv.get("paths_c2")
                p_j = adv.get("paths_c2") if i < j else adv.get("paths_c1")
                c1_desc = f"{cand_names[i]} provides {p_i} equal-cost path(s)"
                c2_desc = f"{cand_names[j]} provides {p_j} equal-cost path(s)"
            else:
                continue

            if priority > best_priority:
                best_priority = priority
                best = {
                    "c1_orig_idx": i,
                    "c2_orig_idx": j,
                    "c1_name": cand_names[i],
                    "c2_name": cand_names[j],
                    "dimension": dimension,
                    "c1_desc": c1_desc,
                    "c2_desc": c2_desc,
                    "adv": adv,
                }

        return best

    # ── Distinguishing question generation ────────────────────────────────────

    def _generate_question(self, clarified_intent: str, diff_info: dict) -> str:
        """Use LLM to convert a Batfish diff into a plain-English operator question."""
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
            if name == "decision_summary.txt":
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


def _is_rejection(answer: str) -> bool:
    """Return True if the operator's answer signals rejection of all options."""
    lower = answer.lower()
    signals = [
        "none of these", "neither", "none of the above", "start over",
        "try again", "restart", "wrong", "i don't like", "not right",
        "none of them", "go back", "redo", "incorrect", "all wrong",
    ]
    return any(s in lower for s in signals)


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

    # Fallback: "option a" / "option b" / "first" / "second" / "1" / "2"
    if re.search(r"\b(option a|first|1|candidate 1)\b", lower):
        return c1_idx, c2_idx
    if re.search(r"\b(option b|second|2|candidate 2)\b", lower):
        return c2_idx, c1_idx

    return None, None


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
