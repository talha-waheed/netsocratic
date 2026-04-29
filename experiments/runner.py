"""
Experiment Runner
=================
Runs the NetSocratic pipeline for every row in a CSV dataset, replacing
the human operator with an LLM that answers based on the correct formal
specification.

CSV format expected:
  row_id, ..., Correct Formal Specification, Ambiguous High Level Intent, ...

Usage
-----
  python experiments/runner.py --csv complex_example_fuzzed_dataset.csv
  python experiments/runner.py --csv ... --limit 10
  python experiments/runner.py --csv ... --row-ids 1425,1426
  python experiments/runner.py --csv ... --skip-selector
  python experiments/runner.py --csv ... --no-strategies
  python experiments/runner.py --csv ... --dry-run

Output
------
  results/experiments/<run_timestamp>/
  ├── summary.csv             ← one row per experiment: metrics + generated rules
  ├── summary.json            ← same data in JSON (more detail)
  └── <row_id>/
      └── <pipeline_timestamp>/
          ├── clarified_intent.txt
          ├── rules.json
          ├── candidates/
          │   ├── candidate_1/rules.json
          │   ├── candidate_2/rules.json
          │   └── candidate_3/rules.json
          └── selection/winner/rules.json  (if selection ran)
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.clarification_agent import ClarificationAgent
from agents.generator_agent import (
    RULES_EXTRACTOR_SYSTEM,
    GeneratorAgent,
)
from agents.selection_agent import SelectionAgent
from interaction.llm_operator import LLMOperator
from llm.base import Message
from llm.openai_client import OpenAIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _normalise_spec(spec: dict) -> dict:
    """Normalise a rules dict for comparison: lowercase keys, sorted lists."""
    out: dict = {}
    reach = spec.get("reachability") or {}
    out["reachability"] = {k.lower(): sorted(v) for k, v in reach.items()}
    wp = spec.get("waypoint") or {}
    out["waypoint"] = {k.lower(): sorted(w.lower() for w in v) for k, v in wp.items()}
    lb = spec.get("loadbalancing") or {}
    out["loadbalancing"] = {k.lower(): int(v) for k, v in lb.items()}
    return out


def evaluate(generated_rules_str: str, correct_spec_str: str) -> dict:
    """
    Compare generated rules JSON to the correct formal specification.

    Returns a dict with per-dimension match flags and detailed diagnostics.
    """
    try:
        gen = _normalise_spec(json.loads(generated_rules_str))
    except Exception:
        return {"error": "could not parse generated rules", "exact_match": False,
                "reachability_match": False, "waypoint_match": False, "loadbalancing_match": False}

    try:
        ref = _normalise_spec(json.loads(correct_spec_str))
    except Exception:
        return {"error": "could not parse correct spec", "exact_match": False,
                "reachability_match": False, "waypoint_match": False, "loadbalancing_match": False}

    def reach_pairs(r: dict) -> set:
        return {(src, pfx) for src, pfxs in r.items() for pfx in pfxs}

    gen_reach = reach_pairs(gen["reachability"])
    ref_reach = reach_pairs(ref["reachability"])
    missing_reach = sorted(ref_reach - gen_reach)
    extra_reach   = sorted(gen_reach - ref_reach)
    reach_ok = (missing_reach == [] and extra_reach == [])

    missing_wp = sorted(set(ref["waypoint"]) - set(gen["waypoint"]))
    wrong_wp   = sorted(
        k for k in ref["waypoint"]
        if k in gen["waypoint"] and gen["waypoint"][k] != ref["waypoint"][k]
    )
    extra_wp = sorted(set(gen["waypoint"]) - set(ref["waypoint"]))
    wp_ok = (missing_wp == [] and wrong_wp == [] and extra_wp == [])

    missing_lb = sorted(set(ref["loadbalancing"]) - set(gen["loadbalancing"]))
    wrong_lb   = sorted(
        k for k in ref["loadbalancing"]
        if k in gen["loadbalancing"] and gen["loadbalancing"][k] != ref["loadbalancing"][k]
    )
    extra_lb = sorted(set(gen["loadbalancing"]) - set(ref["loadbalancing"]))
    lb_ok = (missing_lb == [] and wrong_lb == [] and extra_lb == [])

    return {
        "exact_match":          reach_ok and wp_ok and lb_ok,
        "reachability_match":   reach_ok,
        "waypoint_match":       wp_ok,
        "loadbalancing_match":  lb_ok,
        "missing_reachability": [list(p) for p in missing_reach],
        "extra_reachability":   [list(p) for p in extra_reach],
        "missing_waypoints":    missing_wp,
        "wrong_waypoints":      wrong_wp,
        "extra_waypoints":      extra_wp,
        "missing_lb":           missing_lb,
        "wrong_lb":             wrong_lb,
        "extra_lb":             extra_lb,
    }


# ── Post-clarification rules extraction ───────────────────────────────────────

def extract_rules_neutral(llm, clarified_intent: str) -> str:
    """
    Extract rules from the clarified intent at temperature=0.0 with no strategy hint.
    Used to measure accuracy attributable to the clarification phase alone.
    """
    import re as _re
    messages = [
        Message(role="system", content=RULES_EXTRACTOR_SYSTEM),
        Message(role="user", content=clarified_intent),
    ]
    raw = llm.complete(messages, temperature=0.0, max_tokens=1024)
    raw = _re.sub(r"^```[^\n]*\n", "", raw.strip())
    raw = _re.sub(r"\n```$", "", raw.strip())
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        return "{}"


# ── Post-run stats helpers ─────────────────────────────────────────────────────

def _count_clarify_questions(run_dir: str) -> tuple[int, int]:
    """
    Parse saved questions_round_N.txt files.
    Returns (total_questions, total_rounds).
    """
    total_q = 0
    total_rounds = 0
    i = 1
    while True:
        path = os.path.join(run_dir, f"questions_round_{i}.txt")
        if not os.path.exists(path):
            break
        with open(path) as f:
            content = f.read()
        # Count numbered list items
        total_q += len(re.findall(r"^\d+\.", content, re.MULTILINE))
        total_rounds += 1
        i += 1
    return total_q, total_rounds


def _parse_selection_log(run_dir: str) -> dict:
    """
    Parse selection_log.txt to extract EC count and selection round count.
    Returns dict with n_ecs and n_selection_rounds.
    """
    log_path = os.path.join(run_dir, "selection", "selection_log.txt")
    if not os.path.exists(log_path):
        return {"n_ecs": None, "n_selection_rounds": 0}

    with open(log_path) as f:
        content = f.read()

    n_ecs = None
    m = re.search(r"→\s*(\d+)\s*survivors", content)
    if m:
        n_ecs = int(m.group(1))

    # Count "Round N  Q:" lines
    n_selection_rounds = len(re.findall(r"^Round \d+\s+Q:", content, re.MULTILINE))

    return {"n_ecs": n_ecs, "n_selection_rounds": n_selection_rounds}


# ── Per-row pipeline run ───────────────────────────────────────────────────────

def run_experiment(
    row: dict,
    llm: OpenAIClient,
    base_results_dir: str,
    max_rounds: int,
    max_questions_per_round: int,
    num_candidates: int,
    kb_dir: str,
    topo_dir: str,
    batfish_script_dir: str,
    max_recovery_rounds: int,
    use_strategies: bool,
    skip_selector: bool,
    dry_run: bool,
    verbose_operator: bool,
) -> dict:
    """
    Run the full pipeline for one CSV row.
    Returns an experiment result dict.
    """
    row_id       = row["row_id"]
    intent       = row["Ambiguous High Level Intent"].strip()
    correct_spec = row["Correct Formal Specification"].strip()

    row_dir = os.path.join(base_results_dir, str(row_id))
    os.makedirs(row_dir, exist_ok=True)

    operator = LLMOperator(
        llm=llm,
        correct_spec=correct_spec,
        temperature=0.0,
        verbose=verbose_operator,
    )

    result: dict = {
        "row_id":          row_id,
        "intent":          intent,
        "correct_spec":    correct_spec,
        "status":          "error",
        "error":           None,
        # Clarification
        "clarified_intent":         None,
        "post_clarify_rules":       None,
        "eval_post_clarify":        None,
        "n_clarify_questions":      None,
        "n_clarify_rounds":         None,
        "time_clarify_s":           None,
        # Generation
        "candidate_rules":          {},
        "eval_per_candidate":       {},
        "time_generate_s":          None,
        # Selection
        "winner_rules":                  None,
        "eval_winner":                   None,
        "further_clarified_intent":      None,
        "eval_further_clarified":        None,
        "n_ecs":                         None,
        "n_selection_rounds":            None,
        "time_select_s":                 None,
        # Verification (re-generation from further_clarified_intent)
        "n_verification_ecs":            None,
        "time_verify_s":                 None,
        # Misc
        "pipeline_dir":             None,
    }

    runtime_context: str | None = None

    for recovery_round in range(1, max_recovery_rounds + 2):
        try:
            # ── Phase 1: Clarification ────────────────────────────────────────
            t0 = time.perf_counter()
            clarify_agent = ClarificationAgent(
                llm=llm,
                interactor=operator,
                results_dir=row_dir,
                max_rounds=max_rounds,
                max_questions_per_round=max_questions_per_round,
                dry_run=dry_run,
            )
            clarified = clarify_agent.run(intent, runtime_context=runtime_context)
            result["time_clarify_s"] = round(time.perf_counter() - t0, 2)

            run_dir = clarify_agent._results_dir
            result["pipeline_dir"] = run_dir
            result["clarified_intent"] = clarified

            # Question counts from saved files
            n_q, n_rounds = _count_clarify_questions(run_dir)
            result["n_clarify_questions"] = n_q
            result["n_clarify_rounds"]    = n_rounds

            # Post-clarification accuracy: rules extracted directly at temp=0.0
            if not dry_run:
                post_rules = extract_rules_neutral(llm, clarified)
                result["post_clarify_rules"] = post_rules
                result["eval_post_clarify"]  = evaluate(post_rules, correct_spec)

            # ── Phase 2: Generation ───────────────────────────────────────────
            t0 = time.perf_counter()
            gen_agent = GeneratorAgent(
                llm=llm,
                kb_dir=kb_dir,
                topo_dir=topo_dir,
                num_candidates=num_candidates,
                use_strategies=use_strategies,
                dry_run=dry_run,
            )
            candidates = gen_agent.run(clarified, results_dir=run_dir)
            result["time_generate_s"] = round(time.perf_counter() - t0, 2)

            for i, cand in enumerate(candidates, start=1):
                rules_str = cand.get("__rules__", "")
                cand_name = f"candidate_{i}"
                result["candidate_rules"][cand_name]    = rules_str
                result["eval_per_candidate"][cand_name] = evaluate(rules_str, correct_spec)

            if skip_selector:
                result["status"] = "done_no_selection"
                break

            # ── Phase 3: Selection ────────────────────────────────────────────
            t0 = time.perf_counter()
            sel_agent = SelectionAgent(
                llm=llm,
                interactor=operator,
                batfish_script_dir=batfish_script_dir,
                kb_dir=kb_dir,
                topo_dir=topo_dir,
                dry_run=dry_run,
            )
            winner, further_clarified = sel_agent.run(
                candidates,
                clarified,
                results_dir=run_dir,
                prior_clarification_qa="",
            )
            result["time_select_s"] = round(time.perf_counter() - t0, 2)

            sel_stats = _parse_selection_log(run_dir)
            result["n_ecs"]              = sel_stats["n_ecs"]
            result["n_selection_rounds"] = sel_stats["n_selection_rounds"]

            if winner is not None:
                winner_rules = winner.get("__rules__", "")
                result["winner_rules"]             = winner_rules
                result["eval_winner"]              = evaluate(winner_rules, correct_spec)
                result["further_clarified_intent"] = further_clarified
                if further_clarified and not dry_run:
                    fc_rules = extract_rules_neutral(llm, further_clarified)
                    result["eval_further_clarified"] = evaluate(fc_rules, correct_spec)

                # ── Phase 4: Verification ─────────────────────────────────────
                # Only run when the winner is known-correct (exact match).
                # Generates 3 temperature-only candidates from the further-clarified
                # intent and counts ECs across those regenerated configs only.
                winner_is_correct = (result.get("eval_winner") or {}).get("exact_match")
                if winner_is_correct and not dry_run:
                    t0 = time.perf_counter()
                    verif_dir = os.path.join(run_dir, "verification")
                    verif_gen = GeneratorAgent(
                        llm=llm,
                        kb_dir=kb_dir,
                        topo_dir=topo_dir,
                        num_candidates=3,
                        use_strategies=False,
                        dry_run=dry_run,
                    )
                    verif_gen.run(further_clarified, results_dir=verif_dir)
                    verif_cands_dir = os.path.join(verif_dir, "candidates")

                    n_verif_ecs = sel_agent.count_ecs(
                        n=3,
                        cands_dir=verif_cands_dir,
                        results_dir=verif_dir,
                    )
                    result["n_verification_ecs"] = n_verif_ecs
                    result["time_verify_s"]      = round(time.perf_counter() - t0, 2)

                result["status"] = "done"
                break

            # Recovery
            ctx_path = os.path.join(run_dir, "selection", "runtime_context.txt")
            runtime_context = open(ctx_path).read() if os.path.exists(ctx_path) else None
            if recovery_round > max_recovery_rounds:
                result["status"] = "no_convergence"
                break

        except Exception as exc:
            result["error"] = traceback.format_exc()
            log.error("row_id=%s  error: %s", row_id, exc)
            break

    return result


# ── Summary writers ────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "row_id", "status", "error",
    # Post-clarification accuracy
    "post_clarify_exact", "post_clarify_reach", "post_clarify_wp", "post_clarify_lb",
    # Per-candidate accuracy (best-of-N)
    "exact_match_any_candidate",
    "c1_exact", "c1_reach", "c1_wp", "c1_lb",
    "c2_exact", "c2_reach", "c2_wp", "c2_lb",
    "c3_exact", "c3_reach", "c3_wp", "c3_lb",
    # Winner accuracy
    "winner_exact", "winner_reach", "winner_wp", "winner_lb",
    # Further clarified intent accuracy
    "fc_exact", "fc_reach", "fc_wp", "fc_lb",
    # Counts
    "n_clarify_rounds", "n_clarify_questions", "n_ecs", "n_selection_rounds",
    "n_verification_ecs",
    # Timing (seconds)
    "time_clarify_s", "time_generate_s", "time_select_s", "time_verify_s",
    # Context
    "clarified_intent", "pipeline_dir",
]


def _flat_eval(prefix: str, ev: dict | None) -> dict:
    if not ev:
        return {f"{prefix}_exact": None, f"{prefix}_reach": None,
                f"{prefix}_wp": None, f"{prefix}_lb": None}
    return {
        f"{prefix}_exact": ev.get("exact_match"),
        f"{prefix}_reach": ev.get("reachability_match"),
        f"{prefix}_wp":    ev.get("waypoint_match"),
        f"{prefix}_lb":    ev.get("loadbalancing_match"),
    }


def _to_csv_row(result: dict) -> dict:
    evals = result.get("eval_per_candidate", {})
    row: dict = {
        "row_id":        result["row_id"],
        "status":        result["status"],
        "error":         (result.get("error") or "")[:200],  # truncate long tracebacks
        "exact_match_any_candidate": any(
            e.get("exact_match") for e in evals.values()
        ),
        "n_clarify_rounds":    result.get("n_clarify_rounds"),
        "n_clarify_questions": result.get("n_clarify_questions"),
        "n_ecs":               result.get("n_ecs"),
        "n_selection_rounds":  result.get("n_selection_rounds"),
        "n_verification_ecs":  result.get("n_verification_ecs"),
        "time_clarify_s":      result.get("time_clarify_s"),
        "time_generate_s":     result.get("time_generate_s"),
        "time_select_s":       result.get("time_select_s"),
        "time_verify_s":       result.get("time_verify_s"),
        "clarified_intent": (result.get("clarified_intent") or "").replace("\n", " "),
        "pipeline_dir":     result.get("pipeline_dir") or "",
    }
    row.update(_flat_eval("post_clarify", result.get("eval_post_clarify")))
    for i in range(1, 4):
        row.update(_flat_eval(f"c{i}", evals.get(f"candidate_{i}")))
    row.update(_flat_eval("winner", result.get("eval_winner")))
    row.update(_flat_eval("fc", result.get("eval_further_clarified")))
    return row


def write_summary(results: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(_to_csv_row(r))

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("Summary written to %s", out_dir)


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="experiments/runner.py",
        description="Run NetSocratic experiments against a CSV dataset.",
    )
    p.add_argument("--csv", required=True, metavar="PATH",
                   help="CSV with 'Ambiguous High Level Intent' and "
                        "'Correct Formal Specification' columns.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many rows (useful for quick tests).")
    p.add_argument("--row-ids", metavar="IDS", default=None,
                   help="Comma-separated row_id values to run (e.g. 1425,1426).")
    p.add_argument("--model", default=config.OPENAI_MODEL,
                   help=f"OpenAI model (default: {config.OPENAI_MODEL}).")
    p.add_argument("--max-rounds", type=int, default=config.MAX_CLARIFY_ROUNDS,
                   help=f"Max clarification rounds (default: {config.MAX_CLARIFY_ROUNDS}).")
    p.add_argument("--max-questions", type=int, default=config.MAX_QUESTIONS_PER_ROUND,
                   help=f"Max clarification questions per round (default: {config.MAX_QUESTIONS_PER_ROUND}).")
    p.add_argument("--num-candidates", type=int, default=config.NUM_CANDIDATES,
                   help=f"Candidates per run (default: {config.NUM_CANDIDATES}).")
    p.add_argument("--kb-dir",  default=config.KB_DIR)
    p.add_argument("--topo-dir", default=config.TOPO_DIR)
    p.add_argument("--results-dir",
                   default=os.path.join(config.RESULTS_DIR, "experiments"),
                   help="Root output directory (default: results/experiments).")
    p.add_argument("--batfish-script-dir", default="batfish")
    p.add_argument("--max-recovery-rounds", type=int, default=2)
    p.add_argument("--no-strategies", action="store_true",
                   help="Disable per-candidate strategy hints (temperature-only diversity).")
    p.add_argument("--skip-selector", action="store_true",
                   help="Skip Batfish selection; evaluate generation candidates only.")
    p.add_argument("--verbose-operator", action="store_true",
                   help="Print every LLM operator Q&A to stdout.")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip all LLM/Batfish calls; use canned responses.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.dry_run and not config.OPENAI_API_KEY:
        print(
            "Error: OPENAI_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key, or use --dry-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    llm = OpenAIClient(api_key=config.OPENAI_API_KEY or "dry-run", model=args.model)

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.row_ids:
        wanted = set(s.strip() for s in args.row_ids.split(","))
        rows = [r for r in rows if r["row_id"] in wanted]

    if args.limit:
        rows = rows[:args.limit]

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(args.results_dir, run_timestamp)
    os.makedirs(out_dir, exist_ok=True)

    log.info("Experiment run: %d row(s) → %s", len(rows), out_dir)

    results: list[dict] = []
    experiment_start = time.perf_counter()

    for idx, row in enumerate(rows, start=1):
        row_id = row.get("row_id", f"row{idx}")
        log.info("[%d/%d] row_id=%s", idx, len(rows), row_id)

        result = run_experiment(
            row=row,
            llm=llm,
            base_results_dir=out_dir,
            max_rounds=args.max_rounds,
            max_questions_per_round=args.max_questions,
            num_candidates=args.num_candidates,
            kb_dir=args.kb_dir,
            topo_dir=args.topo_dir,
            batfish_script_dir=args.batfish_script_dir,
            max_recovery_rounds=args.max_recovery_rounds,
            use_strategies=not args.no_strategies,
            skip_selector=args.skip_selector,
            dry_run=args.dry_run,
            verbose_operator=args.verbose_operator,
        )
        results.append(result)

        evals = result.get("eval_per_candidate", {})
        any_exact = any(e.get("exact_match") for e in evals.values())
        pc = result.get("eval_post_clarify") or {}
        log.info(
            "  row_id=%-6s  status=%-20s  post_clarify_exact=%-5s  any_cand_exact=%s"
            "  clarify=%.1fs  gen=%.1fs",
            row_id, result["status"],
            pc.get("exact_match"), any_exact,
            result.get("time_clarify_s") or 0,
            result.get("time_generate_s") or 0,
        )

        write_summary(results, out_dir)

    total_s = time.perf_counter() - experiment_start
    log.info("Done. %d rows in %.1fs (%.1fs/row)", len(results), total_s,
             total_s / len(results) if results else 0)
    _print_aggregate(results)


def _print_aggregate(results: list[dict]) -> None:
    total = len(results)
    if total == 0:
        return

    done = sum(1 for r in results if r["status"] in ("done", "done_no_selection"))

    def _pct(n: int) -> str:
        return f"{n:3d}  ({n/total:.0%})"

    # Post-clarification accuracy
    pc_exact = sum(1 for r in results if (r.get("eval_post_clarify") or {}).get("exact_match"))
    pc_reach = sum(1 for r in results if (r.get("eval_post_clarify") or {}).get("reachability_match"))
    pc_wp    = sum(1 for r in results if (r.get("eval_post_clarify") or {}).get("waypoint_match"))
    pc_lb    = sum(1 for r in results if (r.get("eval_post_clarify") or {}).get("loadbalancing_match"))

    # Best-of-N candidate accuracy
    bn_exact = sum(1 for r in results
                   if any(e.get("exact_match")
                          for e in r.get("eval_per_candidate", {}).values()))
    bn_reach = sum(1 for r in results
                   if any(e.get("reachability_match")
                          for e in r.get("eval_per_candidate", {}).values()))
    bn_wp    = sum(1 for r in results
                   if any(e.get("waypoint_match")
                          for e in r.get("eval_per_candidate", {}).values()))
    bn_lb    = sum(1 for r in results
                   if any(e.get("loadbalancing_match")
                          for e in r.get("eval_per_candidate", {}).values()))

    # Winner accuracy (only for rows that completed selection)
    selected = [r for r in results if r.get("eval_winner")]
    w_exact = sum(1 for r in selected if (r.get("eval_winner") or {}).get("exact_match"))
    w_reach = sum(1 for r in selected if (r.get("eval_winner") or {}).get("reachability_match"))
    w_wp    = sum(1 for r in selected if (r.get("eval_winner") or {}).get("waypoint_match"))
    w_lb    = sum(1 for r in selected if (r.get("eval_winner") or {}).get("loadbalancing_match"))

    # Further-clarified accuracy (subset of selected rows)
    fc_rows = [r for r in results if r.get("eval_further_clarified")]
    fc_exact = sum(1 for r in fc_rows if (r.get("eval_further_clarified") or {}).get("exact_match"))
    fc_reach = sum(1 for r in fc_rows if (r.get("eval_further_clarified") or {}).get("reachability_match"))
    fc_wp    = sum(1 for r in fc_rows if (r.get("eval_further_clarified") or {}).get("waypoint_match"))
    fc_lb    = sum(1 for r in fc_rows if (r.get("eval_further_clarified") or {}).get("loadbalancing_match"))

    # Counts & timing
    def _avg(vals):
        v = [x for x in vals if x is not None]
        return f"{sum(v)/len(v):.1f}" if v else "n/a"

    avg_clarify_q   = _avg([r.get("n_clarify_questions") for r in results])
    avg_clarify_s   = _avg([r.get("time_clarify_s") for r in results])
    avg_generate_s  = _avg([r.get("time_generate_s") for r in results])
    avg_select_s    = _avg([r.get("time_select_s") for r in results])
    avg_verify_s    = _avg([r.get("time_verify_s") for r in results])
    avg_ecs         = _avg([r.get("n_ecs") for r in results])
    avg_sel_rounds  = _avg([r.get("n_selection_rounds") for r in results])

    verif_rows = [r for r in results if r.get("n_verification_ecs") is not None]
    avg_verif_ecs    = _avg([r.get("n_verification_ecs") for r in verif_rows])
    verif_perfect    = sum(1 for r in verif_rows if r.get("n_verification_ecs") == 1)

    w = 76
    print("\n" + "=" * w)
    print("  Experiment Results")
    print("=" * w)
    print(f"  Total rows processed       : {total}")
    print(f"  Pipeline completed         : {_pct(done)}")
    print()
    print(f"  {'Metric':<38}  {'Post-clarify':>12}  {'Best-of-N':>9}  {'Winner':>6}  {'Furt.Clar.':>10}")
    print(f"  {'-'*38}  {'-'*12}  {'-'*9}  {'-'*6}  {'-'*10}")
    for label, pc_v, bn_v, w_v, fc_v in [
        ("Exact match",    pc_exact, bn_exact, w_exact, fc_exact),
        ("Reachability",   pc_reach, bn_reach, w_reach, fc_reach),
        ("Waypoint",       pc_wp,    bn_wp,    w_wp,    fc_wp),
        ("Load balancing", pc_lb,    bn_lb,    w_lb,    fc_lb),
    ]:
        wn  = f"{w_v}/{len(selected)}" if selected else "n/a"
        fcn = f"{fc_v}/{len(fc_rows)}" if fc_rows else "n/a"
        print(f"  {label:<38}  {pc_v:>5}/{total:<6}  {bn_v:>4}/{total:<4}  {wn:>6}  {fcn:>10}")
    print()
    print(f"  Avg clarification questions: {avg_clarify_q}")
    print(f"  Avg ECs after Batfish      : {avg_ecs}")
    print(f"  Avg selection rounds       : {avg_sel_rounds}")
    if verif_rows:
        print(f"  Avg verification ECs       : {avg_verif_ecs}  "
              f"(perfect=1: {verif_perfect}/{len(verif_rows)})")
    print()

    def _fmt_s(v: str) -> str:
        return f"{v}s" if v != "n/a" else v

    print(f"  Avg time — clarify         : {_fmt_s(avg_clarify_s)}")
    print(f"  Avg time — generate        : {_fmt_s(avg_generate_s)}")
    print(f"  Avg time — select          : {_fmt_s(avg_select_s)}")
    if verif_rows:
        print(f"  Avg time — verify          : {_fmt_s(avg_verify_s)}")
    print("=" * w)


if __name__ == "__main__":
    main()
