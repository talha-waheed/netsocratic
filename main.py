"""
NetSocratic — CLI entry point
=============================
Usage:
  python main.py --intent "Ensure connectivity between athens and our servers…"
  python main.py --intent-file path/to/intent.txt
  python main.py --intent "…" --model gpt-4o-mini
  python main.py --intent "…" --dry-run
  python main.py --intent "…" --skip-generator      # clarification only
  python main.py --intent "…" --skip-selector       # clarification + generation only
"""

import argparse
import os
import sys

import config
from agents.clarification_agent import ClarificationAgent
from agents.generator_agent import GeneratorAgent
from agents.selection_agent import SelectionAgent
from interaction.terminal import TerminalInteractor
from llm.openai_client import OpenAIClient


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="netsocratic",
        description="NetSocratic — intent disambiguation for network configuration",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--intent",
        metavar="TEXT",
        help="Vague network intent as a string.",
    )
    group.add_argument(
        "--intent-file",
        metavar="PATH",
        help="Path to a text file containing the vague intent.",
    )
    p.add_argument(
        "--model",
        default=config.OPENAI_MODEL,
        help=f"OpenAI model to use (default: {config.OPENAI_MODEL}).",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=config.MAX_CLARIFY_ROUNDS,
        help=f"Maximum clarification rounds (default: {config.MAX_CLARIFY_ROUNDS}).",
    )
    p.add_argument(
        "--max-questions",
        type=int,
        default=config.MAX_QUESTIONS_PER_ROUND,
        help=f"Maximum clarification questions per round (default: {config.MAX_QUESTIONS_PER_ROUND}).",
    )
    p.add_argument(
        "--num-candidates",
        type=int,
        default=config.NUM_CANDIDATES,
        help=f"Number of generator candidates (default: {config.NUM_CANDIDATES}).",
    )
    p.add_argument(
        "--kb-dir",
        default=config.KB_DIR,
        help=f"Knowledge-base directory (default: {config.KB_DIR}).",
    )
    p.add_argument(
        "--topo-dir",
        default=config.TOPO_DIR,
        help=f"Base topology directory with one .cfg per router (default: {config.TOPO_DIR}).",
    )
    p.add_argument(
        "--results-dir",
        default=config.RESULTS_DIR,
        help=f"Base directory for output files (default: {config.RESULTS_DIR}).",
    )
    p.add_argument(
        "--batfish-script-dir",
        default="batfish",
        help="Directory containing diff_analysis.py / diff_advanced.py (default: batfish).",
    )
    p.add_argument(
        "--max-recovery-rounds",
        type=int,
        default=2,
        help="Max times the full pipeline restarts after a failed selection (default: 2).",
    )
    p.add_argument(
        "--skip-generator",
        action="store_true",
        help="Run clarification only; do not invoke the Generator or Selection Agent.",
    )
    p.add_argument(
        "--skip-selector",
        action="store_true",
        help="Run clarification + generation only; do not invoke the Selection Agent.",
    )
    p.add_argument(
        "--no-strategies",
        action="store_true",
        help="Disable per-candidate strategy hints; use temperature variation only.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; use canned responses for testing without an API key.",
    )
    return p


def load_intent(args: argparse.Namespace) -> str:
    if args.intent:
        return args.intent.strip()
    with open(args.intent_file, encoding="utf-8") as f:
        return f.read().strip()


def _read_runtime_context(run_dir: str) -> str | None:
    """Load runtime context written by SelectionAgent, if it exists."""
    path = os.path.join(run_dir, "selection", "runtime_context.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return None


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    intent = load_intent(args)

    if not args.dry_run:
        if not config.OPENAI_API_KEY:
            print(
                "Error: OPENAI_API_KEY is not set.\n"
                "Copy .env.example to .env and add your key, or set the environment variable.\n"
                "Use --dry-run to run without an API key.",
                file=sys.stderr,
            )
            sys.exit(1)
        llm = OpenAIClient(api_key=config.OPENAI_API_KEY, model=args.model)
    else:
        llm = OpenAIClient(api_key="dry-run", model=args.model)

    interactor = TerminalInteractor()

    runtime_context: str | None = None
    winner = None

    for recovery_round in range(1, args.max_recovery_rounds + 2):
        # ── Phase 1: Clarification ─────────────────────────────────────────
        if recovery_round > 1:
            print("\n" + "=" * 60)
            print(f"RECOVERY ROUND {recovery_round - 1} — restarting clarification")
            print("=" * 60)

        clarify_agent = ClarificationAgent(
            llm=llm,
            interactor=interactor,
            results_dir=args.results_dir,
            max_rounds=args.max_rounds,
            max_questions_per_round=args.max_questions,
            dry_run=args.dry_run,
        )

        clarified = clarify_agent.run(intent, runtime_context=runtime_context)
        run_dir = clarify_agent._results_dir  # timestamped directory for this run

        print("\n" + "=" * 60)
        print("CLARIFIED INTENT")
        print("=" * 60)
        print(clarified)

        if args.skip_generator:
            print(f"\nResults saved to: {run_dir}/")
            return

        # ── Phase 2: Generation ────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("GENERATOR AGENT — synthesising candidate configurations")
        print("=" * 60)

        gen_agent = GeneratorAgent(
            llm=llm,
            kb_dir=args.kb_dir,
            topo_dir=args.topo_dir,
            num_candidates=args.num_candidates,
            use_strategies=not args.no_strategies,
            dry_run=args.dry_run,
        )

        candidates = gen_agent.run(clarified, results_dir=run_dir)

        print(f"\n[Generator] {len(candidates)} candidate(s) generated.")
        for i, candidate in enumerate(candidates, start=1):
            router_names = [k for k in candidate if k not in ("decision_summary.txt", "__rules__")]
            print(f"  Candidate {i}: {len(router_names)} router(s) configured — "
                  f"{', '.join(sorted(router_names))}")

        if args.skip_selector:
            print(f"\nAll results saved to: {run_dir}/")
            return

        # ── Phase 3: Selection ─────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SELECTION AGENT — pruning to a single Network Specification")
        print("=" * 60)

        # Collect clarification Q&A for the runtime context (if recovery is needed)
        prior_qa = _collect_clarification_qa(run_dir)

        sel_agent = SelectionAgent(
            llm=llm,
            interactor=interactor,
            batfish_script_dir=args.batfish_script_dir,
            kb_dir=args.kb_dir,
            topo_dir=args.topo_dir,
            dry_run=args.dry_run,
        )

        winner, further_clarified = sel_agent.run(
            candidates,
            clarified,
            results_dir=run_dir,
            prior_clarification_qa=prior_qa,
        )

        if winner is not None:
            if further_clarified and further_clarified != clarified:
                print("\n" + "=" * 60)
                print("FURTHER CLARIFIED INTENT (after selection)")
                print("=" * 60)
                print(further_clarified)
            break  # success

        # Recovery: load context and loop
        runtime_context = _read_runtime_context(run_dir)
        if recovery_round > args.max_recovery_rounds:
            print(
                f"\nCould not converge after {args.max_recovery_rounds} recovery round(s).\n"
                "Review the runtime context files in the results directory.",
                file=sys.stderr,
            )
            sys.exit(2)

    print(f"\nNetwork Specification saved to: {run_dir}/selection/winner/")
    print(f"All results saved to: {run_dir}/")


def _collect_clarification_qa(run_dir: str) -> str:
    """Read all clarification Q&A files from a run directory into a single string."""
    lines = []
    i = 1
    while True:
        q_path = os.path.join(run_dir, f"questions_round_{i}.txt")
        a_path = os.path.join(run_dir, f"answers_round_{i}.txt")
        if not os.path.exists(q_path):
            break
        lines.append(open(q_path).read())
        if os.path.exists(a_path):
            lines.append(open(a_path).read())
        i += 1
    return "\n".join(lines)


if __name__ == "__main__":
    main()
