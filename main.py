"""
NetSocratic — CLI entry point
=============================
Usage:
  python main.py --intent "Ensure connectivity between athens and our servers…"
  python main.py --intent-file path/to/intent.txt
  python main.py --intent "…" --model gpt-4o-mini
  python main.py --intent "…" --dry-run
"""

import argparse
import sys

import config
from agents.clarification_agent import ClarificationAgent
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
        "--results-dir",
        default=config.RESULTS_DIR,
        help=f"Directory for output files (default: {config.RESULTS_DIR}).",
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
        # In dry-run mode we still construct the client but the agent will not call it
        llm = OpenAIClient(api_key="dry-run", model=args.model)

    interactor = TerminalInteractor()

    agent = ClarificationAgent(
        llm=llm,
        interactor=interactor,
        results_dir=args.results_dir,
        max_rounds=args.max_rounds,
        dry_run=args.dry_run,
    )

    clarified = agent.run(intent)

    print("\n" + "=" * 60)
    print("FINAL CLARIFIED INTENT")
    print("=" * 60)
    print(clarified)
    print(f"\nResults saved to: {agent._results_dir}/")


if __name__ == "__main__":
    main()
