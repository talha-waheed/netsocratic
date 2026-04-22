"""
Clarification Agent
===================
Takes a vague network configuration intent, asks the operator targeted
follow-up questions interactively, and returns a fully clarified intent
that the Generator Agent can act on without guessing.

Two LLM passes per round:
  1. CLARIFY_SYSTEM  – generate the next batch of clarifying questions.
  2. SUFFICIENCY_SYSTEM – decide whether all ambiguity is resolved and,
     if so, synthesise the final clarified intent.
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime

from interaction.terminal import TerminalInteractor
from llm.base import BaseLLMClient, Message


# ── Prompts ───────────────────────────────────────────────────────────────────

CLARIFY_SYSTEM = """\
You are Clarify Agent, a system-role assistant whose only goal is to remove ambiguity before execution.

Mission:
- Given a user intent, identify missing, conflicting, or underspecified details.
- Ask 1 or more clarification questions (as many as needed, as few as possible) so another agent can execute safely and correctly.

The downstream Generator Agent needs ALL three of the following to be fully unambiguous:
1. REACHABILITY   : exact source node and exact destination subnet (CIDR notation) for every pair.
2. WAYPOINTING    : whether each waypoint is mandatory (must always traverse) or a preference
                    (prefer if possible), and the exact waypoint router name.
3. LOAD BALANCING : an exact integer number of equal-cost paths for every destination pair.

Core behavior:
1. Do NOT solve or implement the task.
2. Ask concise, concrete, answerable questions.
3. If the request is already clear enough, state "No clarification needed."
4. Prefer questions that resolve multiple ambiguities at once.
5. Do not ask for stylistic preferences unless they affect execution.
6. Avoid repeating questions for information already provided.
7. Keep output short and skimmable.

Output format:
Return only:

Clarification Questions
1. <question> (Why: <impact>)
2. <question> (Why: <impact>)
"""

SUFFICIENCY_SYSTEM = """\
You are an intent evaluator for network configuration. Given the original vague intent
and all clarification Q&A so far, determine whether the intent is now fully specified.

The downstream Generator Agent needs ALL of the following to be unambiguous:
1. REACHABILITY : exact source node and exact destination subnet (CIDR notation) for every pair.
2. WAYPOINTING  : whether each waypoint is mandatory (must always traverse) or a preference
                  (prefer if possible), and the exact waypoint router name.
3. LOAD BALANCING: an exact integer number of equal-cost paths for every destination pair.

Respond with EXACTLY one of the two formats below — nothing else.

────────────────────────────────────────────────
Format A  (all ambiguity resolved):

CLARIFIED
<the full clarified intent written as precise, imperative sentences — no vague phrases>
────────────────────────────────────────────────
Format B  (critical information still missing):

MORE_QUESTIONS
Clarification Questions
1. <question> (Why: <impact>)
2. <question> (Why: <impact>)
────────────────────────────────────────────────
"""


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class QARound:
    round_number: int
    questions: list[str]
    answers: list[str] = field(default_factory=list)


# ── Agent ─────────────────────────────────────────────────────────────────────

class ClarificationAgent:
    """
    Interactively clarifies a vague network intent until it is fully specified.

    Parameters
    ----------
    llm          : any BaseLLMClient implementation (OpenAI, Anthropic, …)
    interactor   : TerminalInteractor (or any compatible replacement)
    results_dir  : directory where output files are written
    max_rounds   : hard cap on clarification rounds
    dry_run      : if True, skip LLM calls and return canned responses
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        interactor: TerminalInteractor,
        results_dir: str = "results",
        max_rounds: int = 5,
        dry_run: bool = False,
    ) -> None:
        self._llm = llm
        self._interactor = interactor
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._results_dir = os.path.join(results_dir, timestamp)
        self._max_rounds = max_rounds
        self._dry_run = dry_run
        os.makedirs(self._results_dir, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, vague_intent: str) -> str:
        """
        Run the interactive clarification loop.
        Returns the final clarified intent string.
        """
        self._interactor.display_banner("NetSocratic — Clarification Agent")
        self._interactor.display(
            "I will ask you a few questions to resolve any ambiguity in your intent.\n"
            "Please answer each question as precisely as possible."
        )

        self._save_intent(vague_intent)
        history: list[QARound] = []

        for round_num in range(1, self._max_rounds + 1):
            # ── Step 1: generate questions ────────────────────────────────────
            rendered_prompt, questions = self._generate_questions(vague_intent, history, round_num)
            self._save_prompt(rendered_prompt)

            if not questions:
                self._interactor.display("\nNo further clarification needed — synthesising intent…")
                clarified = self._synthesise_clarified_intent(vague_intent, history)
                self._save_clarified(clarified)
                return clarified

            # ── Step 2: show questions to operator ────────────────────────────
            self._interactor.display_section(
                f"Round {round_num} — Clarification Questions",
                f"({len(questions)} question(s))",
            )
            self._save_questions(round_num, questions)

            answers = self._interactor.ask_questions(questions)
            round_record = QARound(round_number=round_num, questions=questions, answers=answers)
            history.append(round_record)
            self._save_answers(round_num, questions, answers)

            # ── Step 3: check whether intent is now fully specified ───────────
            is_done, result = self._check_sufficiency(vague_intent, history)

            if is_done:
                clarified = result
                self._interactor.display_section("Clarified Intent", clarified)
                self._save_clarified(clarified)
                return clarified

            # result is the next batch of questions embedded in MORE_QUESTIONS text;
            # we surface them in the next round via _generate_questions which will
            # receive the updated history and naturally produce follow-ups.
            self._interactor.display(
                "\nThank you. A few more questions are needed to fully resolve the intent…"
            )

        # Reached max rounds — synthesise best-effort clarified intent
        self._interactor.display(
            f"\n[Warning] Reached the maximum of {self._max_rounds} clarification rounds. "
            "Synthesising the best-effort clarified intent from answers so far."
        )
        clarified = self._synthesise_clarified_intent(vague_intent, history)
        self._interactor.display_section("Clarified Intent (best-effort)", clarified)
        self._save_clarified(clarified)
        return clarified

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _generate_questions(
        self,
        vague_intent: str,
        history: list[QARound],
        round_num: int,
    ) -> tuple[str, list[str]]:
        """
        Ask the LLM for the next batch of clarifying questions.
        Returns (rendered_user_message, list_of_question_strings).
        """
        user_content = self._build_clarify_user_message(vague_intent, history, round_num)

        if self._dry_run:
            response = (
                "Clarification Questions\n"
                '1. What exact destination subnet should athens reach? (Why: determines reachability target)\n'
                '2. Is the London waypoint mandatory or a preference? (Why: affects path constraints)\n'
                '3. How many load-balanced paths are required for each source? (Why: specifies redundancy level)\n'
            )
        else:
            messages = [
                Message(role="system", content=CLARIFY_SYSTEM),
                Message(role="user", content=user_content),
            ]
            response = self._llm.complete(messages, temperature=0.3)

        questions = self._parse_questions(response)
        return user_content, questions

    def _check_sufficiency(
        self,
        vague_intent: str,
        history: list[QARound],
    ) -> tuple[bool, str]:
        """
        Ask the LLM whether the accumulated Q&A is enough to fully specify the intent.
        Returns (True, clarified_intent_text) or (False, more_questions_text).
        """
        user_content = self._build_sufficiency_user_message(vague_intent, history)

        if self._dry_run:
            # In dry-run, declare sufficient after the first round
            if history and history[-1].answers:
                return True, (
                    "Athens must be able to reach 100.0.29.0/24. "
                    "Traffic from Athens should prefer paths that go through London. "
                    "The routing must provide exactly 3 load-balanced paths.\n\n"
                    "[dry-run placeholder — replace with real LLM call]"
                )
            return False, ""

        messages = [
            Message(role="system", content=SUFFICIENCY_SYSTEM),
            Message(role="user", content=user_content),
        ]
        response = self._llm.complete(messages, temperature=0.2)
        return self._parse_sufficiency(response)

    def _synthesise_clarified_intent(
        self,
        vague_intent: str,
        history: list[QARound],
    ) -> str:
        """
        Called when the LLM said 'No clarification needed' or we hit max rounds.
        Uses the sufficiency prompt in 'force-clarify' mode to produce the final intent.
        """
        user_content = self._build_sufficiency_user_message(vague_intent, history)
        user_content += (
            "\n\nThe clarification phase has ended. "
            "Produce the best possible CLARIFIED intent from the information gathered."
        )

        if self._dry_run:
            return "[dry-run] Clarified intent would be generated here."

        messages = [
            Message(role="system", content=SUFFICIENCY_SYSTEM),
            Message(role="user", content=user_content),
        ]
        response = self._llm.complete(messages, temperature=0.2)
        _, clarified = self._parse_sufficiency(response)
        return clarified

    # ── Message builders ──────────────────────────────────────────────────────

    def _build_clarify_user_message(
        self,
        vague_intent: str,
        history: list[QARound],
        round_num: int,
    ) -> str:
        parts = [f"Original intent:\n{vague_intent}"]

        if history:
            parts.append("\nClarification history so far:")
            for r in history:
                parts.append(f"\n--- Round {r.round_number} ---")
                for q, a in zip(r.questions, r.answers):
                    parts.append(f"Q: {q}")
                    parts.append(f"A: {a}")

        if round_num > 1:
            parts.append(
                "\nDo not repeat questions already answered above. "
                "Ask only about remaining ambiguities."
            )

        return "\n".join(parts)

    def _build_sufficiency_user_message(
        self,
        vague_intent: str,
        history: list[QARound],
    ) -> str:
        parts = [f"Original intent:\n{vague_intent}"]

        if history:
            parts.append("\nClarification Q&A:")
            for r in history:
                for q, a in zip(r.questions, r.answers):
                    parts.append(f"Q: {q}")
                    parts.append(f"A: {a}")
        else:
            parts.append("\nNo clarification Q&A yet.")

        return "\n".join(parts)

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_questions(self, response: str) -> list[str]:
        """
        Extract question strings from an LLM response that contains a numbered list.
        Returns empty list if the response signals no clarification is needed.
        """
        if re.search(r"no clarification needed", response, re.IGNORECASE):
            return []

        questions: list[str] = []
        # Match lines like "1. question text (Why: ...)" or "1) question text"
        for match in re.finditer(r"^\s*\d+[\.\)]\s+(.+)", response, re.MULTILINE):
            questions.append(match.group(1).strip())

        return questions

    def _parse_sufficiency(self, response: str) -> tuple[bool, str]:
        """
        Parse the SUFFICIENCY_SYSTEM response.
        Returns (True, clarified_intent) or (False, remainder_of_response).
        """
        response = response.strip()

        if response.startswith("CLARIFIED"):
            clarified = response[len("CLARIFIED"):].strip()
            return True, clarified

        if response.startswith("MORE_QUESTIONS"):
            return False, response

        # Fallback: if neither keyword is present, treat as clarified
        return True, response

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _save_intent(self, vague_intent: str) -> None:
        path = os.path.join(self._results_dir, "intent_original.txt")
        self._write(path, vague_intent + "\n")

    def _save_prompt(self, rendered_prompt: str) -> None:
        path = os.path.join(self._results_dir, "clarify_prompt.txt")
        full_content = f"=== CLARIFY AGENT — SYSTEM PROMPT ===\n\n{CLARIFY_SYSTEM}\n\n=== USER MESSAGE ===\n\n{rendered_prompt}\n"
        self._write(path, full_content)

    def _save_questions(self, round_num: int, questions: list[str]) -> None:
        path = os.path.join(self._results_dir, f"questions_round_{round_num}.txt")
        lines = [f"Round {round_num} — Clarification Questions\n"]
        for i, q in enumerate(questions, 1):
            lines.append(f"{i}. {q}")
        self._write(path, "\n".join(lines) + "\n")

    def _save_answers(self, round_num: int, questions: list[str], answers: list[str]) -> None:
        path = os.path.join(self._results_dir, f"answers_round_{round_num}.txt")
        lines = [f"Round {round_num} — Q&A\n"]
        for q, a in zip(questions, answers):
            lines.append(f"Q: {q}")
            lines.append(f"A: {a}\n")
        self._write(path, "\n".join(lines))

    def _save_clarified(self, clarified: str) -> None:
        path = os.path.join(self._results_dir, "clarified_intent.txt")
        self._write(path, clarified + "\n")

    def _write(self, path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
