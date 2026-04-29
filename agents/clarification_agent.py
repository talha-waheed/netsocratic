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

def _build_clarify_system(max_questions: int) -> str:
    return f"""\
You are Clarify Agent. Your job is to fill in concrete facts that are missing from the
intent. You do NOT interpret ambiguous language — the Generator Agent will explore
different valid interpretations of the same phrase as candidate configurations.

## What you must resolve (facts only, not interpretations)

1. REACHABILITY   — exact source router name and exact destination subnet in CIDR
                    notation (e.g. 10.0.0.0/24) for every traffic pair. Ask if a
                    source or destination is vague: "our servers", "the remote network",
                    "the target subnet", "the main subnet", "that network", etc.
2. WAYPOINTING    — the exact router hostname for any waypoint that is named vaguely
                    (e.g. "the london hub", "a transit router"). If the name is already
                    a known router identifier (e.g. "london", "kiev"), do NOT ask.
                    Do NOT ask whether a waypoint is mandatory or preferred — the
                    Generator will explore both interpretations.
3. LOAD BALANCING — an exact integer path count, but ONLY when load balancing is
                    explicitly called for AND the count is missing or unclear (e.g.
                    "split across N paths" with N not stated, "load-balance to X"
                    with no count given). Do NOT ask whether vague phrases like
                    "redundancy", "reliability", "multiple paths", or "fault tolerance"
                    mean load balancing is intended — the Generator interprets these.

## Process — think before you ask

Before writing any question, silently work through these steps:
  a. List every source→destination pair mentioned (even vaguely).
  b. For each pair: is the source a known router name? Is the destination a concrete
     CIDR prefix, or a vague label that needs resolution?
  c. For any waypoint mentioned: is the router hostname concrete and unambiguous?
  d. For any pair with an explicit load-balancing requirement: is the exact path count
     given, or is it missing?
  e. Combine gaps into as few questions as possible.
  f. Write at most {max_questions} question(s) total.

## Question quality rules

- Ask only about concrete missing values: CIDR prefixes, exact router hostnames,
  explicit integer path counts when an LB requirement is clear but the count is absent.
- Do NOT ask whether "via X", "through X", "if possible", "consider routing through X",
  "the path includes X" means mandatory or preferred. This is an interpretation the
  Generator explores.
- Do NOT ask whether vague phrases ("redundancy", "multiple paths", "fault tolerance",
  "reliability") indicate load balancing is intended. This is an interpretation the
  Generator explores.
- Never hint at the answer. Do not embed specific subnet values or router names in the
  question itself.
- Never ask about information already given in the intent or prior answers.

## Example of a GOOD question vs a VAGUE one

VAGUE : "What is the destination subnet?"
GOOD  : "What is the exact destination subnet (in CIDR notation, e.g. 10.0.0.0/24)
         that traffic from athens should reach?"

VAGUE : "What is the waypoint?"
GOOD  : "When you wrote 'route via the london hub', what is the exact router hostname
         you are referring to? (e.g. 'london', 'lon-core-1')"

## Output format — return ONLY this, no prose before or after

Clarification Questions
1. <question> (Why: <impact on reachability / waypointing / load-balancing>)
2. <question> (Why: <impact>)

If nothing is ambiguous, return exactly: No clarification needed.
"""

SUFFICIENCY_SYSTEM = """\
You are an intent evaluator for network configuration. Given the original vague intent
and all clarification Q&A so far, determine whether the concrete facts needed by the
Generator Agent are now known.

The Generator needs these concrete facts to be resolved:
1. REACHABILITY : exact source router name and exact destination subnet (CIDR notation)
                  for every pair. Vague labels like "our servers" or "the remote network"
                  must be replaced with actual values.
2. WAYPOINTING  : the exact router hostname for any named waypoint. The mandatory vs
                  preferred distinction does NOT need to be resolved here — the Generator
                  will produce candidate configurations for both interpretations.
3. LOAD BALANCING: an exact integer path count for any pair where load balancing is
                  explicitly called for AND the count is missing. Vague phrases like
                  "redundancy" or "multiple paths" do NOT need to be resolved here —
                  the Generator will produce candidate configurations for those too.

Respond with EXACTLY one of the two formats below — nothing else.
Do NOT copy the format examples into your response; output only the keyword and content.

Format A (all required concrete facts are known):
CLARIFIED
<the clarified intent — resolve vague references (e.g. replace "our servers" with the
actual CIDR) but PRESERVE the original soft phrasing for waypoints and load balancing.
Do NOT add "mandatory", "must", or "always" to a waypoint unless the original intent
or an operator answer used that exact language. Do NOT invent a path count unless the
operator explicitly stated one.>

Format B (a required concrete fact is still missing):
MORE_QUESTIONS
Clarification Questions
1. <question> (Why: <impact>)
2. <question> (Why: <impact>)
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
    llm                  : any BaseLLMClient implementation (OpenAI, Anthropic, …)
    interactor           : TerminalInteractor (or any compatible replacement)
    results_dir          : directory where output files are written
    max_rounds           : hard cap on clarification rounds
    max_questions_per_round : maximum questions allowed per round (injected into prompt)
    dry_run              : if True, skip LLM calls and return canned responses
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        interactor: TerminalInteractor,
        results_dir: str = "results",
        max_rounds: int = 5,
        max_questions_per_round: int = 5,
        dry_run: bool = False,
    ) -> None:
        self._llm = llm
        self._interactor = interactor
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._results_dir = os.path.join(results_dir, timestamp)
        self._max_rounds = max_rounds
        self._max_questions_per_round = max_questions_per_round
        self._clarify_system = _build_clarify_system(max_questions_per_round)
        self._dry_run = dry_run
        os.makedirs(self._results_dir, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, vague_intent: str, runtime_context: str | None = None) -> str:
        """
        Run the interactive clarification loop.
        Returns the final clarified intent string.

        Parameters
        ----------
        runtime_context : optional text from a previous failed Selection pass.
            When provided it is appended to every LLM prompt so the model knows
            what was already generated, compared, and ruled out.
        """
        self._runtime_context = runtime_context  # stored for use in message builders

        self._interactor.display_banner("NetSocratic — Clarification Agent")
        if runtime_context:
            self._interactor.display(
                "\n[Clarification] Resuming after a failed selection pass. "
                "Previous context has been loaded.\n"
            )
        self._interactor.display(
            "I will ask you a few questions to resolve any ambiguity in your intent.\n"
            "Please answer each question as precisely as possible."
        )

        self._save_intent(vague_intent)
        if runtime_context:
            self._write(
                os.path.join(self._results_dir, "clarify_runtime_context.txt"),
                runtime_context + "\n",
            )
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
                Message(role="system", content=self._clarify_system),
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
        Falls back to the original vague intent if the LLM still returns MORE_QUESTIONS.
        """
        user_content = self._build_sufficiency_user_message(vague_intent, history)
        user_content += (
            "\n\nThe clarification phase has ended. "
            "You MUST respond with CLARIFIED followed by the best-effort intent. "
            "Do NOT output MORE_QUESTIONS."
        )

        if self._dry_run:
            return "[dry-run] Clarified intent would be generated here."

        messages = [
            Message(role="system", content=SUFFICIENCY_SYSTEM),
            Message(role="user", content=user_content),
        ]
        response = self._llm.complete(messages, temperature=0.2)
        is_done, result = self._parse_sufficiency(response)

        if is_done:
            return result

        # LLM returned MORE_QUESTIONS despite the explicit instruction.
        # Fall back to the original vague intent so downstream agents receive
        # something actionable rather than a list of unanswered questions.
        self._interactor.display(
            "\n[Warning] LLM still returned MORE_QUESTIONS during synthesis. "
            "Using the original vague intent as a best-effort fallback."
        )
        return vague_intent

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

        if getattr(self, "_runtime_context", None):
            parts.append(f"\n{self._runtime_context}")

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

        if getattr(self, "_runtime_context", None):
            parts.append(f"\n{self._runtime_context}")

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

        Searches for the keyword anywhere in the response so that preamble text
        (format examples, separator lines, etc. echoed from the system prompt)
        does not prevent correct parsing.
        """
        # Find the first occurrence of either keyword
        clarified_pos = response.find("CLARIFIED")
        more_q_pos    = response.find("MORE_QUESTIONS")

        # Both present — pick whichever comes first
        if clarified_pos != -1 and more_q_pos != -1:
            if clarified_pos < more_q_pos:
                more_q_pos = -1
            else:
                clarified_pos = -1

        if clarified_pos != -1:
            clarified = response[clarified_pos + len("CLARIFIED"):].strip()
            return True, clarified

        if more_q_pos != -1:
            return False, response[more_q_pos:]

        # Fallback: no keyword found — treat entire response as clarified
        return True, response.strip()

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _save_intent(self, vague_intent: str) -> None:
        path = os.path.join(self._results_dir, "intent_original.txt")
        self._write(path, vague_intent + "\n")

    def _save_prompt(self, rendered_prompt: str) -> None:
        path = os.path.join(self._results_dir, "clarify_prompt.txt")
        full_content = f"=== CLARIFY AGENT — SYSTEM PROMPT ===\n\n{self._clarify_system}\n\n=== USER MESSAGE ===\n\n{rendered_prompt}\n"
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
