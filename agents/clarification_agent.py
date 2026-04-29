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
                    Vague destination labels are scoped to the source/policy statement
                    they appear in. Do NOT assume that "the target subnet" or "that
                    network" means the same CIDR for a later source unless the text
                    explicitly says it is the same target.
                    If the intent says "same policy", "same requirements", "similar
                    routing", "other subnets", or "other relevant connections", ask for
                    the exact source→subnet pairs that inherit the policy.
2. WAYPOINTING    — the exact router hostname for any waypoint that is named vaguely
                    (e.g. "the london hub", "a transit router"). If the name is already
                    a known router identifier (e.g. "london", "kiev"), do NOT ask.
                    Do NOT ask whether a waypoint is mandatory or preferred — the
                    Generator will explore both interpretations.
3. LOAD BALANCING — an exact integer path count for each affected source→destination
                    pair when load balancing or redundancy is called for and the count
                    is missing or unclear (e.g. "split across N paths" with N not stated,
                    "load-balance to X", "redundancy", "reliability", "multiple paths",
                    or "fault tolerance"). If multiple flows have missing counts, ask
                    for a per-flow mapping.

## Process — think before you ask

Before writing any question, silently work through these steps:
  a. List every source→destination pair mentioned (even vaguely). Treat repeated vague
     labels as separate unknowns when attached to different sources.
  b. For each pair: is the source a known router name? Is the destination a concrete
     CIDR prefix, or a vague label that needs resolution?
  c. For any waypoint mentioned: is the router hostname concrete and unambiguous?
  d. For any pair with an explicit load-balancing requirement: is the exact path count
     given, or is it missing?
  e. Combine gaps only when the answer can be a clear per-flow mapping. Never combine
     distinct flows into a question that can be answered with one global value unless
     the intent explicitly says the same value applies to all of them.
  f. Write at most {max_questions} question(s) total.

## Question quality rules

- Ask only about concrete missing values: CIDR prefixes, exact router hostnames,
  explicit integer path counts when an LB requirement is clear but the count is absent.
- When a vague destination label appears for multiple sources, ask for each source
  explicitly in the same question or in separate questions. Example: "What exact
  destination subnet should london reach, and what exact destination subnet should
  madrid reach?"
- When a policy is extended to "other subnets" or "other relevant connections", ask
  for the complete list of source→destination subnet pairs covered by that phrase.
- When asking for missing path counts across multiple flows, require a source→prefix→count
  answer. Example: "What exact path count is required for london→100.0.1.0/24,
  basel→100.0.1.0/24, and madrid→100.0.4.0/24?"
- Do NOT ask whether "via X", "through X", "if possible", "consider routing through X",
  "the path includes X" means mandatory or preferred. This is an interpretation the
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
                  must be replaced with actual values. A CIDR answer for one source does
                  not resolve the same vague label for another source unless the answer
                  explicitly says it applies to that other source.
2. WAYPOINTING  : the exact router hostname for any named waypoint. The mandatory vs
                  preferred distinction does NOT need to be resolved here — the Generator
                  will produce candidate configurations for both interpretations.
3. LOAD BALANCING: an exact integer path count for each pair where load balancing or
                  redundancy is called for and the count is missing. A global count
                  answer applies only to the flow named in the question unless the
                  answer explicitly maps counts to other flows.

Respond with EXACTLY one of the two formats below — nothing else.
Do NOT copy the format examples into your response; output only the keyword and content.

Format A (all required concrete facts are known):
CLARIFIED
<the clarified intent — resolve vague references (e.g. replace "our servers" with the
actual CIDR) but PRESERVE the original soft phrasing for waypoints and load balancing.
Do NOT add "mandatory", "must", or "always" to a waypoint unless the original intent
or an operator answer used that exact language. Do NOT invent a path count or copy one
flow's path count to another flow unless the operator explicitly stated that mapping.>

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
        questions = _augment_clarification_questions(
            vague_intent,
            history,
            questions,
            self._max_questions_per_round,
        )
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
        is_done, result = self._parse_sufficiency(response)
        if is_done:
            result = _preserve_answered_path_counts(result, history)
        return is_done, result

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
            return _preserve_answered_path_counts(result, history)

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


def _augment_clarification_questions(
    vague_intent: str,
    history: list[QARound],
    questions: list[str],
    max_questions: int,
) -> list[str]:
    """Add deterministic questions for ambiguity patterns the LLM often collapses."""
    answered_sources = _answered_destination_sources(history)
    generated_sources = _question_destination_sources(questions)

    forced: list[str] = []
    for source, label in _vague_destination_sources(vague_intent):
        if source in answered_sources or source in generated_sources:
            continue
        forced.append(
            f"What is the exact destination subnet in CIDR notation for traffic "
            f"from {source} when the intent says \"{label}\"? "
            "(Why: impacts reachability)"
        )

    for source, known_prefix in _same_policy_sources(vague_intent):
        key = f"same-policy:{source}:{known_prefix or ''}"
        if key in answered_sources or source in generated_sources:
            continue
        if known_prefix:
            forced.append(
                f"Which exact additional destination subnets inherit the same policy "
                f"for traffic from {source} beyond {known_prefix}? Answer as a CIDR "
                "list, or say none. (Why: impacts reachability, waypointing, and "
                "load-balancing)"
            )
        else:
            forced.append(
                f"Which exact source to destination subnet pairs are covered by the "
                f"same-policy or other-subnets wording for {source}? Answer each pair "
                "as source to CIDR. (Why: impacts reachability, waypointing, and "
                "load-balancing)"
            )

    lb_question = _per_flow_load_balance_question(vague_intent, questions)
    if lb_question:
        forced.insert(0, lb_question)
        questions = [q for q in questions if not _is_generic_path_count_question(q)]

    if not forced:
        return questions[:max_questions]

    merged: list[str] = []
    seen: set[str] = set()
    for question in forced + questions:
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(question)
        if len(merged) >= max_questions:
            break
    return merged


def _vague_destination_sources(intent: str) -> list[tuple[str, str]]:
    vague_labels = [
        "target subnet",
        "remote network",
        "main subnet",
        "that network",
        "destination network",
        "our servers",
        "that subnet",
    ]
    results: list[tuple[str, str]] = []
    for line in _intent_units(intent):
        if not line or re.search(r"\d+\.\d+\.\d+\.\d+/\d+", line):
            continue
        label = next((item for item in vague_labels if item in line), None)
        if not label:
            continue
        source = _source_from_line(line)
        if source:
            results.append((source, label))
    return results


def _same_policy_sources(intent: str) -> list[tuple[str, str | None]]:
    """Find source/prefix statements extended by same-policy wording."""
    results: list[tuple[str, str | None]] = []
    trigger = re.compile(
        r"same\s+(?:connectivity\s+)?requirements|same\s+constraints|"
        r"same\s+policy|similar\s+routing|other\s+subnets|"
        r"other\s+relevant\s+connections|related\s+traffic",
        re.IGNORECASE,
    )
    for line in _intent_units(intent):
        if not trigger.search(line):
            continue
        source = _source_from_line(line)
        prefix = _first_prefix(line)
        if source:
            item = (source, prefix)
            if item not in results:
                results.append(item)
    return results


def _intent_units(intent: str) -> list[str]:
    """Split paragraph-style intents into analyzable sentence-like units."""
    normalized = re.sub(r"\s+", " ", intent.strip().lower())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+|;\s+", normalized)
    return [part.strip(" .") for part in parts if part.strip(" .")]


def _source_from_line(line: str) -> str | None:
    patterns = [
        r"([a-z][a-z0-9_-]*)\s+and\s+(?:the\s+)?(?:target\s+subnet|remote\s+network|main\s+subnet|that\s+network|destination\s+network|our\s+servers|that\s+subnet)\s+(?:need|should|must)",
        r"between\s+([a-z][a-z0-9_-]*)\s+and",
        r"from\s+([a-z][a-z0-9_-]*)\s+(?:to|can|should|needs?|must|and)",
        r"(?:subnet|network|servers?)\s+(?:is\s+)?(?:accessible|reachable)\s+from\s+([a-z][a-z0-9_-]*)",
        r"(?:connectivity|access)\s+between\s+([a-z][a-z0-9_-]*)\s+and",
        r"connect\s+([a-z][a-z0-9_-]*)\s+to",
        r"connectivity\s+from\s+([a-z][a-z0-9_-]*)",
        r"([a-z][a-z0-9_-]*)\s+can\s+reach",
        r"([a-z][a-z0-9_-]*)\s+needs?\s+to\s+reach",
        r"([a-z][a-z0-9_-]*)\s+(?:needs?|should)\s+(?:to\s+)?(?:reach|connect)",
        r"([a-z][a-z0-9_-]*)\s+and\s+",
    ]
    for pattern in patterns:
        match = re.search(pattern, line)
        if match and not _is_vague_source_label(match.group(1)):
            return match.group(1)
    return None


def _is_vague_source_label(value: str) -> bool:
    return value in {
        "subnet",
        "network",
        "servers",
        "server",
        "target",
        "remote",
        "main",
        "destination",
    }


def _answered_destination_sources(history: list[QARound]) -> set[str]:
    sources: set[str] = set()
    for round_record in history:
        for question, answer in zip(round_record.questions, round_record.answers):
            if not re.search(r"\d+\.\d+\.\d+\.\d+/\d+", answer):
                continue
            source = _source_from_line(question.lower())
            if source:
                sources.add(source)
    return sources


def _question_destination_sources(questions: list[str]) -> set[str]:
    sources: set[str] = set()
    for question in questions:
        lower = question.lower()
        if "destination subnet" not in lower and "cidr" not in lower:
            continue
        source = _source_from_line(lower)
        if source:
            sources.add(source)
    return sources


def _per_flow_load_balance_question(intent: str, questions: list[str]) -> str | None:
    lower = intent.lower()
    if not re.search(
        r"available paths|load[- ]?balanc|split across|distributed across|"
        r"redundancy|reliability|multiple routes|multiple paths|fault tolerance",
        lower,
    ):
        return None
    if any(_is_per_flow_path_count_question(q) for q in questions):
        return None

    flows = _flows_missing_path_counts(intent)
    if not flows:
        return None
    shown_flows = flows[:12]
    flow_text = ", ".join(shown_flows)
    if len(flows) > len(shown_flows):
        flow_text += ", and any remaining affected flows"
    return (
        "What exact integer path count is required for each affected flow "
        f"({flow_text})? Answer as source to subnet equals count for each flow. "
        "(Why: impacts load-balancing)"
    )


def _mentioned_flows(intent: str) -> list[str]:
    flows: list[str] = []
    current_dest: str | None = None
    for line in _intent_units(intent):
        if not line:
            continue
        source = _source_from_line(line)
        prefix_match = re.search(r"\d+\.\d+\.\d+\.\d+/\d+", line)
        if prefix_match:
            current_dest = prefix_match.group(0)
        else:
            for label in ("target subnet", "remote network", "main subnet", "that network", "destination network"):
                if label in line:
                    current_dest = label
                    break
        if source and current_dest:
            flow = f"{source} to {current_dest}"
            if flow not in flows:
                flows.append(flow)
    return flows


def _flows_missing_path_counts(intent: str) -> list[str]:
    flows = _mentioned_flows(intent)
    if not flows:
        return []

    exact_counts: set[str] = set()
    current_source: str | None = None
    current_dest: str | None = None
    for line in _intent_units(intent):
        source = _source_from_line(line) or current_source
        prefix = _first_prefix(line) or current_dest
        if source:
            current_source = source
        if prefix:
            current_dest = prefix
        if source and prefix and re.search(r"\b\d+\s+(?:equal-cost\s+)?paths?\b", line):
            exact_counts.add(f"{source} to {prefix}")

    return [flow for flow in flows if flow not in exact_counts]


def _is_generic_path_count_question(question: str) -> bool:
    lower = question.lower()
    if "path count" not in lower and "how many" not in lower:
        return False
    return not re.search(
        r"\b[a-z][a-z0-9_-]*\s*(?:to|->|→)\s*"
        r"(?:\d+\.\d+\.\d+\.\d+/\d+|target subnet|remote network|main subnet)",
        lower,
    )


def _is_per_flow_path_count_question(question: str) -> bool:
    lower = question.lower()
    return "for each" in lower and ("path count" in lower or "paths" in lower)


def _preserve_answered_path_counts(clarified: str, history: list[QARound]) -> str:
    """Append explicit per-flow LB counts that were answered but omitted by the LLM."""
    additions: list[str] = []
    clarified_lower = clarified.lower()

    for source, prefix, count in _answered_path_counts(history):
        key = f"({source},{prefix})"
        sentence = (
            f"Traffic from {source} to {prefix} should be load-balanced "
            f"across {count} paths."
        )
        if key in clarified_lower:
            continue
        if re.search(
            rf"from\s+{re.escape(source)}\s+to\s+{re.escape(prefix)}.*"
            rf"(?:load[- ]?balanced|distributed|split).*{count}\s+paths?",
            clarified_lower,
        ):
            continue
        additions.append(sentence)

    if not additions:
        return clarified
    return clarified.rstrip() + " " + " ".join(additions)


def _answered_path_counts(history: list[QARound]) -> list[tuple[str, str, int]]:
    counts: list[tuple[str, str, int]] = []
    for round_record in history:
        for question, answer in zip(round_record.questions, round_record.answers):
            combined = f"{question}\n{answer}"
            if not re.search(r"path count|paths?|load[- ]?balanc|distributed", combined, re.IGNORECASE):
                continue
            for source, prefix, count in _parse_path_count_answer(answer):
                item = (source, prefix, count)
                if item not in counts:
                    counts.append(item)
    return counts


def _parse_path_count_answer(answer: str) -> list[tuple[str, str, int]]:
    results: list[tuple[str, str, int]] = []
    lower = answer.lower()
    patterns = [
        r"([a-z][a-z0-9_-]*)\s+to\s+(\d+\.\d+\.\d+\.\d+/\d+)\s+(?:equals|=|is|:)\s*(\d+)",
        r"([a-z][a-z0-9_-]*)\s*->\s*(\d+\.\d+\.\d+\.\d+/\d+)\s*(?:equals|=|is|:)?\s*(\d+)",
        r"([a-z][a-z0-9_-]*)\s+to\s+(\d+\.\d+\.\d+\.\d+/\d+).*?\b(\d+)\s+paths?",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            results.append((match.group(1), match.group(2), int(match.group(3))))
    return results


def _first_prefix(text: str) -> str | None:
    match = re.search(r"\d+\.\d+\.\d+\.\d+/\d+", text)
    return match.group(0) if match else None
