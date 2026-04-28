"""
LLM Operator
============
A drop-in replacement for TerminalInteractor that answers questions
using an LLM acting as a network operator who knows the correct
formal specification.

Used by the experiment runner to drive the full pipeline without
a human in the loop.
"""

import logging

from llm.base import BaseLLMClient, Message

logger = logging.getLogger(__name__)

# ── Operator system prompts ────────────────────────────────────────────────────

_CLARIFICATION_SYSTEM = """\
You are a network operator participating in a clarification session.
An AI agent is asking you questions to understand your routing intent.

Your exact routing requirements are given below as a formal specification.
Answer each question accurately and concisely based on this specification.
Use plain English as a human operator would — do NOT expose the raw JSON.
Answer only what is asked. Do not volunteer extra information.

Your formal requirements:
{spec}
"""

_SELECTION_SYSTEM = """\
You are a network operator choosing between two routing configurations.
An AI agent describes a behavioral difference and asks which you prefer.

Your exact routing requirements are given below. Answer the question by
selecting the option that best matches your requirements. Be concise.

Your formal requirements:
{spec}
"""


class LLMOperator:
    """
    Implements the same interface as TerminalInteractor but answers via LLM.

    Parameters
    ----------
    llm              : any BaseLLMClient implementation
    correct_spec     : the correct formal specification JSON string
    temperature      : LLM temperature for operator answers (default 0.0)
    verbose          : if True, print all Q&A to stdout (useful for debugging)
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        correct_spec: str,
        temperature: float = 0.0,
        verbose: bool = True,
    ) -> None:
        self._llm = llm
        self._spec = correct_spec
        self._temperature = temperature
        self._verbose = verbose

    # ── Display methods (no-ops in automated mode, log when verbose) ──────────

    def display(self, message: str) -> None:
        if self._verbose:
            print(message)

    def display_section(self, title: str, body: str) -> None:
        if self._verbose:
            print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}\n{body}")

    def display_banner(self, text: str) -> None:
        if self._verbose:
            print(f"\n{'─' * 60}\n  {text}\n{'─' * 60}")

    # ── Input methods — answered by LLM ──────────────────────────────────────

    def ask_questions(self, questions: list[str]) -> list[str]:
        """Answer a list of clarification questions, one LLM call per question."""
        answers: list[str] = []
        system = _CLARIFICATION_SYSTEM.format(spec=self._spec)
        for i, question in enumerate(questions, start=1):
            messages = [
                Message(role="system", content=system),
                Message(role="user", content=question),
            ]
            answer = self._llm.complete(messages, temperature=self._temperature, max_tokens=256)
            answers.append(answer.strip())
            if self._verbose:
                print(f"\n[LLMOperator] Q{i}: {question}")
                print(f"[LLMOperator] A{i}: {answer.strip()}")
            else:
                logger.debug("Q: %s | A: %s", question, answer.strip())
        return answers

    def ask(self, prompt: str) -> str:
        """Answer a single selection question (used by SelectionAgent)."""
        system = _SELECTION_SYSTEM.format(spec=self._spec)
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=prompt),
        ]
        answer = self._llm.complete(messages, temperature=self._temperature, max_tokens=256)
        answer = answer.strip()
        if self._verbose:
            print(f"\n[LLMOperator] Q: {prompt}")
            print(f"[LLMOperator] A: {answer}")
        else:
            logger.debug("Q: %s | A: %s", prompt, answer)
        return answer

    def confirm(self, message: str) -> bool:
        """Answer a yes/no confirmation question."""
        answer = self.ask(message)
        return answer.lower().startswith("y")
