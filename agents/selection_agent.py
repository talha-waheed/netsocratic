"""
Selection Agent — stub
======================
Takes the candidate configurations from the Generator Agent and:
  1. EC-based pruning  — uses Batfish to compute per-attribute equivalence classes
     and drops behaviorally redundant candidates.
  2. Distinguishing Question Generator — finds concrete behavioral differences
     among survivors and converts them into targeted user questions.
  3. User-response-based pruning — removes candidates inconsistent with the
     operator's answers until exactly one candidate remains.

The interactive terminal questions in this agent reuse the same TerminalInteractor
as the Clarification Agent.

Replace this stub with a real implementation when Batfish integration and the
distinguishing-question prompt are available.
"""

from interaction.terminal import TerminalInteractor
from llm.base import BaseLLMClient


class SelectionAgent:
    """
    Prunes candidate configurations via Batfish verification and targeted Q&A.

    Parameters
    ----------
    llm         : LLM client for generating distinguishing questions
    interactor  : shared TerminalInteractor for operator interaction
    max_rounds  : maximum question-answer rounds before giving up
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        interactor: TerminalInteractor,
        max_rounds: int = 10,
    ) -> None:
        self._llm = llm
        self._interactor = interactor
        self._max_rounds = max_rounds

    def run(self, configs: list[str], clarified_intent: str) -> str:
        """
        Select the single configuration that best matches the operator's intent.

        Parameters
        ----------
        configs          : list of candidate configuration strings
        clarified_intent : the output of the Clarification Agent

        Returns
        -------
        str
            The winning configuration string (the Network Specification).
        """
        raise NotImplementedError(
            "SelectionAgent.run() is not yet implemented. "
            "Integrate Batfish EC-based pruning and the distinguishing-question prompt to complete this agent."
        )

    # ── Placeholders for sub-components ──────────────────────────────────────

    def _ec_based_pruning(self, configs: list[str]) -> list[str]:
        """
        Use Batfish to compute per-attribute equivalence classes and drop
        behaviorally identical candidates.
        Attributes checked: reachability, waypointing, load balancing.
        """
        raise NotImplementedError

    def _generate_distinguishing_question(
        self,
        config_a: str,
        config_b: str,
        behavioral_diff: dict,
    ) -> str:
        """
        Ask the LLM to turn a Batfish-detected behavioral difference between
        two candidates into a concise, non-technical user question.
        """
        raise NotImplementedError

    def _prune_by_answer(
        self,
        configs: list[str],
        chosen: str,
        rejected: str,
        discriminating_flows: list[dict],
    ) -> list[str]:
        """
        Remove every candidate whose behavior on discriminating flows matches
        the rejected configuration rather than the chosen one.
        """
        raise NotImplementedError
