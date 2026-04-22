"""
Generator Agent — stub
======================
Receives a clarified intent and a knowledge base (topology, base configs, docs)
and produces one or more candidate network configurations (OSPF format).

The Generator Agent is run multiple times (with varied temperature / sampling)
to build the candidate set that the Selection Agent will compare and prune.

Replace this stub with a real implementation when the prompt and knowledge base
are available.
"""

from llm.base import BaseLLMClient


class GeneratorAgent:
    """
    Generates candidate network configurations from a clarified intent.

    Parameters
    ----------
    llm           : LLM client used to synthesise configurations
    knowledge_base: dict with topology, base configs, documentation, etc.
    num_candidates: how many candidates to generate per intent
    temperatures  : list of temperature values to sample with (one per candidate)
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        knowledge_base: dict | None = None,
        num_candidates: int = 3,
        temperatures: list[float] | None = None,
    ) -> None:
        self._llm = llm
        self._knowledge_base = knowledge_base or {}
        self._num_candidates = num_candidates
        self._temperatures = temperatures or [0.2, 0.7, 1.0]

    def run(self, clarified_intent: str) -> list[str]:
        """
        Generate candidate configurations for the given clarified intent.

        Returns
        -------
        list[str]
            One configuration string per candidate (e.g. FRR OSPF config text).
        """
        raise NotImplementedError(
            "GeneratorAgent.run() is not yet implemented. "
            "Provide the generator prompt and knowledge base to complete this agent."
        )
