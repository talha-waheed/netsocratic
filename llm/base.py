from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: str


class BaseLLMClient(ABC):
    """Abstract LLM client. Swap implementations to change providers."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send messages and return the assistant's text response."""
        ...
