from openai import OpenAI
import time

import config
from .base import BaseLLMClient, Message


class OpenAIClient(BaseLLMClient):
    """LLM client backed by the OpenAI Chat Completions API."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._min_interval_s = config.OPENAI_MIN_SECONDS_BETWEEN_REQUESTS
        self._last_request_at = 0.0

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        self._throttle()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._last_request_at = time.monotonic()
        return response.choices[0].message.content or ""

    def _throttle(self) -> None:
        if self._min_interval_s <= 0 or self._last_request_at <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        delay = self._min_interval_s - elapsed
        if delay > 0:
            time.sleep(delay)
