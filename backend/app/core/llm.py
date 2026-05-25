import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """OpenAI-compatible LLM client supporting any provider (DashScope, OpenAI, local, etc.)."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(f"LLM.{model}")

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            content = await asyncio.to_thread(
                self.client.chat.completions.create, **kwargs
            )
            return content.choices[0].message.content or ""
        except Exception as e:
            self.logger.error("LLM call failed: %s", e)
            raise
