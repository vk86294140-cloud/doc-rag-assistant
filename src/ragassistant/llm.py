"""LLM abstraction over the Anthropic Claude API.

`AnthropicLLM` calls Claude with adaptive thinking. A `Protocol` keeps the RAG
core decoupled from the provider, which also lets tests inject a deterministic
fake LLM with no API key or network.
"""

from __future__ import annotations

from typing import List, Protocol

from .config import settings


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    """Claude-backed LLM. Requires ANTHROPIC_API_KEY in the environment."""

    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        # Imported lazily so the package imports without the SDK installed
        # (e.g. when only the fake LLM is used in tests).
        import anthropic

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or settings.model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()


class EchoLLM:
    """Deterministic offline LLM for tests and `--no-llm` demos.

    It does not call any model — it stitches together a grounded answer from the
    retrieved context so the retrieval pipeline can be exercised end-to-end
    without an API key.
    """

    def complete(self, system: str, user: str) -> str:
        marker = "Context:\n"
        context = user.split(marker, 1)[1] if marker in user else user
        first_line = next((ln.strip() for ln in context.splitlines() if ln.strip()), "")
        return f"[offline answer] Based on the retrieved context: {first_line}"
