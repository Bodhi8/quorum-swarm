"""LLM backends.

Agents call ``backend.complete(...)``. The default ``MockLLM`` costs
nothing and needs no key, so the whole framework runs offline; a domain
supplies its mock behaviour. ``AnthropicLLM`` is used when the ``anthropic``
package is installed and ``ANTHROPIC_API_KEY`` is set.

Token usage is converted to cost units so the controller can trade model
quality against spend.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    usd: float


class LLMBackend(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> LLMResult: ...


class MockLLM:
    """Deterministic stand-in. ``responder(system, prompt) -> text``."""

    name = "mock"

    def __init__(self, responder: Callable[[str, str], str] | None = None, usd_per_call: float = 0.0):
        self.responder = responder or (lambda s, p: "")
        self.usd_per_call = usd_per_call

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> LLMResult:
        text = self.responder(system, prompt)
        return LLMResult(text, len(prompt) // 4, len(text) // 4, self.usd_per_call)


class AnthropicLLM:
    """Claude via the Anthropic SDK. Prices are USD per million tokens."""

    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-5-5", usd_in: float = 1.0, usd_out: float = 5.0):
        import anthropic  # optional dependency

        self.client = anthropic.Anthropic()
        self.model = model
        self.usd_in, self.usd_out = usd_in, usd_out

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> LLMResult:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content)
        tin, tout = msg.usage.input_tokens, msg.usage.output_tokens
        usd = tin * self.usd_in / 1e6 + tout * self.usd_out / 1e6
        return LLMResult(text, tin, tout, usd)


def get_backend(prefer: str = "auto", mock_responder: Callable[[str, str], str] | None = None) -> LLMBackend:
    if prefer in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM(model=os.environ.get("QUORUM_MODEL", "claude-haiku-5-5"))
        except ImportError:
            if prefer == "anthropic":
                raise
    return MockLLM(mock_responder)
