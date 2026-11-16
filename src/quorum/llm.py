#   Copyright 2026 Brian Curry / Vector1 Research
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
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
    """Claude via the Anthropic SDK.

    Prices are USD per million tokens. They are not hard-coded: pass them, or set
    QUORUM_USD_IN / QUORUM_USD_OUT from the current price list. Without them, token
    counts are still recorded and dollar cost is reported as 0.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-5-5", usd_in: float | None = None, usd_out: float | None = None):
        import anthropic  # optional dependency

        self.client = anthropic.Anthropic()
        self.model = model
        self.usd_in = usd_in if usd_in is not None else float(os.environ.get("QUORUM_USD_IN", 0.0))
        self.usd_out = usd_out if usd_out is not None else float(os.environ.get("QUORUM_USD_OUT", 0.0))

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


class MeteredLLM:
    """Wraps any backend and keeps a running tally of calls, tokens and dollars."""

    def __init__(self, inner: LLMBackend):
        self.inner = inner
        self.name = getattr(inner, "name", "llm")
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.usd = 0.0

    def complete(self, system: str, prompt: str, max_tokens: int = 300) -> LLMResult:
        r = self.inner.complete(system, prompt, max_tokens)
        self.calls += 1
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.usd += r.usd
        return r

    def snapshot(self) -> tuple[int, int, int, float]:
        return self.calls, self.input_tokens, self.output_tokens, self.usd


def get_backend(prefer: str = "auto", mock_responder: Callable[[str, str], str] | None = None) -> LLMBackend:
    if prefer in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM(model=os.environ.get("QUORUM_MODEL", "claude-haiku-5-5"))
        except ImportError:
            if prefer == "anthropic":
                raise
    return MockLLM(mock_responder)
