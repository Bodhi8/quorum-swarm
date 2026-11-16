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
"""The domain interface.

A domain is a problem space the swarm works on. Quorum's runtime is
domain-agnostic; a domain tells it how to lay out the field graph, what
stress looks like, which scout strategies exist, and how to evaluate,
stress-test and apply a candidate plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Protocol

import networkx as nx
import numpy as np

from .field import SemanticField
from .llm import LLMBackend


@dataclass
class Candidate:
    cid: str
    plan: Any
    nodes: list[Hashable]  # where on the field this candidate lives
    origin: str  # which scout strategy produced it
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evaluation:
    metrics: dict[str, float]
    ok: bool = True


@dataclass
class Strategy:
    name: str
    cost: float
    fn: Callable[[Any, SemanticField, np.random.Generator, LLMBackend], list[Candidate]]


class Domain(Protocol):
    name: str

    def reset(self, rng: np.random.Generator) -> Any:
        """Start a new scenario; returns an opaque Situation."""

    def field_graph(self, situation: Any) -> nx.Graph: ...

    def noop(self, situation: Any) -> Candidate:
        """The do-nothing plan, used as the baseline."""

    def view(self, situation: Any) -> dict[str, Any]:
        """Optional extra state for dashboards (layout, line loadings...)."""

    def stress(self, situation: Any) -> dict[Hashable, float]: ...

    def strategies(self) -> dict[str, Strategy]: ...

    def evaluate(self, situation: Any, cand: Candidate) -> Evaluation: ...

    def stress_test(self, situation: Any, cand: Candidate, rng: np.random.Generator) -> Evaluation: ...

    def combine(self, situation: Any, cands: list[Candidate]) -> Candidate | None: ...

    def risk_features(self, situation: Any, cand: Candidate, ev: Evaluation, st: Evaluation | None) -> dict[str, float]: ...

    def risk_score(self, features: dict[str, float]) -> float: ...

    def describe(self, situation: Any, cand: Candidate | None) -> str: ...

    def apply(self, situation: Any, cand: Candidate | None) -> dict[str, float]:
        """Apply live (None = safe default). Returns outcome metrics."""

    def resolved(self, ev: Evaluation) -> bool:
        """True when a candidate fully fixes the situation."""

    def operator_policy(self) -> Callable[[Any], bool]:
        """Simulated human preference, for headless runs."""
