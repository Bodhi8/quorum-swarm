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
"""Human intent, compiled into rewards and hard rules.

The human layer does not write prompts for agents. It states goals as
weighted objectives plus hard rules that no optimizer may trade away. The
compiler turns that into:

* ``reward(metrics)``: a scalar the ML layer optimizes (soft), and
* ``violations(features)``: rule breaks that block a plan outright (hard).

Keeping these two apart is what stops the learning layer from "reward
hacking" its way around a safety rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class HardRule:
    name: str
    check: Callable[[dict[str, Any]], bool]  # True means the plan is allowed
    description: str = ""


def max_value(key: str, limit: float, name: str | None = None) -> HardRule:
    return HardRule(
        name=name or f"{key}<={limit}",
        check=lambda f: float(f.get(key, 0.0)) <= limit,
        description=f"{key} must not exceed {limit}",
    )


def forbid(key: str, name: str | None = None) -> HardRule:
    return HardRule(
        name=name or f"no_{key}",
        check=lambda f: not bool(f.get(key, False)),
        description=f"plans with {key} are never allowed",
    )


@dataclass
class Intent:
    goal: str
    objective: dict[str, float]  # metric -> weight (positive = more is better)
    rules: list[HardRule] = field(default_factory=list)
    budget: float = 10.0  # compute/token budget per episode, in cost units
    max_agents: int = 16
    interrupt_budget: int = 3  # max human escalations per episode

    def compile(self) -> "CompiledIntent":
        return CompiledIntent(self)


class CompiledIntent:
    def __init__(self, intent: Intent):
        self.intent = intent
        self.weights = dict(intent.objective)

    def reward(self, metrics: dict[str, float]) -> float:
        return float(sum(w * float(metrics[k]) for k, w in self.weights.items() if k in metrics))

    def violations(self, features: dict[str, Any]) -> list[str]:
        return [r.name for r in self.intent.rules if not r.check(features)]
