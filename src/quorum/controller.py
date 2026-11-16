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
"""The ML control plane (open-core part).

* ``ThompsonBandit``: learns which strategy or model each agent role should
  use, trading quality against cost. Discounted so it tracks drift.
* ``Controller``: one bandit per role plus a pluggable ``FieldPolicy`` that
  proposes changes to the field's physical knobs.
* ``StaticController``: the baseline most frameworks ship, i.e. fixed wiring.

Field policies that learn (and the shadow twin that vets them) plug in
through ``FieldPolicy``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .field import FieldKnobs


@dataclass
class Arm:
    name: str
    cost: float
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class ThompsonBandit:
    """Beta-Bernoulli Thompson sampling with fractional rewards in [0, 1]."""

    def __init__(self, arms: dict[str, float], discount: float = 0.98, prior: tuple[float, float] = (1.0, 1.0)):
        self.arms = {n: Arm(n, c, *prior) for n, c in arms.items()}
        self.discount = discount
        self.prior = prior

    def select(self, rng: np.random.Generator) -> str:
        draws = {n: rng.beta(a.alpha, a.beta) for n, a in self.arms.items()}
        return max(draws, key=draws.get)

    def update(self, name: str, reward01: float) -> None:
        r = float(np.clip(reward01, 0.0, 1.0))
        for a in self.arms.values():  # forget slowly, back toward the prior
            a.alpha = self.prior[0] + self.discount * (a.alpha - self.prior[0])
            a.beta = self.prior[1] + self.discount * (a.beta - self.prior[1])
        arm = self.arms[name]
        arm.alpha += r
        arm.beta += 1.0 - r
        arm.pulls += 1
        arm.total_reward += r

    def summary(self) -> dict[str, dict[str, float]]:
        return {n: {"mean": round(a.mean, 3), "pulls": a.pulls, "cost": a.cost} for n, a in self.arms.items()}


class FieldPolicy(Protocol):
    name: str

    def propose(self, current: FieldKnobs, rng: np.random.Generator) -> FieldKnobs | None:
        """Return candidate knobs to test in the shadow twin, or None."""

    def feedback(self, knobs: FieldKnobs, accepted: bool, delta: float) -> None: ...


class StaticFieldPolicy:
    name = "static"

    def propose(self, current: FieldKnobs, rng: np.random.Generator) -> FieldKnobs | None:
        return None

    def feedback(self, knobs: FieldKnobs, accepted: bool, delta: float) -> None:
        return None


@dataclass
class Controller:
    role_arms: dict[str, dict[str, float]]  # role -> {strategy: cost}
    field_policy: FieldPolicy = field(default_factory=StaticFieldPolicy)
    cost_weight: float = 0.5
    discount: float = 0.98

    def __post_init__(self) -> None:
        self.bandits = {r: ThompsonBandit(a, discount=self.discount) for r, a in self.role_arms.items()}

    name = "quorum"

    def select(self, role: str, rng: np.random.Generator) -> str:
        return self.bandits[role].select(rng)

    def reward(self, role: str, arm: str, quality01: float) -> float:
        """Quality minus cost, squashed to [0, 1]."""
        cost = self.bandits[role].arms[arm].cost
        r = quality01 - self.cost_weight * cost
        self.bandits[role].update(arm, r)
        return r

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        return {r: b.summary() for r, b in self.bandits.items()}


class StaticController(Controller):
    """Fixed wiring: every role always uses one configured strategy."""

    name = "static"

    def __init__(self, role_arms: dict[str, dict[str, float]], fixed: dict[str, str]):
        super().__init__(role_arms=role_arms)
        self.fixed = fixed

    def select(self, role: str, rng: np.random.Generator) -> str:
        return self.fixed[role]
