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
"""The shadow twin: no policy change goes live untested.

Before the controller changes how the swarm behaves, the twin replays the
same scenarios twice in simulation, once with the current knobs and once
with the candidate, using identical random seeds. The change is accepted
only if it does better. Shadow runs never write to the live event history
and never update any learner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .field import FieldKnobs

if TYPE_CHECKING:
    from .swarm import Swarm


@dataclass
class TwinVerdict:
    accept: bool
    delta: float
    current_score: float
    candidate_score: float


class ShadowTwin:
    def __init__(self, swarm: "Swarm"):
        self.swarm = swarm

    def score(self, knobs: FieldKnobs, seeds: list[int]) -> float:
        sw = self.swarm
        lam = sw.controller.cost_weight
        vals = []
        prev_shadow = sw.bus.shadow
        sw.bus.shadow = True
        try:
            for s in seeds:
                sit = sw.domain.reset(np.random.default_rng(s))
                res = sw.search(sit, knobs, np.random.default_rng(s + 1), learn=False)
                scale = max(abs(res.baseline_score), 1e-6)
                vals.append((res.objective - res.baseline_score) / scale - lam * res.cost)
        finally:
            sw.bus.shadow = prev_shadow
        return float(np.mean(vals))

    def evaluate(self, current: FieldKnobs, candidate: FieldKnobs, seeds: list[int]) -> TwinVerdict:
        cur = self.score(current, seeds)
        cand = self.score(candidate, seeds)
        delta = cand - cur
        return TwinVerdict(delta > self.swarm.config.twin_tolerance, delta, cur, cand)
