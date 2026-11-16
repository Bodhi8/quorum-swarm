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
"""Assemble a swarm from plugin names.

``mode="quorum"`` is the full system: bandit routing, the best available
field policy and gate. With Quorum Pro installed that means the learned gate
and the self-tuning field; the open core falls back to the threshold gate
and static field physics and says so. ``mode="static"`` is the baseline most frameworks
ship: fixed wiring (every scout uses the strongest model), fixed field
physics and a fixed risk threshold for human review.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import plugins
from .controller import Controller, StaticController
from .events import EventBus
from .gates import CLIOperator, GateKeeper, SimulatedOperator
from .intent import Intent
from .llm import get_backend
from .swarm import Swarm, SwarmConfig


def _first_available(kind: str, preferred: list[str]) -> str:
    from .pro import notice, pro_feature

    avail = plugins.available()[kind]
    chosen = next(n for n in preferred if n in avail)
    skipped = [n for n in preferred[: preferred.index(chosen)] if pro_feature(kind, n)]
    for n in skipped:
        notice(kind, n, chosen)
    return chosen


def build_swarm(
    domain: str | Any = "toy",
    mode: str = "quorum",
    seed: int = 0,
    events: str | Path | None = None,
    interactive: bool = False,
    gate: str | None = None,
    field_policy: str | None = None,
    llm: str = "auto",
    config: SwarmConfig | None = None,
    domain_kwargs: dict[str, Any] | None = None,
) -> Swarm:
    dom = plugins.get_domain(domain, **(domain_kwargs or {})) if isinstance(domain, str) else domain
    rules = dom.default_rules() if hasattr(dom, "default_rules") else []
    intent = Intent(goal=f"resolve {dom.name} scenarios", objective=dom.default_objective(), rules=rules)
    arms = {"scout": {n: s.cost for n, s in dom.strategies().items()}}

    if mode == "static":
        controller: Controller = StaticController(arms, fixed={"scout": "llm"})
        gate_name = gate or "threshold"
    elif mode == "quorum":
        controller = Controller(arms, field_policy=plugins.get_field_policy(
            field_policy or _first_available("field_policies", ["adaptive", "static"])))
        gate_name = gate or _first_available("gates", ["learned", "threshold"])
    else:
        raise ValueError("mode must be 'quorum' or 'static'")

    policy = dom.operator_policy()
    operator = CLIOperator() if interactive else SimulatedOperator(policy)
    keeper = GateKeeper(plugins.get_gate(gate_name), operator, oracle=policy)
    responder = getattr(dom, "mock_llm", None)
    return Swarm(dom, intent, controller, keeper, bus=EventBus(events), llm=get_backend(llm, responder),
                 config=config or SwarmConfig(twin_every=3 if mode == "quorum" else 0), seed=seed)
