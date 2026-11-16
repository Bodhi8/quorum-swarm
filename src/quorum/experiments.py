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
"""Reproducible experiments behind the demo and the papers.

* ``compare``: static wiring vs. Quorum on identical scenario sequences.
* ``gate_study``: the same stream of plans pushed through different gate
  policies, measuring human interruptions against plans wrongly let through.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import plugins
from .build import build_swarm
from .gates import Proposal, SimulatedOperator
from .swarm import SwarmConfig


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def compare(domain: str = "toy", episodes: int = 40, seeds: tuple[int, ...] = (0, 1, 2),
            modes: tuple[str, ...] = ("static", "quorum"), events_dir: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    shared = plugins.get_domain(domain)  # build once; scenario catalogs can be slow
    for mode in modes:
        rows = []
        for s in seeds:
            ev = f"{events_dir}/{mode}_seed{s}.jsonl" if events_dir else None
            sw = build_swarm(shared, mode=mode, seed=s, events=ev, llm="mock")
            sw.run(episodes)
            sw.bus.close()
            rows.append(sw.history)
        half = episodes // 2

        def agg(key: str, sl: slice) -> float:
            return _mean([h.metrics[key] for hist in rows for h in hist[sl]])

        out[mode] = {
            "survival": agg("survived", slice(None)),
            "survival_late": agg("survived", slice(half, None)),
            "compute_cost": agg("compute_cost", slice(None)),
            "compute_cost_late": agg("compute_cost", slice(half, None)),
            "interrupts": agg("interrupts", slice(None)),
            "interrupts_late": agg("interrupts", slice(half, None)),
            "misses": agg("misses", slice(None)),
            "reward": _mean([h.reward for hist in rows for h in hist]),
            "reward_late": _mean([h.reward for hist in rows for h in hist[half:]]),
            "curve_reward": [round(_mean([hist[i].reward for hist in rows]), 3) for i in range(episodes)],
            "curve_cost": [round(_mean([hist[i].metrics["compute_cost"] for hist in rows]), 3) for i in range(episodes)],
            "curve_interrupts": [round(_mean([hist[i].metrics["interrupts"] for hist in rows]), 3) for i in range(episodes)],
        }
    return out


ABLATION_LADDER: list[dict[str, Any]] = [
    {"name": "static", "label": "Fixed wiring (baseline)", "mode": "static", "gate": "threshold",
     "field_policy": "static", "gate_aware": False},
    {"name": "+routing", "label": "+ learned strategy routing", "mode": "quorum", "gate": "threshold",
     "field_policy": "static", "gate_aware": False},
    {"name": "+field", "label": "+ self-tuning field (Pro)", "mode": "quorum", "gate": "threshold",
     "field_policy": "adaptive", "gate_aware": False},
    {"name": "+gate", "label": "+ learned gate (Pro)", "mode": "quorum", "gate": "learned",
     "field_policy": "adaptive", "gate_aware": False},
    {"name": "full", "label": "+ gate-aware plan choice (full Quorum)", "mode": "quorum", "gate": "learned",
     "field_policy": "adaptive", "gate_aware": True},
]

ABLATION_METRICS = ("survived", "compute_cost", "interrupts", "misses")


def _ci(xs: list[float]) -> tuple[float, float]:
    """Mean and 95% half-width over seeds (normal approximation)."""
    a = np.asarray(xs, dtype=float)
    if len(a) < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


def ablation(domain: str = "grid", episodes: int = 40, seeds: tuple[int, ...] = tuple(range(10)),
             ladder: list[dict[str, Any]] | None = None, progress=None) -> dict[str, Any]:
    """Add one component at a time, on identical scenarios, and measure what each one buys."""
    ladder = ladder or ABLATION_LADDER
    avail = plugins.available()
    shared = plugins.get_domain(domain)
    out: dict[str, Any] = {"domain": domain, "episodes": episodes, "seeds": list(seeds), "rungs": []}
    for rung in ladder:
        if rung["gate"] not in avail["gates"] or rung["field_policy"] not in avail["field_policies"]:
            out["rungs"].append({**rung, "skipped": "needs Quorum Pro"})
            continue
        per_seed: dict[str, list[float]] = {m: [] for m in ABLATION_METRICS + ("reward",)}
        for s in seeds:
            cfg = SwarmConfig(twin_every=3 if rung["field_policy"] != "static" else 0, gate_aware=rung["gate_aware"])
            sw = build_swarm(shared, mode=rung["mode"], seed=s, llm="mock", gate=rung["gate"],
                             field_policy=rung["field_policy"], config=cfg)
            sw.run(episodes)
            for m in ABLATION_METRICS:
                per_seed[m].append(_mean([h.metrics[m] for h in sw.history]))
            per_seed["reward"].append(_mean([h.reward for h in sw.history]))
            if progress:
                progress(rung["name"], s)
        out["rungs"].append({**rung, **{m: _ci(v) for m, v in per_seed.items()}, "per_seed": per_seed})
    return out


@dataclass
class GateStats:
    gate: str
    interrupts: int
    misses: int  # auto-approved, human would have rejected
    lost: int  # blocked, human would have approved
    n: int
    interrupts_late: int
    misses_late: int


def proposal_stream(domain: str = "toy", scenarios: int = 60, per_scenario: int = 3, seed: int = 0) -> tuple[list[Proposal], Any]:
    """Plans a Quorum swarm actually produces, recorded once and replayed for every gate."""
    sw = build_swarm(domain, mode="quorum", seed=seed, llm="mock", gate="auto",
                     config=SwarmConfig(twin_every=0))
    stream: list[Proposal] = []
    for i in range(scenarios):
        sit = sw.domain.reset(sw.scenario_rng)
        res = sw.search(sit, sw.knobs, sw.rng, learn=True)
        for j, s in enumerate(res.top[:per_scenario]):
            feats = sw.domain.risk_features(sit, s.cand, s.ev, s.st)
            stream.append(Proposal(f"s{i}c{j}", s.cand.plan, sw.domain.describe(sit, s.cand), feats,
                                   sw.domain.risk_score(feats)))
    return stream, sw.domain


def gate_study(domain: str = "toy", scenarios: int = 60, seed: int = 0,
               gates: tuple[str, ...] = ("auto", "always_ask", "threshold", "learned")) -> list[GateStats]:
    stream, dom = proposal_stream(domain, scenarios, seed=seed)
    truth = dom.operator_policy()
    operator = SimulatedOperator(truth)
    avail = plugins.available()["gates"]
    out = []
    half = len(stream) // 2
    for name in gates:
        if name not in avail:
            continue
        gate = plugins.get_gate(name)
        inter = miss = lost = inter_late = miss_late = 0
        for k, p in enumerate(stream):
            d = gate.decide(p, interrupts_left=10**6)
            ok = truth(p)
            if d.action == "escalate":
                inter += 1
                inter_late += k >= half
                gate.observe(p, operator.review(p))
            elif d.action == "auto" and not ok:
                miss += 1
                miss_late += k >= half
            elif d.action == "block" and ok:
                lost += 1
        out.append(GateStats(name, inter, miss, lost, len(stream), inter_late, miss_late))
    return out
