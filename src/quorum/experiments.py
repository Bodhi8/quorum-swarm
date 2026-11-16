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
