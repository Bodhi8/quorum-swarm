"""Fault hunt: a tiny, dependency-free domain for tests and first contact.

A fault sits somewhere on a network. Sensors give noisy stress readings
that rise near it. The swarm must find the fault and choose a fix:

* ``patch``    cheap, safe, fixes the fault only if applied exactly there
* ``reroute``  cheap, partly fixes the fault from a neighbouring node
* ``replace``  expensive and risky, fixes the fault fully

The simulated operator rejects ``replace`` on critical nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx
import numpy as np

from ..domain import Candidate, Evaluation, Strategy
from ..field import SemanticField
from ..llm import LLMBackend

FIXES = {"patch": 0.1, "reroute": 0.2, "replace": 1.0}  # fix -> cost


@dataclass
class ToySituation:
    graph: nx.Graph
    fault: int
    severity: float
    sensors: dict[int, float]
    critical: set[int]


class ToyDomain:
    name = "toy"

    def __init__(self, n_nodes: int = 30, seed: int = 7):
        self.base_graph = nx.connected_watts_strogatz_graph(n_nodes, 4, 0.2, seed=seed)
        self.dist = dict(nx.all_pairs_shortest_path_length(self.base_graph))
        self.layout = {n: tuple(map(float, p)) for n, p in nx.spring_layout(self.base_graph, seed=seed).items()}

    # -- scenario ---------------------------------------------------------------
    def reset(self, rng: np.random.Generator) -> ToySituation:
        g = self.base_graph
        fault = int(rng.integers(g.number_of_nodes()))
        sensors = {n: float(np.exp(-0.9 * self.dist[fault][n]) + rng.normal(0, 0.08)) for n in g.nodes}
        critical = set(map(int, rng.choice(g.number_of_nodes(), size=5, replace=False)))
        return ToySituation(g, fault, float(rng.uniform(0.6, 1.0)), sensors, critical)

    def field_graph(self, sit: ToySituation) -> nx.Graph:
        return sit.graph

    def stress(self, sit: ToySituation) -> dict[int, float]:
        return {n: max(0.0, v) for n, v in sit.sensors.items()}

    def noop(self, sit: ToySituation) -> Candidate:
        return Candidate("noop", None, [], "noop")

    def view(self, sit: ToySituation) -> dict[str, Any]:
        return {"layout": {str(n): p for n, p in self.layout.items()}}

    # -- strategies ---------------------------------------------------------------
    def _cands_at(self, nodes: list[int], rng: np.random.Generator) -> list[Candidate]:
        out = []
        for n in nodes:
            for fix in ("patch", "reroute"):
                out.append(Candidate(f"{fix}@{n}", (fix, int(n)), [n], ""))
            if rng.random() < 0.3:
                out.append(Candidate(f"replace@{n}", ("replace", int(n)), [n], ""))
        return out

    def strategies(self) -> dict[str, Strategy]:
        def random_(sit, fld: SemanticField, rng, llm):
            return self._cands_at(list(rng.choice(fld.nodes, size=2, replace=False)), rng)

        def field_(sit, fld: SemanticField, rng, llm):
            return self._cands_at(list(dict.fromkeys(fld.sample(rng, size=2))), rng)

        def local(sit, fld, rng, llm):
            top = max(sit.sensors, key=sit.sensors.get)
            return self._cands_at([top] + list(sit.graph.neighbors(top))[:1], rng)

        def llm_(sit, fld, rng, llm: LLMBackend):
            readings = ", ".join(f"{n}:{v:.2f}" for n, v in sorted(sit.sensors.items(), key=lambda kv: -kv[1])[:6])
            text = llm.complete("Pick the most likely fault node. Reply with node ids.", readings).text
            ids = [int(t) for t in text.replace(",", " ").split() if t.isdigit() and int(t) in sit.graph]
            return self._cands_at(ids[:2] or [max(sit.sensors, key=sit.sensors.get)], rng)

        return {
            "random": Strategy("random", 0.05, random_),
            "field": Strategy("field", 0.1, field_),
            "local": Strategy("local", 0.1, local),
            "llm": Strategy("llm", 1.0, llm_),
        }

    @staticmethod
    def mock_llm(system: str, prompt: str) -> str:
        first = prompt.split(",")[0]
        return first.split(":")[0].strip()

    # -- evaluation ---------------------------------------------------------------
    def _residual(self, sit: ToySituation, plan: tuple[str, int] | None) -> float:
        if plan is None:
            return sit.severity
        fix, node = plan
        d = self.dist[sit.fault][node]
        if fix == "patch":
            return 0.0 if d == 0 else sit.severity
        if fix == "reroute":
            return sit.severity * (0.3 if d <= 1 else 1.0)
        return 0.0 if d <= 1 else sit.severity  # replace

    def evaluate(self, sit: ToySituation, cand: Candidate) -> Evaluation:
        res = self._residual(sit, cand.plan)
        cost = FIXES[cand.plan[0]] if cand.plan else 0.0
        return Evaluation({"residual": res, "cost": cost})

    def stress_test(self, sit: ToySituation, cand: Candidate, rng: np.random.Generator) -> Evaluation:
        ev = self.evaluate(sit, cand)
        noisy = dict(ev.metrics)
        noisy["residual"] = min(1.0, noisy["residual"] + abs(rng.normal(0, 0.05)))
        return Evaluation(noisy)

    def combine(self, sit: ToySituation, cands: list[Candidate]) -> Candidate | None:
        return None

    def resolved(self, ev: Evaluation) -> bool:
        return ev.metrics.get("residual", 1.0) < 0.1

    # -- gating ---------------------------------------------------------------------
    def risk_features(self, sit, cand, ev, st) -> dict[str, float]:
        fix, node = cand.plan
        return {
            "is_replace": float(fix == "replace"),
            "critical_node": float(node in sit.critical),
            "cost": FIXES[fix],
            "residual": ev.metrics["residual"],
        }

    def risk_score(self, f: dict[str, float]) -> float:
        return 0.5 * f["is_replace"] + 0.3 * f["critical_node"] + 0.2 * f["residual"]

    def operator_policy(self) -> Callable[[Any], bool]:
        return lambda p: not (p.features["is_replace"] and p.features["critical_node"])

    # -- live -------------------------------------------------------------------------
    def apply(self, sit: ToySituation, cand: Candidate | None) -> dict[str, float]:
        res = self._residual(sit, None if cand is None else cand.plan)
        return {"residual": res, "survived": float(res < 0.5), "cost": FIXES[cand.plan[0]] if cand else 0.0}

    def describe(self, sit: ToySituation, cand: Candidate | None) -> str:
        if cand is None:
            return f"Fault somewhere in a {sit.graph.number_of_nodes()}-node network"
        return f"Apply {cand.plan[0]} at node {cand.plan[1]}"

    @staticmethod
    def default_objective() -> dict[str, float]:
        return {"residual": -3.0, "cost": -0.3, "survived": 2.0, "interrupts": -0.5,
                "compute_cost": -0.2, "misses": -3.0}
