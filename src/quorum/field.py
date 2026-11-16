"""The physics-informed semantic field.

Agents do not message each other. They coordinate through a shared field
laid over a graph of the problem space (stigmergy). Each channel on the
field evolves as an advection-diffusion-reaction (ADR) system on the graph:

    dp/dt = -D * L p           diffusion: good news spreads to neighbours
            + A(s) p           advection: drift toward high-stress regions
            - k * p            decay: stale signals fade
            + r * p (1 - p)    reaction: recruitment saturates
            + deposits         sources: what agents report

The ML controller tunes a handful of bounded physical knobs (drift,
diffusion, decay, recruitment, temperature) instead of thousands of opaque
parameters, so every policy change is interpretable. ``stability_margin``
is a CFL-style bound on the explicit update; the field refuses knobs that
would make it oscillate or blow up.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Hashable

import networkx as nx
import numpy as np

CHANNELS = ("promise", "dead_end", "stress", "density")


@dataclass(frozen=True)
class FieldKnobs:
    drift: float = 0.3  # advection strength toward stress
    diffusion: float = 0.2  # spread along graph edges
    decay: float = 0.15  # forgetting rate
    recruitment: float = 0.3  # logistic self-reinforcement of promise
    temperature: float = 3.0  # how sharply agents follow the field
    crowding: float = 0.5  # repulsion from where agents already are
    dt: float = 1.0

    BOUNDS = {
        "drift": (0.0, 2.0),
        "diffusion": (0.0, 1.5),
        "decay": (0.01, 0.9),
        "recruitment": (0.0, 1.0),
        "temperature": (0.2, 10.0),
        "crowding": (0.0, 3.0),
    }

    def clipped(self) -> "FieldKnobs":
        vals = {k: float(np.clip(getattr(self, k), lo, hi)) for k, (lo, hi) in self.BOUNDS.items()}
        return replace(self, **vals)

    def as_dict(self) -> dict[str, float]:
        return {k: v for k, v in asdict(self).items()}


class SemanticField:
    def __init__(self, graph: nx.Graph, knobs: FieldKnobs | None = None):
        if graph.number_of_nodes() == 0:
            raise ValueError("field graph needs at least one node")
        self.graph = graph
        self.nodes: list[Hashable] = list(graph.nodes)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.n = len(self.nodes)
        # combinatorial Laplacian L = D - A, built with numpy only (no scipy dependency)
        A = np.zeros((self.n, self.n))
        for u, v in graph.edges:
            if u != v:
                A[self.index[u], self.index[v]] = A[self.index[v], self.index[u]] = 1.0
        self.L = np.diag(A.sum(axis=1)) - A
        self.lambda_max = float(np.max(np.linalg.eigvalsh(self.L))) if self.n > 1 else 0.0
        self.max_degree = max((d for _, d in graph.degree()), default=0)
        self.values = {c: np.zeros(self.n) for c in CHANNELS}
        self._pending = {c: np.zeros(self.n) for c in ("promise", "dead_end")}
        self.knobs = (knobs or FieldKnobs()).clipped()
        self.knobs = self.stabilize(self.knobs)

    # -- stability --------------------------------------------------------
    def stability_margin(self, knobs: FieldKnobs | None = None) -> float:
        """1 - (spectral bound of the explicit update). Must stay > 0."""
        k = knobs or self.knobs
        rate = k.diffusion * self.lambda_max + k.decay + k.recruitment + k.drift * self.max_degree
        return 1.0 - k.dt * rate

    def stabilize(self, knobs: FieldKnobs, margin: float = 0.05) -> FieldKnobs:
        """Shrink the time step until the update is provably stable."""
        k = knobs
        while self.stability_margin(k) < margin and k.dt > 1e-3:
            k = replace(k, dt=k.dt * 0.8)
        return k

    def set_knobs(self, knobs: FieldKnobs) -> FieldKnobs:
        self.knobs = self.stabilize(knobs.clipped())
        return self.knobs

    # -- sources ------------------------------------------------------------
    def deposit(self, node: Hashable, channel: str, amount: float) -> None:
        if node in self.index and channel in self._pending:
            self._pending[channel][self.index[node]] += float(amount)

    def set_channel(self, channel: str, mapping: dict[Hashable, float]) -> None:
        vec = np.zeros(self.n)
        for node, v in mapping.items():
            if node in self.index:
                vec[self.index[node]] = float(v)
        self.values[channel] = vec

    # -- dynamics -------------------------------------------------------------
    def _advection(self, p: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Upwind transport of p along edges, from low to high stress."""
        out = np.zeros_like(p)
        for u, v in self.graph.edges:
            i, j = self.index[u], self.index[v]
            if s[j] > s[i]:
                flux = (s[j] - s[i]) * p[i]
                out[i] -= flux
                out[j] += flux
            elif s[i] > s[j]:
                flux = (s[i] - s[j]) * p[j]
                out[j] -= flux
                out[i] += flux
        return out

    def step(self) -> None:
        k = self.knobs
        s = self.values["stress"]
        s_norm = s / (s.max() + 1e-9) if s.max() > 0 else s

        p = self.values["promise"]
        dp = (
            -k.diffusion * self.L @ p
            + k.drift * self._advection(p, s_norm)
            - k.decay * p
            + k.recruitment * p * (1.0 - np.clip(p, 0, None))
        )
        self.values["promise"] = np.clip(p + k.dt * dp + self._pending["promise"], 0.0, 5.0)

        d = self.values["dead_end"]
        dd = -0.5 * k.diffusion * self.L @ d - 0.5 * k.decay * d
        self.values["dead_end"] = np.clip(d + k.dt * dd + self._pending["dead_end"], 0.0, 5.0)

        self.values["density"] *= 0.5
        for c in self._pending:
            self._pending[c][:] = 0.0

    # -- reading --------------------------------------------------------------
    def attractiveness(self, stress_weight: float = 1.0) -> np.ndarray:
        v = self.values
        s = v["stress"] / (v["stress"].max() + 1e-9) if v["stress"].max() > 0 else v["stress"]
        return v["promise"] - v["dead_end"] + stress_weight * s - self.knobs.crowding * v["density"]

    def distribution(self, stress_weight: float = 1.0) -> np.ndarray:
        a = self.knobs.temperature * self.attractiveness(stress_weight)
        a = a - a.max()
        w = np.exp(a)
        return w / w.sum()

    def sample(self, rng: np.random.Generator, size: int = 1, stress_weight: float = 1.0) -> list[Hashable]:
        p = self.distribution(stress_weight)
        idx = rng.choice(self.n, size=size, p=p, replace=size > self.n)
        for i in np.atleast_1d(idx):
            self.values["density"][i] += 1.0
        return [self.nodes[i] for i in np.atleast_1d(idx)]

    def entropy(self) -> float:
        p = self.distribution()
        return float(-(p * np.log(p + 1e-12)).sum() / np.log(self.n)) if self.n > 1 else 0.0

    def snapshot(self) -> dict[str, list[float]]:
        return {c: [round(float(x), 4) for x in self.values[c]] for c in ("promise", "dead_end", "stress")}

    def reset(self) -> None:
        for c in CHANNELS:
            self.values[c][:] = 0.0
        for c in self._pending:
            self._pending[c][:] = 0.0
