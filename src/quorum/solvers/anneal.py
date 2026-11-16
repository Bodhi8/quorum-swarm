"""Simulated annealing: the free, always-available classical default."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import QUBO, Solution


class SimulatedAnnealingSolver:
    name = "anneal"

    def __init__(self, sweeps: int = 1000, reads: int = 32, t_start: float | None = None, t_end: float = 1e-3, seed: int = 0):
        self.sweeps, self.reads, self.t_start, self.t_end, self.seed = sweeps, reads, t_start, t_end, seed

    def solve(self, qubo: QUBO, **kw: Any) -> Solution:
        rng = np.random.default_rng(kw.get("seed", self.seed))
        Q = (qubo.Q + qubo.Q.T) / 2.0  # symmetric form; energy unchanged
        n = qubo.n
        diag = np.diag(Q).copy()
        t0 = self.t_start or max(1.0, float(np.abs(Q).sum(axis=1).max()))
        temps = np.geomspace(t0, self.t_end, self.sweeps)
        best_x, best_e = None, np.inf
        start = time.perf_counter()
        for _ in range(self.reads):
            x = rng.integers(0, 2, n).astype(float)
            field = Q @ x  # local fields
            e = float(x @ Q @ x)
            for T in temps:
                for i in rng.permutation(n):
                    # energy change of flipping bit i: E(x + d e_i) - E(x)
                    d = 1.0 - 2.0 * x[i]
                    de = 2.0 * d * field[i] + diag[i] * d * d
                    if de <= 0 or rng.random() < np.exp(-de / T):
                        x[i] += d
                        field += d * Q[:, i]
                        e += de
            if e < best_e:
                best_x, best_e = x.copy(), e
        return Solution(best_x, float(best_e + qubo.offset), self.name,
                        {"reads": self.reads, "sweeps": self.sweeps, "seconds": time.perf_counter() - start})
