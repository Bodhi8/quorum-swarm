"""One interface for combinatorial solvers: classical by default, quantum as plug-ins.

Problems are expressed as QUBOs (quadratic unconstrained binary
optimization): minimise  x^T Q x + offset  over binary x. Unit commitment,
team-to-task assignment and many scheduling problems map onto this form,
and so does every current quantum optimizer (annealers and QAOA).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


@dataclass
class QUBO:
    Q: np.ndarray  # upper-triangular or symmetric; energy uses x^T Q x
    offset: float = 0.0
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.Q = np.asarray(self.Q, dtype=float)
        if self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError("Q must be square")
        if not self.labels:
            self.labels = [f"x{i}" for i in range(self.n)]

    @property
    def n(self) -> int:
        return self.Q.shape[0]

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def to_dict(self) -> dict[tuple[str, str], float]:
        """Ocean/dimod-style {(u, v): bias} form."""
        out: dict[tuple[str, str], float] = {}
        Qs = np.triu(self.Q + self.Q.T - np.diag(np.diag(self.Q)))
        for i, j in zip(*np.nonzero(Qs)):
            out[(self.labels[i], self.labels[j])] = float(Qs[i, j])
        return out


@dataclass
class Solution:
    x: np.ndarray
    energy: float
    solver: str
    meta: dict[str, Any] = field(default_factory=dict)


class Solver(Protocol):
    name: str

    def solve(self, qubo: QUBO, **kw: Any) -> Solution: ...


class ExactSolver:
    """Brute force. Ground truth for small benchmark instances (n <= 22)."""

    name = "exact"

    def solve(self, qubo: QUBO, **kw: Any) -> Solution:
        if qubo.n > 22:
            raise ValueError("exact solver is limited to 22 variables")
        best_x, best_e = None, np.inf
        for bits in itertools.product((0, 1), repeat=qubo.n):
            x = np.array(bits, dtype=float)
            e = qubo.energy(x)
            if e < best_e:
                best_x, best_e = x, e
        return Solution(best_x, float(best_e), self.name, {"evaluated": 2**qubo.n})


_REGISTRY: dict[str, Callable[[], Solver]] = {}


def register_solver(name: str, factory: Callable[[], Solver]) -> None:
    _REGISTRY[name] = factory


def get_solver(name: str) -> Solver:
    if name not in _REGISTRY:
        raise KeyError(f"unknown solver {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available_solvers() -> list[str]:
    return sorted(_REGISTRY)
