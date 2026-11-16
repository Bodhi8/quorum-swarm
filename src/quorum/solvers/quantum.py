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
"""Quantum solver plug-ins. Optional, import-guarded, never on the critical path.

* ``DWaveSolver``: D-Wave Leap (annealing). ``pip install dwave-ocean-sdk``
  and set ``DWAVE_API_TOKEN``.
* ``QAOASolver``: QAOA via Qiskit. Runs on the local simulator by default;
  pass a backend for IBM hardware (``pip install qiskit qiskit-ibm-runtime``).

Both return the same ``Solution`` type as the classical solvers, so a
benchmark can swap them in and compare like for like.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import QUBO, Solution


class QuantumUnavailable(RuntimeError):
    pass


class DWaveSolver:
    name = "dwave"

    def __init__(self, hybrid: bool = False, num_reads: int = 200):
        self.hybrid, self.num_reads = hybrid, num_reads

    def solve(self, qubo: QUBO, **kw: Any) -> Solution:
        try:
            import dimod  # type: ignore
            if self.hybrid:
                from dwave.system import LeapHybridSampler  # type: ignore
                sampler = LeapHybridSampler()
            else:
                from dwave.system import DWaveSampler, EmbeddingComposite  # type: ignore
                sampler = EmbeddingComposite(DWaveSampler())
        except Exception as exc:  # pragma: no cover - needs credentials/hardware
            raise QuantumUnavailable(f"D-Wave not available: {exc}") from exc
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.to_dict(), offset=qubo.offset)
        kwargs = {} if self.hybrid else {"num_reads": self.num_reads}
        ss = sampler.sample(bqm, **kwargs)
        best = ss.first
        x = np.array([best.sample[label] for label in qubo.labels], dtype=float)
        return Solution(x, float(best.energy), self.name, {"timing": dict(ss.info.get("timing", {}))})


class QAOASolver:
    name = "qaoa"

    def __init__(self, reps: int = 2, backend: Any = None, shots: int = 1024):
        self.reps, self.backend, self.shots = reps, backend, shots

    def solve(self, qubo: QUBO, **kw: Any) -> Solution:
        try:
            from qiskit.circuit.library import QAOAAnsatz  # type: ignore
            from qiskit.quantum_info import SparsePauliOp  # type: ignore
            from qiskit.primitives import StatevectorSampler  # type: ignore
            from scipy.optimize import minimize  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise QuantumUnavailable(f"Qiskit not available: {exc}") from exc
        # QUBO -> Ising: x = (1 - z) / 2
        Q = (qubo.Q + qubo.Q.T) / 2.0
        n = qubo.n
        terms, coeffs = [], []
        for i in range(n):
            h = -0.5 * Q[i, i] - 0.5 * sum(Q[i, j] for j in range(n) if j != i)
            if abs(h) > 1e-12:
                terms.append("".join("Z" if k == i else "I" for k in reversed(range(n))))
                coeffs.append(h)
            for j in range(i + 1, n):
                if abs(Q[i, j]) > 1e-12:
                    terms.append("".join("Z" if k in (i, j) else "I" for k in reversed(range(n))))
                    coeffs.append(0.5 * Q[i, j])
        op = SparsePauliOp(terms or ["I" * n], coeffs or [0.0])
        ansatz = QAOAAnsatz(op, reps=self.reps)
        ansatz.measure_all()
        sampler = StatevectorSampler() if self.backend is None else self.backend

        def expected(params: np.ndarray) -> float:
            counts = sampler.run([(ansatz, params)], shots=self.shots).result()[0].data.meas.get_counts()
            total = sum(counts.values())
            return sum(c * qubo.energy(np.array([int(b) for b in reversed(k)])) for k, c in counts.items()) / total

        x0 = np.full(ansatz.num_parameters, 0.5)
        opt = minimize(expected, x0, method="COBYLA", options={"maxiter": 60})
        counts = sampler.run([(ansatz, opt.x)], shots=self.shots).result()[0].data.meas.get_counts()
        best_bits = min(counts, key=lambda k: qubo.energy(np.array([int(b) for b in reversed(k)])))
        x = np.array([int(b) for b in reversed(best_bits)], dtype=float)
        return Solution(x, qubo.energy(x), self.name, {"reps": self.reps, "evals": int(opt.nfev)})
