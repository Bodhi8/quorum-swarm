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
from .anneal import SimulatedAnnealingSolver
from .base import QUBO, ExactSolver, Solution, Solver, available_solvers, get_solver, register_solver
from .quantum import DWaveSolver, QAOASolver, QuantumUnavailable

register_solver("exact", ExactSolver)
register_solver("anneal", SimulatedAnnealingSolver)
register_solver("dwave", DWaveSolver)
register_solver("dwave-hybrid", lambda: DWaveSolver(hybrid=True))
register_solver("qaoa", QAOASolver)

__all__ = [
    "QUBO", "Solution", "Solver", "ExactSolver", "SimulatedAnnealingSolver", "DWaveSolver", "QAOASolver",
    "QuantumUnavailable", "available_solvers", "get_solver", "register_solver",
]
