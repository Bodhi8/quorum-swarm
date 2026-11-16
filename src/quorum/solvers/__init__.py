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
