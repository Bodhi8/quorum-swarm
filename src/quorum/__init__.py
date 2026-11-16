"""Quorum: a learned control plane for agent swarms.

ML steers the swarm; humans steer the ML.
"""
from .controller import Controller, StaticController, StaticFieldPolicy, ThompsonBandit
from .domain import Candidate, Domain, Evaluation, Strategy
from .events import Event, EventBus, read_events
from .field import FieldKnobs, SemanticField
from .gates import CLIOperator, GateKeeper, Proposal, SimulatedOperator, ThresholdGate
from .intent import HardRule, Intent, forbid, max_value
from .llm import AnthropicLLM, MockLLM, get_backend
from .safety import Budget, CircuitBreaker, KillSwitch, SafetyMonitor, SwarmHalted
from .swarm import EpisodeResult, Swarm, SwarmConfig
from .twin import ShadowTwin

__version__ = "0.1.0"

__all__ = [
    "Candidate", "CircuitBreaker", "CLIOperator", "Controller", "Domain", "Evaluation", "Event", "EventBus",
    "EpisodeResult", "FieldKnobs", "GateKeeper", "HardRule", "Intent", "KillSwitch", "MockLLM", "AnthropicLLM",
    "Proposal", "SafetyMonitor", "SemanticField", "ShadowTwin", "SimulatedOperator", "StaticController",
    "StaticFieldPolicy", "Strategy", "Swarm", "SwarmConfig", "SwarmHalted", "ThompsonBandit", "ThresholdGate",
    "Budget", "forbid", "get_backend", "max_value", "read_events",
]
