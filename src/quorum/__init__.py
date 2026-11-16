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

__version__ = "0.2.0"

__all__ = [
    "Candidate", "CircuitBreaker", "CLIOperator", "Controller", "Domain", "Evaluation", "Event", "EventBus",
    "EpisodeResult", "FieldKnobs", "GateKeeper", "HardRule", "Intent", "KillSwitch", "MockLLM", "AnthropicLLM",
    "Proposal", "SafetyMonitor", "SemanticField", "ShadowTwin", "SimulatedOperator", "StaticController",
    "StaticFieldPolicy", "Strategy", "Swarm", "SwarmConfig", "SwarmHalted", "ThompsonBandit", "ThresholdGate",
    "Budget", "forbid", "get_backend", "max_value", "read_events",
]
