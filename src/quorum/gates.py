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
"""Human gates.

A gate looks at a proposed plan and decides one of:

* ``auto``     apply it without asking anyone
* ``escalate`` interrupt a human, who approves or rejects
* ``block``    do not apply it; fall back to the safe default

Open core ships a fixed ``ThresholdGate``. A gate that *learns* when to
interrupt people (treating human attention as a scarce resource) plugs in
through the same interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class Proposal:
    pid: str
    plan: Any
    summary: str
    features: dict[str, float]
    risk: float


@dataclass
class GateDecision:
    action: str  # "auto" | "escalate" | "block"
    p_approve: float
    reason: str


class Gate(Protocol):
    name: str

    def decide(self, proposal: Proposal, interrupts_left: int) -> GateDecision: ...

    def observe(self, proposal: Proposal, approved: bool) -> None: ...


class ThresholdGate:
    """Escalate anything whose risk score is at or above a fixed threshold."""

    name = "threshold"

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def decide(self, proposal: Proposal, interrupts_left: int) -> GateDecision:
        if proposal.risk >= self.threshold:
            if interrupts_left > 0:
                return GateDecision("escalate", 0.5, f"risk {proposal.risk:.2f} >= {self.threshold}")
            return GateDecision("block", 0.5, "risky and no interrupt budget left")
        return GateDecision("auto", 1.0, f"risk {proposal.risk:.2f} < {self.threshold}")

    def observe(self, proposal: Proposal, approved: bool) -> None:
        return None


class AlwaysAskGate:
    """Human review of every plan: the usual enterprise human-in-the-loop setup."""

    name = "always_ask"

    def decide(self, proposal: Proposal, interrupts_left: int) -> GateDecision:
        if interrupts_left > 0:
            return GateDecision("escalate", 0.5, "policy: always ask")
        return GateDecision("block", 0.5, "no interrupt budget left")

    def observe(self, proposal: Proposal, approved: bool) -> None:
        return None


class AutoGate:
    """Full autonomy: never asks anyone."""

    name = "auto"

    def decide(self, proposal: Proposal, interrupts_left: int) -> GateDecision:
        return GateDecision("auto", 1.0, "policy: full autonomy")

    def observe(self, proposal: Proposal, approved: bool) -> None:
        return None


class Operator(Protocol):
    def review(self, proposal: Proposal) -> bool: ...


class CLIOperator:
    """A real person at the terminal."""

    def review(self, proposal: Proposal) -> bool:
        print("\n=== Approval needed ===")
        print(proposal.summary)
        for k, v in proposal.features.items():
            print(f"  {k:>22}: {v:.3f}")
        ans = input("Approve? [y/N] ").strip().lower()
        return ans in ("y", "yes")


class SimulatedOperator:
    """A stand-in operator for headless runs and experiments.

    Its policy is hidden from the gate; the gate only sees its decisions.
    """

    def __init__(self, policy: Callable[[Proposal], bool]):
        self.policy = policy

    def review(self, proposal: Proposal) -> bool:
        return bool(self.policy(proposal))


@dataclass
class GateRecord:
    pid: str
    decision: str
    approved: bool
    p_approve: float
    would_approve: bool | None  # ground truth, only known in simulation


@dataclass
class GateKeeper:
    gate: Any
    operator: Any
    oracle: Callable[[Proposal], bool] | None = None  # sim-only ground truth for scoring misses
    records: list[GateRecord] = field(default_factory=list)

    def review(self, proposal: Proposal, interrupts_left: int) -> tuple[bool, GateDecision, bool]:
        """Returns (approved, decision, interrupted)."""
        decision = self.gate.decide(proposal, interrupts_left)
        interrupted = False
        if decision.action == "escalate":
            approved = bool(self.operator.review(proposal))
            interrupted = True
            self.gate.observe(proposal, approved)
        else:
            approved = decision.action == "auto"
        truth = self.oracle(proposal) if self.oracle else None
        self.records.append(GateRecord(proposal.pid, decision.action, approved, decision.p_approve, truth))
        return approved, decision, interrupted

    @staticmethod
    def is_miss(rec: GateRecord) -> bool:
        """An auto-approved plan the human would have rejected."""
        return rec.decision == "auto" and rec.would_approve is False
