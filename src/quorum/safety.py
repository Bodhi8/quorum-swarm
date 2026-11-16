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
"""Guardrails that sit outside the learning loop.

Nothing here is learned or tunable by the controller: budgets, circuit
breakers and the kill switch are fixed by the human layer.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path


class SwarmHalted(RuntimeError):
    """Raised when a guardrail stops the swarm."""


class KillSwitch:
    """Stops the swarm from code, another thread, or by touching a file."""

    def __init__(self, flag_file: str | Path | None = None):
        self._event = threading.Event()
        self.flag_file = Path(flag_file) if flag_file else None
        self.reason = ""

    def trip(self, reason: str = "manual") -> None:
        self.reason = reason
        self._event.set()

    @property
    def tripped(self) -> bool:
        if self.flag_file is not None and self.flag_file.exists():
            self.reason = self.reason or f"flag file {self.flag_file}"
            self._event.set()
        return self._event.is_set()


@dataclass
class CircuitBreaker:
    """Opens after too many consecutive bad live outcomes."""

    max_consecutive_failures: int = 3
    failures: int = 0
    open: bool = False

    def record(self, ok: bool) -> None:
        self.failures = 0 if ok else self.failures + 1
        if self.failures >= self.max_consecutive_failures:
            self.open = True


@dataclass
class Budget:
    limit: float
    spent: float = 0.0
    log: list[tuple[str, float]] = field(default_factory=list)

    def charge(self, what: str, cost: float) -> None:
        self.spent += cost
        self.log.append((what, cost))

    @property
    def remaining(self) -> float:
        return self.limit - self.spent

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.limit


class SafetyMonitor:
    def __init__(self, kill_switch: KillSwitch | None = None, breaker: CircuitBreaker | None = None):
        self.kill_switch = kill_switch or KillSwitch()
        self.breaker = breaker or CircuitBreaker()

    def check(self) -> None:
        if self.kill_switch.tripped:
            raise SwarmHalted(f"kill switch: {self.kill_switch.reason}")
        if self.breaker.open:
            raise SwarmHalted(
                f"circuit breaker open after {self.breaker.failures} consecutive failed outcomes"
            )
