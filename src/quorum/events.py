"""Event-sourced bus.

Every decision in a Quorum run (field updates, scout choices, evaluations,
gate decisions, policy changes) is published as an Event. Subscribers get
them live (dashboards, the replay viewer) and a JSONL sink keeps the full
record so any run can be replayed, audited or used as training data.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class Event:
    kind: str
    episode: int
    tick: int
    data: dict[str, Any]
    ts: float = field(default_factory=time.time)
    shadow: bool = False


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


Handler = Callable[[Event], None]


class EventBus:
    """In-process pub/sub with an optional JSONL sink."""

    def __init__(self, sink: str | Path | None = None, keep: bool = True):
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self.episode = 0
        self.tick = 0
        self.shadow = False
        self.keep = keep
        self.history: list[Event] = []
        self._fh = None
        if sink is not None:
            path = Path(sink)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")

    def subscribe(self, kind: str, handler: Handler) -> None:
        """Subscribe to one event kind, or '*' for everything."""
        self._subs[kind].append(handler)

    def publish(self, kind: str, **data: Any) -> Event | None:
        if self.shadow:
            # Shadow-twin runs are evaluated, not recorded as live history.
            return None
        ev = Event(kind=kind, episode=self.episode, tick=self.tick, data=_jsonable(data))
        if self.keep:
            self.history.append(ev)
        if self._fh is not None:
            self._fh.write(json.dumps(asdict(ev)) + "\n")
        for h in self._subs.get(kind, []) + self._subs.get("*", []):
            h(ev)
        return ev

    def of(self, kind: str) -> list[Event]:
        return [e for e in self.history if e.kind == kind]

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_events(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
