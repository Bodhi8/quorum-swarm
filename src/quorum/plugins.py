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
"""Plugin registry.

Domains, gates and field policies register here by name. Third-party
packages (including ``quorum_pro``) register themselves through the
``quorum.plugins`` entry-point group: each entry point is a callable that
receives this module and calls the ``register_*`` functions.
"""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Callable

_DOMAINS: dict[str, Callable[..., Any]] = {}
_GATES: dict[str, Callable[..., Any]] = {}
_FIELD_POLICIES: dict[str, Callable[..., Any]] = {}
_loaded = False


def register_domain(name: str, factory: Callable[..., Any]) -> None:
    _DOMAINS[name] = factory


def register_gate(name: str, factory: Callable[..., Any]) -> None:
    _GATES[name] = factory


def register_field_policy(name: str, factory: Callable[..., Any]) -> None:
    _FIELD_POLICIES[name] = factory


def load_plugins() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    import sys

    from . import plugins as me
    from .controller import StaticFieldPolicy
    from .domains.toy import ToyDomain
    from .gates import AlwaysAskGate, AutoGate, ThresholdGate

    register_domain("toy", ToyDomain)
    register_gate("threshold", ThresholdGate)
    register_gate("always_ask", AlwaysAskGate)
    register_gate("auto", AutoGate)
    register_field_policy("static", StaticFieldPolicy)
    for ep in entry_points(group="quorum.plugins"):
        try:
            ep.load()(me)
        except Exception as exc:  # a broken plugin must not take down the core
            print(f"[quorum] plugin {ep.name!r} failed to load: {exc}", file=sys.stderr)


class ProFeatureUnavailable(KeyError):
    """Raised when a Quorum Pro component is requested but Quorum Pro is not installed."""


def _get(table: dict[str, Callable[..., Any]], kind: str, name: str, group: str, **kw: Any) -> Any:
    load_plugins()
    if name not in table:
        from .pro import pro_feature, pro_message

        if pro_feature(group, name):
            raise ProFeatureUnavailable(pro_message(group, name))
        raise KeyError(f"unknown {kind} {name!r}; available: {sorted(table)}")
    return table[name](**kw)


def get_domain(name: str, **kw: Any) -> Any:
    return _get(_DOMAINS, "domain", name, "domains", **kw)


def get_gate(name: str, **kw: Any) -> Any:
    return _get(_GATES, "gate", name, "gates", **kw)


def get_field_policy(name: str, **kw: Any) -> Any:
    return _get(_FIELD_POLICIES, "field policy", name, "field_policies", **kw)


def available() -> dict[str, list[str]]:
    load_plugins()
    return {"domains": sorted(_DOMAINS), "gates": sorted(_GATES), "field_policies": sorted(_FIELD_POLICIES)}
