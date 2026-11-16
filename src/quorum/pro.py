"""Quorum Pro: Vector1's commercial extensions to this open core.

The open core is the full runtime and every interface. Quorum Pro plugs in
through the same plugin registry and adds what the published results depend
on:

* ``learned`` gate: learns when a human actually needs to be asked
* ``adaptive`` field policy: self-tuning field physics, vetted by the shadow twin
* ``grid`` domain: grid stress response on Grid2Op, unit commitment as a QUBO,
  Southwest Power Pool market data

Quorum Pro lives in a private repository (Bodhi8/quorum-pro). Access is
available to design partners, pilots and research collaborators via
https://vector1.ai.
"""
from __future__ import annotations

import sys
from importlib.util import find_spec

PRO_URL = "https://vector1.ai"
PRO_REPO = "github.com/Bodhi8/quorum-pro (private)"

PRO_FEATURES = {
    "gates": {"learned": "learned gate: asks a human only when it is unsure"},
    "field_policies": {"adaptive": "self-tuning field physics, vetted by the shadow twin"},
    "domains": {"grid": "grid stress response (Grid2Op), unit commitment QUBO, SPP market data"},
}

_told: set[str] = set()
_LABEL = {"gates": "gate", "field_policies": "field policy", "domains": "domain"}


def pro_installed() -> bool:
    return find_spec("quorum_pro") is not None


def pro_feature(kind: str, name: str) -> str | None:
    return PRO_FEATURES.get(kind, {}).get(name)


def pro_message(kind: str, name: str) -> str:
    desc = pro_feature(kind, name) or name
    return (f"'{name}' is a Quorum Pro feature ({desc}). Quorum Pro is Vector1's commercial extension "
            f"to this open core; request access at {PRO_URL}.")


def notice(kind: str, name: str, fallback: str) -> None:
    """Tell the user once per feature that a Pro component was replaced by an open one."""
    key = f"{kind}:{name}"
    if key in _told:
        return
    _told.add(key)
    print(f"[quorum] using '{fallback}' {_LABEL.get(kind, kind)}; {pro_message(kind, name)}", file=sys.stderr)
