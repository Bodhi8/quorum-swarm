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
"""Run the comparison with a real LLM scout instead of the offline mock.

Needs: pip install anthropic, ANTHROPIC_API_KEY, and (for dollar figures) the
current per-million-token prices in QUORUM_USD_IN / QUORUM_USD_OUT.

    export ANTHROPIC_API_KEY=...
    export QUORUM_MODEL=claude-haiku-5-5 QUORUM_USD_IN=... QUORUM_USD_OUT=...
    python examples/real_model.py --domain grid --episodes 20 --seeds 0 1 2

Reports, per configuration: survival, real LLM calls, tokens and dollars per episode.
The grid domain needs Quorum Pro; the toy domain runs on the open core.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from quorum import plugins  # noqa: E402
from quorum.build import build_swarm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="toy")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/real-model.json")
    a = ap.parse_args()
    dom = plugins.get_domain(a.domain)
    results = {}
    for mode in ("static", "quorum"):
        rows = []
        for s in a.seeds:
            sw = build_swarm(dom, mode=mode, seed=s, llm="anthropic")
            sw.run(a.episodes)
            rows += [h.metrics for h in sw.history]
        results[mode] = {k: float(np.mean([r[k] for r in rows]))
                         for k in ("survived", "llm_calls", "llm_tokens", "llm_usd", "interrupts", "misses")}
        print(mode, json.dumps({k: round(v, 4) for k, v in results[mode].items()}))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
