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
"""Reproduce the v0.1 results table (docs/results-v0.1.md).

    python examples/benchmark.py --out runs/benchmark
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore")

from quorum.experiments import compare, gate_study  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/benchmark")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--scenarios", type=int, default=80)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from quorum.pro import pro_installed, pro_message

    domains = ["toy"]
    if pro_installed():
        domains.append("grid")
    else:
        print("Running the toy domain only. " + pro_message("domains", "grid"))
    results = {}
    for dom in domains:
        t = time.time()
        results[dom] = {
            "compare": compare(dom, episodes=a.episodes, seeds=tuple(a.seeds)),
            "gates": [asdict(g) for g in gate_study(dom, scenarios=a.scenarios)],
            "seconds": round(time.time() - t, 1),
        }
        print(dom, "done in", results[dom]["seconds"], "s", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("wrote", out / "results.json")


if __name__ == "__main__":
    main()
