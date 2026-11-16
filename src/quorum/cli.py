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
"""Command line: run, compare, study gates, benchmark solvers, render replays."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _table(headers: list[str], rows: list[list[str]]) -> str:
    w = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = lambda r: "| " + " | ".join(str(x).ljust(n) for x, n in zip(r, w)) + " |"  # noqa: E731
    return "\n".join([line(headers), "|" + "|".join("-" * (n + 2) for n in w) + "|"] + [line(r) for r in rows])


def cmd_run(a: argparse.Namespace) -> None:
    from .build import build_swarm
    from .viz import render

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sw = build_swarm(a.domain, mode=a.mode, seed=a.seed, events=out / "events.jsonl", interactive=a.interactive,
                     llm=a.llm)

    def progress(r) -> None:
        m = r.metrics
        print(f"ep {r.episode:>3}  survived={m.get('survived', 1):.0f}  gate={r.decision:<8} "
              f"interrupts={m['interrupts']:.0f}  cost={m['compute_cost']:.2f}  reward={r.reward:+.2f}")

    sw.run(a.episodes, on_episode=progress)
    sw.bus.close()
    html = render(out / "events.jsonl", out / "replay.html", title=f"Quorum · {a.domain} · {a.mode}",
                  meta={"domain": a.domain, "mode": a.mode})
    summary = {"controller": sw.controller.summary(), "knobs": sw.knobs.as_dict()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nreplay: {html}\nevents: {out / 'events.jsonl'}")


def cmd_compare(a: argparse.Namespace) -> None:
    from .experiments import compare

    res = compare(a.domain, episodes=a.episodes, seeds=tuple(a.seeds), events_dir=a.out)
    rows = []
    for mode, v in res.items():
        rows.append([mode, f"{v['survival']:.1%}", f"{v['compute_cost']:.2f}", f"{v['interrupts']:.2f}",
                     f"{v['interrupts_late']:.2f}", f"{v['misses']:.3f}", f"{v['reward']:.2f}"])
    print(_table(["mode", "survival", "compute/ep", "interrupts/ep", "interrupts (2nd half)", "misses/ep", "reward"], rows))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "compare.json").write_text(json.dumps(res, indent=2))


def cmd_gates(a: argparse.Namespace) -> None:
    from .experiments import gate_study

    stats = gate_study(a.domain, scenarios=a.scenarios, seed=a.seed)
    rows = [[s.gate, s.n, s.interrupts, s.interrupts_late, s.misses, s.misses_late, s.lost] for s in stats]
    print(_table(["gate", "plans", "interrupts", "interrupts (2nd half)", "misses", "misses (2nd half)",
                  "good plans blocked"], [[str(x) for x in r] for r in rows]))
    if "learned" not in {s.gate for s in stats}:
        from .pro import pro_message

        print("\n" + pro_message("gates", "learned"))


def cmd_ablate(a: argparse.Namespace) -> None:
    from .experiments import ablation

    res = ablation(a.domain, episodes=a.episodes, seeds=tuple(range(a.seeds)),
                   progress=lambda r, s: print(f"  {r:<9} seed {s}", file=sys.stderr, flush=True))
    rows = []
    for r in res["rungs"]:
        if "skipped" in r:
            rows.append([r["label"], "-", "-", "-", "-", r["skipped"]])
            continue
        f = lambda m, pct=False: (f"{100*r[m][0]:.1f}% ±{100*r[m][1]:.1f}" if pct  # noqa: E731
                                  else f"{r[m][0]:.3f} ±{r[m][1]:.3f}")
        rows.append([r["label"], f("survived", True), f("compute_cost"), f("interrupts"), f("misses"), ""])
    print(_table(["configuration", "survived", "compute/ep", "interrupts/ep", "misses/ep", "note"], rows))
    print(f"{len(res['seeds'])} seeds x {res['episodes']} episodes, mean ±95% CI over seeds")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))


def cmd_uc(a: argparse.Namespace) -> None:
    from . import plugins
    from .solvers import QuantumUnavailable, get_solver

    plugins.load_plugins()
    try:
        from quorum_pro.energy.unit_commitment import build_qubo, decode, from_grid
    except ImportError:
        from .pro import pro_message

        sys.exit(pro_message("domains", "grid") + "\nThe solver interface itself (exact, anneal, dwave, qaoa) "
                 "is open: see quorum.solvers.")
    dom = plugins.get_domain("grid")
    inst = from_grid(dom, hours=a.hours, start=a.start)
    q = build_qubo(inst)
    print(f"Unit commitment: {inst.G} units x {inst.T} hours -> QUBO with {q.n} binary variables")
    rows = []
    for name in a.solvers:
        try:
            sol = get_solver(name).solve(q)
        except (QuantumUnavailable, ValueError) as exc:
            rows.append([name, "-", "-", "-", f"unavailable: {exc}"[:60]])
            continue
        d = decode(inst, sol)
        sched = " ".join(f"{k.split('_')[0]}:{''.join(map(str, v))}" for k, v in d["schedule"].items())
        rows.append([name, f"{d['energy']:.2f}", f"{d['cost']:.2f}", str(d["feasible"]), sched])
    print(_table(["solver", "QUBO energy", "cost", "feasible", "schedule (1=on per hour)"], rows))
    print(f"net demand per hour (MW): {inst.demand.round(1).tolist()}")


def cmd_replay(a: argparse.Namespace) -> None:
    from .viz import render

    print(render(a.events, a.out, title=a.title))


def cmd_plugins(a: argparse.Namespace) -> None:
    from . import plugins
    from .pro import _LABEL, PRO_FEATURES, PRO_URL, pro_installed
    from .solvers import available_solvers

    print(json.dumps({**plugins.available(), "solvers": available_solvers()}, indent=2))
    if pro_installed():
        print("\nQuorum Pro: installed")
    else:
        print(f"\nQuorum Pro: not installed. It adds (request access at {PRO_URL}):")
        for kind, feats in PRO_FEATURES.items():
            for name, desc in feats.items():
                print(f"  {_LABEL[kind]:>12}  {name:<9} {desc}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="quorum", description="Quorum: a learned control plane for agent swarms")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a swarm and write a replay")
    r.add_argument("--domain", default="toy")
    r.add_argument("--mode", choices=["quorum", "static"], default="quorum")
    r.add_argument("--episodes", type=int, default=30)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="runs/latest")
    r.add_argument("--llm", choices=["auto", "mock", "anthropic"], default="mock")
    r.add_argument("--interactive", action="store_true", help="you approve escalated plans at the terminal")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="static wiring vs Quorum on identical scenarios")
    c.add_argument("--domain", default="toy")
    c.add_argument("--episodes", type=int, default=30)
    c.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    c.add_argument("--out", default=None)
    c.set_defaults(fn=cmd_compare)

    g = sub.add_parser("gates", help="replay one stream of plans through different gate policies")
    g.add_argument("--domain", default="toy")
    g.add_argument("--scenarios", type=int, default=60)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_gates)

    ab = sub.add_parser("ablate", help="add one component at a time and measure what each buys")
    ab.add_argument("--domain", default="grid")
    ab.add_argument("--episodes", type=int, default=40)
    ab.add_argument("--seeds", type=int, default=10, help="number of seeds (0..n-1)")
    ab.add_argument("--out", default=None, help="write full results as JSON")
    ab.set_defaults(fn=cmd_ablate)

    u = sub.add_parser("uc", help="unit commitment QUBO benchmark (classical vs quantum solvers)")
    u.add_argument("--hours", type=int, default=3)
    u.add_argument("--start", type=int, default=96)
    u.add_argument("--solvers", nargs="+", default=["exact", "anneal"])
    u.set_defaults(fn=cmd_uc)

    rp = sub.add_parser("replay", help="render an event log to HTML")
    rp.add_argument("events")
    rp.add_argument("-o", "--out", default="replay.html")
    rp.add_argument("--title", default="Quorum run replay")
    rp.set_defaults(fn=cmd_replay)

    pl = sub.add_parser("plugins", help="list registered domains, gates, field policies and solvers")
    pl.set_defaults(fn=cmd_plugins)

    a = p.parse_args(argv)
    from .plugins import ProFeatureUnavailable

    try:
        a.fn(a)
    except ProFeatureUnavailable as exc:
        sys.exit(str(exc.args[0]))


if __name__ == "__main__":
    main()
