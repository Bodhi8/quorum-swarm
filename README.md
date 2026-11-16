# Quorum

**A learned control plane for agent swarms. ML steers the swarm; humans steer the ML.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Status: research preview](https://img.shields.io/badge/status-v0.1%20research%20preview-orange.svg)

> **Quorum comes in two parts.** This repository is the **open core** (Apache-2.0): the full runtime,
> every interface, and a dependency-free demo domain. **[Quorum Pro](#quorum-pro)** is
> [Vector1](https://vector1.ai)'s commercial extension. It adds the **learned human gate**, the
> **self-tuning field physics** and the **energy domain pack** (the grid stress-response domain used
> in the results below). Quorum Pro lives in a private repository; [request access](#get-quorum-pro).

Most agent frameworks wire agents together by hand, or let an LLM decide who talks to whom. Quorum
splits the jobs. LLM agents do the work. Conventional, auditable machine learning decides how the swarm
is organised. Humans set goals and hard limits, and a gate decides when a human actually needs to be
asked.

```
 Human orchestration   goals → rewards + hard rules · budgets · approvals         [core]  learned gate [Pro]
          │
 ML control plane      bandit routing · shadow twin vets every change              [core]  self-tuning field [Pro]
          │  policy deltas ↓   ↑ telemetry
 Semantic field        problem graph with signals that drift, spread, fade, recruit [core]
          │  signals ↓   ↑ traces
 Agent swarm           scouts · analysts · skeptics · planner (any LLM)            [core]  energy domain pack [Pro]

 Guardrails (outside the learning loop): hard rules · circuit breaker · kill switch · event log   [core]
```

## What is new here

1. **Coordination through a physics-informed field (stigmergy).** Agents don't message each other.
   They read and write signals on a shared graph that evolve by advection–diffusion–reaction. A small
   set of interpretable physical knobs governs the swarm, and a stability bound stops the field from
   oscillating. *(Core: the field and its stability bound. Pro: knobs that tune themselves.)*
2. **No policy change goes live untested.** A shadow twin replays the same scenarios with current and
   proposed settings and accepts a change only if it wins. *(Core.)*
3. **Human attention is treated as a scarce resource.** The swarm learns which plans need a human and
   prefers plans the operator will accept. *(Core: the gate interface, threshold, always-ask and
   autonomy gates, gate-aware plan selection. Pro: the learned gate.)*
4. **Goals compile into rewards plus hard rules.** The optimizer can trade off objectives, but it can
   never trade away a rule. *(Core.)*
5. **Classical and quantum solvers share one interface.** Any QUBO can be solved exactly, by simulated
   annealing, on D-Wave or with QAOA. *(Core: the interface and all four solvers. Pro: the unit
   commitment model built from a real grid.)*

## Open core vs. Quorum Pro

| | Open core (this repo) | Quorum Pro |
|---|:---:|:---:|
| Swarm runtime: fast, slow and human loops | ✓ | ✓ |
| Intent compiler: weighted goals + hard rules | ✓ | ✓ |
| Physics-informed semantic field with stability bound | ✓ | ✓ |
| Thompson-sampling strategy/model routing | ✓ | ✓ |
| Shadow twin (every policy change tested first) | ✓ | ✓ |
| Guardrails: budgets, circuit breaker, kill switch | ✓ | ✓ |
| Event-sourced log and HTML replay viewer | ✓ | ✓ |
| Gates: threshold, always-ask, full autonomy | ✓ | ✓ |
| QUBO interface: exact, simulated annealing, D-Wave, QAOA | ✓ | ✓ |
| Toy domain (fault hunt) | ✓ | ✓ |
| **Learned gate**: asks a human only when unsure or facing something new | | ✓ |
| **Self-tuning field physics**, vetted by the shadow twin | | ✓ |
| **Energy domain pack**: grid stress response on Grid2Op (AC power flow) | | ✓ |
| **Unit commitment as a QUBO**, built from the grid's own units and load | | ✓ |
| **Southwest Power Pool market data bridge** (prices, load, fuel mix) | | ✓ |
| Design-partner support and roadmap input | | ✓ |

Quorum Pro registers itself through the same plugin system the core exposes (`quorum.plugins`), so
installing it upgrades an existing setup without code changes: `mode="quorum"` automatically uses the
learned gate and self-tuning field when they are present. Without Pro, the core falls back to the
threshold gate and fixed field physics, and tells you so.

## Results (v0.2 ablation, simulated)

Grid stress response on the IEEE-14 test grid (Quorum Pro's energy domain): a line trips during high
load, and doing nothing ends in a cascade blackout within ten minutes. Each row adds one component, on
identical scenario sequences, 10 seeds × 40 episodes, ±95% interval:

| | Grid survived | Compute / episode | Bad plans auto-applied |
|---|---|---|---|
| Fixed wiring (baseline) | 83.5% ±3.1 | 5.28 | 0.013 / episode |
| + learned strategy routing (open core) | **98.5% ±0.8** | **2.31** (−56%) | 0.003 / episode |
| + self-tuning field (Pro) | 98.2% ±1.3 | 2.33 | 0.003 / episode |
| + learned gate (Pro) | 98.2% ±1.3 | 2.33 | **0** |

The open core's routing carries most of the gain. The learned gate brings wrongly auto-applied plans to
zero but asks the operator more often than a hand-tuned risk threshold. The self-tuning field shows no
measurable effect at this grid size; larger grids are next. Against always-ask, the learned gate cut
interruptions by 63% in the v0.1 gate study. Full tables, the toy domain and caveats (mock LLM, simulated
operator) are in [docs/results-v0.2.md](docs/results-v0.2.md); v0.1 numbers in
[docs/results-v0.1.md](docs/results-v0.1.md).

**What the open core reproduces on its own:** the full toy-domain ablation (`quorum ablate --domain toy`:
routing cuts compute by 73% at equal outcomes) and the open gates in the gate study. Rerunning the grid
rows needs Quorum Pro's energy domain.

## Quick start (open core)

```bash
pip install -e .                         # numpy + networkx only, no API keys

quorum plugins                           # what is registered (and what Pro would add)
quorum run --domain toy --episodes 30    # writes runs/latest/replay.html
quorum compare --domain toy              # fixed wiring vs Quorum
quorum gates --domain toy                # gate policy study
quorum run --domain toy --interactive    # you approve escalated plans yourself
quorum run --domain toy --llm anthropic  # real LLM scout (needs ANTHROPIC_API_KEY)
python examples/quickstart.py            # build a swarm by hand
```

### With Quorum Pro

```bash
pip install -e ./quorum-pro              # from the private repo
quorum run --domain grid --episodes 30   # grid stress response, learned gate, self-tuning field
quorum compare --domain grid             # fixed wiring vs Quorum + Pro
quorum uc --hours 3 --solvers exact anneal dwave qaoa   # unit commitment QUBO benchmark
quorum ablate --domain grid             # reproduces docs/results-v0.2.md
python examples/benchmark.py             # reproduces docs/results-v0.1.md in full
```

## Quorum Pro

Quorum Pro is where the research turns into something you can run in production:

* **Learned gate.** It learns your operators' approval patterns from their decisions. It asks only when
  it is unsure or facing an unfamiliar plan, and it blocks what it is confident they would reject. In
  testing it auto-applied no plan the operator would have rejected, and asked 63% less often than
  reviewing every plan.
* **Self-tuning field.** The swarm's coordination physics adapt to your problem. Every change is proven
  in the shadow twin before it goes live, and every change is explainable in physical terms. (On the
  14-substation test grid the default physics were already adequate; larger grids are being tested.)
* **Energy domain pack.** Grid stress response on real AC power-flow simulation, unit commitment
  compiled for classical and quantum solvers, and a live data bridge to the Southwest Power Pool market.
  Built for utilities, co-ops, grid operators and energy traders.
* **A direct line to the Vector1 research team**, including priority on the roadmap (async runtime,
  larger grids, power-price and storage optimization).

### Get Quorum Pro

Quorum Pro is available to **design partners, pilot customers and research collaborators**. Code is in
the private repository `Bodhi8/quorum-pro`. Request access at **[vector1.ai](https://vector1.ai)** with
a line on your use case.

## Repository layout

```
src/quorum/
  intent.py         goals → rewards + hard rules
  field.py          physics-informed semantic field (ADR on a graph, stability bound)
  controller.py     Thompson-sampling routing, field-policy interface, static baseline
  twin.py           shadow twin: tests every policy change on replayed scenarios
  gates.py          gates (threshold, always-ask, autonomy), operators, gate keeper
  safety.py         budgets, circuit breaker, kill switch
  swarm.py          the runtime: fast, slow and human loops
  solvers/          QUBO interface: exact, simulated annealing, D-Wave, QAOA
  domains/toy.py    dependency-free demo domain
  viz/replay.py     self-contained HTML replay of any run
  experiments.py    compare + gate study
  pro.py            what Quorum Pro adds, and how to get it
docs/               architecture, results
examples/           quickstart, benchmark
```

## Roadmap

| Version | Open core | Quorum Pro |
|---|---|---|
| v0.2 | ablation harness ✓; LLM cost metering ✓; async runtime | real-model runs on the grid; learned-gate calibration to cut interruptions |
| v0.3 | multi-step plans; live web dashboard | 36- and 118-substation grids; does field tuning pay off at scale? |
| v0.4 | GNN topology control; credit maps | quantum vs. classical unit-commitment benchmark |
| Phase 2 | | power-price forecasting and storage arbitrage on SPP data |

## License and contributing

The open core is licensed under [Apache-2.0](LICENSE). Quorum Pro is proprietary and is not covered by
this license (see [NOTICE](NOTICE)). Contributions to the open core are welcome under Apache-2.0. Please
sign off your commits (`git commit -s`) to certify the
[Developer Certificate of Origin](https://developercertificate.org/).

## Citing

```bibtex
@software{quorum2026,
  author = {Curry, Brian James and {Vector1}},
  title  = {Quorum: a learned control plane for agent swarms},
  year   = {2026},
  url    = {https://github.com/Bodhi8/quorum-swarm},
  note   = {Open core. Quorum Pro extensions by Vector1.}
}
```
