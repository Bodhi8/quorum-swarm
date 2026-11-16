# v0.1 results

> The grid-domain rows and the learned-gate rows require **Quorum Pro** (Vector1's commercial extension;
> see the README). Everything else reproduces with the open core alone.

Reproduce with `python examples/benchmark.py` (the toy domain with the open core; the grid domain with Quorum Pro) (about 80 seconds on a laptop CPU, no API keys).
All runs use the mock LLM scout, so compute cost is in abstract units, where one LLM call = 1.0.

## 1. Fixed wiring vs. Quorum, identical scenario sequences

3 seeds × 40 episodes per configuration. Every configuration sees the same scenarios in the same order.

**Static** is the usual setup: every scout uses the strongest model, the field physics are fixed, and
risky plans go to a human above a fixed risk threshold. **Quorum** adds bandit routing, the self-tuning
field vetted by the shadow twin, the learned gate, and gate-aware plan selection.

### Energy: grid stress response (IEEE-14 grid, line outage under load), Quorum Pro

| | Static | Quorum | Change |
|---|---|---|---|
| Grid survived the episode | 82.5% | **98.3%** | +15.8 pts |
| Compute cost per episode | 5.43 | **2.26** | −58% |
| Human interrupts per episode (2nd half) | 0.067 | **0.033** | −50% |
| Plans auto-applied that the operator would have rejected | 0.017 | **0.000** | |

Doing nothing ends in a cascade blackout within two 5-minute steps in these scenarios. The episodes
static wiring loses come from its narrow search: the strongest model always looks at the ends of the
most overloaded line, and sometimes the fix is elsewhere.

### Toy: fault hunt (30-node network)

| | Static | Quorum |
|---|---|---|
| Survived | 100% | 100% |
| Compute cost per episode | 3.33 | **0.94** (−72%) |
| Interrupts per episode (2nd half) | 0.150 | **0.017** |

## 2. Gate study: the same stream of plans, four gate policies

Plans produced by a Quorum swarm are recorded once, then replayed through each gate against the same
simulated operator. A *miss* is a plan the gate auto-applied that the operator would have rejected.

| Gate | Grid: interrupts | Grid: misses | Toy: interrupts | Toy: misses |
|---|---|---|---|---|
| Full autonomy (never ask) | 0 | 41 | 0 | 10 |
| Always ask | 158 | 0 | 163 | 0 |
| Fixed risk threshold | 36 | 5 | 88 | 0 |
| **Learned gate** (Quorum Pro) | 59 | **1** | **24** | **0** |

(158 grid plans, 163 toy plans.)

How to read this honestly:

* On the toy domain the learned gate dominates: 73% fewer interruptions than the threshold, with no misses.
* On the grid it is a different trade-off, not a free win. It asks more often than a tuned threshold
  (59 vs. 36) but lets through far fewer bad plans (1 vs. 5), and its one miss came during warm-up. Against
  always-ask, the usual enterprise setting, it cuts interruptions by 63%.
* The threshold here was tuned by hand for this domain. The learned gate was not tuned at all.

## 3. Unit commitment QUBO (quantum benchmark harness), model in Quorum Pro, solvers in the open core

3 dispatchable units × 3 hours → 18 binary variables (including slack bits).

| Solver | Cost | Notes |
|---|---|---|
| Exact (brute force) | 221.0 | ground truth |
| Simulated annealing, 32 reads | 221.0 | matches exact; with 8 reads it settled at 236.0 |
| D-Wave / QAOA | — | adapters included, not run (no credentials or Qiskit here) |

Even this small instance shows the penalty landscape is rugged enough that a weak classical baseline
looks bad. Any quantum comparison must be against a well-tuned classical solver.

## Caveats before anything is published

* **Bundle, not ablation.** "Quorum" changes four things at once. The paper needs ablations: bandit only,
  plus field tuning, plus learned gate, plus gate-aware selection.
* **Mock LLM.** The LLM scout is a deterministic stand-in. Rerun with a real model (`--llm anthropic`)
  and report real token cost.
* **Simulated operator.** Its preferences are hand-written (no reconfiguring the nuclear plant's
  substation, plans must survive the stress test, redispatch ≤ 15 MW). Real operator feedback is the
  real test.
* **Small grid, short horizon.** IEEE-14, one remedial action, 30-minute horizon. Next: the 36- and
  118-substation grids bundled with Grid2Op.
* **Three seeds.** Enough for direction, not for confidence intervals. Use 10+ seeds before publishing
  numbers.
