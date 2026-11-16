# v0.2 results: ablation

Which part of Quorum earns the gain? Each row adds one component to the row above it,
on identical scenario sequences. 10 seeds × 40 episodes per row; ± is the 95% interval
over seeds. Reproduce with `quorum ablate --domain grid` (the grid domain needs Quorum
Pro) or `quorum ablate --domain toy` (open core). Raw results: `docs/benchmark-v0.2/`.
The LLM scout is the offline mock, so compute is in abstract units (one LLM call = 1.0).

## Energy: grid stress response (IEEE-14 grid, line outage under load)

| Configuration | Grid survived | Compute / episode | Interrupts / episode | Bad plans auto-applied / episode |
|---|---|---|---|---|
| Fixed wiring (baseline) | 83.5% ±3.1 | 5.28 ±0.26 | 0.045 ±0.018 | 0.013 ±0.015 |
| + learned strategy routing (core) | **98.5% ±0.8** | **2.31 ±0.14** | 0.122 ±0.038 | 0.003 ±0.005 |
| + self-tuning field (Pro) | 98.2% ±1.3 | 2.33 ±0.19 | 0.125 ±0.037 | 0.003 ±0.005 |
| + learned gate (Pro) | 98.2% ±1.3 | 2.33 ±0.19 | 0.200 ±0.044 | **0.000** |
| + gate-aware plan choice (core) | 97.7% ±1.4 | 2.33 ±0.19 | 0.185 ±0.043 | **0.000** |

## Toy: fault hunt (30-node network)

| Configuration | Survived | Compute / episode | Interrupts / episode |
|---|---|---|---|
| Fixed wiring (baseline) | 100.0% | 3.35 ±0.09 | 0.172 ±0.038 |
| + learned strategy routing (core) | 99.5% ±0.7 | **0.89 ±0.07** | 0.180 ±0.040 |
| + self-tuning field (Pro) | 99.8% ±0.5 | 0.93 ±0.05 | 0.165 ±0.039 |
| + learned gate (Pro) | 99.8% ±0.5 | 0.93 ±0.05 | **0.147 ±0.014** |
| + gate-aware plan choice (core) | 99.8% ±0.5 | 0.93 ±0.05 | 0.147 ±0.014 |

## What this says

* **Learned routing does almost all of the work, and it is in the open core.** On the
  grid it lifts survival from 83.5% to 98.5% and cuts compute by 56%. Fixed wiring
  always sends the strongest scout to the ends of the most overloaded line. That is a
  good guess most of the time, but when the fix lies elsewhere the swarm never looks
  there. The bandit mixes cheap, broad scouts with focused ones and pays for the
  expensive one only when it earns its cost.
* **The self-tuning field shows no measurable effect at this scale.** Its rows match
  routing alone within the intervals on both domains. On a 14-substation grid with a
  few ticks per episode, the field's default physics are already adequate. Whether
  tuning pays off on larger grids (36 and 118 substations) is the open question for
  v0.3. Until that is tested, it is not a claimed benefit.
* **The learned gate trades interruptions for safety.** On the grid it brought plans
  wrongly auto-applied to zero (from 0.003 per episode, about 1 in 300), but asked the
  operator more often than the hand-tuned risk threshold (0.200 vs. 0.125 per
  episode, averaged over all 40 episodes including its warm-up). On the toy domain it
  asked slightly less than the threshold with no misses. Against always-ask, the usual
  enterprise setting, it cut interruptions by 63% in the v0.1 gate study.
* **Gate-aware plan choice** trims interruptions a little (0.200 to 0.185) at a cost
  in survival that is within the noise (98.2% to 97.7%).

## Caveats

* Mock LLM scout; costs are abstract units. `examples/real_model.py` reruns the
  comparison with a real model and reports real calls, tokens and dollars.
* The operator is simulated with a hand-written policy.
* IEEE-14 grid, one remedial action per episode, 30-minute horizon.
