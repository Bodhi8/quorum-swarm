# Quorum architecture (v0.1)

**ML steers the swarm; humans steer the ML.**

Quorum is a control plane for agent swarms. LLM agents do the work. A layer of
conventional, auditable machine learning decides how they are organised. Humans
set the goals and the hard limits, and decide when the system has to ask them.

## Layers

| Layer | What it does | Code |
|---|---|---|
| Human orchestration | Goals as weighted objectives, hard rules, budgets, interrupt budget, approvals | `intent.py`, `gates.py`; learned gate in Quorum Pro |
| ML control plane | Bandit routing of strategies/models, field-physics tuning, shadow-twin vetting | `controller.py`, `twin.py`; self-tuning field in Quorum Pro |
| Semantic field | Shared graph with signals that drift, spread, fade and recruit (ADR dynamics) | `field.py` |
| Agent swarm | Scouts, analysts, skeptics, planner, coordinating only through the field | `swarm.py`, domain strategies |
| Guardrails (outside the learning loop) | Hard rules, budgets, circuit breaker, kill switch | `intent.py`, `safety.py` |
| Observability | Event-sourced log of every decision; HTML replay | `events.py`, `viz/replay.py` |

## Three loops, three speeds

1. **Fast loop (ticks).** The field updates. Scouts pick a strategy (chosen by the
   bandit) and propose candidates where the field is attractive. Analysts score
   candidates in simulation. Skeptics stress-test the leaders. Results go back into
   the field as *promise* or *dead-end* signal. The loop stops when a plan resolves
   the situation under stress or the budget runs out.
2. **Slow loop (episodes).** Bandits update from each scout's quality minus cost.
   Every few episodes the field policy proposes new physics knobs, and the shadow
   twin replays the same scenarios with current and candidate knobs (common random
   numbers). The change goes live only if it wins by a margin.
3. **Human loop.** The best plan goes to the gate. Hard-rule violations are blocked
   outright. Otherwise the gate auto-approves, blocks, or interrupts a person. A
   learned gate turns every escalation into a training label, so it needs to ask
   less over time.

## The physics-informed field

Each channel evolves on the problem graph as

```
dp/dt = -D·L·p  +  drift·A(stress)·p  -  decay·p  +  r·p·(1-p)  +  deposits
```

* **Diffusion** (`D`, graph Laplacian `L`) spreads good news to neighbouring regions.
* **Advection** (`drift`) carries promise up the stress gradient, toward where the
  problem is.
* **Decay** forgets stale signals. **Recruitment** (`r`) reinforces what is working,
  and the logistic term keeps it from growing without limit.
* **Temperature** and **crowding** set how sharply agents follow the field and how
  much they avoid piling onto one spot.

`stability_margin()` is a CFL-style bound on the explicit update
(`dt·(D·λmax + decay + r + drift·deg_max) < 1`). The field shrinks `dt` rather than
accept knobs that would oscillate. In the energy domain the field graph *is* the
transmission grid (substations and lines), so diffusion follows the real network.

## Extension points

Everything plugs in by name through `quorum.plugins`:

* **Domain**: `reset`, `field_graph`, `stress`, `strategies`, `evaluate`,
  `stress_test`, `combine`, `risk_features`, `risk_score`, `apply`, `operator_policy`
  (see `domain.py`, and `domains/toy.py` for a minimal example).
* **Gate**: `decide(proposal, interrupts_left)` and `observe(proposal, approved)`.
* **Field policy**: `propose(current_knobs, rng)` and `feedback(knobs, accepted, delta)`.
* **Solver**: `solve(QUBO) -> Solution`. Classical (`exact`, `anneal`) and quantum
  (`dwave`, `dwave-hybrid`, `qaoa`) backends share one interface.
* **LLM backend**: `complete(system, prompt)`. `MockLLM` runs offline; `AnthropicLLM`
  is used when a key is present.

External packages register through the `quorum.plugins` entry-point group. That is
how **Quorum Pro** (Vector1's commercial extension, private repo `Bodhi8/quorum-pro`)
adds the learned gate, the self-tuning field and the energy domain pack without the
core depending on it. `quorum/pro.py` lists what Pro adds; when a Pro component is
requested but not installed, the core falls back to an open one and says so.

## Event schema

Every event is `{kind, episode, tick, data, ts, shadow}`, written as JSON lines.

| kind | data |
|---|---|
| `episode_start` | scenario description |
| `search_start` | baseline score, field nodes and edges, domain view, knobs, stability margin |
| `tick` | field snapshot, entropy, scout pulls with quality, evaluations so far, spend, best plan |
| `gate` | decision, approved, p(approve), reason, risk, features, plan summary |
| `episode_end` | outcome metrics, reward, decision, bandit state, strategy usage |
| `policy_delta` | candidate knobs, twin scores, accepted |

Shadow-twin runs publish nothing, so the live record stays clean.

## Known limits of v0.1

* The loop is synchronous. Agents run in sequence within a tick. An asyncio runtime
  with real concurrent LLM calls is planned for v0.2.
* The LLM scout runs on a mock by default. Costs are abstract units, not dollars.
* One remedial action per episode on a 14-substation test grid. Larger grids
  (`l2rpn_icaps_2021`, 36 substations; `l2rpn_idf_2023`, 118) are bundled with
  Grid2Op and are the next step.
* The simulated operator's policy is hand-written. Real operator feedback is what
  the learned gate is ultimately for.
