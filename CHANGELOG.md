# Changelog

## 0.2.0 (unreleased)

- `quorum ablate`: adds one component at a time (routing, self-tuning field, learned
  gate, gate-aware plan choice) on identical scenarios and reports mean and 95%
  intervals over seeds.
- Real LLM metering: every episode records LLM calls, tokens and dollars
  (`llm_calls`, `llm_tokens`, `llm_usd`); `examples/real_model.py` runs the comparison
  with a real model.
- `SwarmConfig.gate_aware` switch for ablations.
- Apache 2.0 header on every source file, with a test; NOTICE, TRADEMARKS, CITATION,
  CONTRIBUTING and SECURITY files; CI on Linux, Windows and macOS; PyPI trusted
  publishing on version tags.

## 0.1.0 (Nov 16, 2026)

First public release of the open core: the swarm runtime with fast, slow and human
loops; the physics-informed semantic field with its stability bound; Thompson-sampling
routing; the shadow twin; threshold, always-ask and autonomy gates; guardrails; the QUBO
interface with exact, simulated-annealing, D-Wave and QAOA solvers; the event log and
HTML replay; the toy domain.
