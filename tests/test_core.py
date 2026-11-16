import json

import networkx as nx
import numpy as np
import pytest

from quorum import EventBus, FieldKnobs, Intent, SemanticField, ThompsonBandit, forbid, max_value
from quorum.build import build_swarm
from quorum.gates import AlwaysAskGate, AutoGate, GateKeeper, Proposal, SimulatedOperator, ThresholdGate
from quorum.safety import CircuitBreaker, KillSwitch, SafetyMonitor, SwarmHalted
from quorum.solvers import QUBO, ExactSolver, SimulatedAnnealingSolver
from quorum.swarm import SwarmConfig


# ---------------------------------------------------------------- field
def test_field_stability_clamps_dt():
    g = nx.complete_graph(12)
    f = SemanticField(g, FieldKnobs(diffusion=1.5, drift=2.0, dt=1.0))
    assert f.stability_margin() > 0
    assert f.knobs.dt < 1.0


def test_field_promise_diffuses_to_neighbours_and_decays():
    g = nx.path_graph(5)
    f = SemanticField(g, FieldKnobs(diffusion=0.3, decay=0.1, recruitment=0.0, drift=0.0))
    f.deposit(2, "promise", 1.0)
    f.step()
    f.step()
    p = f.values["promise"]
    assert p[2] > p[1] > p[0] >= 0 and p[1] > 0
    total = p.sum()
    for _ in range(20):
        f.step()
    assert f.values["promise"].sum() < total


def test_field_advection_moves_promise_toward_stress():
    g = nx.path_graph(3)
    f = SemanticField(g, FieldKnobs(diffusion=0.0, decay=0.0, recruitment=0.0, drift=0.5))
    f.set_channel("stress", {0: 0.0, 1: 0.5, 2: 1.0})  # a gradient to flow along
    f.deposit(0, "promise", 1.0)
    f.step()
    for _ in range(4):
        f.step()
    assert f.values["promise"][2] > 0


def test_field_sampling_prefers_attractive_nodes():
    g = nx.path_graph(4)
    f = SemanticField(g, FieldKnobs(temperature=8.0, crowding=0.0))
    f.deposit(3, "promise", 2.0)
    f.step()
    picks = f.sample(np.random.default_rng(0), size=200)
    assert picks.count(3) > 120


# ---------------------------------------------------------------- intent / rules
def test_intent_reward_and_hard_rules():
    it = Intent("x", {"a": 2.0, "b": -1.0}, rules=[max_value("mw", 10), forbid("protected")]).compile()
    assert it.reward({"a": 1, "b": 3, "ignored": 9}) == -1.0
    assert it.violations({"mw": 5, "protected": False}) == []
    assert set(it.violations({"mw": 50, "protected": True})) == {"mw<=10", "no_protected"}


# ---------------------------------------------------------------- bandit
def test_bandit_learns_best_arm():
    rng = np.random.default_rng(0)
    b = ThompsonBandit({"good": 0.1, "bad": 0.1}, discount=1.0)
    for _ in range(200):
        arm = b.select(rng)
        b.update(arm, 0.8 if arm == "good" else 0.2)
    assert b.arms["good"].pulls > 150


# ---------------------------------------------------------------- gates
def _prop(risk):
    return Proposal("p", None, "", {"risk": risk}, risk)


def test_threshold_gate_and_keeper_counts_misses():
    keeper = GateKeeper(ThresholdGate(0.5), SimulatedOperator(lambda p: p.risk < 0.3), oracle=lambda p: p.risk < 0.3)
    ok, d, inter = keeper.review(_prop(0.8), 1)
    assert d.action == "escalate" and inter and not ok
    ok, d, inter = keeper.review(_prop(0.4), 1)
    assert d.action == "auto" and ok and GateKeeper.is_miss(keeper.records[-1])
    assert ThresholdGate(0.5).decide(_prop(0.9), 0).action == "block"


def test_always_and_auto_gates():
    assert AlwaysAskGate().decide(_prop(0), 1).action == "escalate"
    assert AutoGate().decide(_prop(1), 0).action == "auto"


# ---------------------------------------------------------------- safety
def test_kill_switch_and_breaker(tmp_path):
    ks = KillSwitch(tmp_path / "STOP")
    mon = SafetyMonitor(ks, CircuitBreaker(2))
    mon.check()
    (tmp_path / "STOP").write_text("")
    with pytest.raises(SwarmHalted):
        mon.check()
    mon2 = SafetyMonitor(breaker=CircuitBreaker(2))
    mon2.breaker.record(False)
    mon2.breaker.record(False)
    with pytest.raises(SwarmHalted):
        mon2.check()


# ---------------------------------------------------------------- solvers
def test_annealer_matches_exact_on_small_qubo():
    rng = np.random.default_rng(3)
    Q = np.triu(rng.normal(0, 1, (10, 10)))
    q = QUBO(Q, offset=1.5)
    ex = ExactSolver().solve(q)
    sa = SimulatedAnnealingSolver(sweeps=300, reads=8).solve(q)
    assert sa.energy == pytest.approx(ex.energy, abs=1e-6)
    assert q.energy(ex.x) == pytest.approx(ex.energy)


# ---------------------------------------------------------------- events + end to end
def test_events_jsonl_roundtrip(tmp_path):
    bus = EventBus(tmp_path / "e.jsonl")
    bus.publish("hello", x=np.float32(1.5), arr=np.arange(3))
    bus.shadow = True
    assert bus.publish("hidden") is None
    bus.close()
    rows = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["data"] == {"x": 1.5, "arr": [0, 1, 2]}


def test_toy_swarm_runs_end_to_end(tmp_path):
    sw = build_swarm("toy", mode="quorum", seed=0, events=tmp_path / "ev.jsonl", gate="threshold",
                     field_policy="static", config=SwarmConfig(ticks=4, twin_every=0))
    hist = sw.run(8)
    sw.bus.close()
    assert len(hist) == 8
    assert all("compute_cost" in h.metrics for h in hist)
    kinds = {json.loads(l)["kind"] for l in (tmp_path / "ev.jsonl").read_text().splitlines()}
    assert {"episode_start", "tick", "gate", "episode_end"} <= kinds


def test_static_baseline_uses_fixed_wiring():
    sw = build_swarm("toy", mode="static", seed=0, config=SwarmConfig(ticks=3, twin_every=0))
    sw.run(3)
    assert set(sw.controller.summary()["scout"]) >= {"llm"}
    assert sw.controller.summary()["scout"]["random"]["pulls"] == 0


def test_replay_renders(tmp_path):
    from quorum.viz import render

    sw = build_swarm("toy", seed=1, events=tmp_path / "ev.jsonl", gate="threshold", field_policy="static",
                     config=SwarmConfig(ticks=3, twin_every=0))
    sw.run(3)
    sw.bus.close()
    out = render(tmp_path / "ev.jsonl", tmp_path / "r.html")
    html = out.read_text()
    assert "<svg" in html and '"episodes"' in html and "__DATA__" not in html
