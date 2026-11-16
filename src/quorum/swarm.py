"""The swarm runtime: three loops at three speeds.

* fast loop (ticks):     scouts read the field, analysts evaluate, skeptics
                         stress-test, everyone writes back to the field
* slow loop (episodes):  bandits update; every few episodes the field policy
                         proposes new knobs, which the shadow twin must
                         approve before they go live
* human loop:            the gate decides when a person is interrupted
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .controller import Controller
from .domain import Candidate, Domain, Evaluation
from .events import EventBus
from .field import FieldKnobs, SemanticField
from .gates import GateDecision, GateKeeper, Proposal
from .intent import Intent
from .llm import LLMBackend, MockLLM
from .safety import Budget, SafetyMonitor


@dataclass
class SwarmConfig:
    ticks: int = 8
    scouts_per_tick: int = 3
    skeptic_top_k: int = 2
    eval_cost: float = 0.02
    skeptic_cost: float = 0.04
    twin_every: int = 3  # episodes between knob proposals
    twin_scenarios: int = 5
    twin_tolerance: float = 0.05  # candidate must win by this margin (normalized objective)


@dataclass
class Scored:
    cand: Candidate
    ev: Evaluation
    score: float
    st: Evaluation | None = None
    robust: float | None = None


@dataclass
class SearchResult:
    best: Scored | None
    baseline_score: float
    cost: float
    ticks_used: int
    n_evals: int
    resolved: bool
    top: list[Scored] = field(default_factory=list)

    @property
    def objective(self) -> float:
        if self.best is None:
            return self.baseline_score
        return self.best.robust if self.best.robust is not None else self.best.score


@dataclass
class EpisodeResult:
    episode: int
    metrics: dict[str, float]
    reward: float
    decision: str
    approved: bool
    knobs: dict[str, float]
    arm_usage: dict[str, int] = field(default_factory=dict)


class Swarm:
    def __init__(
        self,
        domain: Domain,
        intent: Intent,
        controller: Controller,
        gatekeeper: GateKeeper,
        bus: EventBus | None = None,
        llm: LLMBackend | None = None,
        safety: SafetyMonitor | None = None,
        config: SwarmConfig | None = None,
        knobs: FieldKnobs | None = None,
        seed: int = 0,
    ):
        self.domain = domain
        self.intent = intent
        self.compiled = intent.compile()
        self.controller = controller
        self.gatekeeper = gatekeeper
        self.bus = bus or EventBus(keep=False)
        self.llm = llm or MockLLM()
        self.safety = safety or SafetyMonitor()
        self.config = config or SwarmConfig()
        self.knobs = knobs or FieldKnobs()
        self.rng = np.random.default_rng(seed)
        # scenarios come from their own stream, so every configuration sees the same sequence
        self.scenario_rng = np.random.default_rng(10_000 + seed)
        self.seed = seed
        self.strategies = domain.strategies()
        self.history: list[EpisodeResult] = []
        from .twin import ShadowTwin  # local import avoids a cycle

        self.twin = ShadowTwin(self)

    # ------------------------------------------------------------------ fast loop
    def search(self, sit: Any, knobs: FieldKnobs, rng: np.random.Generator, learn: bool = True) -> SearchResult:
        d, cfg, bus = self.domain, self.config, self.bus
        fld = SemanticField(d.field_graph(sit), knobs)
        budget = Budget(self.intent.budget)
        base_ev = d.evaluate(sit, d.noop(sit))
        base = self.compiled.reward(base_ev.metrics)
        scale = max(abs(base), 1e-6)
        seen: dict[str, Scored] = {}
        arm_usage: dict[str, int] = {}
        resolved = False
        t = 0
        if learn:
            bus.publish("search_start", baseline=base, nodes=[str(n) for n in fld.nodes],
                        edges=[[str(u), str(v)] for u, v in fld.graph.edges], view=d.view(sit),
                        knobs=fld.knobs.as_dict(), stability_margin=fld.stability_margin())

        for t in range(cfg.ticks):
            self.safety.check()
            bus.tick = t
            fld.set_channel("stress", d.stress(sit))
            fld.step()
            tick_scouts = []
            for _ in range(cfg.scouts_per_tick):
                if budget.exhausted:
                    break
                arm = self.controller.select("scout", rng)
                strat = self.strategies[arm]
                budget.charge(f"scout:{arm}", strat.cost)
                arm_usage[arm] = arm_usage.get(arm, 0) + 1
                cands = [c for c in strat.fn(sit, fld, rng, self.llm) if c.cid not in seen]
                best_imp = 0.0
                for c in cands:
                    c.origin = arm
                    budget.charge("eval", cfg.eval_cost)
                    ev = d.evaluate(sit, c)
                    sc = self.compiled.reward(ev.metrics)
                    seen[c.cid] = Scored(c, ev, sc)
                    imp = sc - base
                    best_imp = max(best_imp, imp)
                    amount = math.tanh(abs(imp) / scale)
                    for node in c.nodes:
                        fld.deposit(node, "promise" if imp > 0 else "dead_end", amount)
                quality = min(1.0, max(0.0, best_imp / scale))
                if learn:
                    self.controller.reward("scout", arm, quality)
                tick_scouts.append({"arm": arm, "n": len(cands), "nodes": sorted({str(n) for c in cands for n in c.nodes}),
                                    "quality": round(quality, 3)})

            # planner: try combining the best partial fixes
            top = sorted(seen.values(), key=lambda s: s.score, reverse=True)[:4]
            combo = d.combine(sit, [s.cand for s in top])
            if combo is not None and combo.cid not in seen:
                budget.charge("eval", cfg.eval_cost)
                ev = d.evaluate(sit, combo)
                seen[combo.cid] = Scored(combo, ev, self.compiled.reward(ev.metrics))

            # skeptics: stress-test the leaders
            leaders = [s for s in sorted(seen.values(), key=lambda s: s.score, reverse=True) if s.st is None]
            for s in leaders[: cfg.skeptic_top_k]:
                budget.charge("skeptic", cfg.skeptic_cost)
                s.st = d.stress_test(sit, s.cand, rng)
                s.robust = min(s.score, self.compiled.reward(s.st.metrics))
                if s.robust < s.score - 0.5 * scale:
                    for node in s.cand.nodes:
                        fld.deposit(node, "dead_end", 0.5)

            tested = [s for s in seen.values() if s.robust is not None]
            best_now = max(tested, key=lambda s: s.robust) if tested else None
            if learn:
                bus.publish("tick", field=fld.snapshot(), entropy=fld.entropy(), scouts=tick_scouts,
                            n_evals=len(seen), spent=budget.spent,
                            best=None if best_now is None else {"cid": best_now.cand.cid, "robust": best_now.robust,
                                                                "nodes": [str(n) for n in best_now.cand.nodes]})
            if best_now is not None and d.resolved(best_now.ev) and best_now.st is not None and d.resolved(best_now.st):
                resolved = True
                break
            if budget.exhausted:
                break

        tested = [s for s in seen.values() if s.robust is not None]
        pool = tested or list(seen.values())
        key = lambda s: s.robust if s.robust is not None else s.score  # noqa: E731
        ranked = [s for s in sorted(pool, key=key, reverse=True) if key(s) > base]  # must beat doing nothing
        self._last_arm_usage = arm_usage
        return SearchResult(ranked[0] if ranked else None, base, budget.spent, t + 1, len(seen), resolved,
                            top=ranked[:6])

    def choose(self, sit: Any, res: SearchResult) -> Scored | None:
        """Gate-aware choice: among near-best plans, prefer the one a human is likeliest to approve.

        Only applies when the gate can predict approvals (a learned gate).
        """
        if res.best is None or not hasattr(self.gatekeeper.gate, "predict"):
            return res.best
        scale = max(abs(res.baseline_score), 1e-6)
        best_val = res.best.robust if res.best.robust is not None else res.best.score
        near = [s for s in res.top if s.robust is not None and s.robust >= best_val - 0.15 * scale] or [res.best]

        def p_ok(s: Scored) -> float:
            feats = self.domain.risk_features(sit, s.cand, s.ev, s.st)
            return self.gatekeeper.gate.predict(Proposal("probe", s.cand.plan, "", feats, 0.0))

        pick = max(near, key=lambda s: (round(p_ok(s), 2), s.robust))
        if p_ok(pick) < 0.5:  # likely rejected: settle for a weaker plan the human would accept
            fallback = [s for s in res.top if p_ok(s) >= 0.5]
            if fallback:
                return max(fallback, key=lambda s: s.robust if s.robust is not None else s.score)
        return pick

    # ------------------------------------------------------------------ episode
    def run_episode(self, ep: int) -> EpisodeResult:
        d, bus = self.domain, self.bus
        bus.episode, bus.tick = ep, 0
        sit = d.reset(self.scenario_rng)
        bus.publish("episode_start", scenario=d.describe(sit, None))
        res = self.search(sit, self.knobs, self.rng, learn=True)

        interrupted = False
        approved = False
        decision = GateDecision("block", 0.0, "no candidate beats doing nothing")
        proposal = None
        miss = 0
        best = self.choose(sit, res)
        if best is not None:
            feats = d.risk_features(sit, best.cand, best.ev, best.st)
            proposal = Proposal(pid=f"ep{ep}", plan=best.cand.plan, summary=d.describe(sit, best.cand),
                                features=feats, risk=d.risk_score(feats))
            violations = self.compiled.violations(feats)
            if violations:
                decision = GateDecision("block", 0.0, "hard rule: " + ", ".join(violations))
            else:
                approved, decision, interrupted = self.gatekeeper.review(proposal, self.intent.interrupt_budget)
                if self.gatekeeper.records and GateKeeper.is_miss(self.gatekeeper.records[-1]):
                    miss = 1
            bus.publish("gate", decision=decision.action, approved=approved, p_approve=decision.p_approve,
                        reason=decision.reason, risk=proposal.risk, features=feats, summary=proposal.summary)

        outcome = d.apply(sit, best.cand if (best is not None and approved) else None)
        self.safety.breaker.record(outcome.get("survived", 1.0) > 0.5)
        metrics = dict(outcome)
        metrics.update(interrupts=float(interrupted), compute_cost=res.cost, misses=float(miss),
                       ticks=float(res.ticks_used), resolved_in_search=float(res.resolved))
        reward = self.compiled.reward(metrics)
        result = EpisodeResult(ep, metrics, reward, decision.action, approved, self.knobs.as_dict(),
                               dict(self._last_arm_usage))
        self.history.append(result)
        bus.publish("episode_end", metrics=metrics, reward=reward, decision=decision.action, approved=approved,
                    arms=self.controller.summary(), arm_usage=result.arm_usage)

        if self.config.twin_every and (ep + 1) % self.config.twin_every == 0:
            self.tune_field(ep)
        return result

    # ------------------------------------------------------------------ slow loop
    def tune_field(self, ep: int) -> None:
        candidate = self.controller.field_policy.propose(self.knobs, self.rng)
        if candidate is None:
            return
        seeds = [int(self.seed * 1000 + ep * 10 + i) for i in range(self.config.twin_scenarios)]
        verdict = self.twin.evaluate(self.knobs, candidate, seeds)
        self.controller.field_policy.feedback(candidate, verdict.accept, verdict.delta)
        if verdict.accept:
            self.knobs = candidate
        self.bus.publish("policy_delta", accepted=verdict.accept, delta=verdict.delta,
                         current=verdict.current_score, candidate=verdict.candidate_score,
                         knobs=candidate.as_dict(), live_knobs=self.knobs.as_dict())

    def run(self, episodes: int, on_episode=None) -> list[EpisodeResult]:
        for ep in range(len(self.history), len(self.history) + episodes):
            r = self.run_episode(ep)
            if on_episode:
                on_episode(r)
        return self.history
