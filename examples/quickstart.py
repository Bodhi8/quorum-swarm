"""Build a swarm by hand instead of through build_swarm, to see the moving parts.

    python examples/quickstart.py
"""
from quorum import (Controller, EventBus, GateKeeper, Intent, SimulatedOperator, Swarm, SwarmConfig,
                    ThresholdGate, max_value)
from quorum.domains.toy import ToyDomain
from quorum.llm import MockLLM

domain = ToyDomain()

# 1. Human layer: goals as weights, plus rules no optimizer may trade away.
intent = Intent(
    goal="find and fix the fault cheaply, without surprising the operator",
    objective=ToyDomain.default_objective(),
    rules=[max_value("cost", 1.0)],
    budget=4.0,
    interrupt_budget=1,
)

# 2. Control plane: one bandit over scout strategies (name -> cost).
controller = Controller({"scout": {n: s.cost for n, s in domain.strategies().items()}})

# 3. Gate: escalate risky plans to a (here simulated) human.
policy = domain.operator_policy()
gate = GateKeeper(ThresholdGate(0.4), SimulatedOperator(policy), oracle=policy)

swarm = Swarm(domain, intent, controller, gate, bus=EventBus("runs/quickstart/events.jsonl"),
              llm=MockLLM(ToyDomain.mock_llm), config=SwarmConfig(ticks=6, twin_every=0), seed=0)

for r in swarm.run(20):
    print(f"episode {r.episode:>2}: {r.decision:<8} reward {r.reward:+.2f}  cost {r.metrics['compute_cost']:.2f}")
swarm.bus.close()
print("\nWhat the bandit learned:", swarm.controller.summary()["scout"])
