"""
examples/quickstart.py
~~~~~~~~~~~~~~~~~~~~~~
The simplest possible EvoAgent example.

Run:
    python examples/quickstart.py
"""

from evoagent import Agent, AgentConfig
from evoagent.evaluator import XORParity
from evoagent.meta import MetaConfig

cfg = AgentConfig(
    layer_sizes=[4, 16, 8, 2],
    activation="relu",
    task=XORParity(bits=4),
    meta=MetaConfig(
        stop_accuracy=1.0,
        max_generations=500,
        patience=100,
        n_candidates=8,
    ),
    seed=42,
    verbose=True,
    print_every=20,
)

agent = Agent(cfg)
result = agent.run()

print(f"\nFinal accuracy : {result.accuracy:.2%}")
print(f"Stop reason    : {result.stop_reason}")
print(f"Generations    : {result.generation}")
print(f"Parameters     : {agent.network.total_parameters()}")
