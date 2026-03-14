"""
EvoAgent
~~~~~~~~
A self-evolving neural agent that reads and modifies its own weights
to improve performance — without backpropagation.

Quick start
-----------
>>> from evoagent import Agent, AgentConfig
>>> from evoagent.evaluator import XORParity
>>> agent = Agent(AgentConfig(task=XORParity(bits=4), seed=42))
>>> agent.run()
"""

from .agent import Agent, AgentConfig
from .evaluator import BinarySymmetry, CountOnes, Evaluator, Task, XORParity
from .meta import MetaConfig, MetaController, StepResult, Strategy
from .network import Network, WeightStats

__version__ = "1.0.0"
__author__ = "EvoAgent Contributors"
__license__ = "MIT"

__all__ = [
    "Agent",
    "AgentConfig",
    "Evaluator",
    "Task",
    "XORParity",
    "BinarySymmetry",
    "CountOnes",
    "MetaConfig",
    "MetaController",
    "StepResult",
    "Strategy",
    "Network",
    "WeightStats",
]
