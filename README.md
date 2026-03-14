# EvoAgent

> A self-evolving neural agent that **reads and modifies its own weights** to improve — without backpropagation.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

---

## What is EvoAgent?

EvoAgent is a lightweight research library that demonstrates **self-modifying neural networks**.  Unlike standard gradient-based training, EvoAgent lets an agent:

1. **Introspect** its own weight matrices (read mean, std, dead-neuron fraction, L2 norm, etc.)
2. **Diagnose** its current state (collapsing weights, saturation, stagnation)
3. **Self-modify** by adaptively perturbing its weights based on that diagnosis
4. **Self-select** the best mutation from a pool of candidates
5. **Stop itself** automatically when it meets a performance target or exhausts its budget

This is a toy implementation of ideas from **evolutionary strategies (ES)**, **neuroevolution**, and **meta-learning** — the same family of approaches used in large-scale self-improving systems.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Agent                            │
│  ┌───────────┐   ┌────────────────┐   ┌─────────────┐  │
│  │  Network  │◄──│ MetaController │──►│  Evaluator  │  │
│  │           │   │                │   │             │  │
│  │ weights[] │   │ 1. Introspect  │   │  Task       │  │
│  │ biases[]  │   │ 2. Diagnose    │   │  Score      │  │
│  │           │   │ 3. Mutate      │   │  Loss       │  │
│  │ forward() │   │ 4. Select best │   │             │  │
│  │ inspect() │   │ 5. Check stop  │   │             │  │
│  └───────────┘   └────────────────┘   └─────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Modules

| Module | Responsibility |
|---|---|
| `evoagent.network` | Feedforward MLP with full weight access & introspection |
| `evoagent.evaluator` | Dataset generation and network scoring |
| `evoagent.meta` | Adaptive meta-controller: diagnosis, mutation, stopping |
| `evoagent.agent` | Orchestrator: wires everything together, handles I/O |

---

## Installation

```bash
git clone https://github.com/yourorg/evoagent.git
cd evoagent
pip install -e .
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.24 (no other dependencies).

---

## Quickstart

```python
from evoagent import Agent, AgentConfig
from evoagent.evaluator import XORParity
from evoagent.meta import MetaConfig

cfg = AgentConfig(
    layer_sizes=[4, 16, 8, 2],   # input → hidden → hidden → output
    activation="relu",
    task=XORParity(bits=4),
    meta=MetaConfig(
        stop_accuracy=1.0,        # stop when 100% accuracy reached
        max_generations=500,      # hard ceiling
        patience=100,             # stop if no improvement for 100 gens
        n_candidates=8,           # mutants evaluated per generation
    ),
    seed=42,
)

agent = Agent(cfg)
result = agent.run()

print(f"Accuracy  : {result.accuracy:.2%}")
print(f"Stopped   : {result.stop_reason}")
print(f"Gen       : {result.generation}")
```

Output:
```
  gen    10 │ [████░░░░░░░░░░░░░░░░] 37.5% │ loss= 0.6931 │ strat=moderate    │    0.3s
  gen    20 │ [████████░░░░░░░░░░░░] 50.0% │ loss= 0.6180 │ strat=moderate    │    0.6s
  ...
  gen   140 │ [████████████████████] 100.0% │ loss= 0.0812 │ strat=fine_tune  │    4.2s

  ┌─────────────────────────────────────────────┐
  │  Agent stopped after 147 generations
  │  Reason   : target accuracy 100.0% reached
  │  Accuracy : 100.00%
  │  Loss     : 0.0812
  │  Elapsed  : 4.4s
  └─────────────────────────────────────────────┘
```

---

## Stopping Conditions

The meta-controller checks four stopping conditions after every generation:

| Condition | Config key | Default | Description |
|---|---|---|---|
| Target accuracy | `stop_accuracy` | `1.0` | Stop when accuracy ≥ threshold |
| Max generations | `max_generations` | `1000` | Hard ceiling on generation count |
| Patience | `patience` | `100` | Stop if no meaningful improvement for N gens |
| Loss plateau | `min_loss_delta` | `0.001` | Minimum improvement to reset patience |

The first condition that fires wins.  The `StepResult.stop_reason` field explains which one triggered.

---

## Meta-Controller Strategies

The meta-controller selects one of four mutation regimes each generation:

| Strategy | σ (sigma) | Mutation rate | When used |
|---|---|---|---|
| `fine_tune` | 0.008 | 5% | Accuracy ≥ 90% |
| `fine` | 0.02 | 10% | Improving steadily |
| `moderate` | 0.12 | 25% | Early or mid training |
| `aggressive` | 0.35 | 50% | Stagnated, or weights collapsed |

Sigma and mutation rate are further scaled per layer if a layer's weights are detected as collapsed (std < 0.05).

---

## Built-in Tasks

| Task | Class | Description |
|---|---|---|
| XOR Parity | `XORParity(bits=4)` | Predict even/odd bit count |
| Binary Symmetry | `BinarySymmetry(length=6)` | Detect palindrome bit strings |
| Count Ones | `CountOnes(bits=6)` | Classify low/mid/high 1-bit count |

### Custom Task

```python
from evoagent.evaluator import Task, Sample

class MyTask(Task):
    @property
    def n_inputs(self): return 4
    @property
    def n_classes(self): return 2
    @property
    def name(self): return "my-task"

    def generate(self) -> list[Sample]:
        return [
            Sample(input=[0, 0, 0, 0], target=0),
            Sample(input=[1, 1, 1, 1], target=1),
            # ...
        ]

agent = Agent(AgentConfig(task=MyTask(), layer_sizes=[4, 8, 2]))
agent.run()
```

---

## Callbacks

```python
def on_step(result):
    print(f"Gen {result.generation}: {result.accuracy:.1%} [{result.strategy.value}]")

def on_done(result):
    print(f"Done! {result.stop_reason}")

cfg = AgentConfig(
    ...,
    on_step=on_step,
    on_done=on_done,
)
```

---

## Weight Introspection

```python
agent = Agent(cfg)
agent.step()  # run one generation

for stats in agent.network.inspect_all():
    print(stats)
    print(f"  collapsed? {stats.is_collapsed()}")
    print(f"  saturated? {stats.has_saturation()}")
```

Output:
```
WeightStats(layer=0, mean=-0.0123, std=0.4821, dead=3.1%)
  collapsed? False
  saturated? False
WeightStats(layer=1, mean=0.0089, std=0.3201, dead=8.7%)
  collapsed? False
  saturated? False
```

---

## Save & Load

```python
# Save best weights
agent.save("best_agent.json")

# Restore later
loaded = Agent.load("best_agent.json", cfg)
acc, loss = loaded.evaluator.evaluate(loaded.network)
print(f"Restored accuracy: {acc:.2%}")
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 30+ unit tests cover: network initialisation, forward pass, introspection, snapshot/restore, every stopping condition, callbacks, persistence, and end-to-end agent runs.

---

## Project Structure

```
evoagent/
├── evoagent/
│   ├── __init__.py      # Public API
│   ├── network.py       # MLP with weight introspection
│   ├── evaluator.py     # Tasks & scoring
│   ├── meta.py          # Meta-controller & stopping logic
│   └── agent.py         # Top-level orchestrator
├── examples/
│   ├── quickstart.py    # Minimal working example
│   └── advanced.py      # Custom task, callbacks, multi-task comparison
├── tests/
│   └── test_evoagent.py # Full test suite (30+ tests)
├── setup.py
└── README.md
```

---

## How It Works — Deep Dive

### The Self-Modification Loop

Each call to `agent.step()` runs the following pipeline:

```
1. Introspect   net.inspect_all()
                → per-layer mean, std, L2 norm, dead fraction

2. Diagnose     meta._choose_strategy(layer_stats)
                → Strategy: fine | fine_tune | moderate | aggressive

3. Generate     for i in range(n_candidates):
                    clone = incumbent.snapshot()
                    mutate(clone, sigma, rate)
                    score = evaluator.evaluate(clone)

4. Select       if best_candidate > incumbent:
                    net.load_snapshot(best_candidate)
                    update best_accuracy, best_loss

5. Stop check   if accuracy >= target → done
                if generation >= max → done
                if stagnant >= patience → done
```

### Why No Gradients?

EvoAgent intentionally avoids backpropagation to explore the space of *gradient-free* self-improvement.  This makes it:

- **Transparent** — every weight change is auditable
- **Flexible** — works on any differentiable or non-differentiable objective
- **Inspectable** — the agent explains its own decisions via `StepResult.log_lines`

In practice, gradient methods are far more sample-efficient for large networks.  EvoAgent is a research toy and educational tool, not a replacement for SGD.

---

## License

MIT — see [LICENSE](LICENSE).
