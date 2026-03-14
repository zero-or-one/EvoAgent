"""
examples/advanced.py
~~~~~~~~~~~~~~~~~~~~
Advanced usage: custom callbacks, weight introspection, saving/loading,
and a custom task.

Run:
    python examples/advanced.py
"""

from __future__ import annotations

from evoagent import Agent, AgentConfig
from evoagent.evaluator import BinarySymmetry, CountOnes, Evaluator, Task, XORParity, Sample
from evoagent.meta import MetaConfig, StepResult
from evoagent.network import Network


# ---------------------------------------------------------------------------
# 1. Custom task — detect majority (more 1s than 0s)
# ---------------------------------------------------------------------------

class MajorityVote(Task):
    """Binary classification: is there a majority of 1-bits?"""

    def __init__(self, bits: int = 5) -> None:
        self._bits = bits

    @property
    def n_inputs(self) -> int:
        return self._bits

    @property
    def n_classes(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return f"majority-vote-{self._bits}"

    def generate(self) -> list[Sample]:
        samples = []
        for i in range(2 ** self._bits):
            bits = [(i >> j) & 1 for j in range(self._bits)]
            label = 1 if sum(bits) > self._bits / 2 else 0
            samples.append(Sample(input=bits, target=label))
        return samples


# ---------------------------------------------------------------------------
# 2. Callback that prints detailed introspection every 50 gens
# ---------------------------------------------------------------------------

def verbose_callback(result: StepResult) -> None:
    if result.generation % 50 == 0:
        print(f"\n  ── Weight introspection at gen {result.generation} ──")
        for stats in result.layer_stats:
            flag = " [COLLAPSED]" if stats.is_collapsed() else ""
            print(
                f"    layer {stats.layer_index}: "
                f"mean={stats.mean:+.4f}  std={stats.std:.4f}  "
                f"dead={stats.dead_fraction:.1%}  L2={stats.l2_norm:.3f}{flag}"
            )
        for line in result.log_lines:
            print("   ", line)


# ---------------------------------------------------------------------------
# 3. Run three tasks back-to-back and compare results
# ---------------------------------------------------------------------------

def run_task(task: Task, label: str) -> None:
    print(f"\n{'='*55}")
    print(f"  Task: {label}")
    print(f"{'='*55}")

    cfg = AgentConfig(
        layer_sizes=[task.n_inputs, 16, 8, task.n_classes],
        activation="relu",
        task=task,
        meta=MetaConfig(
            stop_accuracy=1.0,
            max_generations=600,
            patience=120,
            n_candidates=10,
        ),
        seed=0,
        verbose=True,
        print_every=50,
        on_step=verbose_callback,
    )
    agent = Agent(cfg)
    result = agent.run()

    print(f"\n  Network: {agent.network}")
    print(f"  Best accuracy : {result.accuracy:.2%}")
    print(f"  Stop reason   : {result.stop_reason}")

    # Save the best weights
    out_path = f"best_{task.name}.json"
    agent.save(out_path)
    print(f"  Saved weights → {out_path}")

    # Demonstrate load + verify
    loaded_agent = Agent.load(out_path, cfg)
    acc2, loss2 = loaded_agent.evaluator.evaluate(loaded_agent.network)
    print(f"  Loaded & re-evaluated: acc={acc2:.2%}  loss={loss2:.4f}")


if __name__ == "__main__":
    run_task(XORParity(bits=4),     "XOR Parity (4-bit)")
    run_task(BinarySymmetry(length=6), "Binary Symmetry (6-bit)")
    run_task(MajorityVote(bits=5),  "Majority Vote (5-bit)")
