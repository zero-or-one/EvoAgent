"""
evoagent.agent
~~~~~~~~~~~~~~
Top-level orchestrator that wires together a :class:`~evoagent.network.Network`,
an :class:`~evoagent.evaluator.Evaluator`, and a
:class:`~evoagent.meta.MetaController` into a single runnable agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .evaluator import Evaluator, Task, XORParity
from .meta import MetaConfig, MetaController, StepResult
from .network import Network

# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """
    Full configuration for an :class:`Agent` run.

    Parameters
    ----------
    layer_sizes:
        Node counts per layer (default: ``[4, 16, 8, 2]``).
    activation:
        Hidden-layer activation function (default: ``"relu"``).
    task:
        Task instance to solve (default: :class:`~evoagent.evaluator.XORParity`).
    meta:
        Meta-controller configuration (default: :class:`~evoagent.meta.MetaConfig`).
    seed:
        Global RNG seed for reproducibility (default: ``None``).
    verbose:
        Print progress to stdout during :meth:`Agent.run` (default: ``True``).
    print_every:
        Print a summary line every N generations (default: ``10``).
    on_step:
        Optional callback called after every generation with the
        :class:`~evoagent.meta.StepResult`.
    on_done:
        Optional callback called once when the agent stops, with the final
        :class:`~evoagent.meta.StepResult`.
    """

    layer_sizes: List[int] = field(default_factory=lambda: [4, 16, 8, 2])
    activation: str = "relu"
    task: Task = field(default_factory=XORParity)
    meta: MetaConfig = field(default_factory=MetaConfig)
    seed: Optional[int] = None
    verbose: bool = True
    print_every: int = 10
    on_step: Optional[Callable[[StepResult], None]] = None
    on_done: Optional[Callable[[StepResult], None]] = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """
    Self-evolving neural agent.

    The agent owns a :class:`~evoagent.network.Network` and a
    :class:`~evoagent.meta.MetaController`.  On each step the controller
    introspects the network's weights, decides how to mutate them, evaluates
    candidates, and keeps the best one.

    Parameters
    ----------
    config:
        :class:`AgentConfig` instance.
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.cfg = config or AgentConfig()
        self._build()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build(self) -> None:
        self.network = Network(
            layer_sizes=self.cfg.layer_sizes,
            activation=self.cfg.activation,
            seed=self.cfg.seed,
        )
        self.evaluator = Evaluator(self.cfg.task)
        self.meta = MetaController(
            config=self.cfg.meta,
            seed=self.cfg.seed,
        )
        self._last_result: Optional[StepResult] = None
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        """True once a stopping condition has been triggered."""
        return self._last_result is not None and self._last_result.done

    @property
    def generation(self) -> int:
        return self.meta.generation

    @property
    def best_accuracy(self) -> float:
        return self.meta.best_accuracy

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def step(self) -> StepResult:
        """
        Execute one generation of self-modification.

        Returns
        -------
        StepResult
            Full audit record for this generation.

        Raises
        ------
        RuntimeError
            If called after the agent has already stopped.
        """
        if self.done:
            raise RuntimeError(
                "Agent has already stopped. Call reset() to start a new run."
            )

        if self._start_time is None:
            self._start_time = time.time()

        result = self.meta.step(self.network, self.evaluator)
        self._last_result = result

        if self.cfg.on_step:
            self.cfg.on_step(result)

        if result.done and self.cfg.on_done:
            self.cfg.on_done(result)

        return result

    def run(self) -> StepResult:
        """
        Run the agent until a stopping condition is met.

        This is a *blocking* call.  For non-blocking integration (e.g. a GUI
        event loop), call :meth:`step` in your own loop instead.

        Returns
        -------
        StepResult
            The final result that triggered the stop.
        """
        while not self.done:
            result = self.step()

            if self.cfg.verbose and result.generation % self.cfg.print_every == 0:
                _print_progress(result, self.elapsed_seconds)

        if self.cfg.verbose:
            _print_summary(self._last_result, self.elapsed_seconds)

        return self._last_result

    def reset(self) -> None:
        """Rebuild the agent from scratch (new random weights, reset counters)."""
        self._build()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the current network weights to a JSON file.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``"best_agent.json"``).
        """
        self.network.save(path)

    @classmethod
    def load(cls, path: str, config: Optional[AgentConfig] = None) -> "Agent":
        """
        Restore an agent from a saved network file.

        Parameters
        ----------
        path:
            Path to a JSON file previously written by :meth:`save`.
        config:
            Optional config.  Only ``task`` and ``meta`` settings are used;
            ``layer_sizes`` and ``activation`` are inferred from the file.
        """
        net = Network.load(path)
        cfg = config or AgentConfig(
            layer_sizes=net.layer_sizes,
            activation=net.activation_name,
        )
        agent = cls(cfg)
        agent.network = net
        return agent

    def __repr__(self) -> str:
        return (
            f"Agent(network={self.network}, "
            f"task={self.cfg.task.name}, "
            f"gen={self.generation}, "
            f"acc={self.best_accuracy:.2%})"
        )


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------


def _progress_bar(fraction: float, width: int = 20) -> str:
    filled = int(fraction * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _print_progress(result: StepResult, elapsed: float) -> None:
    bar = _progress_bar(result.accuracy)
    print(
        f"  gen {result.generation:>5} │ {bar} {result.accuracy:>6.1%} │ "
        f"loss={result.loss:>7.4f} │ strat={result.strategy.value:<10} │ "
        f"{elapsed:>6.1f}s"
    )


def _print_summary(result: StepResult, elapsed: float) -> None:
    print()
    print("  ┌─────────────────────────────────────────────┐")
    print(f"  │  Agent stopped after {result.generation} generations")
    print(f"  │  Reason   : {result.stop_reason}")
    print(f"  │  Accuracy : {result.accuracy:.2%}")
    print(f"  │  Loss     : {result.loss:.4f}")
    print(f"  │  Elapsed  : {elapsed:.1f}s")
    print("  └─────────────────────────────────────────────┘")
    print()
