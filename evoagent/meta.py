"""
evoagent.meta
~~~~~~~~~~~~~
The meta-controller: the agent's self-awareness module.

Given real-time weight statistics and recent performance history, the
meta-controller decides *how* to mutate the network's weights.  It implements
an adaptive strategy that responds to stagnation, weight collapse, and
saturation — mimicking the role of a learning-rate scheduler + weight
regulariser, but without any gradients.

Algorithm overview
------------------
1.  **Introspect** — read per-layer weight statistics from the network.
2.  **Diagnose** — classify the current training state into one of four
    regimes: *fine*, *fine_tune*, *moderate*, *aggressive*.
3.  **Plan** — compute per-layer mutation hyperparameters (σ, mutation rate,
    clip bounds).
4.  **Mutate** — apply Gaussian perturbations, accept if the mutant is better,
    otherwise revert (1+λ evolutionary strategy with λ candidates).
5.  **Log** — record every decision so the caller can display / audit it.

Stopping conditions
-------------------
The meta-controller checks several stopping criteria after each generation:

* **Target accuracy reached** — configurable via ``stop_accuracy``.
* **Max generations** — hard ceiling via ``max_generations``.
* **Convergence** — no improvement over the last ``patience`` generations.
* **Loss plateau** — loss improvement below ``min_loss_delta`` for
  ``patience`` generations.

Example
-------
>>> from evoagent.meta import MetaController, MetaConfig
>>> cfg = MetaConfig(stop_accuracy=1.0, max_generations=500, patience=80)
>>> meta = MetaController(cfg)
>>> result = meta.step(net, evaluator)
>>> print(result.strategy, result.accepted)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from .evaluator import Evaluator
from .network import Network, WeightStats


# ---------------------------------------------------------------------------
# Enums & small data classes
# ---------------------------------------------------------------------------

class Strategy(str, Enum):
    """Mutation intensity regime chosen by the meta-controller."""
    FINE = "fine"
    FINE_TUNE = "fine_tune"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class StepResult:
    """Everything the meta-controller learned and did during one generation."""

    generation: int
    accuracy: float
    loss: float
    strategy: Strategy
    accepted: bool                        # did the mutant improve on the incumbent?
    mutations_applied: int                # total weight updates across all layers
    layer_stats: List[WeightStats]
    log_lines: List[str] = field(default_factory=list)

    # Stopping information
    done: bool = False
    stop_reason: Optional[str] = None


@dataclass
class MetaConfig:
    """
    Hyperparameters that govern the meta-controller's behaviour.

    Parameters
    ----------
    stop_accuracy:
        Halt when accuracy >= this value (default 1.0 = 100 %).
    max_generations:
        Hard ceiling on the number of generations (default 1000).
    patience:
        Stop if accuracy has not improved by ``min_acc_delta`` for this many
        consecutive generations (default 100).
    min_acc_delta:
        Minimum accuracy improvement to reset the patience counter (default
        0.005 = 0.5 pp).
    min_loss_delta:
        Minimum loss improvement to consider as progress (default 0.001).
    n_candidates:
        Number of random mutants evaluated per generation (default 8).
    weight_clip:
        Hard clip applied to all weights after mutation (default 3.0).
    sigma_fine:
        Gaussian σ used in the *fine* regime (default 0.02).
    sigma_fine_tune:
        Gaussian σ in the *fine_tune* regime (default 0.008).
    sigma_moderate:
        Gaussian σ in the *moderate* regime (default 0.12).
    sigma_aggressive:
        Gaussian σ in the *aggressive* regime (default 0.35).
    """

    stop_accuracy: float = 1.0
    max_generations: int = 1000
    patience: int = 100
    min_acc_delta: float = 0.005
    min_loss_delta: float = 0.001
    n_candidates: int = 8
    weight_clip: float = 3.0
    sigma_fine: float = 0.02
    sigma_fine_tune: float = 0.008
    sigma_moderate: float = 0.12
    sigma_aggressive: float = 0.35


# ---------------------------------------------------------------------------
# MetaController
# ---------------------------------------------------------------------------

class MetaController:
    """
    Adaptive self-modifying controller.

    Parameters
    ----------
    config:
        A :class:`MetaConfig` instance.  Pass ``MetaConfig()`` for defaults.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(self, config: Optional[MetaConfig] = None, seed: Optional[int] = None) -> None:
        self.cfg = config or MetaConfig()
        self._rng = np.random.default_rng(seed)

        # State
        self._generation: int = 0
        self._best_accuracy: float = 0.0
        self._best_loss: float = math.inf
        self._stagnant_gens: int = 0       # gens since last accuracy improvement
        self._history: List[StepResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def best_accuracy(self) -> float:
        return self._best_accuracy

    @property
    def history(self) -> List[StepResult]:
        return list(self._history)

    def step(self, net: Network, evaluator: Evaluator) -> StepResult:
        """
        Run one generation of self-modification.

        1. Introspect the network's current weights.
        2. Diagnose the training regime.
        3. Generate ``n_candidates`` mutants and keep the best one if it
           improves on the incumbent.
        4. Check stopping conditions.
        5. Return a :class:`StepResult` with full audit information.

        Parameters
        ----------
        net:
            The live network to mutate (modified in-place if a mutant is
            accepted).
        evaluator:
            Evaluator used to score each candidate.
        """
        self._generation += 1
        gen = self._generation
        logs: List[str] = []

        # ── 1. Introspect ───────────────────────────────────────────────
        layer_stats = net.inspect_all()
        logs.append(f"[gen {gen}] inspecting {len(layer_stats)} layers")

        for s in layer_stats:
            if s.is_collapsed():
                logs.append(f"  ⚠ layer {s.layer_index} collapsed (std={s.std:.4f})")
            if s.has_saturation():
                logs.append(f"  ⚠ layer {s.layer_index} saturated (mean={s.mean:.4f})")

        # ── 2. Diagnose ─────────────────────────────────────────────────
        prev_acc = self._best_accuracy
        strategy = self._choose_strategy(layer_stats)
        logs.append(f"  strategy → {strategy.value}")

        # ── 3. Mutate & select ──────────────────────────────────────────
        incumbent_snap = net.snapshot()
        best_snap = incumbent_snap
        best_acc, best_loss = evaluator.evaluate(net)   # score incumbent
        best_candidate_acc = best_acc
        best_candidate_loss = best_loss
        accepted = False
        total_mutations = 0

        for _ in range(self.cfg.n_candidates):
            net.load_snapshot(incumbent_snap)
            n_muts = self._apply_mutation(net, strategy, layer_stats)
            acc, loss = evaluator.evaluate(net)

            score_new = acc - 0.05 * loss
            score_old = best_candidate_acc - 0.05 * best_candidate_loss

            if score_new > score_old:
                best_snap = net.snapshot()
                best_candidate_acc = acc
                best_candidate_loss = loss
                total_mutations = n_muts
                accepted = True

        net.load_snapshot(best_snap)

        if accepted and (best_candidate_acc > self._best_accuracy or
                         (best_candidate_acc == self._best_accuracy and
                          best_candidate_loss < self._best_loss)):
            self._best_accuracy = best_candidate_acc
            self._best_loss = best_candidate_loss
            logs.append(f"  ✓ accepted: acc={best_candidate_acc:.4f} loss={best_candidate_loss:.4f} ({total_mutations} muts)")
        else:
            accepted = False
            logs.append(f"  ✗ no improvement (acc={best_candidate_acc:.4f})")

        # ── 4. Patience tracking ────────────────────────────────────────
        if self._best_accuracy - prev_acc >= self.cfg.min_acc_delta:
            self._stagnant_gens = 0
        else:
            self._stagnant_gens += 1

        # ── 5. Stopping conditions ──────────────────────────────────────
        done, stop_reason = self._check_stopping()

        result = StepResult(
            generation=gen,
            accuracy=self._best_accuracy,
            loss=self._best_loss,
            strategy=strategy,
            accepted=accepted,
            mutations_applied=total_mutations if accepted else 0,
            layer_stats=layer_stats,
            log_lines=logs,
            done=done,
            stop_reason=stop_reason,
        )
        self._history.append(result)
        return result

    def reset(self) -> None:
        """Reset all state counters (does not reset the network itself)."""
        self._generation = 0
        self._best_accuracy = 0.0
        self._best_loss = math.inf
        self._stagnant_gens = 0
        self._history.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _choose_strategy(self, layer_stats: List[WeightStats]) -> Strategy:
        """Select mutation intensity based on current performance and weight health."""
        acc = self._best_accuracy
        stagnant = self._stagnant_gens
        pat = self.cfg.patience

        collapsed = any(s.is_collapsed() for s in layer_stats)

        if collapsed:
            return Strategy.AGGRESSIVE
        if acc >= 0.90:
            return Strategy.FINE_TUNE
        if acc >= 0.70 and stagnant < pat // 2:
            return Strategy.FINE
        if stagnant > pat // 2:
            return Strategy.AGGRESSIVE
        return Strategy.MODERATE

    def _sigma_for_strategy(self, strategy: Strategy) -> float:
        """Return base σ for the chosen strategy."""
        return {
            Strategy.FINE: self.cfg.sigma_fine,
            Strategy.FINE_TUNE: self.cfg.sigma_fine_tune,
            Strategy.MODERATE: self.cfg.sigma_moderate,
            Strategy.AGGRESSIVE: self.cfg.sigma_aggressive,
        }[strategy]

    def _apply_mutation(
        self,
        net: Network,
        strategy: Strategy,
        layer_stats: List[WeightStats],
    ) -> int:
        """
        Perturb the network's weights in-place.

        Returns the total number of individual weight values modified.
        """
        base_sigma = self._sigma_for_strategy(strategy)

        # Mutation rates per regime
        rate_map = {
            Strategy.FINE: 0.10,
            Strategy.FINE_TUNE: 0.05,
            Strategy.MODERATE: 0.25,
            Strategy.AGGRESSIVE: 0.50,
        }
        base_rate = rate_map[strategy]

        total = 0
        for i, (W, b, stats) in enumerate(zip(net.weights, net.biases, layer_stats)):
            # Increase σ for collapsed layers
            layer_sigma = base_sigma * (2.0 if stats.is_collapsed() else 1.0)
            layer_rate = min(base_rate * (1.5 if stats.is_collapsed() else 1.0), 0.9)

            mask = self._rng.random(W.shape) < layer_rate
            noise = self._rng.normal(0.0, layer_sigma, W.shape)
            W[mask] += noise[mask]
            np.clip(W, -self.cfg.weight_clip, self.cfg.weight_clip, out=W)
            total += int(mask.sum())

            # Smaller bias perturbation
            bias_mask = self._rng.random(b.shape) < layer_rate * 0.4
            b_noise = self._rng.normal(0.0, layer_sigma * 0.3, b.shape)
            b[bias_mask] += b_noise[bias_mask]

        return total

    def _check_stopping(self) -> Tuple[bool, Optional[str]]:
        """Return (done, reason) based on configured stopping criteria."""
        if self._best_accuracy >= self.cfg.stop_accuracy:
            return True, f"target accuracy {self.cfg.stop_accuracy:.1%} reached"
        if self._generation >= self.cfg.max_generations:
            return True, f"max generations ({self.cfg.max_generations}) reached"
        if self._stagnant_gens >= self.cfg.patience:
            return True, f"no improvement for {self.cfg.patience} generations (patience exhausted)"
        return False, None
