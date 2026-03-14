"""
evoagent.network
~~~~~~~~~~~~~~~~
Lightweight feedforward neural network with full weight introspection.

The Network class intentionally exposes its weights as plain NumPy arrays so
the meta-controller can read, analyse, and rewrite them directly — no gradient
tape required.

Example
-------
>>> from evoagent.network import Network
>>> net = Network(layer_sizes=[4, 8, 4, 2], activation="relu")
>>> out = net.forward([0, 1, 0, 1])
>>> snap = net.snapshot()          # serialisable copy of all weights + biases
>>> net.load_snapshot(snap)        # restore a previous state
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x >= 0, x, alpha * x)


ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": _relu,
    "sigmoid": _sigmoid,
    "tanh": _tanh,
    "leaky_relu": _leaky_relu,
}


# ---------------------------------------------------------------------------
# WeightStats — lightweight introspection report
# ---------------------------------------------------------------------------

@dataclass
class WeightStats:
    """Per-layer statistics used by the meta-controller for self-inspection."""

    layer_index: int
    mean: float
    std: float
    min_val: float
    max_val: float
    dead_fraction: float   # fraction of weights with |w| < 1e-3
    l2_norm: float

    def is_collapsed(self, threshold: float = 0.05) -> bool:
        """Return True if the layer's weights have collapsed toward zero."""
        return self.std < threshold

    def has_saturation(self, threshold: float = 2.5) -> bool:
        """Return True if many weights are pushed to extreme values."""
        return abs(self.mean) > threshold or self.max_val > threshold * 1.5

    def __repr__(self) -> str:
        return (
            f"WeightStats(layer={self.layer_index}, "
            f"mean={self.mean:.4f}, std={self.std:.4f}, "
            f"dead={self.dead_fraction:.1%})"
        )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network:
    """
    A fully-connected feedforward neural network whose weights are directly
    accessible and mutable at runtime.

    Parameters
    ----------
    layer_sizes:
        List of node counts per layer, e.g. ``[4, 8, 4, 2]``.
    activation:
        Hidden-layer activation — one of ``"relu"``, ``"sigmoid"``,
        ``"tanh"``, ``"leaky_relu"``.  The output layer always uses sigmoid.
    seed:
        Optional RNG seed for reproducible weight initialisation.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        seed: Optional[int] = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Network requires at least an input and output layer.")
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(ACTIVATIONS)}")

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self._act_fn = ACTIVATIONS[activation]
        self._rng = np.random.default_rng(seed)

        # Weight matrices and bias vectors — stored as plain numpy arrays so
        # the meta-controller can manipulate them directly.
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._initialise_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise_weights(self) -> None:
        """He (Kaiming) initialisation for relu; Xavier/Glorot for others."""
        self.weights.clear()
        self.biases.clear()

        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]

            if self.activation_name in ("relu", "leaky_relu"):
                scale = math.sqrt(2.0 / fan_in)
            else:
                scale = math.sqrt(2.0 / (fan_in + fan_out))

            W = self._rng.normal(0.0, scale, size=(fan_out, fan_in))
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Sequence[float]) -> np.ndarray:
        """
        Run a forward pass and return the output layer activations.

        Parameters
        ----------
        x:
            Input vector; must match ``layer_sizes[0]`` in length.

        Returns
        -------
        np.ndarray
            Output activations (always passed through sigmoid).
        """
        h = np.asarray(x, dtype=float)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ h + b
            is_last = i == len(self.weights) - 1
            h = _sigmoid(z) if is_last else self._act_fn(z)
        return h

    # ------------------------------------------------------------------
    # Weight introspection
    # ------------------------------------------------------------------

    def inspect_layer(self, layer_index: int) -> WeightStats:
        """
        Return detailed statistics for a single weight matrix.

        Parameters
        ----------
        layer_index:
            Zero-based index into ``self.weights``.
        """
        W = self.weights[layer_index].ravel()
        dead = float(np.mean(np.abs(W) < 1e-3))
        return WeightStats(
            layer_index=layer_index,
            mean=float(np.mean(W)),
            std=float(np.std(W)),
            min_val=float(W.min()),
            max_val=float(W.max()),
            dead_fraction=dead,
            l2_norm=float(np.linalg.norm(W)),
        )

    def inspect_all(self) -> List[WeightStats]:
        """Return ``WeightStats`` for every layer."""
        return [self.inspect_layer(i) for i in range(len(self.weights))]

    def total_parameters(self) -> int:
        """Total number of trainable parameters (weights + biases)."""
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Return a deep copy of the network state as a plain dict.

        The snapshot is JSON-serialisable (via ``numpy.ndarray.tolist``).
        """
        return {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation_name,
            "weights": [W.tolist() for W in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    def load_snapshot(self, snap: dict) -> None:
        """Restore weights and biases from a previously captured snapshot."""
        self.weights = [np.array(W) for W in snap["weights"]]
        self.biases = [np.array(b) for b in snap["biases"]]

    def save(self, path: str) -> None:
        """Serialise the network to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.snapshot(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Network":
        """Deserialise a network from a JSON file produced by :meth:`save`."""
        with open(path) as f:
            snap = json.load(f)
        net = cls(snap["layer_sizes"], snap["activation"])
        net.load_snapshot(snap)
        return net

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sizes = " → ".join(str(s) for s in self.layer_sizes)
        return f"Network({sizes}, activation={self.activation_name}, params={self.total_parameters()})"
