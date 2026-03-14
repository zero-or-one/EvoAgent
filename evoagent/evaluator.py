"""
evoagent.evaluator
~~~~~~~~~~~~~~~~~~
Dataset generation and network scoring.

All built-in tasks are deterministic, meaning that the same network will
always receive the same score — vital for fair comparisons when the
meta-controller decides whether to accept a mutation.

Adding a custom task
--------------------
Subclass :class:`Task` and implement :meth:`generate` and :meth:`score_label`.

Example
-------
>>> from evoagent.evaluator import Evaluator, XORParity
>>> task = XORParity(bits=4)
>>> ev   = Evaluator(task)
>>> acc, loss = ev.evaluate(my_network)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .network import Network


# ---------------------------------------------------------------------------
# Data sample
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    input: List[float]
    target: int          # class index


# ---------------------------------------------------------------------------
# Task base class
# ---------------------------------------------------------------------------

class Task(ABC):
    """Abstract base class for all classification tasks."""

    @abstractmethod
    def generate(self) -> List[Sample]:
        """Return the complete dataset for this task."""

    @property
    @abstractmethod
    def n_inputs(self) -> int:
        """Number of input features."""

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """Number of output classes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable task name."""


# ---------------------------------------------------------------------------
# Built-in tasks
# ---------------------------------------------------------------------------

class XORParity(Task):
    """
    N-bit parity classification.

    The goal is to predict whether the number of 1-bits in the input is
    even (class 0) or odd (class 1).  This requires the network to learn
    non-linear feature interactions — a classic challenge for small networks.

    Parameters
    ----------
    bits:
        Number of input bits (default 4 → 16 samples).
    """

    def __init__(self, bits: int = 4) -> None:
        self._bits = bits

    @property
    def n_inputs(self) -> int:
        return self._bits

    @property
    def n_classes(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return f"XOR-parity-{self._bits}"

    def generate(self) -> List[Sample]:
        samples = []
        for i in range(2 ** self._bits):
            bits = [(i >> j) & 1 for j in range(self._bits)]
            parity = sum(bits) % 2
            samples.append(Sample(input=bits, target=parity))
        return samples


class BinarySymmetry(Task):
    """
    Detect whether a binary vector is symmetric (palindrome).

    Parameters
    ----------
    length:
        Must be even.  Default is 6.
    """

    def __init__(self, length: int = 6) -> None:
        if length % 2 != 0:
            raise ValueError("length must be even")
        self._length = length

    @property
    def n_inputs(self) -> int:
        return self._length

    @property
    def n_classes(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return f"binary-symmetry-{self._length}"

    def generate(self) -> List[Sample]:
        samples = []
        for i in range(2 ** self._length):
            bits = [(i >> j) & 1 for j in range(self._length)]
            label = 1 if bits == bits[::-1] else 0
            samples.append(Sample(input=bits, target=label))
        return samples


class CountOnes(Task):
    """
    Count the number of 1-bits and classify into low / mid / high buckets.

    Parameters
    ----------
    bits:
        Number of input bits (default 6).
    """

    def __init__(self, bits: int = 6) -> None:
        self._bits = bits

    @property
    def n_inputs(self) -> int:
        return self._bits

    @property
    def n_classes(self) -> int:
        return 3

    @property
    def name(self) -> str:
        return f"count-ones-{self._bits}"

    def generate(self) -> List[Sample]:
        samples = []
        low_thresh = self._bits // 3
        high_thresh = 2 * self._bits // 3
        for i in range(2 ** self._bits):
            bits = [(i >> j) & 1 for j in range(self._bits)]
            ones = sum(bits)
            if ones <= low_thresh:
                label = 0
            elif ones <= high_thresh:
                label = 1
            else:
                label = 2
            samples.append(Sample(input=bits, target=label))
        return samples


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evaluates a :class:`~evoagent.network.Network` against a :class:`Task`.

    Parameters
    ----------
    task:
        The task to evaluate against.
    loss:
        Loss function name — ``"cross_entropy"`` (default) or ``"mse"``.
    """

    def __init__(self, task: Task, loss: str = "cross_entropy") -> None:
        self.task = task
        self._loss_name = loss
        self._dataset: List[Sample] = task.generate()

    def evaluate(self, net: Network) -> Tuple[float, float]:
        """
        Score a network.

        Returns
        -------
        accuracy:
            Fraction of samples correctly classified (0–1).
        loss:
            Mean per-sample loss (lower is better).
        """
        correct = 0
        total_loss = 0.0
        eps = 1e-9

        for sample in self._dataset:
            probs = net.forward(sample.input)
            pred = int(np.argmax(probs))
            if pred == sample.target:
                correct += 1

            if self._loss_name == "cross_entropy":
                p = float(np.clip(probs[sample.target], eps, 1 - eps))
                total_loss += -math.log(p)
            else:  # mse
                one_hot = np.zeros(len(probs))
                one_hot[sample.target] = 1.0
                total_loss += float(np.mean((probs - one_hot) ** 2))

        n = len(self._dataset)
        return correct / n, total_loss / n

    def sample_count(self) -> int:
        """Number of samples in the dataset."""
        return len(self._dataset)
