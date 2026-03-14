"""
tests/test_evoagent.py
~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the EvoAgent library.

Run with:  python -m pytest tests/ -v
"""

import json
import math
import os
import tempfile

import numpy as np
import pytest

from evoagent import (
    Agent,
    AgentConfig,
    BinarySymmetry,
    CountOnes,
    Evaluator,
    MetaConfig,
    MetaController,
    Network,
    WeightStats,
    XORParity,
)
from evoagent.meta import Strategy


# ===========================================================================
# Network tests
# ===========================================================================

class TestNetwork:
    def test_init_shape(self):
        net = Network([4, 8, 2])
        assert len(net.weights) == 2
        assert net.weights[0].shape == (8, 4)
        assert net.weights[1].shape == (2, 8)

    def test_forward_output_shape(self):
        net = Network([4, 8, 4, 2])
        out = net.forward([0, 1, 0, 1])
        assert len(out) == 2

    def test_forward_output_range(self):
        """Output layer uses sigmoid so values must be in (0, 1)."""
        net = Network([4, 8, 2], seed=0)
        for _ in range(20):
            x = np.random.default_rng().integers(0, 2, 4).tolist()
            out = net.forward(x)
            assert all(0.0 < v < 1.0 for v in out)

    def test_total_parameters(self):
        net = Network([4, 8, 2])
        # weights: 8*4 + 2*8 = 32+16 = 48; biases: 8 + 2 = 10
        assert net.total_parameters() == 58

    def test_inspect_layer_returns_stats(self):
        net = Network([4, 8, 2], seed=1)
        stats = net.inspect_layer(0)
        assert isinstance(stats, WeightStats)
        assert stats.layer_index == 0
        assert stats.std >= 0.0
        assert 0.0 <= stats.dead_fraction <= 1.0

    def test_inspect_all_length(self):
        net = Network([4, 8, 4, 2])
        all_stats = net.inspect_all()
        assert len(all_stats) == 3

    def test_snapshot_roundtrip(self):
        net = Network([4, 8, 2], seed=42)
        snap = net.snapshot()
        original_w0 = net.weights[0].copy()

        # Corrupt weights
        net.weights[0] += 99.0

        # Restore
        net.load_snapshot(snap)
        assert np.allclose(net.weights[0], original_w0)

    def test_save_and_load(self):
        net = Network([4, 8, 2], seed=7)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            net.save(path)
            loaded = Network.load(path)
            assert loaded.layer_sizes == net.layer_sizes
            assert np.allclose(loaded.weights[0], net.weights[0])
        finally:
            os.unlink(path)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError):
            Network([4, 2], activation="gelu")

    def test_too_few_layers_raises(self):
        with pytest.raises(ValueError):
            Network([4])

    def test_all_activations_run(self):
        for act in ("relu", "sigmoid", "tanh", "leaky_relu"):
            net = Network([4, 8, 2], activation=act, seed=0)
            out = net.forward([1, 0, 1, 0])
            assert len(out) == 2

    def test_repr(self):
        net = Network([4, 8, 2])
        r = repr(net)
        assert "Network" in r
        assert "relu" in r


# ===========================================================================
# WeightStats tests
# ===========================================================================

class TestWeightStats:
    def _make_stats(self, std=0.5, mean=0.0, max_val=1.0):
        return WeightStats(
            layer_index=0, mean=mean, std=std,
            min_val=-max_val, max_val=max_val,
            dead_fraction=0.0, l2_norm=1.0
        )

    def test_is_collapsed_true(self):
        assert self._make_stats(std=0.01).is_collapsed()

    def test_is_collapsed_false(self):
        assert not self._make_stats(std=0.5).is_collapsed()

    def test_has_saturation_true(self):
        assert self._make_stats(mean=3.0).has_saturation()

    def test_has_saturation_false(self):
        assert not self._make_stats(mean=0.1).has_saturation()


# ===========================================================================
# Task & Evaluator tests
# ===========================================================================

class TestTasks:
    def test_xor_parity_dataset_size(self):
        task = XORParity(bits=4)
        ds = task.generate()
        assert len(ds) == 16

    def test_xor_parity_labels(self):
        task = XORParity(bits=3)
        ds = task.generate()
        for s in ds:
            expected = sum(s.input) % 2
            assert s.target == expected

    def test_binary_symmetry_odd_length_raises(self):
        with pytest.raises(ValueError):
            BinarySymmetry(length=5)

    def test_binary_symmetry_correct_labels(self):
        task = BinarySymmetry(length=4)
        ds = task.generate()
        for s in ds:
            assert s.target == (1 if s.input == s.input[::-1] else 0)

    def test_count_ones_dataset_size(self):
        task = CountOnes(bits=4)
        assert len(task.generate()) == 16

    def test_count_ones_classes(self):
        task = CountOnes(bits=6)
        labels = {s.target for s in task.generate()}
        assert labels == {0, 1, 2}


class TestEvaluator:
    def test_returns_valid_range(self):
        net = Network([4, 8, 2], seed=0)
        ev = Evaluator(XORParity(bits=4))
        acc, loss = ev.evaluate(net)
        assert 0.0 <= acc <= 1.0
        assert loss >= 0.0

    def test_mse_loss(self):
        net = Network([4, 8, 2], seed=0)
        ev = Evaluator(XORParity(bits=4), loss="mse")
        acc, loss = ev.evaluate(net)
        assert loss >= 0.0

    def test_sample_count(self):
        ev = Evaluator(XORParity(bits=4))
        assert ev.sample_count() == 16


# ===========================================================================
# MetaController tests
# ===========================================================================

class TestMetaController:
    def _setup(self):
        net = Network([4, 8, 4, 2], seed=0)
        ev = Evaluator(XORParity(bits=4))
        cfg = MetaConfig(max_generations=5, patience=50, n_candidates=3)
        meta = MetaController(cfg, seed=0)
        return net, ev, meta

    def test_step_returns_result(self):
        net, ev, meta = self._setup()
        result = meta.step(net, ev)
        assert result.generation == 1
        assert 0.0 <= result.accuracy <= 1.0
        assert result.loss >= 0.0

    def test_strategy_is_valid(self):
        net, ev, meta = self._setup()
        result = meta.step(net, ev)
        assert isinstance(result.strategy, Strategy)

    def test_stops_at_max_generations(self):
        net = Network([4, 8, 4, 2], seed=0)
        ev = Evaluator(XORParity(bits=4))
        cfg = MetaConfig(max_generations=3, patience=200)
        meta = MetaController(cfg, seed=0)
        for _ in range(3):
            result = meta.step(net, ev)
        assert result.done
        assert "max generations" in result.stop_reason

    def test_stops_at_accuracy_target(self):
        """Force an immediately achievable target and verify stopping."""
        net = Network([4, 8, 4, 2], seed=0)
        ev = Evaluator(XORParity(bits=4))
        cfg = MetaConfig(stop_accuracy=0.0, max_generations=100)  # already satisfied
        meta = MetaController(cfg, seed=0)
        result = meta.step(net, ev)
        assert result.done

    def test_stops_at_patience(self):
        net = Network([4, 8, 4, 2], seed=0)
        ev = Evaluator(XORParity(bits=4))
        cfg = MetaConfig(patience=3, max_generations=1000, min_acc_delta=1.0)  # never improves enough
        meta = MetaController(cfg, seed=0)
        result = None
        for _ in range(10):
            result = meta.step(net, ev)
            if result.done:
                break
        assert result.done
        assert "patience" in result.stop_reason

    def test_reset_clears_state(self):
        net, ev, meta = self._setup()
        for _ in range(3):
            meta.step(net, ev)
        meta.reset()
        assert meta.generation == 0
        assert meta.best_accuracy == 0.0

    def test_error_after_done(self):
        from evoagent.agent import Agent, AgentConfig
        cfg = AgentConfig(
            meta=MetaConfig(stop_accuracy=0.0),
            seed=0
        )
        agent = Agent(cfg)
        agent.step()
        with pytest.raises(RuntimeError):
            agent.step()


# ===========================================================================
# Agent integration tests
# ===========================================================================

class TestAgent:
    def test_run_terminates(self):
        cfg = AgentConfig(
            layer_sizes=[4, 8, 2],
            task=XORParity(bits=4),
            meta=MetaConfig(max_generations=10, n_candidates=2),
            seed=0,
            verbose=False,
        )
        agent = Agent(cfg)
        result = agent.run()
        assert result.done

    def test_on_step_callback(self):
        calls = []
        cfg = AgentConfig(
            layer_sizes=[4, 8, 2],
            task=XORParity(bits=4),
            meta=MetaConfig(max_generations=5, n_candidates=2),
            seed=0,
            verbose=False,
            on_step=lambda r: calls.append(r.generation),
        )
        Agent(cfg).run()
        assert calls == [1, 2, 3, 4, 5]

    def test_on_done_callback(self):
        done_results = []
        cfg = AgentConfig(
            layer_sizes=[4, 8, 2],
            task=XORParity(bits=4),
            meta=MetaConfig(max_generations=3, n_candidates=2),
            seed=0,
            verbose=False,
            on_done=lambda r: done_results.append(r),
        )
        Agent(cfg).run()
        assert len(done_results) == 1
        assert done_results[0].done

    def test_save_and_load(self):
        cfg = AgentConfig(
            layer_sizes=[4, 8, 2],
            task=XORParity(bits=4),
            meta=MetaConfig(max_generations=5, n_candidates=2),
            seed=0,
            verbose=False,
        )
        agent = Agent(cfg)
        agent.run()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            loaded = Agent.load(path, cfg)
            assert loaded.network.layer_sizes == agent.network.layer_sizes
        finally:
            os.unlink(path)

    def test_repr(self):
        agent = Agent(AgentConfig(verbose=False))
        r = repr(agent)
        assert "Agent" in r

    def test_reset(self):
        cfg = AgentConfig(
            meta=MetaConfig(max_generations=3, n_candidates=2),
            verbose=False,
            seed=0,
        )
        agent = Agent(cfg)
        agent.run()
        assert agent.done
        agent.reset()
        assert not agent.done
        assert agent.generation == 0
