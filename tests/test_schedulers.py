"""Unit tests for dreamplace_ext/schedulers.py — pure Python, no GPU needed."""

import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dreamplace_ext.schedulers import (
    gamma_baseline_linear,
    gamma_exponential,
    gamma_overflow_adaptive,
    gamma_cosine_annealing,
    lambda_baseline_exponential,
    lambda_gradient_adaptive,
    lambda_plateau_detector,
)


GAMMA_SCHEDULES = [
    gamma_baseline_linear,
    gamma_exponential,
    gamma_overflow_adaptive,
    gamma_cosine_annealing,
]

LAMBDA_SCHEDULES = [
    lambda_baseline_exponential,
    lambda_gradient_adaptive,
    lambda_plateau_detector,
]


class TestGammaSchedules:
    @pytest.mark.parametrize("fn", GAMMA_SCHEDULES)
    def test_returns_float(self, fn):
        val = fn(0, 1000, 0.9, [])
        assert isinstance(val, float)

    @pytest.mark.parametrize("fn", GAMMA_SCHEDULES)
    def test_in_valid_range(self, fn):
        for t in [0, 100, 500, 999]:
            for ovfl in [0.0, 0.5, 1.0]:
                val = fn(t, 1000, ovfl, [])
                assert 0.01 <= val <= 20.0, f"{fn.__name__} returned {val} at t={t}, ovfl={ovfl}"

    @pytest.mark.parametrize("fn", GAMMA_SCHEDULES)
    def test_start_higher_than_end(self, fn):
        start = fn(0, 1000, 0.9, [])
        end = fn(999, 1000, 0.05, [])
        assert start > end, f"{fn.__name__}: start={start} not > end={end}"

    def test_linear_midpoint(self):
        # At t=0.5: should be midpoint of [8.0, 0.5] = 4.25
        val = gamma_baseline_linear(500, 1000, 0.5, [])
        assert val == pytest.approx(4.25, rel=0.05)

    def test_cosine_start_equals_max(self):
        val = gamma_cosine_annealing(0, 1000, 0.9, [])
        assert val == pytest.approx(8.0, rel=0.01)

    def test_overflow_adaptive_responds_to_overflow(self):
        # Higher overflow → higher gamma (smooth gradients when cells are clustered)
        # Low overflow → low gamma (accurate WL gradients when cells are spread out)
        high_ovfl = gamma_overflow_adaptive(100, 1000, 0.9, [])
        low_ovfl = gamma_overflow_adaptive(100, 1000, 0.1, [])
        assert high_ovfl > low_ovfl  # high overflow → higher gamma (DreamPlace convention)

    @pytest.mark.parametrize("fn", GAMMA_SCHEDULES)
    def test_handles_zero_total_iterations(self, fn):
        # Should not divide by zero
        val = fn(0, 0, 0.5, [])
        assert math.isfinite(val)

    @pytest.mark.parametrize("fn", GAMMA_SCHEDULES)
    def test_handles_empty_history(self, fn):
        val = fn(50, 1000, 0.3, [])
        assert math.isfinite(val)

    @pytest.mark.parametrize("fn", GAMMA_SCHEDULES)
    def test_handles_long_history(self, fn):
        history = list(range(1000))
        val = fn(500, 1000, 0.5, history)
        assert math.isfinite(val)


class TestLambdaSchedules:
    @pytest.mark.parametrize("fn", LAMBDA_SCHEDULES)
    def test_returns_float(self, fn):
        val = fn(0, 0.9, [], 0.5, 1.0)
        assert isinstance(val, float)

    @pytest.mark.parametrize("fn", LAMBDA_SCHEDULES)
    def test_positive(self, fn):
        for ovfl in [0.0, 0.3, 0.7, 1.0]:
            val = fn(100, ovfl, [], 0.1, 1.0)
            assert val > 0, f"{fn.__name__} returned non-positive {val}"

    @pytest.mark.parametrize("fn", LAMBDA_SCHEDULES)
    def test_bounded_by_lambda_max(self, fn):
        # Starting from a very large lambda, should not exceed lambda_max=1e6
        val = fn(0, 0.9, [], 0.5, 9e5)
        assert val <= 1e6 + 1e-9

    @pytest.mark.parametrize("fn", LAMBDA_SCHEDULES)
    def test_increases_with_high_overflow(self, fn):
        lo = fn(100, 0.05, [], 0.1, 1.0)
        hi = fn(100, 0.95, [], 0.1, 1.0)
        assert hi >= lo

    def test_gradient_adaptive_boosts_when_flat(self):
        # Very small gradient norm → should push lambda harder
        small_grad = lambda_gradient_adaptive(100, 0.9, [], gradient_norm=1e-5, current_lambda=1.0)
        large_grad = lambda_gradient_adaptive(100, 0.9, [], gradient_norm=2.0, current_lambda=1.0)
        assert small_grad > large_grad

    def test_plateau_detector_doubles_on_plateau(self):
        # Build a flat overflow history (plateau condition)
        flat_history = [0.95] * 25
        boosted = lambda_plateau_detector(100, 0.95, flat_history, 0.1, 1.0)
        baseline = lambda_baseline_exponential(100, 0.95, flat_history, 0.1, 1.0)
        assert boosted > baseline  # plateau should cause stronger boost

    def test_plateau_detector_normal_without_plateau(self):
        # Decreasing overflow → no plateau → normal increase
        decreasing = [0.9 - i * 0.01 for i in range(25)]
        val = lambda_plateau_detector(100, 0.5, decreasing, 0.1, 1.0)
        assert val == pytest.approx(1.0 * 1.02)


class TestSchedulerMonotonicity:
    """Gamma schedules should be monotonically non-increasing (in expectation)."""

    @pytest.mark.parametrize("fn", GAMMA_SCHEDULES)
    def test_generally_decreasing(self, fn):
        """Schedule value at t=0 should exceed t=total_iterations-1."""
        total = 1000
        vals = [fn(t, total, max(0.05, 1.0 - t/total), []) for t in range(0, total, 100)]
        # At least the first should exceed the last
        assert vals[0] >= vals[-1]
