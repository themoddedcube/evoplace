"""Unit tests for evaluator/metrics.py — pure numpy, no GPU needed."""

import math
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluator.metrics import (
    compute_hpwl,
    compute_overflow,
    compute_tns_proxy,
    normalize_hpwl,
    compute_all_metrics,
)


# ── HPWL ──────────────────────────────────────────────────────────────────────

class TestHPWL:
    def test_single_net_two_pins(self):
        # Two cells at (0,0) and (3,4): HPWL = 3+4 = 7
        x = np.array([0.0, 3.0])
        y = np.array([0.0, 4.0])
        nets = [[0, 1]]
        assert compute_hpwl(x, y, nets) == pytest.approx(7.0)

    def test_single_net_four_pins(self):
        # Bounding box: x in [0,4], y in [0,3] → HPWL = 4+3 = 7
        x = np.array([0.0, 4.0, 2.0, 1.0])
        y = np.array([0.0, 3.0, 1.0, 2.0])
        nets = [[0, 1, 2, 3]]
        assert compute_hpwl(x, y, nets) == pytest.approx(7.0)

    def test_two_independent_nets(self):
        x = np.array([0.0, 1.0, 10.0, 12.0])
        y = np.array([0.0, 0.0, 5.0, 5.0])
        nets = [[0, 1], [2, 3]]
        # Net 0: HPWL=1+0=1, Net 1: HPWL=2+0=2 → total=3
        assert compute_hpwl(x, y, nets) == pytest.approx(3.0)

    def test_single_pin_net_skipped(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        nets = [[0], [0, 1]]
        # Single-pin net contributes 0; two-pin net = sqrt(2) HPWL = 1+1=2
        assert compute_hpwl(x, y, nets) == pytest.approx(2.0)

    def test_net_weights(self):
        x = np.array([0.0, 4.0])
        y = np.array([0.0, 0.0])
        nets = [[0, 1]]
        weights = np.array([2.5])
        assert compute_hpwl(x, y, nets, net_weights=weights) == pytest.approx(10.0)

    def test_zero_hpwl_same_position(self):
        x = np.array([5.0, 5.0, 5.0])
        y = np.array([3.0, 3.0, 3.0])
        nets = [[0, 1, 2]]
        assert compute_hpwl(x, y, nets) == pytest.approx(0.0)

    def test_large_design_consistency(self):
        rng = np.random.default_rng(42)
        N = 10000
        x = rng.uniform(0, 1000, N)
        y = rng.uniform(0, 1000, N)
        nets = [list(rng.choice(N, size=rng.integers(2, 8), replace=False))
                for _ in range(2000)]
        hpwl = compute_hpwl(x, y, nets)
        assert hpwl > 0
        assert math.isfinite(hpwl)


# ── Overflow ──────────────────────────────────────────────────────────────────

class TestOverflow:
    def test_no_overflow_sparse(self):
        # 4 cells in a 100×100 die, each 1×1 → very low density
        x = np.array([10.0, 30.0, 70.0, 90.0])
        y = np.array([10.0, 30.0, 70.0, 90.0])
        w = np.ones(4)
        h = np.ones(4)
        mean_ovfl, top5_ovfl = compute_overflow(x, y, w, h, 100, 100, num_bins_x=10, num_bins_y=10)
        assert mean_ovfl == pytest.approx(0.0, abs=1e-9)

    def test_overflow_detected_when_dense(self):
        # Many cells stacked at center → should overflow
        N = 500
        x = np.full(N, 50.0)
        y = np.full(N, 50.0)
        w = np.ones(N) * 2.0
        h = np.ones(N) * 2.0
        mean_ovfl, top5_ovfl = compute_overflow(x, y, w, h, 100, 100, num_bins_x=10, num_bins_y=10)
        assert mean_ovfl > 0.0
        assert top5_ovfl >= mean_ovfl

    def test_top5_geq_mean(self):
        rng = np.random.default_rng(0)
        N = 100
        x = rng.uniform(0, 100, N)
        y = rng.uniform(0, 100, N)
        w = np.ones(N) * 3
        h = np.ones(N) * 3
        mean_ovfl, top5 = compute_overflow(x, y, w, h, 100, 100)
        assert top5 >= mean_ovfl - 1e-9


# ── TNS Proxy ──────────────────────────────────────────────────────────────────

class TestTNSProxy:
    def test_zero_when_no_critical_nets(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])
        nets = [[0, 1]]
        critical = np.array([False])
        tns = compute_tns_proxy(x, y, nets, critical)
        assert tns == pytest.approx(0.0)

    def test_positive_for_critical_long_net(self):
        x = np.array([0.0, 1000.0])
        y = np.array([0.0, 0.0])
        nets = [[0, 1]]
        critical = np.array([True])
        tns = compute_tns_proxy(x, y, nets, critical, timing_slack_threshold=0.0)
        assert tns > 0.0

    def test_high_fanout_nets_flagged_by_default(self):
        # Default: nets with >= 4 pins are treated as critical
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.zeros(4)
        nets = [[0, 1, 2, 3]]  # 4 pins → flagged as critical by default
        tns = compute_tns_proxy(x, y, nets, critical_net_flags=None, timing_slack_threshold=0.0)
        assert tns > 0.0


# ── Normalization ──────────────────────────────────────────────────────────────

class TestNormalize:
    def test_equal_returns_one(self):
        assert normalize_hpwl(100.0, 100.0) == pytest.approx(1.0)

    def test_better_returns_lt_one(self):
        assert normalize_hpwl(90.0, 100.0) == pytest.approx(0.9)

    def test_worse_returns_gt_one(self):
        assert normalize_hpwl(110.0, 100.0) == pytest.approx(1.1)

    def test_zero_baseline_returns_inf(self):
        assert normalize_hpwl(1.0, 0.0) == float("inf")


# ── compute_all_metrics ────────────────────────────────────────────────────────

class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        x = np.array([0.0, 10.0, 20.0])
        y = np.array([0.0, 5.0, 10.0])
        w = np.ones(3) * 2
        h = np.ones(3) * 2
        nets = [[0, 1], [1, 2]]
        metrics = compute_all_metrics(x, y, w, h, nets, die_w=100, die_h=100)
        assert "hpwl" in metrics
        assert "mean_overflow" in metrics
        assert "top5_overflow" in metrics
        assert "tns_proxy" in metrics

    def test_normalized_hpwl_added_when_baseline_given(self):
        x = np.array([0.0, 5.0])
        y = np.array([0.0, 0.0])
        nets = [[0, 1]]
        metrics = compute_all_metrics(
            x, y, np.ones(2), np.ones(2), nets,
            die_w=100, die_h=100, baseline_hpwl=10.0
        )
        assert "normalized_hpwl" in metrics
        assert metrics["normalized_hpwl"] == pytest.approx(0.5)
