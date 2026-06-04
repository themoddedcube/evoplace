"""
Integration tests for the full stub pipeline.

These tests exercise the complete evaluation → evolution → logging cycle
without a GPU or DreamPlace build.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluator.run_placement import run_placement, run_cascade_evaluation, PlacementResult
from evaluator.benchmark_suite import run_suite, aggregate_metrics
from dreamplace_ext.schedulers import gamma_baseline_linear, gamma_cosine_annealing
from graphs.plot_utils import (
    plot_benchmark_bars, plot_convergence, plot_pareto_frontier,
    plot_schedule_trajectory,
)


# ── PlacementResult ────────────────────────────────────────────────────────────

class TestPlacementResult:
    def test_fitness_penalizes_divergence(self):
        r_clean = PlacementResult({"hpwl": 1e9}, 60.0, divergence_events=0, converged=True)
        r_div   = PlacementResult({"hpwl": 1e9}, 60.0, divergence_events=3, converged=False)
        assert r_div.fitness > r_clean.fitness

    def test_to_dict_serializable(self):
        r = PlacementResult({"hpwl": 1e9, "mean_overflow": 0.07}, 45.0, 1, True)
        d = r.to_dict()
        assert "metrics" in d and "fitness" in d
        json.dumps(d)  # should not raise

    def test_repr_doesnt_crash(self):
        r = PlacementResult({"hpwl": 4.2e8, "mean_overflow": 0.06, "tns_proxy": 1e7}, 30.0, 0, True)
        s = repr(r)
        assert "hpwl" in s.lower() or "PlacementResult" in s


# ── Stub run_placement ─────────────────────────────────────────────────────────

class TestRunPlacementStub:
    @pytest.fixture
    def tmpdir(self):
        d = Path(tempfile.mkdtemp())
        yield d
        shutil.rmtree(d)

    def test_returns_valid_result(self, tmpdir):
        r = run_placement(tmpdir, tmpdir, use_stub=True)
        assert isinstance(r, PlacementResult)
        assert r.metrics["hpwl"] > 0
        assert 0.0 <= r.metrics["mean_overflow"] <= 1.0
        assert r.runtime_s > 0

    def test_deterministic_with_same_seed(self, tmpdir):
        r1 = run_placement(tmpdir, tmpdir / "r1", use_stub=True, seed=42)
        r2 = run_placement(tmpdir, tmpdir / "r2", use_stub=True, seed=42)
        assert r1.metrics["hpwl"] == pytest.approx(r2.metrics["hpwl"])

    def test_different_seeds_give_different_results(self, tmpdir):
        r1 = run_placement(tmpdir, tmpdir / "r1", use_stub=True, seed=1)
        r2 = run_placement(tmpdir, tmpdir / "r2", use_stub=True, seed=2)
        assert r1.metrics["hpwl"] != r2.metrics["hpwl"]

    def test_good_gamma_schedule_improves_hpwl(self, tmpdir):
        """Stub rewards well-behaved monotone-decreasing gamma schedules."""
        baseline = run_placement(tmpdir, tmpdir / "b", use_stub=True, seed=99)
        improved = run_placement(tmpdir, tmpdir / "i", use_stub=True, seed=99,
                                 gamma_schedule_fn=gamma_cosine_annealing)
        # cosine_annealing is monotone-decreasing → should get schedule_factor=0.97
        assert improved.metrics["hpwl"] <= baseline.metrics["hpwl"] * 1.05

    def test_output_json_saved(self, tmpdir):
        out = tmpdir / "result_test"
        run_placement(tmpdir, out, use_stub=True)
        assert (out / "result.json").exists()


# ── Cascade Evaluation ────────────────────────────────────────────────────────

class TestCascadeEvaluation:
    @pytest.fixture
    def tmpdir(self):
        d = Path(tempfile.mkdtemp())
        yield d
        shutil.rmtree(d)

    def test_passes_with_good_candidate(self, tmpdir):
        # Baseline HPWL close to what stub returns → threshold not triggered
        baseline = run_placement(tmpdir, tmpdir / "base", use_stub=True, seed=0)
        baseline_hpwl = baseline.metrics["hpwl"]
        result = run_cascade_evaluation(
            tmpdir, tmpdir / "cascade",
            baseline_hpwl=baseline_hpwl,
            use_stub=True,
        )
        assert result is not None

    def test_eliminated_with_very_tight_threshold(self, tmpdir):
        # Set baseline HPWL unrealistically high so even a good run looks bad
        result = run_cascade_evaluation(
            tmpdir, tmpdir / "cascade_elim",
            baseline_hpwl=1e3,  # 1000x too small → norm_hpwl >> threshold
            use_stub=True,
        )
        assert result is None


# ── Benchmark Suite ────────────────────────────────────────────────────────────

class TestBenchmarkSuite:
    @pytest.fixture
    def tmpdir(self):
        d = Path(tempfile.mkdtemp())
        yield d
        shutil.rmtree(d)

    def test_small_suite_returns_results(self, tmpdir):
        results = run_suite(
            benchmark_dir=tmpdir / "benchmarks",
            output_dir=tmpdir / "out",
            suite="small",
            use_stub=True,
        )
        assert len(results) == 2
        assert "fft_1" in results
        assert "fft_2" in results

    def test_aggregate_metrics_finite(self, tmpdir):
        results = run_suite(tmpdir / "b", tmpdir / "o", suite="small", use_stub=True)
        agg = aggregate_metrics(results)
        assert all(
            isinstance(v, (int, float)) and (not isinstance(v, float) or v == v)
            for v in agg.values()
        )

    def test_baselines_normalize_correctly(self, tmpdir):
        results = run_suite(tmpdir / "b", tmpdir / "o", suite="small", use_stub=True)
        baselines = {name: r.metrics["hpwl"] for name, r in results.items()}
        results2 = run_suite(tmpdir / "b", tmpdir / "o2", suite="small",
                              use_stub=True, baselines=baselines)
        for name, r in results2.items():
            assert "normalized_hpwl" in r.metrics
            assert r.metrics["normalized_hpwl"] == pytest.approx(1.0, rel=0.15)


# ── Graph Generation ──────────────────────────────────────────────────────────

class TestGraphGeneration:
    @pytest.fixture
    def tmpdir(self):
        d = Path(tempfile.mkdtemp())
        yield d
        shutil.rmtree(d)

    def make_stub_results(self):
        from evaluator.run_placement import PlacementResult
        results = {}
        for name, hpwl in [("fft_1", 4.2e8), ("fft_2", 3.8e8), ("des_perf_1", 2.3e9)]:
            results[name] = PlacementResult(
                {"hpwl": hpwl, "normalized_hpwl": 1.0,
                 "mean_overflow": 0.06, "tns_proxy": hpwl * 0.02},
                runtime_s=60.0, divergence_events=0, converged=True,
            )
        return results

    def test_bar_chart_created(self, tmpdir):
        results = self.make_stub_results()
        out = tmpdir / "bar.png"
        plot_benchmark_bars(
            {"DREAMPlace4.0": results},
            output_path=out,
            title="Test Bar Chart",
            metric="hpwl",
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_comparison_bar_chart(self, tmpdir):
        results = self.make_stub_results()
        # Simulate an improved method
        improved = {}
        for k, v in results.items():
            r = PlacementResult(
                {"hpwl": v.metrics["hpwl"] * 0.95, "normalized_hpwl": 0.95,
                 "mean_overflow": 0.05, "tns_proxy": v.metrics["tns_proxy"] * 0.9},
                runtime_s=55.0, divergence_events=0, converged=True,
            )
            improved[k] = r
        out = tmpdir / "comparison.png"
        plot_benchmark_bars(
            {"DREAMPlace4.0": results, "EvoPlace": improved},
            output_path=out, title="Comparison", metric="hpwl",
        )
        assert out.exists()

    def test_convergence_plot(self, tmpdir):
        iters = 500
        hpwl_curve = [4e8 * (1 + 0.5 * (1 - t/iters)) for t in range(iters)]
        ovfl_curve = [max(0.05, 1.0 - 0.95 * t/iters) for t in range(iters)]
        out = tmpdir / "convergence.png"
        plot_convergence(hpwl_curve, ovfl_curve, out, title="Test Convergence")
        assert out.exists()

    def test_convergence_with_baseline_overlay(self, tmpdir):
        iters = 300
        curve = [4e8 * (1 - 0.5 * t/iters) for t in range(iters)]
        ovfl = [max(0.05, 1 - t/iters) for t in range(iters)]
        out = tmpdir / "conv_overlay.png"
        plot_convergence(curve, ovfl, out, title="With Overlay",
                         baseline_hpwl=curve, baseline_overflow=ovfl)
        assert out.exists()

    def test_pareto_plot(self, tmpdir):
        results = [
            {"normalized_hpwl": 0.95 + 0.1 * i, "runtime_s": 30 + 10 * i}
            for i in range(20)
        ]
        out = tmpdir / "pareto.png"
        plot_pareto_frontier(results, out)
        assert out.exists()

    def test_schedule_trajectory_plot(self, tmpdir):
        out = tmpdir / "schedule.png"
        plot_schedule_trajectory(gamma_baseline_linear, output_path=out)
        assert out.exists()

    def test_all_schedules_plot(self, tmpdir):
        from dreamplace_ext.schedulers import (
            gamma_baseline_linear, gamma_exponential,
            gamma_overflow_adaptive, gamma_cosine_annealing,
        )
        for name, fn in [
            ("linear", gamma_baseline_linear),
            ("exponential", gamma_exponential),
            ("overflow_adaptive", gamma_overflow_adaptive),
            ("cosine", gamma_cosine_annealing),
        ]:
            out = tmpdir / f"sched_{name}.png"
            plot_schedule_trajectory(fn, output_path=out, title=f"γ: {name}")
            assert out.exists()
