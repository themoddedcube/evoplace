"""
Fixed placement evaluation harness.

This file is NEVER MODIFIED during experiments. It provides a stable,
reproducible interface to run DREAMPlace on a benchmark and return metrics.

Usage:
    python evaluator/run_placement.py --benchmark benchmarks/ispd2015/fft_1
    python evaluator/run_placement.py --benchmark benchmarks/ispd2015/fft_1 \\
        --gamma-schedule evaluator_wrapper.gamma_fn \\
        --output experiments/exp01_wl_smoothing/run_001/
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DreamPlace Python API wrapper
# ---------------------------------------------------------------------------

# Use the no-space symlink path (~/evoplace) if available — DreamPlace's C++ place_io
# parser splits file paths on whitespace, so paths with spaces cause assertion failures.
_SYMLINK = Path.home() / "evoplace"
_EFFECTIVE_ROOT = _SYMLINK if _SYMLINK.exists() else PROJECT_ROOT
DREAMPLACE_ROOT = _EFFECTIVE_ROOT / "vendor" / "dreamplace"
DREAMPLACE_INSTALL = DREAMPLACE_ROOT / "install"

_dreamplace_loaded = False


def load_dreamplace():
    """Import DreamPlace modules from the CMake install tree.

    Prefers vendor/dreamplace/install (Python sources + compiled ops); the
    bare source tree has no compiled ops and is only a fallback for
    stub-development machines. DREAMPlace mixes flat intra-package imports
    (import Params) with package imports (import dreamplace.ops...), so both
    the install root and the package dir must be on sys.path — the same
    layout Placer.py runs with.
    """
    global _dreamplace_loaded
    if not DREAMPLACE_ROOT.exists():
        raise RuntimeError(
            f"DreamPlace not found at {DREAMPLACE_ROOT}. "
            "Run: git submodule update --init vendor/dreamplace"
        )
    install_dir = (DREAMPLACE_INSTALL
                   if (DREAMPLACE_INSTALL / "dreamplace").exists()
                   else DREAMPLACE_ROOT)
    for p in [str(install_dir), str(install_dir / "dreamplace")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import Params
        import PlaceDB
        import NonLinearPlace
        _dreamplace_loaded = True
        return Params, PlaceDB, NonLinearPlace
    except ImportError as e:
        install_cmd = (
            "mkdir -p vendor/dreamplace/build && "
            "cd vendor/dreamplace/build && "
            "cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_CXX_ABI=1 "
            "-DPython_EXECUTABLE=$(which python3) && "
            "make -j$(nproc) && make install"
        )
        raise RuntimeError(
            f"Failed to import DreamPlace: {e}. "
            f"See README 'Manual Build', or:\n  {install_cmd}"
        ) from e


def build_dreamplace_params(
    Params,
    benchmark_dir: Path,
    output_dir: Path,
    max_iterations: int = 2000,
    seed: int = 42,
):
    """
    Build the DreamPlace Params object from the benchmark's JSON config.

    Schedule/init/timing callables are NOT passed here — they are injected
    via the dreamplace_ext hooks module, which the patched PlaceObj.py
    consults each iteration.
    """
    # Find the .json config file in the benchmark dir
    json_files = sorted(benchmark_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json config found in {benchmark_dir}")
    base_config_path = json_files[0]

    params = Params.Params()
    params.load(str(base_config_path))

    # Resolve config-relative input paths against the project root (via the
    # no-space symlink when available) so runs work regardless of cwd.
    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else str(_EFFECTIVE_ROOT / p)

    for key in ("def_input", "verilog_input", "aux_input", "lib_input",
                "sdc_input", "early_lib_input", "late_lib_input"):
        val = getattr(params, key, None)
        if isinstance(val, str) and val:
            setattr(params, key, _resolve(val))
    if getattr(params, "lef_input", None):
        params.lef_input = [_resolve(p) for p in params.lef_input]

    params.result_dir = str(output_dir)
    params.random_seed = seed
    # Override iteration count only; keep the config's bins / lr / optimizer /
    # gpu flag (the per-benchmark JSON decides CPU vs GPU).
    params.global_place_stages[0]["iteration"] = max_iterations

    return params


# ---------------------------------------------------------------------------
# Path-group timing weight injection (Exp 4, Variant A)
# ---------------------------------------------------------------------------

def _apply_path_group_weights(placedb, params,
                              activation_overflow: float = 0.3):
    """
    Classify nets by timing path group and register criticality weights for
    phased activation: weights are applied to data_collections.net_weights only
    once overflow drops below `activation_overflow` (default 0.3).

    Applying weights during early global spreading (overflow > 0.3) biases
    cell distribution before bins have room to spread, causing density hotspots
    near timing-critical macro clusters.  Phased activation avoids this.

    Silently no-ops if no Liberty/SDC inputs are configured (ISPD 2015
    benchmarks have no timing constraints).
    """
    sdc_path = getattr(params, "sdc_input", None) or None
    lib_paths = (getattr(params, "lib_input", None)
                 or getattr(params, "late_lib_input", None) or [])
    if isinstance(lib_paths, str):
        lib_paths = [lib_paths]

    if not sdc_path or not lib_paths:
        return

    try:
        from models.path_group_classifier import (
            PathGroupConfig, classify_nets, parse_liberty_cell_types,
        )
        from dreamplace_ext import hooks

        cell_type_map: Dict[str, str] = {}
        for lp in lib_paths:
            cell_type_map.update(parse_liberty_cell_types(lp))

        pg_data = classify_nets(placedb, cell_type_map, sdc_path, PathGroupConfig())
        hooks.set_path_group_data(pg_data)
        hooks.set_deferred_net_weights(pg_data.net_weights, threshold=activation_overflow)
        logger.info(
            f"Path-group weights registered (activate at overflow ≤ {activation_overflow}): "
            f"{pg_data.has_timing_constraints} timing nets"
        )

    except Exception as e:
        logger.warning(f"Path-group weight setup failed (falling back to plain WL): {e}")


# ---------------------------------------------------------------------------
# Cascade evaluation stages
# ---------------------------------------------------------------------------

class PlacementResult:
    """Holds the result of a placement run."""

    def __init__(self, metrics: Dict[str, float], runtime_s: float,
                 divergence_events: int, converged: bool, log_path: Optional[Path] = None):
        self.metrics = metrics
        self.runtime_s = runtime_s
        self.divergence_events = divergence_events
        self.converged = converged
        self.log_path = log_path
        self.stage = 0  # set by cascade evaluator

    @property
    def fitness(self) -> float:
        """Primary fitness score (lower = better)."""
        hpwl = self.metrics.get("hpwl", float("inf"))
        penalty = 1.0 + 5.0 * self.divergence_events
        return hpwl * penalty

    def to_dict(self) -> Dict:
        return {
            "metrics": self.metrics,
            "runtime_s": self.runtime_s,
            "divergence_events": self.divergence_events,
            "converged": self.converged,
            "stage": self.stage,
            "fitness": self.fitness,
        }

    def __repr__(self):
        m = self.metrics
        return (
            f"PlacementResult(hpwl={m.get('hpwl', '?'):.3e}, "
            f"overflow={m.get('mean_overflow', '?'):.4f}, "
            f"tns={m.get('tns_proxy', '?'):.3e}, "
            f"t={self.runtime_s:.1f}s, div={self.divergence_events})"
        )


def run_placement_stub(
    benchmark_dir: Path,
    output_dir: Path,
    gamma_schedule_fn: Optional[Callable] = None,
    lambda_schedule_fn: Optional[Callable] = None,
    init_positions_fn: Optional[Callable] = None,
    timing_loss_fn: Optional[Callable] = None,
    max_iterations: int = 500,
    seed: int = 42,
) -> PlacementResult:
    """
    Stub that returns synthetic results when DreamPlace is not built yet.

    Replace this function with the real DreamPlace call once the build is ready.
    The stub generates realistic-looking metrics so the rest of the framework
    can be developed and tested without hardware.
    """
    rng = np.random.default_rng(seed + hash(str(benchmark_dir)) % 10000)

    # Synthetic baseline HPWL (varies by benchmark). fft_1/fft_2 match the
    # real measured GPU baselines so stub results stay on the same scale as
    # the cascade baselines in evolve/evaluator_wrapper.py; the rest are
    # rough guesses pending measurement.
    benchmark_name = benchmark_dir.name
    hpwl_baselines = {
        "fft_1": 2.180e6, "fft_2": 1.921e6, "fft_a": 2.8e6, "fft_b": 2.7e6,
        "des_perf_1": 1.2e7, "matrix_mult_1": 9.5e6, "matrix_mult_2": 8.5e6,
        "matrix_mult_a": 1.1e7, "superblue12": 4.5e7, "superblue14": 3.8e7,
        "superblue19": 4.8e7,
    }
    base_hpwl = hpwl_baselines.get(benchmark_name, 5.0e6)

    # Simulate schedule effect: a custom gamma_schedule_fn can improve HPWL
    schedule_factor = 1.0
    if gamma_schedule_fn is not None:
        # Evaluate the schedule at a few key points; assess quality
        test_overflows = [0.9, 0.7, 0.5, 0.3, 0.1]
        gammas = [gamma_schedule_fn(int(i * max_iterations / 5), max_iterations,
                                    ovfl, []) for i, ovfl in enumerate(test_overflows)]
        # Good schedule: starts high, ends low; check monotone decrease
        if all(g2 <= g1 for g1, g2 in zip(gammas, gammas[1:])):
            schedule_factor *= 0.97  # 3% improvement for well-behaved schedule

    hpwl = base_hpwl * schedule_factor * (1 + rng.normal(0, 0.01))
    overflow = max(0.0, rng.normal(0.06, 0.01))
    tns = base_hpwl * 0.02 * (1 + rng.normal(0, 0.1))
    divergence_events = int(rng.poisson(1.5))
    runtime = rng.uniform(30, 90)

    metrics = {
        "hpwl": float(hpwl),
        "mean_overflow": float(overflow),
        "top5_overflow": float(overflow * 2.5),
        "tns_proxy": float(tns),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result = PlacementResult(
        metrics=metrics,
        runtime_s=float(runtime),
        divergence_events=divergence_events,
        converged=overflow < 0.1,
    )
    import json as _json
    with open(output_dir / "result.json", "w") as _f:
        _json.dump(result.to_dict(), _f, indent=2)
    return result


def run_placement(
    benchmark_dir: Path,
    output_dir: Path,
    gamma_schedule_fn: Optional[Callable] = None,
    lambda_schedule_fn: Optional[Callable] = None,
    init_positions_fn: Optional[Callable] = None,
    timing_loss_fn: Optional[Callable] = None,
    max_iterations: int = 2000,
    seed: int = 42,
    use_stub: bool = False,
) -> PlacementResult:
    """
    Run DreamPlace on a benchmark and return metrics.

    Set use_stub=True during development before DreamPlace is built.
    """
    benchmark_dir = Path(benchmark_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_stub:
        return run_placement_stub(
            benchmark_dir, output_dir, gamma_schedule_fn,
            lambda_schedule_fn, init_positions_fn, timing_loss_fn,
            max_iterations, seed
        )

    # Real DreamPlace path (requires built vendor/dreamplace)
    try:
        Params, PlaceDB, NonLinearPlace = load_dreamplace()
    except RuntimeError:
        logger.warning("DreamPlace not available; falling back to stub")
        return run_placement_stub(
            benchmark_dir, output_dir, gamma_schedule_fn,
            lambda_schedule_fn, init_positions_fn, timing_loss_fn,
            max_iterations, seed
        )

    # Inject custom hooks before running. The patched PlaceObj.py in the
    # DREAMPlace fork consults these each iteration.
    from dreamplace_ext import hooks
    hooks.set_gamma_schedule(gamma_schedule_fn)
    hooks.set_lambda_schedule(lambda_schedule_fn)
    hooks.set_init_positions(init_positions_fn)
    hooks.set_timing_loss(timing_loss_fn)

    params = build_dreamplace_params(
        Params, benchmark_dir, output_dir, max_iterations, seed
    )

    t0 = time.perf_counter()
    divergence = 0
    try:
        # GB10 / driver 580.142 / CUDA 13.0: building PlaceDB on GPU triggers
        # Xid 31 GPU MMU faults on large benchmarks (superblue15 confirmed)
        # and hard-hangs the box. Build the DB on CPU, then restore the JSON
        # config's gpu flag before constructing the placer. See
        # CRASH_DIAGNOSIS.md.
        wants_gpu = int(getattr(params, "gpu", 0))
        params.gpu = 0
        db = PlaceDB.PlaceDB()
        db(params)
        params.gpu = wants_gpu

        # Path-group timing weights (Exp 4 / Variant A).
        # Runs only when both Liberty and SDC inputs are present in params.
        # On ISPD 2015 (no SDC) this is a no-op; all net_weights remain 1.0.
        _apply_path_group_weights(db, params)

        placer = NonLinearPlace.NonLinearPlace(params, db, None)
        learning_rate = params.global_place_stages[0].get("learning_rate", 0.01)
        all_metrics = placer(params, db, learning_rate)
        runtime = time.perf_counter() - t0
        divergence = hooks.get_divergence_count()

        # all_metrics is a (possibly nested) list of EvalMetrics; the last
        # entry carrying an HPWL is the final state of the run.
        def _iter_metrics(m):
            if isinstance(m, (list, tuple)):
                for x in m:
                    yield from _iter_metrics(x)
            elif m is not None:
                yield m

        # hpwl and overflow live on different metric entries (the final
        # legalization metric has hpwl but no overflow), so track the last
        # non-None value of each independently.
        final = None
        overflow = None
        for m in _iter_metrics(all_metrics):
            if getattr(m, "hpwl", None) is not None:
                final = m
            m_ovfl = getattr(m, "goverflow", None)
            if m_ovfl is None:
                m_ovfl = getattr(m, "overflow", None)
            if m_ovfl is not None:
                overflow = m_ovfl
        if final is None:
            raise RuntimeError("DreamPlace returned no HPWL metrics")
        mean_overflow = (
            float(np.mean([float(v) for v in np.atleast_1d(
                overflow.cpu().numpy() if hasattr(overflow, "cpu") else overflow)]))
            if overflow is not None else 1.0
        )

        # TNS: real value only when a timing-driven run produced one (Exp 4
        # with ICCAD benchmarks); 0.0 otherwise — no synthetic proxy.
        tns_attr = getattr(final, "tns", None)
        metrics = {
            "hpwl": float(final.hpwl),
            "mean_overflow": mean_overflow,
            "top5_overflow": 0.0,   # not produced by DREAMPlace; kept for schema stability
            "tns_proxy": float(tns_attr) if tns_attr is not None else 0.0,
        }
        converged = mean_overflow <= float(getattr(params, "stop_overflow", 0.07)) + 1e-3
        result = PlacementResult(metrics, runtime, divergence, converged=converged)
        with open(output_dir / "result.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        return result

    except Exception as e:
        runtime = time.perf_counter() - t0
        logger.exception(f"DreamPlace failed: {e}")
        return PlacementResult(
            {"hpwl": float("inf"), "mean_overflow": 1.0,
             "top5_overflow": 1.0, "tns_proxy": float("inf")},
            runtime, divergence_events=divergence + 99, converged=False
        )
    finally:
        hooks.reset()


# ---------------------------------------------------------------------------
# Cascade evaluator
# ---------------------------------------------------------------------------

CASCADE_THRESHOLDS = [2.0, 1.3]  # Stage 1, Stage 2 — normalized to baseline
STAGE_ITERATIONS = [50, 300, 2000]


def run_cascade_evaluation(
    benchmark_dir: Path,
    output_dir: Path,
    baseline_hpwl,
    gamma_schedule_fn: Optional[Callable] = None,
    lambda_schedule_fn: Optional[Callable] = None,
    init_positions_fn: Optional[Callable] = None,
    timing_loss_fn: Optional[Callable] = None,
    use_stub: bool = False,
) -> Optional[PlacementResult]:
    """
    Three-stage cascade evaluation. Returns None if candidate is eliminated early.

    Stage 0 (50 iters):  Reject if normalized HPWL > thresholds[0].
    Stage 1 (300 iters): Reject if normalized HPWL > thresholds[1].
    Stage 2 (full):      Full evaluation with all metrics.

    baseline_hpwl may be a single float (used for every stage) or a sequence
    of one baseline per stage. Truncated runs land at a very different HPWL
    scale than converged ones (a 50-iter HPWL is ~10x the converged value),
    so stage-matched baselines are required for the early thresholds to mean
    "no worse than k times the default schedule at the same iteration budget".
    """
    if isinstance(baseline_hpwl, (int, float)):
        stage_baselines = [float(baseline_hpwl)] * len(STAGE_ITERATIONS)
    else:
        stage_baselines = [float(b) for b in baseline_hpwl]
        assert len(stage_baselines) == len(STAGE_ITERATIONS), (
            f"need one baseline per stage ({len(STAGE_ITERATIONS)}), "
            f"got {len(stage_baselines)}"
        )

    for stage, (n_iter, threshold) in enumerate(
        zip(STAGE_ITERATIONS, CASCADE_THRESHOLDS + [float("inf")])
    ):
        result = run_placement(
            benchmark_dir,
            output_dir / f"stage_{stage}",
            gamma_schedule_fn=gamma_schedule_fn,
            lambda_schedule_fn=lambda_schedule_fn,
            init_positions_fn=init_positions_fn,
            timing_loss_fn=timing_loss_fn,
            max_iterations=n_iter,
            use_stub=use_stub,
        )
        result.stage = stage

        if stage < 2:  # not final stage
            norm_hpwl = result.metrics["hpwl"] / stage_baselines[stage]
            if norm_hpwl > threshold:
                logger.info(f"Stage {stage} eliminated: norm_hpwl={norm_hpwl:.3f} > {threshold}")
                return None

        logger.info(f"Stage {stage} passed: {result}")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run DreamPlace on a benchmark")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark directory")
    parser.add_argument("--output", default="experiments/tmp/", help="Output directory")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--stub", action="store_true", help="Use stub (no DreamPlace build needed)")
    parser.add_argument("--baseline-hpwl", type=float, default=None,
                        help="Baseline HPWL for normalization")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    benchmark_dir = Path(args.benchmark)
    output_dir = Path(args.output)

    result = run_placement(
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        max_iterations=args.max_iterations,
        use_stub=args.stub,
    )

    print("\n=== Placement Result ===")
    print(f"  HPWL:          {result.metrics['hpwl']:.6e}")
    print(f"  Mean Overflow: {result.metrics['mean_overflow']:.4f}")
    print(f"  Top5 Overflow: {result.metrics['top5_overflow']:.4f}")
    print(f"  TNS Proxy:     {result.metrics['tns_proxy']:.6e}")
    print(f"  Runtime:       {result.runtime_s:.1f}s")
    print(f"  Divergences:   {result.divergence_events}")
    print(f"  Converged:     {result.converged}")
    if args.baseline_hpwl:
        print(f"  Norm. HPWL:    {result.metrics['hpwl'] / args.baseline_hpwl:.4f}")

    # Save result JSON
    out_json = output_dir / "result.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nResult saved to {out_json}")


if __name__ == "__main__":
    main()
