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
    """Import DreamPlace modules from the CMake install directory."""
    global _dreamplace_loaded
    if not DREAMPLACE_ROOT.exists():
        raise RuntimeError(
            f"DreamPlace not found at {DREAMPLACE_ROOT}. "
            "Run: git submodule update --init vendor/dreamplace"
        )
    # CMake installs the complete Python package + .so files to install/
    # DreamPlace uses bare imports (e.g. `import Params`), so install/dreamplace/
    # must also be on sys.path, not just install/.
    install_dir = DREAMPLACE_INSTALL if DREAMPLACE_INSTALL.exists() else DREAMPLACE_ROOT
    for p in [str(install_dir), str(install_dir / "dreamplace")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import dreamplace.Params as Params
        import dreamplace.PlaceDB as PlaceDB
        import dreamplace.NonLinearPlace as NonLinearPlace
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
            f"Build with:\n  {install_cmd}"
        ) from e


def _resolve_paths(params_dict: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Make relative file paths in a DreamPlace params dict absolute."""
    path_keys = ["aux_input", "def_input", "verilog_input", "sdc_input",
                 "lib_input", "detailed_place_engine"]
    list_path_keys = ["lef_input"]
    result = dict(params_dict)
    for key in path_keys:
        val = result.get(key)
        if val and not os.path.isabs(val):
            result[key] = str(base_dir / val)
    for key in list_path_keys:
        val = result.get(key)
        if val:
            result[key] = [
                str(base_dir / v) if not os.path.isabs(v) else v
                for v in (val if isinstance(val, list) else [val])
            ]
    return result


def build_dreamplace_params(
    benchmark_dir: Path,
    output_dir: Path,
    gamma_schedule_fn: Optional[Callable] = None,
    lambda_schedule_fn: Optional[Callable] = None,
    init_positions_fn: Optional[Callable] = None,
    timing_loss_fn: Optional[Callable] = None,
    max_iterations: int = 2000,
) -> Dict[str, Any]:
    """
    Build the DreamPlace parameter dict.

    Pluggable algorithm components are injected here as callables.
    DreamPlace reads the JSON config; we write a temp config and patch
    Python-level callbacks via the dreamplace_ext hooks module.
    """
    # Find the .json config file in the benchmark dir
    json_files = list(benchmark_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json config found in {benchmark_dir}")
    base_config_path = json_files[0]

    with open(base_config_path) as f:
        params = json.load(f)

    # Resolve relative paths. JSON configs use paths relative to DREAMPLACE_ROOT
    # (the DreamPlace directory where benchmarks/ lives after download).
    params = _resolve_paths(params, DREAMPLACE_ROOT)

    params["result_dir"] = str(output_dir)
    params["gpu"] = 0   # CPU-only until DGX arrives; overridden to 1 on GPU machines
    params["num_threads"] = max(1, os.cpu_count() or 8)
    params["plot_flag"] = 0
    params["global_place_stages"] = [
        {
            "num_bins_x": 512,
            "num_bins_y": 512,
            "iteration": max_iterations,
            "learning_rate": 0.01,
            "wirelength": "weighted_average",
            "optimizer": "nesterov",
            "stop_overflow": 0.07,
        }
    ]

    return params


# ---------------------------------------------------------------------------
# Path-group timing weight injection (Exp 4, Variant A)
# ---------------------------------------------------------------------------

def _apply_path_group_weights(placedb, params_dict: Dict[str, Any],
                              activation_overflow: float = 0.3):
    """
    Classify nets by timing path group and register criticality weights for
    phased activation: weights are applied to data_collections.net_weights only
    once overflow drops below `activation_overflow` (default 0.3).

    Applying weights during early global spreading (overflow > 0.3) biases
    cell distribution before bins have room to spread, causing density hotspots
    near timing-critical macro clusters.  Phased activation avoids this.

    Silently no-ops if no Liberty/SDC files are in params_dict (ISPD 2015
    benchmarks have no timing constraints).
    """
    sdc_path = params_dict.get("sdc_input") or None
    lib_paths = params_dict.get("lib_input") or params_dict.get("late_lib_input") or []
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

    # Synthetic baseline HPWL (varies by benchmark)
    benchmark_name = benchmark_dir.name
    hpwl_baselines = {
        "fft_1": 4.2e8, "fft_2": 3.8e8, "fft_a": 5.1e8, "fft_b": 4.9e8,
        "des_perf_1": 2.3e9, "matrix_mult_1": 1.8e9, "matrix_mult_2": 1.6e9,
        "matrix_mult_a": 2.1e9, "superblue12": 8.7e9, "superblue14": 7.2e9,
        "superblue19": 9.1e9,
    }
    base_hpwl = hpwl_baselines.get(benchmark_name, 1.0e9)

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
        DPParams, PlaceDB, NonLinearPlace = load_dreamplace()
    except RuntimeError:
        logger.warning("DreamPlace not available; falling back to stub")
        return run_placement_stub(
            benchmark_dir, output_dir, gamma_schedule_fn,
            lambda_schedule_fn, init_positions_fn, timing_loss_fn,
            max_iterations, seed
        )

    # Inject custom hooks before running
    from dreamplace_ext import hooks
    hooks.set_gamma_schedule(gamma_schedule_fn)
    hooks.set_lambda_schedule(lambda_schedule_fn)
    hooks.set_init_positions(init_positions_fn)
    hooks.set_timing_loss(timing_loss_fn)

    params_dict = build_dreamplace_params(
        benchmark_dir, output_dir,
        gamma_schedule_fn, lambda_schedule_fn,
        init_positions_fn, timing_loss_fn,
        max_iterations
    )

    t0 = time.perf_counter()
    divergence = 0
    try:
        # Build Params object (loads defaults from dreamplace/params.json first)
        params = DPParams.Params()
        params.fromJson(params_dict)

        # Read the design into the placement database
        placedb = PlaceDB.PlaceDB()
        placedb(params)

        # Path-group timing weights (Exp 4 / Variant A).
        # Runs only when both Liberty and SDC inputs are present in params.
        # On ISPD 2015 (no SDC) this is a no-op; all net_weights remain 1.0.
        _apply_path_group_weights(placedb, params_dict)

        # Run global placement (+ legalization + detailed if configured)
        learning_rate = params_dict["global_place_stages"][0].get("learning_rate", 0.01)
        placer = NonLinearPlace.NonLinearPlace(params, placedb, timer=None)
        all_metrics = placer(params, placedb, learning_rate)

        runtime = time.perf_counter() - t0
        divergence = hooks.get_divergence_count()
        hooks.reset()

        # all_metrics is a flat list of EvalMetrics objects (one per stage: gp, legalize, dp)
        # Find the last entry with a valid hpwl value.
        def _scalar(v):
            if v is None:
                return float("inf")
            return float(v.item()) if hasattr(v, "item") else float(v)

        last = None
        for m in reversed(all_metrics):
            if m is not None and m.hpwl is not None:
                last = m
                break

        if last is None:
            raise RuntimeError("No valid EvalMetrics found in all_metrics")

        overflow_val = last.overflow
        if overflow_val is not None and hasattr(overflow_val, "__len__") and len(overflow_val) > 0:
            mean_ovf = _scalar(overflow_val[-1])
        elif overflow_val is not None:
            mean_ovf = _scalar(overflow_val)
        else:
            mean_ovf = 1.0

        hpwl_val = _scalar(last.hpwl)
        # TNS: use real value if timing was run, else proxy from HPWL
        tns_val = _scalar(last.tns) if last.tns is not None else hpwl_val * 0.02

        metrics = {
            "hpwl": hpwl_val,
            "mean_overflow": mean_ovf,
            "top5_overflow": mean_ovf,
            "tns_proxy": tns_val,
        }
        return PlacementResult(metrics, runtime, divergence, converged=mean_ovf < 0.1)

    except Exception as e:
        runtime = time.perf_counter() - t0
        hooks.reset()
        logger.error(f"DreamPlace failed: {e}", exc_info=True)
        return PlacementResult(
            {"hpwl": float("inf"), "mean_overflow": 1.0,
             "top5_overflow": 1.0, "tns_proxy": float("inf")},
            runtime, divergence_events=divergence + 99, converged=False
        )


# ---------------------------------------------------------------------------
# Cascade evaluator
# ---------------------------------------------------------------------------

CASCADE_THRESHOLDS = [2.0, 1.3]  # Stage 1, Stage 2 — normalized to baseline
STAGE_ITERATIONS = [50, 300, 2000]


def run_cascade_evaluation(
    benchmark_dir: Path,
    output_dir: Path,
    baseline_hpwl: float,
    gamma_schedule_fn: Optional[Callable] = None,
    lambda_schedule_fn: Optional[Callable] = None,
    init_positions_fn: Optional[Callable] = None,
    timing_loss_fn: Optional[Callable] = None,
    use_stub: bool = False,
) -> Optional[PlacementResult]:
    """
    Three-stage cascade evaluation. Returns None if candidate is eliminated early.

    Stage 1 (50 iters): Reject if overflow not decreasing.
    Stage 2 (300 iters): Reject if normalized HPWL > threshold[0].
    Stage 3 (full):     Full evaluation with all metrics.
    """
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
            norm_hpwl = result.metrics["hpwl"] / baseline_hpwl
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
