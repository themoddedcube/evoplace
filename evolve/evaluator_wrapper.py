"""
OpenEvolve evaluator wrapper for placement experiments.

OpenEvolve calls evaluate(program_path) and expects a dict of metrics.
This wrapper loads the evolved program, runs the placement cascade, and
returns metrics in the format OpenEvolve expects.

FIXED: This file is not modified during evolution.
"""

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Benchmark to use for evolution fitness (small = fast iteration)
EVOLUTION_BENCHMARK = "fft_1"
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks" / "ispd2015_no_region"

# DREAMPlace baseline HPWL on fft_1 (set after Exp 0 baseline run)
# Update this after running experiments/exp00_baseline/run.py
BASELINE_HPWL = {
    "fft_1": 4.2e8,
    "fft_2": 3.8e8,
    "des_perf_1": 2.3e9,
}


def load_evolved_function(program_path: str, function_name: str):
    """Dynamically load the evolved function from a file path."""
    spec = importlib.util.spec_from_file_location("evolved_module", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise ValueError(f"{function_name} is not callable in {program_path}")
    return fn


def evaluate(program_path: str, experiment: str = "exp01_wl_smoothing") -> Dict[str, Any]:
    """
    OpenEvolve evaluator entry point.

    Called with the path to the evolved program file.
    Returns a dict with 'score' (primary metric) and 'metrics' (all metrics).

    The 'score' is negated HPWL (OpenEvolve maximizes score).
    """
    from evaluator.run_placement import run_cascade_evaluation, PlacementResult

    experiment_configs = {
        "exp01_wl_smoothing": {
            "function_name": "gamma_schedule",
            "hook_arg": "gamma_schedule_fn",
        },
        "exp02_density_schedule": {
            "function_name": "lambda_schedule",
            "hook_arg": "lambda_schedule_fn",
        },
    }

    config = experiment_configs.get(experiment, experiment_configs["exp01_wl_smoothing"])
    function_name = config["function_name"]
    hook_arg = config["hook_arg"]

    # Load the evolved function
    try:
        evolved_fn = load_evolved_function(program_path, function_name)
    except Exception as e:
        logger.error(f"Failed to load evolved function: {e}")
        return {"score": -float("inf"), "metrics": {"error": str(e)}}

    # Quick sanity check: call the function with dummy args
    try:
        if function_name == "gamma_schedule":
            val = evolved_fn(0, 1000, 0.9, [])
            if not (0.01 <= val <= 20.0):
                return {"score": -float("inf"), "metrics": {"error": f"gamma out of range: {val}"}}
        elif function_name == "lambda_schedule":
            val = evolved_fn(0, 0.9, [], 0.1, 1.0)
            if not (0.0 < val < 1e8):
                return {"score": -float("inf"), "metrics": {"error": f"lambda out of range: {val}"}}
    except Exception as e:
        return {"score": -float("inf"), "metrics": {"error": f"Sanity check failed: {e}"}}

    # Run cascade evaluation
    bench_dir = BENCHMARK_ROOT / EVOLUTION_BENCHMARK
    output_dir = PROJECT_ROOT / "experiments" / experiment / "evolution_runs" / "tmp"
    baseline_hpwl = BASELINE_HPWL.get(EVOLUTION_BENCHMARK, 1.0)

    kwargs = {hook_arg: evolved_fn}
    use_stub = not (PROJECT_ROOT / "vendor" / "dreamplace" / "dreamplace").exists()

    result = run_cascade_evaluation(
        benchmark_dir=bench_dir,
        output_dir=output_dir,
        baseline_hpwl=baseline_hpwl,
        use_stub=use_stub,
        **kwargs,
    )

    if result is None:
        # Eliminated by cascade
        return {
            "score": -float("inf"),
            "metrics": {"eliminated": True, "stage": "cascade"},
            "artifacts": {"status": "eliminated_by_cascade"},
        }

    norm_hpwl = result.metrics["hpwl"] / baseline_hpwl
    score = -norm_hpwl  # OpenEvolve maximizes score; lower HPWL = higher score

    return {
        "score": score,
        "metrics": {
            "normalized_hpwl": norm_hpwl,
            "hpwl": result.metrics["hpwl"],
            "overflow": result.metrics["mean_overflow"],
            "tns_proxy": result.metrics["tns_proxy"],
            "runtime_s": result.runtime_s,
            "divergence_events": result.divergence_events,
        },
        "artifacts": {
            "stage_reached": result.stage,
            "converged": result.converged,
        },
    }


if __name__ == "__main__":
    # Test the evaluator with the baseline program
    import sys
    prog = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "evolve" / "initial_program.py")
    result = evaluate(prog)
    print(json.dumps(result, indent=2))
