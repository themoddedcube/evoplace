"""
Batch benchmark evaluation across ISPD 2015 and ICCAD 2015 suites.

Runs all benchmarks and returns aggregate metrics for comparison tables.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np

from evaluator.run_placement import run_placement, PlacementResult

logger = logging.getLogger(__name__)

ISPD2015_BENCHMARKS = [
    "mgc_des_perf_a", "mgc_des_perf_b", "mgc_edit_dist_a",
    "mgc_matrix_mult_b", "mgc_matrix_mult_c", "mgc_pci_bridge32_a",
    "mgc_pci_bridge32_b", "mgc_superblue11_a", "mgc_superblue16_a",
]

ISPD2015_NO_REGION = [
    "des_perf_1", "fft_1", "fft_2", "fft_a", "fft_b",
    "matrix_mult_1", "matrix_mult_2", "matrix_mult_a",
    "superblue12", "superblue14", "superblue19",
]

ICCAD2015_BENCHMARKS = [
    "des_perf_1", "des_perf_b", "edit_dist_1", "fft_2",
    "pci_bridge32_a", "superblue12",
]

SMALL_BENCHMARKS = ["fft_1", "fft_2"]  # For fast iteration during evolution


def run_suite(
    benchmark_dir: Path,
    output_dir: Path,
    suite: str = "ispd2015_no_region",
    gamma_schedule_fn: Optional[Callable] = None,
    lambda_schedule_fn: Optional[Callable] = None,
    init_positions_fn: Optional[Callable] = None,
    timing_loss_fn: Optional[Callable] = None,
    max_iterations: int = 2000,
    use_stub: bool = False,
    baselines: Optional[Dict[str, float]] = None,
) -> Dict[str, PlacementResult]:
    """
    Run all benchmarks in a suite and return per-benchmark results.

    Args:
        benchmark_dir: Root directory containing benchmark subdirectories
        output_dir: Directory to write per-benchmark results
        suite: One of 'ispd2015', 'ispd2015_no_region', 'iccad2015', 'small'
        baselines: Dict mapping benchmark_name → baseline HPWL for normalization

    Returns:
        Dict mapping benchmark_name → PlacementResult
    """
    suites = {
        "ispd2015": ISPD2015_BENCHMARKS,
        "ispd2015_no_region": ISPD2015_NO_REGION,
        "iccad2015": ICCAD2015_BENCHMARKS,
        "small": SMALL_BENCHMARKS,
    }
    if suite not in suites:
        raise ValueError(f"Unknown suite: {suite}. Choose from {list(suites)}")

    benchmarks = suites[suite]
    results = {}

    for name in benchmarks:
        bench_path = benchmark_dir / name
        if not bench_path.exists():
            if use_stub:
                # Create empty dir so stub can generate synthetic results
                bench_path.mkdir(parents=True, exist_ok=True)
            else:
                logger.warning(f"Benchmark not found: {bench_path} — skipping")
                continue

        logger.info(f"Running {name}...")
        result = run_placement(
            benchmark_dir=bench_path,
            output_dir=output_dir / name,
            gamma_schedule_fn=gamma_schedule_fn,
            lambda_schedule_fn=lambda_schedule_fn,
            init_positions_fn=init_positions_fn,
            timing_loss_fn=timing_loss_fn,
            max_iterations=max_iterations,
            use_stub=use_stub,
        )

        if baselines and name in baselines:
            result.metrics["normalized_hpwl"] = result.metrics["hpwl"] / baselines[name]

        results[name] = result
        logger.info(f"  {name}: {result}")

    return results


def aggregate_metrics(results: Dict[str, PlacementResult]) -> Dict[str, float]:
    """Compute mean and geomean normalized HPWL and other metrics across suite."""
    hpwls = [r.metrics.get("normalized_hpwl", r.metrics["hpwl"]) for r in results.values()]
    overflows = [r.metrics["mean_overflow"] for r in results.values()]
    tnss = [r.metrics["tns_proxy"] for r in results.values()]
    runtimes = [r.runtime_s for r in results.values()]
    divergences = [r.divergence_events for r in results.values()]

    geomean_hpwl = float(np.exp(np.mean(np.log([max(h, 1e-9) for h in hpwls]))))

    return {
        "mean_normalized_hpwl": float(np.mean(hpwls)),
        "geomean_normalized_hpwl": geomean_hpwl,
        "mean_overflow": float(np.mean(overflows)),
        "mean_tns_proxy": float(np.mean(tnss)),
        "mean_runtime_s": float(np.mean(runtimes)),
        "total_divergence_events": int(np.sum(divergences)),
        "num_benchmarks": len(results),
    }


def save_comparison_table(
    results_by_method: Dict[str, Dict[str, PlacementResult]],
    output_path: Path,
):
    """
    Save a comparison table of multiple methods across all benchmarks.

    results_by_method: {"DREAMPlace4.0": {bench: result}, "EvoPlace": {bench: result}}
    """
    import csv
    all_benches = sorted(set(b for r in results_by_method.values() for b in r))
    methods = list(results_by_method.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["benchmark"] + [f"{m}_hpwl" for m in methods] + [f"{m}_norm_hpwl" for m in methods]
        writer.writerow(header)
        for bench in all_benches:
            row = [bench]
            for m in methods:
                r = results_by_method[m].get(bench)
                row.append(f"{r.metrics['hpwl']:.4e}" if r else "N/A")
            for m in methods:
                r = results_by_method[m].get(bench)
                row.append(f"{r.metrics.get('normalized_hpwl', 'N/A'):.4f}" if r else "N/A")
            writer.writerow(row)

    logger.info(f"Comparison table saved to {output_path}")
