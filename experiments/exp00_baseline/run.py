"""
Experiment 0: Baseline DREAMPlace 4.0 Reproduction

Runs DREAMPlace with default settings on ISPD 2015 (with and without region)
and ICCAD 2015. Records HPWL, overflow, TNS proxy, and runtime.
These numbers are the denominator for all future normalized comparisons.

Expected runtime: 4–8 GPU-hours for full suite.
Use --stub flag for a quick sanity check without hardware.

Usage:
    python experiments/exp00_baseline/run.py --stub
    python experiments/exp00_baseline/run.py  # real run on GPU
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.benchmark_suite import run_suite, aggregate_metrics, save_comparison_table
from graphs.plot_utils import plot_benchmark_bars, plot_comparison_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "results"
GRAPHS_DIR = PROJECT_ROOT / "graphs" / "exp00_baseline"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true", help="Use stub (no GPU needed)")
    parser.add_argument("--suite", default="ispd2015_no_region",
                        choices=["ispd2015", "ispd2015_no_region", "iccad2015", "small"])
    args = parser.parse_args()

    benchmark_dir = PROJECT_ROOT / "benchmarks"
    output_dir = OUTPUT_DIR / args.suite
    output_dir.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running Exp 0 baseline on suite: {args.suite}")
    logger.info(f"Using stub: {args.stub}")

    results = run_suite(
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        suite=args.suite,
        use_stub=args.stub,
    )

    if not results:
        logger.error("No benchmarks ran. Check that benchmark_dir contains the right files.")
        logger.info(f"Expected: {benchmark_dir / args.suite}")
        if args.stub:
            logger.info("Stub mode creates synthetic results even without benchmark files.")
        return

    # Save per-benchmark results
    baselines = {}
    all_metrics = {}
    for name, result in results.items():
        baselines[name] = result.metrics["hpwl"]
        all_metrics[name] = result.to_dict()
        logger.info(f"  {name}: HPWL={result.metrics['hpwl']:.4e}, "
                    f"OVF={result.metrics['mean_overflow']:.4f}, "
                    f"TNS={result.metrics['tns_proxy']:.4e}, "
                    f"t={result.runtime_s:.1f}s")

    # Save baselines JSON (used to normalize all future experiments)
    baselines_file = OUTPUT_DIR / f"baselines_{args.suite}.json"
    with open(baselines_file, "w") as f:
        json.dump(baselines, f, indent=2)
    logger.info(f"Baselines saved to {baselines_file}")

    # Save all metrics
    metrics_file = output_dir / "all_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Aggregate
    agg = aggregate_metrics(results)
    logger.info(f"\nAggregate metrics:")
    for k, v in agg.items():
        logger.info(f"  {k}: {v}")

    agg_file = output_dir / "aggregate.json"
    with open(agg_file, "w") as f:
        json.dump(agg, f, indent=2)

    # Generate graphs
    plot_benchmark_bars(
        results={"DREAMPlace4.0": results},
        output_path=GRAPHS_DIR / f"hpwl_baseline_{args.suite}.png",
        title=f"DREAMPlace 4.0 Baseline — {args.suite}",
        metric="hpwl",
        normalize=False,
    )
    logger.info(f"Graphs saved to {GRAPHS_DIR}")

    # Update autoresearch/evaluate.py baseline
    if args.suite == "ispd2015_no_region" and "fft_1" in baselines:
        baseline_fft1 = baselines["fft_1"]
        logger.info(f"\nUpdate BASELINE_HPWL_FFT1 in autoresearch/evaluate.py to: {baseline_fft1:.6e}")
        logger.info("Update BASELINE_HPWL in evolve/evaluator_wrapper.py as well.")


if __name__ == "__main__":
    main()
