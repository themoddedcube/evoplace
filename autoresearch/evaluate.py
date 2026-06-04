"""
FIXED evaluation harness for the autoresearch loop.

This file is NEVER MODIFIED during autonomous experiments.
It provides a stable interface: given a train.py path, it evaluates
the placement config it defines and returns normalized HPWL.

Autoresearch loop pattern:
    evaluate.py (FIXED) ← this file
    train.py    (EDITABLE by AI agent)
    results.tsv (APPEND-ONLY log)
    program.md  (human research directions)
"""

import csv
import hashlib
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

RESULTS_TSV = PROJECT_ROOT / "autoresearch" / "results.tsv"
TRAIN_PY = PROJECT_ROOT / "autoresearch" / "train.py"
TIME_BUDGET_SECONDS = 600  # 10 minutes per experiment

# Updated after Exp 0 baseline run
BASELINE_HPWL_FFT1 = 4.2e8  # placeholder; update after Exp 0


def load_train_module():
    """Load autoresearch/train.py as a module."""
    spec = importlib.util.spec_from_file_location("train_module", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_evaluation(
    benchmark: str = "fft_1",
    use_stub: bool = True,
) -> Dict:
    """
    Load train.py, extract the placement config, run evaluation.

    Returns a dict with all metrics.
    """
    from evaluator.run_placement import run_placement

    # Load train.py to get the current config
    try:
        train = load_train_module()
    except Exception as e:
        logger.error(f"Failed to load train.py: {e}")
        return {"error": str(e), "normalized_hpwl": float("inf")}

    # train.py must define: gamma_schedule (optional), lambda_schedule (optional)
    gamma_fn = getattr(train, "gamma_schedule", None)
    lambda_fn = getattr(train, "lambda_schedule", None)
    init_fn = getattr(train, "init_positions", None)
    timing_fn = getattr(train, "timing_loss", None)

    bench_dir = PROJECT_ROOT / "benchmarks" / "ispd2015_no_region" / benchmark
    output_dir = PROJECT_ROOT / "autoresearch" / "runs" / f"{int(time.time())}"

    t0 = time.perf_counter()
    result = run_placement(
        benchmark_dir=bench_dir,
        output_dir=output_dir,
        gamma_schedule_fn=gamma_fn,
        lambda_schedule_fn=lambda_fn,
        init_positions_fn=init_fn,
        timing_loss_fn=timing_fn,
        max_iterations=500,
        use_stub=use_stub,
    )
    runtime = time.perf_counter() - t0

    norm_hpwl = result.metrics["hpwl"] / BASELINE_HPWL_FFT1

    return {
        "normalized_hpwl": norm_hpwl,
        "hpwl": result.metrics["hpwl"],
        "mean_overflow": result.metrics["mean_overflow"],
        "tns_proxy": result.metrics["tns_proxy"],
        "runtime_s": runtime,
        "divergence_events": result.divergence_events,
        "converged": result.converged,
    }


def log_result(metrics: Dict, code_hash: str):
    """Append result to results.tsv."""
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)

    # Write header if file doesn't exist
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "timestamp", "code_hash", "normalized_hpwl", "hpwl",
                "mean_overflow", "tns_proxy", "runtime_s", "divergence_events", "converged"
            ])

    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            int(time.time()),
            code_hash,
            f"{metrics.get('normalized_hpwl', 'N/A'):.6f}",
            f"{metrics.get('hpwl', 'N/A'):.6e}",
            f"{metrics.get('mean_overflow', 'N/A'):.4f}",
            f"{metrics.get('tns_proxy', 'N/A'):.6e}",
            f"{metrics.get('runtime_s', 'N/A'):.1f}",
            metrics.get("divergence_events", "N/A"),
            metrics.get("converged", "N/A"),
        ])


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with open(TRAIN_PY) as f:
        code = f.read()
    code_hash = hashlib.sha256(code.encode()).hexdigest()[:8]

    use_stub = not (PROJECT_ROOT / "vendor" / "dreamplace" / "dreamplace").exists()
    if use_stub:
        logger.info("DreamPlace not built; using stub evaluator")

    logger.info(f"Evaluating train.py (hash={code_hash})...")
    metrics = run_evaluation(use_stub=use_stub)

    if "error" in metrics:
        logger.error(f"Evaluation failed: {metrics['error']}")
        log_result({"normalized_hpwl": float("inf")}, code_hash)
        print(f"val_norm_hpwl: inf")
        return

    log_result(metrics, code_hash)

    # Print in autoresearch-compatible format (key: value)
    print(f"val_norm_hpwl:     {metrics['normalized_hpwl']:.6f}")
    print(f"hpwl:              {metrics['hpwl']:.6e}")
    print(f"mean_overflow:     {metrics['mean_overflow']:.4f}")
    print(f"tns_proxy:         {metrics['tns_proxy']:.6e}")
    print(f"runtime_seconds:   {metrics['runtime_s']:.1f}")
    print(f"divergence_events: {metrics['divergence_events']}")


if __name__ == "__main__":
    main()
