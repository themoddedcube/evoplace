"""
Graph generation utilities for EvoPlace experiments.

Auto-generates publication-quality plots after each experiment run.
All plots saved as both PNG (for viewing) and PDF (for papers).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def _get_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        return plt, mpatches
    except ImportError:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")


def plot_benchmark_bars(
    results: Dict[str, Any],
    output_path: Path,
    title: str,
    metric: str = "normalized_hpwl",
    normalize: bool = True,
    figsize=(12, 5),
):
    """
    Bar chart: one group of bars per benchmark, one bar per method.

    results: {"method_name": {bench_name: PlacementResult or dict}}
    """
    plt, mpatches = _get_matplotlib()
    import numpy as np

    methods = list(results.keys())
    all_benches = sorted(set(b for r in results.values() for b in r))

    n_benches = len(all_benches)
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    x = np.arange(n_benches)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods):
        vals = []
        for bench in all_benches:
            r = results[method].get(bench)
            if r is None:
                vals.append(float("nan"))
                continue
            if hasattr(r, "metrics"):
                val = r.metrics.get(metric, r.metrics.get("hpwl", float("nan")))
            elif isinstance(r, dict):
                m = r.get("metrics", r)
                val = m.get(metric, m.get("hpwl", float("nan")))
            else:
                val = float("nan")
            vals.append(val)

        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=method, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_benches, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Normalized HPWL" if normalize else "HPWL")
    ax.set_title(title)
    ax.legend()
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved bar chart: {output_path}")


def plot_convergence(
    iteration_hpwl: List[float],
    iteration_overflow: List[float],
    output_path: Path,
    title: str,
    baseline_hpwl: Optional[List[float]] = None,
    baseline_overflow: Optional[List[float]] = None,
    figsize=(10, 4),
):
    """
    Dual-axis convergence plot: HPWL + overflow vs. iteration.
    Optionally overlay baseline curves.
    """
    plt, _ = _get_matplotlib()
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    iters = list(range(len(iteration_hpwl)))

    # HPWL plot
    ax1.plot(iters, iteration_hpwl, "b-", linewidth=1.5, label="EvoPlace")
    if baseline_hpwl:
        ax1.plot(iters[:len(baseline_hpwl)], baseline_hpwl, "gray", linewidth=1,
                 linestyle="--", label="DREAMPlace 4.0")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("HPWL (×10⁶)")
    ax1.set_title("Wirelength Convergence")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))

    # Overflow plot
    ax2.plot(iters, iteration_overflow, "r-", linewidth=1.5, label="EvoPlace")
    if baseline_overflow:
        ax2.plot(iters[:len(baseline_overflow)], baseline_overflow, "gray",
                 linewidth=1, linestyle="--", label="DREAMPlace 4.0")
    ax2.axhline(0.1, color="green", linestyle=":", linewidth=1, alpha=0.7, label="target (0.1)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Density Overflow")
    ax2.set_title("Overflow Convergence")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved convergence plot: {output_path}")


def plot_pareto_frontier(
    results: List[Dict],
    output_path: Path,
    title: str = "Quality vs. Runtime Pareto Frontier",
    x_metric: str = "normalized_hpwl",
    y_metric: str = "runtime_s",
    highlight_best: bool = True,
    figsize=(7, 5),
):
    """
    Scatter plot of (HPWL quality, runtime) for all evolved variants.
    Highlights Pareto-optimal points.
    """
    plt, mpatches = _get_matplotlib()
    import numpy as np

    xs = [r.get(x_metric, float("nan")) for r in results]
    ys = [r.get(y_metric, float("nan")) for r in results]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xs, ys, alpha=0.5, s=20, c="steelblue", label="All variants")

    # Find Pareto frontier (minimize both x and y)
    pareto = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if any(xs[j] <= x and ys[j] <= y and (xs[j] < x or ys[j] < y)
               for j in range(len(xs)) if i != j):
            continue
        pareto.append((x, y))

    if pareto:
        pareto.sort()
        px, py = zip(*pareto)
        ax.scatter(px, py, s=60, c="red", zorder=5, label="Pareto optimal")
        ax.step(px, py, "r-", alpha=0.5, where="post")

    ax.set_xlabel(x_metric.replace("_", " ").title())
    ax.set_ylabel(y_metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Pareto plot: {output_path}")


def plot_schedule_trajectory(
    schedule_fn,
    total_iterations: int = 1000,
    output_path: Path = None,
    title: str = "Schedule Trajectory",
    figsize=(8, 4),
):
    """
    Visualize how a schedule function behaves over iterations.
    Useful for understanding evolved gamma/lambda schedules.
    """
    plt, _ = _get_matplotlib()
    import numpy as np
    import inspect

    iters = list(range(total_iterations))
    sig = inspect.signature(schedule_fn)
    params = list(sig.parameters.keys())

    values = []
    for t in iters:
        # Call with dummy state
        try:
            if "overflow" in params:
                # Simulate decreasing overflow over time
                ovfl = max(0.05, 1.0 - t / total_iterations)
                val = schedule_fn(t, total_iterations, ovfl, [])
            else:
                val = schedule_fn(t, 0.5, [], 0.1, 1.0)
            values.append(float(val))
        except Exception:
            values.append(float("nan"))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iters, values, "b-", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Schedule Value")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()
        logger.info(f"Saved schedule trajectory: {output_path}")
    else:
        plt.show()


def plot_comparison_table(
    results_by_method: Dict[str, Dict],
    output_path: Path,
    title: str = "Method Comparison",
    metric: str = "normalized_hpwl",
    lower_is_better: bool = True,
):
    """
    Heatmap-style table showing relative performance across benchmarks and methods.
    Green = better than baseline, red = worse.
    """
    plt, _ = _get_matplotlib()
    import numpy as np

    methods = list(results_by_method.keys())
    all_benches = sorted(set(b for r in results_by_method.values() for b in r))

    data = np.full((len(methods), len(all_benches)), fill_value=float("nan"))
    for i, method in enumerate(methods):
        for j, bench in enumerate(all_benches):
            r = results_by_method[method].get(bench)
            if r is None:
                continue
            if hasattr(r, "metrics"):
                val = r.metrics.get(metric, float("nan"))
            elif isinstance(r, dict):
                m = r.get("metrics", r)
                val = m.get(metric, float("nan"))
            else:
                val = float("nan")
            data[i, j] = val

    fig, ax = plt.subplots(figsize=(max(8, len(all_benches) * 1.2), len(methods) + 1))

    # Normalize columns to baseline row (row 0)
    baseline = data[0]
    with np.errstate(invalid="ignore"):
        norm_data = data / baseline

    im = ax.imshow(norm_data, cmap="RdYlGn_r" if lower_is_better else "RdYlGn",
                    vmin=0.9, vmax=1.1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Relative to baseline")

    ax.set_xticks(range(len(all_benches)))
    ax.set_xticklabels(all_benches, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title(title)

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(all_benches)):
            val = norm_data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison table: {output_path}")
