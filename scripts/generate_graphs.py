"""
Generate all publication-quality graphs from stub data.

Produces the full set of figures we'll need for the paper,
using synthetic but realistic data. Real experiment data will
replace these when the DGX Spark runs complete.

Run: python scripts/generate_graphs.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from graphs.plot_utils import (
    plot_benchmark_bars, plot_convergence, plot_pareto_frontier,
    plot_schedule_trajectory, plot_comparison_table,
)
from evaluator.run_placement import PlacementResult

GRAPHS_DIR = Path(__file__).parent.parent / "graphs" / "paper_figures"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# ── Benchmark names ────────────────────────────────────────────────────────────
BENCHMARKS = [
    "fft_1", "fft_2", "fft_a", "fft_b", "des_perf_1",
    "matrix_mult_1", "matrix_mult_2", "matrix_mult_a",
    "superblue12", "superblue14", "superblue19",
]

HPWL_BASELINES = {
    "fft_1": 4.2e8, "fft_2": 3.8e8, "fft_a": 5.1e8, "fft_b": 4.9e8,
    "des_perf_1": 2.3e9, "matrix_mult_1": 1.8e9, "matrix_mult_2": 1.6e9,
    "matrix_mult_a": 2.1e9, "superblue12": 8.7e9, "superblue14": 7.2e9,
    "superblue19": 9.1e9,
}


def make_results(method_factors):
    """Build PlacementResult dicts from per-method normalization factors."""
    out = {}
    for method, (hpwl_f, tns_f, rt_f) in method_factors.items():
        out[method] = {}
        for b in BENCHMARKS:
            base = HPWL_BASELINES[b]
            noise = 1 + rng.normal(0, 0.008)
            h = base * hpwl_f * noise
            out[method][b] = PlacementResult(
                metrics={
                    "hpwl": h,
                    "normalized_hpwl": hpwl_f * noise,
                    "mean_overflow": rng.uniform(0.055, 0.075),
                    "tns_proxy": base * 0.02 * tns_f * noise,
                },
                runtime_s=rng.uniform(45, 75) * rt_f,
                divergence_events=rng.integers(0, 4),
                converged=True,
            )
    return out


# ── Figure 1: HPWL comparison across all methods (w/o region) ─────────────────
print("Generating Figure 1: HPWL comparison (w/o region)...")
results = make_results({
    "Eh?Placer":      (1.17, 1.25, 14.0),
    "NTUplace4dr":    (1.07, 1.12, 37.0),
    "DREAMPlace":     (1.01, 1.03, 1.0),
    "DREAMPlace4.0":  (1.00, 1.00, 1.0),
    "EvoPlace (ours)":(0.95, 0.88, 0.97),  # projected improvement
})
plot_benchmark_bars(
    results,
    GRAPHS_DIR / "fig1_hpwl_comparison.png",
    title="Normalized HPWL — ISPD 2015 (w/o Region Constraints)",
    metric="normalized_hpwl",
)
print("  ✓ fig1_hpwl_comparison.png")


# ── Figure 2: TNS comparison ───────────────────────────────────────────────────
print("Generating Figure 2: TNS proxy comparison...")
plot_benchmark_bars(
    results,
    GRAPHS_DIR / "fig2_tns_comparison.png",
    title="Normalized TNS Proxy — ISPD 2015 (w/o Region Constraints)",
    metric="tns_proxy",
    normalize=False,
)
print("  ✓ fig2_tns_comparison.png")


# ── Figure 3: Convergence curves (DREAMPlace4 vs EvoPlace) ────────────────────
print("Generating Figure 3: Convergence curves...")
total_iters = 1000
t = np.arange(total_iters)

# DREAMPlace4.0 baseline convergence
base_hpwl = 4.2e8
dp4_hpwl  = base_hpwl * (1 + 0.8 * np.exp(-4.0 * t/total_iters) + rng.normal(0, 0.005, total_iters).cumsum()/500)
dp4_ovfl  = np.maximum(0.04, 1.0 * np.exp(-5.5 * t/total_iters) + rng.normal(0, 0.003, total_iters))

# EvoPlace (evolved gamma + lambda schedules)
evo_hpwl  = base_hpwl * (1 + 0.8 * np.exp(-5.5 * t/total_iters) + rng.normal(0, 0.004, total_iters).cumsum()/600)
evo_hpwl  *= 0.95  # projected 5% HPWL improvement
evo_ovfl  = np.maximum(0.04, 1.0 * np.exp(-6.5 * t/total_iters) + rng.normal(0, 0.002, total_iters))

# Mark a divergence event in DREAMPlace4
dp4_hpwl[400:420] *= 1.08
dp4_ovfl[400:420] *= 1.15

plot_convergence(
    evo_hpwl.tolist(), evo_ovfl.tolist(),
    GRAPHS_DIR / "fig3_convergence_fft1.png",
    title="Convergence on fft_1 — EvoPlace vs DREAMPlace 4.0",
    baseline_hpwl=dp4_hpwl.tolist(),
    baseline_overflow=dp4_ovfl.tolist(),
)
print("  ✓ fig3_convergence_fft1.png")


# ── Figure 4: All 4 gamma schedules compared ─────────────────────────────────
print("Generating Figure 4: Gamma schedule trajectories...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dreamplace_ext.schedulers import (
    gamma_baseline_linear, gamma_exponential,
    gamma_overflow_adaptive, gamma_cosine_annealing,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
total = 1000
iters = np.arange(total)

schedules = [
    ("Linear (DREAMPlace default)", gamma_baseline_linear, "#888888", "--"),
    ("Exponential decay",           gamma_exponential,      "#2196F3", "-"),
    ("Overflow-adaptive",           gamma_overflow_adaptive,"#FF9800", "-"),
    ("Cosine annealing",            gamma_cosine_annealing, "#4CAF50", "-"),
]

for ax, (title, ovfl_func) in zip(axes, [
    ("γ schedule comparison (overflow=0.9→0.05)", lambda t: max(0.05, 1 - t/total)),
    ("λ schedule effect on HPWL trajectory", None),
]):
    if ovfl_func is None:
        break
    for name, fn, color, ls in schedules:
        vals = [fn(int(i), total, ovfl_func(i), []) for i in iters]
        ax.plot(iters, vals, label=name, color=color, linestyle=ls, linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("γ (WA-WL smoothness)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Second panel: lambda trajectories
ax2 = axes[1]
from dreamplace_ext.schedulers import (
    lambda_baseline_exponential, lambda_gradient_adaptive, lambda_plateau_detector
)
lambda_schedules = [
    ("Exponential (DREAMPlace)", lambda_baseline_exponential, "#888888", "--"),
    ("Gradient-adaptive",        lambda_gradient_adaptive,    "#2196F3", "-"),
    ("Plateau-detector",         lambda_plateau_detector,     "#E91E63", "-"),
]
lam_vals = {name: [1.0] for name, *_ in lambda_schedules}
ovfl_hist = []
for t_i in range(200):
    ovfl = max(0.05, 1 - t_i / 200)
    ovfl_hist.append(ovfl)
    for name, fn, _, _ in lambda_schedules:
        prev = lam_vals[name][-1]
        new = fn(t_i, ovfl, ovfl_hist[-20:], 0.1, prev)
        lam_vals[name].append(new)

for name, fn, color, ls in lambda_schedules:
    ax2.plot(lam_vals[name], label=name, color=color, linestyle=ls, linewidth=1.8)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("λ (density weight)")
ax2.set_title("λ schedule trajectories (simulated)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_yscale("log")

plt.tight_layout()
out = GRAPHS_DIR / "fig4_schedule_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close()
print("  ✓ fig4_schedule_comparison.png")


# ── Figure 5: Pareto frontier from evolution ──────────────────────────────────
print("Generating Figure 5: Evolution Pareto frontier...")
n_evolved = 150
# Simulate evolution results: trade-off between quality and runtime
evolved_results = []
for i in range(n_evolved):
    quality = rng.uniform(0.93, 1.08)  # normalized HPWL
    runtime = 30 + (quality - 0.93) * 200 + rng.normal(0, 5)  # faster = lower quality generally
    evolved_results.append({
        "normalized_hpwl": quality,
        "runtime_s": max(10, runtime),
        "divergence_events": rng.integers(0, 5),
    })

# Add a few "evolved best" points
for hpwl, rt in [(0.94, 52), (0.95, 45), (0.93, 65), (0.96, 40)]:
    evolved_results.append({"normalized_hpwl": hpwl, "runtime_s": rt, "divergence_events": 0})

plot_pareto_frontier(
    evolved_results,
    GRAPHS_DIR / "fig5_evolution_pareto.png",
    title="Evolution Pareto Frontier — Quality vs. Runtime (Exp 1, fft_1)",
    x_metric="normalized_hpwl",
    y_metric="runtime_s",
)
print("  ✓ fig5_evolution_pareto.png")


# ── Figure 6: GNN initialization benefit ─────────────────────────────────────
print("Generating Figure 6: GNN initialization benefit...")
fig, ax = plt.subplots(figsize=(8, 4))
benchmarks_small = ["fft_1", "fft_2", "fft_a", "matrix_mult_1", "des_perf_1"]
iters_center = [480, 510, 530, 620, 580]
iters_gnn    = [290, 320, 310, 380, 350]  # ~35% fewer iterations
x = np.arange(len(benchmarks_small))
w = 0.35
ax.bar(x - w/2, iters_center, w, label="Center init (DREAMPlace default)", color="#888888", alpha=0.85)
ax.bar(x + w/2, iters_gnn,    w, label="GNN warm init (Exp 3)",          color="#4CAF50", alpha=0.85)
for xi, (a, b) in zip(x, zip(iters_center, iters_gnn)):
    pct = (a - b) / a * 100
    ax.text(xi + w/2, b + 5, f"-{pct:.0f}%", ha="center", va="bottom", fontsize=8, color="#2e7d32")
ax.set_xticks(x)
ax.set_xticklabels(benchmarks_small)
ax.set_ylabel("Iterations to Overflow < 0.2")
ax.set_title("GNN Warm Initialization Reduces Convergence Iterations by ~35%\n[Projected — Exp 3]")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = GRAPHS_DIR / "fig6_gnn_init_benefit.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close()
print("  ✓ fig6_gnn_init_benefit.png")


# ── Figure 7: Full system comparison heatmap ─────────────────────────────────
print("Generating Figure 7: Full system heatmap...")
methods_heatmap = {
    "RePlAce (OpenROAD)": {b: PlacementResult({"normalized_hpwl": 1.04 + rng.normal(0,0.01)}, 120, 0, True) for b in BENCHMARKS},
    "DREAMPlace 3.0":     {b: PlacementResult({"normalized_hpwl": 1.01 + rng.normal(0,0.008)}, 55, 1, True) for b in BENCHMARKS},
    "DREAMPlace 4.0":     {b: PlacementResult({"normalized_hpwl": 1.00},                       55, 0, True) for b in BENCHMARKS},
    "AutoDMP":            {b: PlacementResult({"normalized_hpwl": 0.99 + rng.normal(0,0.007)}, 60, 0, True) for b in BENCHMARKS},
    "EvoPlace (ours)":    {b: PlacementResult({"normalized_hpwl": 0.95 + rng.normal(0,0.005)}, 54, 0, True) for b in BENCHMARKS},
}
plot_comparison_table(
    methods_heatmap,
    GRAPHS_DIR / "fig7_full_comparison_heatmap.png",
    title="Normalized HPWL vs DREAMPlace 4.0 — Full System Comparison [Projected]",
    metric="normalized_hpwl",
)
print("  ✓ fig7_full_comparison_heatmap.png")


# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\nAll figures saved to: {GRAPHS_DIR}")
print("Note: All figures show projected/synthetic data.")
print("They will be updated with real experimental results from the DGX Spark.")

figs = list(GRAPHS_DIR.glob("*.png"))
print(f"\nGenerated {len(figs)} figures:")
for f in sorted(figs):
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name} ({size_kb} KB)")
