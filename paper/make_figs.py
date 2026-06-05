"""Generate paper figures from campaign data. Run from repo root."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FIGS = ROOT / "paper" / "figs"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

SIGMA = 0.0015  # measured single-seed noise floor (sigma, norm. HPWL)

# --------------------------------------------------------------- Fig 1
it_sc = np.loadtxt(FIGS / "exp1_scores.tsv")
its, scores = it_sc[:, 0], it_sc[:, 1]
fig, ax = plt.subplots(figsize=(3.45, 2.2))
ax.axhspan(1 - 3 * SIGMA, 1 + 3 * SIGMA, color="0.85", zorder=0,
           label=r"$\pm3\sigma$ single-seed noise band")
ax.axhline(1.0, color="0.4", lw=0.7, ls="--", zorder=1)
surv = ax.scatter(its, scores, s=18, c="#1f77b4", zorder=3,
                  label="cascade survivors (14/200)")
elim = sorted(set(range(1, 201)) - set(its.astype(int)))
ax.scatter(elim, [1.0455] * len(elim), marker="|", s=12, c="#d62728",
           alpha=0.55, zorder=2, label="eliminated by cascade (186/200)")
for it, sc, name, dy in [(91, 0.9956, "cand. 0090", -0.0035),
                         (118, 0.9972, "cand. 0117", -0.0033)]:
    ax.annotate(name, (it, sc), xytext=(it + 6, sc + dy), fontsize=6.5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
ax.set_xlabel("evolution iteration")
ax.set_ylabel("normalized HPWL (single seed)")
ax.set_xlim(0, 202)
ax.set_ylim(0.9905, 1.0495)
ax.legend(loc="center right", framealpha=0.9)
fig.savefig(FIGS / "fig_evolution.pdf")
plt.close(fig)

# --------------------------------------------------------------- Fig 2
# (program, single-seed score, multiseed mean, std, n)
RERANK = [
    ("cand. 0090", 0.99563, 1.00014, 0.00851, 10),
    ("cand. 0117", 0.99716, 0.99685, 0.00273, 10),
    ("seed (default)", 1.00010, 1.00057, 0.00735, 10),
    ("rejected A", None, 1.02244, 0.00589, 10),
    ("rejected B", None, 1.06626, 0.00464, 10),
    ("rejected C", None, 1.07746, 0.01232, 10),
]
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(3.45, 2.3), width_ratios=[3, 2], sharey=False)
xs = np.arange(3)
for i, (name, ss, mu, sd, n) in enumerate(RERANK[:3]):
    sem95 = 1.96 * sd / np.sqrt(n)
    ax1.errorbar(i + 0.13, mu, yerr=sem95, fmt="o", ms=4, c="#1f77b4",
                 capsize=2.5, lw=1,
                 label="multi-seed mean ±95% CI" if i == 0 else None)
    ax1.plot(i - 0.13, ss, "s", ms=4, c="#d62728",
             label="single-seed score" if i == 0 else None)
    ax1.plot([i - 0.13, i + 0.13], [ss, mu], c="0.6", lw=0.6, zorder=0)
ax1.axhspan(1 - 3 * SIGMA, 1 + 3 * SIGMA, color="0.88", zorder=0)
ax1.axhline(1.0, color="0.4", lw=0.7, ls="--")
ax1.set_xticks(xs, [r[0] for r in RERANK[:3]], rotation=12)
ax1.set_ylabel("HPWL ratio vs. default")
ax1.set_ylim(0.988, 1.012)
ax1.legend(loc="upper left", framealpha=0.9)
xs2 = np.arange(3)
for i, (name, ss, mu, sd, n) in enumerate(RERANK[3:]):
    sem95 = 1.96 * sd / np.sqrt(n)
    ax2.errorbar(i, mu, yerr=sem95, fmt="o", ms=4, c="#7f7f7f", capsize=2.5, lw=1)
ax2.axhline(1.0, color="0.4", lw=0.7, ls="--")
ax2.set_xticks(xs2, [r[0] for r in RERANK[3:]], rotation=12)
ax2.set_title("negative controls", fontsize=7.5)
ax2.set_ylim(0.99, 1.10)
ax2.ticklabel_format(axis="y", useOffset=False)
fig.subplots_adjust(wspace=0.42)
fig.savefig(FIGS / "fig_rerank.pdf")
plt.close(fig)

# --------------------------------------------------------------- Fig 3
sys.path.insert(0, str(ROOT / "experiments/exp01_wl_smoothing/evolution_runs"))
import importlib.util


def load_fn(path):
    spec = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.gamma_schedule


cand = load_fn(ROOT / "experiments/exp01_wl_smoothing/evolution_runs/candidate_0117.py")
seed = load_fn(ROOT / "evolve/initial_program.py")

T = 1000
t = np.arange(T)
# illustrative overflow trajectory (exponential relaxation to stop_overflow)
ov = 0.07 + 0.93 * np.exp(-4.0 * t / T)
g_seed = [seed(int(i), T, float(o), []) for i, o in zip(t, ov)]
g_cand = [cand(int(i), T, float(o), []) for i, o in zip(t, ov)]
g_dp = 4.0 * 10 ** ((ov - 0.1) * 20 / 9 - 1)  # DREAMPlace built-in heuristic

fig, (ax, axo) = plt.subplots(
    2, 1, figsize=(3.45, 2.6), sharex=True, height_ratios=[3, 1])
ax.plot(t, g_dp, c="0.55", lw=1, ls=":", label="DREAMPlace built-in")
ax.plot(t, g_seed, c="#d62728", lw=1, label="seed program (default-matched)")
ax.plot(t, g_cand, c="#1f77b4", lw=1.2, label="evolved cand. 0117")
ax.set_yscale("log")
ax.set_ylabel(r"$\gamma$ (dimensionless)")
ax.legend(framealpha=0.9)
axo.plot(t, ov, c="0.3", lw=0.9)
axo.set_ylabel("overflow")
axo.set_xlabel("global-placement iteration")
fig.savefig(FIGS / "fig_schedules.pdf")
plt.close(fig)

print("figures written:", sorted(p.name for p in FIGS.glob("*.pdf")))

# --------------------------------------------------------------- Fig 4
# Lambda guard-branch ablation: paired ratios per design (NOTES 2026-06-05)
LAM = {
    "fft_1": [0.98999, 0.99024, 0.99004, 0.98637, 0.98948],
    "fft_2": [0.90820, 0.92117, 0.91732, 0.90257, 0.91551],
    "matrix_mult_1": [0.98294, 0.98277],
    "des_perf_1": [0.99053, 0.99034],
}
fig, ax = plt.subplots(figsize=(3.45, 2.1))
ax.axhspan(1 - 3 * SIGMA, 1 + 3 * SIGMA, color="0.88", zorder=0,
           label=r"$\pm3\sigma$ noise band")
ax.axhline(1.0, color="0.4", lw=0.7, ls="--")
for i, (name, ratios) in enumerate(LAM.items()):
    r = np.asarray(ratios)
    ax.scatter([i] * len(r) + np.linspace(-0.08, 0.08, len(r)), r,
               s=12, c="#1f77b4", alpha=0.65, zorder=3, lw=0)
    mean = r.mean()
    ci = 1.96 * r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0
    ax.errorbar(i + 0.22, mean, yerr=ci, fmt="D", ms=4, c="#d62728",
                capsize=2.5, lw=1, zorder=4,
                label="mean ± 95% CI" if i == 0 else None)
ax.set_xticks(range(len(LAM)), [k.replace("_", "\\_") if False else k
                                for k in LAM], fontsize=7)
ax.set_ylabel("HPWL ratio vs.\ndefault $\\lambda$ update")
ax.set_ylim(0.895, 1.012)
ax.legend(loc="lower left", framealpha=0.9, fontsize=6.5)
fig.savefig(FIGS / "fig_lambda.pdf")
plt.close(fig)
print("fig_lambda.pdf written")
