"""
Side-by-side placement animation: evolved gamma schedule vs vanilla DREAMPlace.

Produces a GIF in the style of the macro-place-challenge demos
(https://github.com/partcleda/macro-place-challenge-2026): each frame shows
both placements at the same global-placement iteration, with a live metric
strip (HPWL curves + the gamma schedules) underneath, so the *why* is visible:
the evolved schedule keeps gradients smooth while cells spread, then anneals
gamma for accurate wirelength when it matters.

Positions are captured by monkeypatching NonLinearPlace.plot (which receives
the raw position array every `plot_interval` iterations); HPWL is recomputed
per frame from the netlist, identically for both runs, so the curves are
directly comparable. Requires the plot_interval patch in vendor/dreamplace
(evoplace-hooks branch) and a `make install` after it.

Usage (do NOT run while an evolution run owns the GPU):
    python scripts/make_comparison_gif.py \
        --program experiments/exp01_wl_smoothing/evolution_runs/best_program.py \
        --benchmark fft_1 --seed 42 --interval 25 \
        --out graphs/exp01_wl_smoothing/evolved_vs_default_fft_1.gif
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from evaluator import run_placement as rp
from evolve.evaluator_wrapper import load_evolved_function


def capture_run(program_path, bench_dir, out_dir, seed, interval, max_iterations):
    """Run a placement, capturing (iteration, x, y) snapshots and the gamma
    trace. Returns (snapshots, gamma_trace, netlist, result)."""
    Params, PlaceDB, NonLinearPlace = rp.load_dreamplace()

    snapshots = []   # (iteration, x[num_physical], y[num_physical])
    netlist = {}

    def capture_plot(self, params, placedb, iteration, pos):
        n = placedb.num_physical_nodes
        x = pos[: n].copy()
        y = pos[placedb.num_nodes: placedb.num_nodes + n].copy()
        snapshots.append((int(iteration), x, y))
        if not netlist:
            netlist.update(
                num_movable=int(placedb.num_movable_nodes),
                num_physical=int(n),
                node_size_x=np.asarray(placedb.node_size_x[:n], dtype=np.float64),
                node_size_y=np.asarray(placedb.node_size_y[:n], dtype=np.float64),
                xl=float(placedb.xl), yl=float(placedb.yl),
                xh=float(placedb.xh), yh=float(placedb.yh),
                pin2node=np.asarray(placedb.pin2node_map, dtype=np.int64),
                pin_off_x=np.asarray(placedb.pin_offset_x, dtype=np.float64),
                pin_off_y=np.asarray(placedb.pin_offset_y, dtype=np.float64),
                flat_net2pin=np.asarray(placedb.flat_net2pin_map, dtype=np.int64),
                net2pin_start=np.asarray(placedb.flat_net2pin_start_map, dtype=np.int64),
            )

    gamma_trace = []  # (iteration, gamma, overflow)
    schedule_fn = None
    if program_path is not None:
        inner = load_evolved_function(str(program_path), "gamma_schedule")

        def schedule_fn(iteration, total_iterations, overflow, hpwl_history):
            g = inner(iteration, total_iterations, overflow, hpwl_history)
            gamma_trace.append((int(iteration), float(g), float(overflow)))
            return g

    orig_plot = NonLinearPlace.NonLinearPlace.plot
    orig_build = rp.build_dreamplace_params

    def build_with_plot(*a, **kw):
        params = orig_build(*a, **kw)
        params.plot_flag = 1
        params.plot_interval = interval
        return params

    NonLinearPlace.NonLinearPlace.plot = capture_plot
    rp.build_dreamplace_params = build_with_plot
    try:
        result = rp.run_placement(
            bench_dir, out_dir, gamma_schedule_fn=schedule_fn,
            max_iterations=max_iterations, seed=seed,
        )
    finally:
        NonLinearPlace.NonLinearPlace.plot = orig_plot
        rp.build_dreamplace_params = orig_build
    return snapshots, gamma_trace, netlist, result


def hpwl_of(x, y, nl):
    """Exact HPWL from pin positions (same formula for both runs)."""
    # guard: snapshots cover physical nodes only; pins on fillers don't exist
    px = x[nl["pin2node"]] + nl["pin_off_x"]
    py = y[nl["pin2node"]] + nl["pin_off_y"]
    px = px[nl["flat_net2pin"]]
    py = py[nl["flat_net2pin"]]
    starts = nl["net2pin_start"][:-1]
    return float(
        (np.maximum.reduceat(px, starts) - np.minimum.reduceat(px, starts)).sum()
        + (np.maximum.reduceat(py, starts) - np.minimum.reduceat(py, starts)).sum()
    )


def draw_placement(ax, x, y, nl, title, color):
    n_mov = nl["num_movable"]
    ax.set_xlim(nl["xl"], nl["xh"])
    ax.set_ylim(nl["yl"], nl["yh"])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    # fixed cells/macros as gray rectangles
    for i in range(n_mov, nl["num_physical"]):
        ax.add_patch(Rectangle((x[i], y[i]), nl["node_size_x"][i],
                               nl["node_size_y"][i], color="0.75", lw=0))
    # movable cells as points at their centers
    cx = x[:n_mov] + nl["node_size_x"][:n_mov] / 2
    cy = y[:n_mov] + nl["node_size_y"][:n_mov] / 2
    ax.scatter(cx, cy, s=0.3, c=color, lw=0, alpha=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True)
    ap.add_argument("--benchmark", default="fft_1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--interval", type=int, default=25)
    ap.add_argument("--max-iterations", type=int, default=2000)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench_dir = PROJECT_ROOT / "benchmarks" / args.benchmark
    out = Path(args.out) if args.out else (
        PROJECT_ROOT / "graphs" / "comparisons"
        / f"evolved_vs_default_{args.benchmark}_s{args.seed}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/gif_runs")

    print("Running evolved schedule ...")
    snap_e, gtrace, nl, res_e = capture_run(
        Path(args.program), bench_dir, tmp / "evolved", args.seed,
        args.interval, args.max_iterations)
    print(f"  {len(snap_e)} frames, final HPWL {res_e.metrics['hpwl']:.4e}")
    print("Running default schedule ...")
    snap_d, _, nl_d, res_d = capture_run(
        None, bench_dir, tmp / "default", args.seed,
        args.interval, args.max_iterations)
    print(f"  {len(snap_d)} frames, final HPWL {res_d.metrics['hpwl']:.4e}")
    nl = nl or nl_d

    hp_e = [(it, hpwl_of(x, y, nl)) for it, x, y in snap_e]
    hp_d = [(it, hpwl_of(x, y, nl)) for it, x, y in snap_d]
    gamma_by_iter = dict((it, g) for it, g, _ in gtrace)

    n_frames = max(len(snap_e), len(snap_d))
    final_ratio = res_e.metrics["hpwl"] / res_d.metrics["hpwl"]
    images = []
    for f in range(n_frames):
        e = snap_e[min(f, len(snap_e) - 1)]
        d = snap_d[min(f, len(snap_d) - 1)]
        fig = plt.figure(figsize=(9, 6.4), dpi=110)
        gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1], hspace=0.18, wspace=0.06)
        ax_e = fig.add_subplot(gs[0, 0])
        ax_d = fig.add_subplot(gs[0, 1])
        ax_m = fig.add_subplot(gs[1, :])

        he = hp_e[min(f, len(hp_e) - 1)][1]
        hd = hp_d[min(f, len(hp_d) - 1)][1]
        g_now = gamma_by_iter.get(e[0])
        draw_placement(ax_e, e[1], e[2], nl,
                       f"EvoPlace schedule — iter {e[0]}  HPWL {he:.3e}"
                       + (f"  γ={g_now:.2f}" if g_now is not None else ""),
                       "#1f77b4")
        draw_placement(ax_d, d[1], d[2], nl,
                       f"DREAMPlace default — iter {d[0]}  HPWL {hd:.3e}",
                       "#d62728")

        # live metric strip: HPWL trajectories up to current frame
        ax_m.plot([it for it, _ in hp_e[: f + 1]], [v for _, v in hp_e[: f + 1]],
                  c="#1f77b4", label=f"evolved (final {res_e.metrics['hpwl']:.4e})")
        ax_m.plot([it for it, _ in hp_d[: f + 1]], [v for _, v in hp_d[: f + 1]],
                  c="#d62728", label=f"default (final {res_d.metrics['hpwl']:.4e})")
        ax_m.set_xlim(0, max(hp_e[-1][0], hp_d[-1][0]))
        all_v = [v for _, v in hp_e + hp_d]
        ax_m.set_ylim(min(all_v) * 0.97, np.percentile(all_v, 95) * 1.1)
        ax_m.set_xlabel("global-placement iteration", fontsize=8)
        ax_m.set_ylabel("HPWL", fontsize=8)
        ax_m.tick_params(labelsize=7)
        ax_m.legend(fontsize=7, loc="upper right")
        delta = (1 - he / hd) * 100 if hd else 0.0
        fig.suptitle(
            f"{args.benchmark} (seed {args.seed}) — evolved vs default γ schedule   "
            f"current Δ {delta:+.2f}%   final Δ {(1 - final_ratio) * 100:+.2f}%",
            fontsize=11)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    # hold the final frame for 2 seconds
    durations = [1000 // args.fps] * len(images)
    durations[-1] = 2000
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"\nSaved {len(images)}-frame GIF: {out}")
    print(f"final HPWL: evolved {res_e.metrics['hpwl']:.6e} vs "
          f"default {res_d.metrics['hpwl']:.6e}  ({(1 - final_ratio) * 100:+.3f}%)")


if __name__ == "__main__":
    main()
