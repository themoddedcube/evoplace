"""
DREAMPlace-style placement animations with live metrics.

Two modes:
  --program <evolved.py>  : side-by-side evolved-vs-default comparison GIF
                            (cells in official DREAMPlace colors, live HPWL
                            strip, per-frame gamma annotation)
  (no --program)          : single-run convergence showcase GIF of the
                            default schedule (bigblue4-nofiller style)

With --fields density|all, also renders animated 3D surface GIFs of the
bin density map (and electric potential / field magnitude with `all`),
matching the official README's density_map_SLD.gif / potential_map_SLD.gif.

Positions are captured by monkeypatching NonLinearPlace.plot (called every
`params.plot_interval` iterations — requires the plot_interval patch in
vendor/dreamplace AND a `make install`). Field tensors are read from the
model's ElectricPotential op via its public spectral-transform attributes;
no submodule modifications needed. HPWL is recomputed per frame from the
netlist, identically for all runs.

Usage (do NOT run while an evolution/re-rank run owns the GPU):
    python scripts/make_comparison_gif.py \
        --program experiments/exp01_wl_smoothing/evolution_runs/best_program.py \
        --benchmark fft_1 --seed 42 --interval 25 --fields all \
        --out-dir graphs/comparisons/fft_1_s42
    python scripts/make_comparison_gif.py \
        --benchmark superblue12 --interval 50 --fields density \
        --out-dir graphs/comparisons/superblue12_showcase
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
from matplotlib.collections import PolyCollection
from PIL import Image

from evaluator import run_placement as rp
from evolve.evaluator_wrapper import load_evolved_function

# Official DREAMPlace draw_place colors (ops/draw_place/src/PlaceDrawer.h)
MOVABLE_COLOR = (0.0, 0.0, 1.0, 0.5)
FIXED_COLOR = (1.0, 0.0, 0.0, 0.5)
# Above this many movable cells, rectangles are too slow per frame — fall
# back to a center-point scatter in the same color (superblue-class designs).
RECT_RENDER_LIMIT = 50_000


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def _capture_field_maps(model, pos_t, want_all):
    """Read density (and potential / field magnitude) bin maps from the
    model's ElectricPotential op using its public attributes. Returns a dict
    of numpy (num_bins_x, num_bins_y) arrays; missing entries mean the op
    state didn't support them (e.g. fast_mode has no idct2)."""
    import torch
    from dreamplace.ops.electric_potential.electric_potential import (
        ElectricDensityMapFunction,
    )

    dop = getattr(model.op_collections, "density_op", None)
    if dop is None:
        return {}
    with torch.no_grad():
        if dop.initial_density_map is None:
            dop.forward(pos_t, mode="overflow")  # one-time op init
        # raw density map, no fillers (mirrors the mode=="overflow" branch)
        density_map = ElectricDensityMapFunction.forward(
            pos_t, dop.node_size_x_clamped, dop.node_size_y_clamped,
            dop.offset_x, dop.offset_y, dop.ratio, dop.bin_center_x,
            dop.bin_center_y, dop.initial_density_map, dop.target_density,
            dop.xl, dop.yl, dop.xh, dop.yh, dop.bin_size_x, dop.bin_size_y,
            dop.num_movable_nodes, 0, dop.padding, dop.padding_mask,
            dop.num_bins_x, dop.num_bins_y,
            dop.num_movable_impacted_bins_x, dop.num_movable_impacted_bins_y,
            dop.num_filler_impacted_bins_x, dop.num_filler_impacted_bins_y,
            dop.deterministic_flag, dop.sorted_node_map)
        bin_area = float(dop.bin_size_x * dop.bin_size_y)
        out = {"density": (density_map / bin_area).cpu().numpy()}
        if want_all:
            # potential / field via the op's precomputed spectral transforms
            # (replicates ElectricPotentialFunction.forward's DCT block)
            auv = dop.dct2.forward(density_map / bin_area)
            fx = dop.idxst_idct.forward(auv.mul(dop.wu_by_wu2_plus_wv2_half))
            fy = dop.idct_idxst.forward(auv.mul(dop.wv_by_wu2_plus_wv2_half))
            out["field"] = (fx.pow(2) + fy.pow(2)).sqrt().cpu().numpy()
            if dop.idct2 is not None:  # absent in fast_mode
                out["potential"] = (
                    dop.idct2.forward(auv.mul(dop.inv_wu2_plus_wv2))
                    .cpu().numpy())
    return out


def capture_run(program_path, bench_dir, out_dir, seed, interval,
                max_iterations, fields="none"):
    """Run a placement, capturing (iteration, x, y) snapshots, field maps,
    and the gamma trace. Returns (snapshots, field_snaps, gamma_trace,
    netlist, result)."""
    import torch
    Params, PlaceDB, NonLinearPlace = rp.load_dreamplace()

    snapshots = []    # (iteration, x[num_physical], y[num_physical])
    field_snaps = []  # (iteration, {"density": ..., "potential": ..., ...})
    netlist = {}

    def capture_plot(self, params, placedb, iteration, pos):
        n = placedb.num_physical_nodes
        x = pos[: n].copy()
        y = pos[placedb.num_nodes: placedb.num_nodes + n].copy()
        snapshots.append((int(iteration), x, y))
        if fields != "none":
            pos_t = torch.as_tensor(
                pos, dtype=self.pos[0].dtype, device=self.pos[0].device)
            try:
                maps = _capture_field_maps(self, pos_t, fields == "all")
                if maps:
                    field_snaps.append((int(iteration), maps))
            except Exception as e:  # never let viz capture kill the run
                print(f"  field capture failed at iter {iteration}: {e}")
        if not netlist:
            netlist.update(
                num_movable=int(placedb.num_movable_nodes),
                num_physical=int(n),
                node_size_x=np.asarray(placedb.node_size_x[:n], dtype=np.float64),
                node_size_y=np.asarray(placedb.node_size_y[:n], dtype=np.float64),
                xl=float(placedb.xl), yl=float(placedb.yl),
                xh=float(placedb.xh), yh=float(placedb.yh),
                num_bins_x=int(placedb.num_bins_x),
                num_bins_y=int(placedb.num_bins_y),
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
    return snapshots, field_snaps, gamma_trace, netlist, result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def hpwl_of(x, y, nl):
    """Exact HPWL from pin positions (same formula for all runs)."""
    px = x[nl["pin2node"]] + nl["pin_off_x"]
    py = y[nl["pin2node"]] + nl["pin_off_y"]
    px = px[nl["flat_net2pin"]]
    py = py[nl["flat_net2pin"]]
    starts = nl["net2pin_start"][:-1]
    return float(
        (np.maximum.reduceat(px, starts) - np.minimum.reduceat(px, starts)).sum()
        + (np.maximum.reduceat(py, starts) - np.minimum.reduceat(py, starts)).sum()
    )


def _rect_quads(x, y, w, h):
    """(n,4,2) vertex array of axis-aligned rectangles for PolyCollection."""
    quads = np.empty((len(x), 4, 2))
    quads[:, 0, 0] = x;     quads[:, 0, 1] = y
    quads[:, 1, 0] = x + w; quads[:, 1, 1] = y
    quads[:, 2, 0] = x + w; quads[:, 2, 1] = y + h
    quads[:, 3, 0] = x;     quads[:, 3, 1] = y + h
    return quads


def draw_placement(ax, x, y, nl, title, iteration, show_bins=False):
    """Official DREAMPlace look: movable blue rects, fixed red rects,
    white bg, no fillers, iteration counter."""
    n_mov = nl["num_movable"]
    ax.set_xlim(nl["xl"], nl["xh"])
    ax.set_ylim(nl["yl"], nl["yh"])
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    if show_bins:
        for k in range(0, nl["num_bins_x"] + 1, max(1, nl["num_bins_x"] // 32)):
            ax.axvline(nl["xl"] + k * (nl["xh"] - nl["xl"]) / nl["num_bins_x"],
                       color="0.85", lw=0.3, zorder=0)
        for k in range(0, nl["num_bins_y"] + 1, max(1, nl["num_bins_y"] // 32)):
            ax.axhline(nl["yl"] + k * (nl["yh"] - nl["yl"]) / nl["num_bins_y"],
                       color="0.85", lw=0.3, zorder=0)
    # fixed cells/macros: red rectangles
    if nl["num_physical"] > n_mov:
        ax.add_collection(PolyCollection(
            _rect_quads(x[n_mov:], y[n_mov:], nl["node_size_x"][n_mov:],
                        nl["node_size_y"][n_mov:]),
            facecolor=FIXED_COLOR, edgecolor="black", lw=0.2))
    # movable cells: blue rectangles (scatter fallback for huge designs)
    if n_mov <= RECT_RENDER_LIMIT:
        ax.add_collection(PolyCollection(
            _rect_quads(x[:n_mov], y[:n_mov], nl["node_size_x"][:n_mov],
                        nl["node_size_y"][:n_mov]),
            facecolor=MOVABLE_COLOR, edgecolor=(0, 0, 0.8, 0.8), lw=0.1))
    else:
        cx = x[:n_mov] + nl["node_size_x"][:n_mov] / 2
        cy = y[:n_mov] + nl["node_size_y"][:n_mov] / 2
        ax.scatter(cx, cy, s=0.2, color=MOVABLE_COLOR, lw=0)
    # iteration counter, draw_place style
    ax.text(0.02, 0.97, "{:04d}".format(iteration), transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", color="0.25")


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def assemble_gif(images, out, fps):
    durations = [1000 // fps] * len(images)
    durations[-1] = 2000  # hold final frame
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"Saved {len(images)}-frame GIF: {out}")


def render_surface_gif(field_snaps, key, out_path, zlabel, fps):
    """Animated 3D surface like the official density_map_SLD.gif."""
    snaps = [(it, maps[key]) for it, maps in field_snaps if key in maps]
    if not snaps:
        print(f"  no {key} frames captured, skipping {out_path}")
        return
    zmax = max(float(m.max()) for _, m in snaps)
    zmin = min(float(m.min()) for _, m in snaps)
    nbx, nby = snaps[0][1].shape
    # upstream plots map[x][y] against meshgrid(ys, xs) (see
    # electric_overflow.plot); mgrid keeps axis order consistent
    xs, ys = np.meshgrid(np.arange(nby), np.arange(nbx))
    images = []
    for it, m in snaps:
        fig = plt.figure(figsize=(6.4, 5.2), dpi=100)
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(xs, ys, m, alpha=0.8, cmap="viridis")
        ax.set_zlim(zmin, zmax * 1.02)
        ax.set_xlabel("bin y", fontsize=8)
        ax.set_ylabel("bin x", fontsize=8)
        ax.set_zlabel(zlabel, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{zlabel} — iteration {it:04d}", fontsize=10)
        images.append(fig_to_image(fig))
    assemble_gif(images, out_path, fps)


def render_comparison_gif(snap_e, snap_d, hp_e, hp_d, gamma_by_iter, nl,
                          res_e, res_d, bench, seed, out, fps):
    n_frames = max(len(snap_e), len(snap_d))
    final_ratio = res_e.metrics["hpwl"] / res_d.metrics["hpwl"]
    images = []
    for f in range(n_frames):
        e = snap_e[min(f, len(snap_e) - 1)]
        d = snap_d[min(f, len(snap_d) - 1)]
        fig = plt.figure(figsize=(9, 6.4), dpi=110)
        gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1], hspace=0.18,
                              wspace=0.06)
        ax_e = fig.add_subplot(gs[0, 0])
        ax_d = fig.add_subplot(gs[0, 1])
        ax_m = fig.add_subplot(gs[1, :])

        he = hp_e[min(f, len(hp_e) - 1)][1]
        hd = hp_d[min(f, len(hp_d) - 1)][1]
        g_now = gamma_by_iter.get(e[0])
        draw_placement(ax_e, e[1], e[2], nl,
                       f"EvoPlace schedule — HPWL {he:.3e}"
                       + (f"  γ={g_now:.2f}" if g_now is not None else ""),
                       e[0])
        draw_placement(ax_d, d[1], d[2], nl,
                       f"DREAMPlace default — HPWL {hd:.3e}", d[0])

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
            f"{bench} (seed {seed}) — evolved vs default γ schedule   "
            f"current Δ {delta:+.2f}%   final Δ {(1 - final_ratio) * 100:+.2f}%",
            fontsize=11)
        images.append(fig_to_image(fig))
    assemble_gif(images, out, fps)
    print(f"final HPWL: evolved {res_e.metrics['hpwl']:.6e} vs "
          f"default {res_d.metrics['hpwl']:.6e}  ({(1 - final_ratio) * 100:+.3f}%)")


def render_single_gif(snaps, hp, nl, result, bench, seed, out, fps):
    """Single-run convergence showcase (bigblue4-nofiller style)."""
    images = []
    for f, (it, x, y) in enumerate(snaps):
        fig = plt.figure(figsize=(7.2, 7.8), dpi=110)
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.15)
        ax_p = fig.add_subplot(gs[0])
        ax_m = fig.add_subplot(gs[1])
        draw_placement(ax_p, x, y, nl,
                       f"{bench} — HPWL {hp[f][1]:.3e}", it)
        ax_m.plot([i for i, _ in hp[: f + 1]], [v for _, v in hp[: f + 1]],
                  c="#1f77b4")
        ax_m.set_xlim(0, hp[-1][0])
        all_v = [v for _, v in hp]
        ax_m.set_ylim(min(all_v) * 0.97, np.percentile(all_v, 95) * 1.1)
        ax_m.set_xlabel("iteration", fontsize=8)
        ax_m.set_ylabel("HPWL", fontsize=8)
        ax_m.tick_params(labelsize=7)
        fig.suptitle(f"{bench} (seed {seed}) — DREAMPlace convergence, "
                     f"final HPWL {result.metrics['hpwl']:.4e}", fontsize=11)
        images.append(fig_to_image(fig))
    assemble_gif(images, out, fps)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", default=None,
                    help="evolved program; omit for single-run showcase mode")
    ap.add_argument("--benchmark", default="fft_1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--interval", type=int, default=25)
    ap.add_argument("--max-iterations", type=int, default=2000)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--fields", choices=["none", "density", "all"],
                    default="none",
                    help="also render 3D surface GIFs of bin maps")
    ap.add_argument("--out", default=None, help="comparison GIF path (legacy)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    bench_dir = PROJECT_ROOT / "benchmarks" / args.benchmark
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "graphs" / "comparisons"
        / f"{args.benchmark}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path("/tmp/gif_runs")

    if args.program:
        print("Running evolved schedule ...")
        snap_e, fields_e, gtrace, nl, res_e = capture_run(
            Path(args.program), bench_dir, tmp / "evolved", args.seed,
            args.interval, args.max_iterations, args.fields)
        print(f"  {len(snap_e)} frames, final HPWL {res_e.metrics['hpwl']:.4e}")
        print("Running default schedule ...")
        snap_d, _, _, nl_d, res_d = capture_run(
            None, bench_dir, tmp / "default", args.seed,
            args.interval, args.max_iterations, "none")
        print(f"  {len(snap_d)} frames, final HPWL {res_d.metrics['hpwl']:.4e}")
        nl = nl or nl_d
        hp_e = [(it, hpwl_of(x, y, nl)) for it, x, y in snap_e]
        hp_d = [(it, hpwl_of(x, y, nl)) for it, x, y in snap_d]
        out = Path(args.out) if args.out else out_dir / "comparison.gif"
        render_comparison_gif(snap_e, snap_d, hp_e, hp_d,
                              dict((it, g) for it, g, _ in gtrace), nl,
                              res_e, res_d, args.benchmark, args.seed,
                              out, args.fps)
        field_snaps = fields_e
    else:
        print("Running default schedule (showcase mode) ...")
        snaps, field_snaps, _, nl, result = capture_run(
            None, bench_dir, tmp / "showcase", args.seed,
            args.interval, args.max_iterations, args.fields)
        print(f"  {len(snaps)} frames, final HPWL {result.metrics['hpwl']:.4e}")
        hp = [(it, hpwl_of(x, y, nl)) for it, x, y in snaps]
        out = Path(args.out) if args.out else out_dir / "convergence.gif"
        render_single_gif(snaps, hp, nl, result, args.benchmark, args.seed,
                          out, args.fps)

    if args.fields != "none":
        render_surface_gif(field_snaps, "density",
                           out_dir / "density.gif", "density", args.fps)
        if args.fields == "all":
            render_surface_gif(field_snaps, "potential",
                               out_dir / "potential.gif", "potential", args.fps)
            render_surface_gif(field_snaps, "field",
                               out_dir / "field.gif", "|E| field", args.fps)


if __name__ == "__main__":
    main()
