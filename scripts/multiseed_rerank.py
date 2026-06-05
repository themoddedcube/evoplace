"""
Multi-seed re-ranking of top Exp 1 candidates.

Single-seed fitness has a noise floor of sigma ~ 0.15% norm_hpwl (NOTES.md),
so candidates within ~0.3% of each other cannot be ordered from one run.
This script re-scores the top-N surviving candidates with a PAIRED design:
each candidate and the default DREAMPlace schedule are run on the same
(benchmark, seed) pairs, and the ratio candidate/default is averaged.
Pairing cancels the per-seed placement noise that dominates at this margin.

Usage (after an evolution run completes):
    python scripts/multiseed_rerank.py --experiment exp01_wl_smoothing \
        --top 5 --seeds 42 43 44 45 46 --benchmarks fft_1 fft_2
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluator.run_placement import run_placement
from evolve.evaluator_wrapper import load_evolved_function


def top_candidates(results_tsv: Path, n: int):
    """Return [(iteration, norm_hpwl)] of the n best finite-score rows,
    deduplicated by code_hash (identical programs score once)."""
    rows = []
    seen_hashes = set()
    with open(results_tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                score = float(row["norm_hpwl"])
            except (KeyError, ValueError):
                continue
            if score == float("inf") or row["code_hash"] in seen_hashes:
                continue
            seen_hashes.add(row["code_hash"])
            rows.append((int(row["iteration"]), score))
    rows.sort(key=lambda r: r[1])
    return rows[:n]


def run_one(program_path, bench, seed, out_root):
    """Full placement run; returns converged HPWL."""
    fn = load_evolved_function(str(program_path), "gamma_schedule") if program_path else None
    out = out_root / f"{bench}_s{seed}_{program_path.stem if program_path else 'default'}"
    r = run_placement(
        PROJECT_ROOT / "benchmarks" / bench, out,
        gamma_schedule_fn=fn, seed=seed,
    )
    return r.metrics["hpwl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="exp01_wl_smoothing")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--benchmarks", nargs="+", default=["fft_1", "fft_2"])
    ap.add_argument("--out", default=None, help="output TSV (default: <exp>/multiseed_rerank.tsv)")
    args = ap.parse_args()

    exp_dir = PROJECT_ROOT / "experiments" / args.experiment
    runs_dir = exp_dir / "evolution_runs"
    out_root = Path("/tmp/multiseed_rerank")
    out_tsv = Path(args.out) if args.out else exp_dir / "multiseed_rerank.tsv"

    candidates = top_candidates(runs_dir / "results.tsv", args.top)
    if not candidates:
        sys.exit("no finite-score candidates in results.tsv")
    print(f"Top {len(candidates)} single-seed candidates: "
          + ", ".join(f"#{i} ({s:.4f})" for i, s in candidates))

    # Reference: default DREAMPlace schedule on every (bench, seed) cell.
    default_hpwl = {}
    for bench in args.benchmarks:
        for seed in args.seeds:
            default_hpwl[(bench, seed)] = run_one(None, bench, seed, out_root)
            print(f"default  {bench} seed={seed}: {default_hpwl[(bench, seed)]:.6e}")

    # Candidates (plus the seed program as a calibration row — its paired
    # ratio should be ~1.000 since it reproduces the default linear decay).
    programs = [("seed_program", PROJECT_ROOT / "evolve" / "initial_program.py")]
    programs += [(f"candidate_{it:04d}", runs_dir / f"candidate_{it:04d}.py")
                 for it, _ in candidates]

    results = []
    for name, path in programs:
        if not path.exists():
            print(f"!! {name}: {path} missing, skipped")
            continue
        ratios = []
        for bench in args.benchmarks:
            for seed in args.seeds:
                hpwl = run_one(path, bench, seed, out_root)
                ratio = hpwl / default_hpwl[(bench, seed)]
                ratios.append(ratio)
                print(f"{name}  {bench} seed={seed}: {hpwl:.6e}  ratio={ratio:.5f}")
        mean = statistics.mean(ratios)
        std = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
        results.append((name, mean, std, ratios))

    results.sort(key=lambda r: r[1])
    with open(out_tsv, "w") as f:
        f.write("rank\tprogram\tmean_ratio\tstd\tn\timprovement_pct\n")
        print(f"\n=== Paired multi-seed ranking (ratio vs default, < 1.0 is better) ===")
        for rank, (name, mean, std, ratios) in enumerate(results, 1):
            line = (f"{rank}\t{name}\t{mean:.5f}\t{std:.5f}\t{len(ratios)}"
                    f"\t{(1 - mean) * 100:+.3f}")
            f.write(line + "\n")
            print(f"{rank}. {name:18s} mean={mean:.5f} ±{std:.5f}  "
                  f"({(1 - mean) * 100:+.3f}% vs default)")
    print(f"\nSaved: {out_tsv}")


if __name__ == "__main__":
    main()
