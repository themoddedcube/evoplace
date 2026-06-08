"""
PlaceDB MMU-fault probe (companion to CRASH_DIAGNOSIS.md).

Calls DREAMPlace's PlaceDB(p) on a single benchmark, with optional
override of params.gpu to test the proposed workaround. Designed to
be driven from a sweep that brackets benchmark size — see Q4 in
CRASH_DIAGNOSIS.md.

Print a one-line JSON record so the wrapper can grep for status.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def load_dreamplace():
    from covenant.harness.run_placement import load_dreamplace as _ld
    return _ld()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir", required=True, type=Path)
    ap.add_argument("--gpu", choices=("0", "1"), required=True,
                    help="value to force into params.gpu before db(p)")
    args = ap.parse_args()

    json_files = sorted(args.benchmark_dir.glob("*.json"))
    if not json_files:
        print(json.dumps({"status": "config-missing", "dir": str(args.benchmark_dir)}))
        return 2
    cfg = json_files[0]

    Params, PlaceDB, _ = load_dreamplace()
    p = Params.Params()
    p.load(str(cfg))

    # Resolve aux_input relative to the benchmark dir (the json stores a
    # bare filename).
    if getattr(p, "aux_input", None):
        aux = Path(p.aux_input)
        if not aux.is_absolute():
            p.aux_input = str(args.benchmark_dir / aux)

    forced_gpu = int(args.gpu)
    original_gpu = int(getattr(p, "gpu", 1))
    p.gpu = forced_gpu

    t0 = time.perf_counter()
    record = {
        "benchmark": args.benchmark_dir.name,
        "forced_gpu": forced_gpu,
        "json_gpu": original_gpu,
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", ""),
    }
    try:
        db = PlaceDB.PlaceDB()
        db(p)
        dt = time.perf_counter() - t0
        record.update({
            "status": "ok",
            "elapsed_s": round(dt, 3),
            "num_nodes": int(getattr(db, "num_nodes", -1)),
            "num_nets": int(getattr(db, "num_nets", -1)),
            "num_movable_nodes": int(getattr(db, "num_movable_nodes", -1)),
        })
        print(json.dumps(record))
        return 0
    except Exception as e:
        dt = time.perf_counter() - t0
        record.update({
            "status": "exception",
            "elapsed_s": round(dt, 3),
            "exc_type": type(e).__name__,
            "exc_msg": str(e)[:400],
        })
        print(json.dumps(record))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
