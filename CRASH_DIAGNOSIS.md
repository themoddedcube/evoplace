# DGX Spark crash diagnosis — DREAMPlace GPU MMU faults

**Host:** `spark-bd9f` (NVIDIA DGX Spark, GB10, aarch64)
**Kernel:** `6.17.0-1014-nvidia`, **Driver:** `580.142`, **CUDA:** `13.0`, **BIOS:** `5.36_0ACUM018`
**First diagnosed:** 2026-06-08

## TL;DR

The box keeps hard-hanging during DREAMPlace `PlaceDB` construction on large benchmarks (confirmed on `superblue15`). Root cause is **NVIDIA Xid 31 (GPU MMU fault) on the GRAPHICS engine**, which on GB10's unified-memory architecture can take the whole interconnect down before the kernel can flush logs — so reboots show up in `wtmp` as `crash` with no smoking gun in the current boot's journal.

**Workaround:** keep `PlaceDB(p)` on CPU; only move to GPU for the actual placement step.

## Evidence

### 1. Reboot pattern (`last -x`)
Multiple short uptimes (2–4 h) ending in `crash` markers on Jun 5–6. No `shutdown` records for those — unclean.

### 2. Xid 31 storms in `kern.log.1`
Dozens of entries on 2026-05-21, all the same shape:

```
NVRM: Xid (PCI:000f:01:00): 31, pid=<...>, name=bench_run, channel 0x10,
  MMU Fault: ENGINE GRAPHICS GPC<n> GPCCLIENT_T1_<n> faulted @ 0x3_2f<...>000.
  Fault is of type FAULT_PTE ACCESS_TYPE_VIRT_READ
```

Xid 31 = GPU tried to read an unmapped virtual address (GPU-side segfault).
Fault virtual addresses cluster in `0x3_2f0xx000`–`0x3_2f3xx000` — looks like a contiguous region of dangling/early-freed device pointers, not random corruption.

### 3. Apport crash on 2026-06-06 21:54
`/var/crash/_usr_bin_python3.12.1007.crash` — Signal 6 (SIGABRT), 885 MB core. `ProcCmdline`:

```python
from covenant.harness.run_placement import load_dreamplace
Params, PlaceDB, _ = load_dreamplace()
p = Params.Params(); p.load('benchmarks_routability/superblue15/superblue15.json')
p.aux_input = 'benchmarks_routability/superblue15/superblue15.aux'
db = PlaceDB.PlaceDB(); db(p)   # <-- died here, took the box with it
```

Journal for that boot stops at 21:55:16 mid-sentence — consistent with a hard GPU-side hang that prevented log flush. No OOM, no panic, no Xid line in *that* boot's journal (the May 21 Xid logs survived because the May 21 hangs were less severe).

### 4. What's NOT broken
- NVMe: 0 critical warnings, 0 media errors, 0% endurance used, 41 °C
- RAM: 4 / 121 GB used at diagnosis time; no OOM in any boot
- Thermal: CPU 41–43 °C, GPU 42 °C idle
- Disk: 17 % of 3.7 TB
- No MCE, no hung-task, no softlockup, no NVMe errors anywhere in logs

## Suspected root cause

DREAMPlace's `PlaceDB.__call__` (`db(p)`) does heavy C++/pybind work that allocates and copies large structures to the GPU during DB construction. On the GB10 / aarch64 / driver 580.142 / CUDA 13.0 combo this is hitting a GPU MMU fault — most likely either:

- A DREAMPlace CUDA path that was built against a different CUDA toolkit than the running driver, leaving a kernel referencing stale or never-mapped virtual addresses, **or**
- A unified-memory ownership/migration bug surfaced only on GB10's shared-memory model (this is new hardware, the Xid pattern doesn't match anything in DREAMPlace's pre-Spark issue tracker as of writing).

The May 21 Xid 31s named `bench_run`; the Jun 6 SIGABRT came from the same `load_dreamplace()` code path — same component, same fault class.

## Mitigations

### Immediate: keep DB construction on CPU
In any harness that calls `db(p)`, force CPU before the call:

```python
p.gpu = 0
db = PlaceDB.PlaceDB()
db(p)
# only after db is fully built:
p.gpu = 1   # if/when you need GPU for the actual placer
```

Most of the Xid faults come from inside the DB build, not the optimizer — moving just that step to CPU should keep the box up.

### Bracket benchmark size
Verify with progressively larger benchmarks before touching `superblue15`:

1. `adaptec1` (~210k cells) — should always pass
2. `superblue1` (~850k cells) — passes on Spark today?
3. `superblue15` (~2M cells) — the known-bad case

If 1–2 pass and 3 fails, it's a size threshold inside DREAMPlace's CUDA paths, not a driver bug.

### Make faults surface synchronously
Run with `CUDA_LAUNCH_BLOCKING=1` and `cuda-gdb` attached when reproducing — async kernel launches hide which call site triggered the MMU fault.

```
CUDA_LAUNCH_BLOCKING=1 python -m covenant.harness.run_placement <args>
```

### Verify DREAMPlace was built against the running CUDA
Check `dreamplace_build/`:

```
nvcc --version    # should match driver's CUDA 13.0
ldd <built .so> | grep -i cuda
```

A toolkit/driver mismatch is the most common Xid 31 generator. If `dreamplace_ext` was built against a CUDA older than 13.0, rebuild it.

## Reproducing the diagnosis

```bash
# Recent reboot pattern
last -x | head -25

# Xid / hardware events (needs sudo for /var/log/kern.log*)
sudo grep -iE 'xid|nvrm|hardware error|mce|panic' /var/log/kern.log.1 | tail -50

# Unpack the apport crash
sudo apport-unpack /var/crash/_usr_bin_python3.12.1007.crash /tmp/crash
cat /tmp/crash/ProcCmdline
cat /tmp/crash/Signal

# SSD health
sudo nvme smart-log /dev/nvme0 | grep -E 'critical|percentage|temperature|media_errors'

# Currently-resident GPU processes
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

## Findings (2026-06-08 follow-up)

All four open questions probed via `scripts/probe_placedb_crash.py`; raw logs in
`/tmp/sb*_gpu*.{out,err}` during the diagnosis session.

### Q1 — Workaround confirmed on `superblue15`
`PlaceDB(p)` with `p.gpu = 0` forced before the call completes cleanly on
`superblue15` in **12.4 s**:

```
{"benchmark": "superblue15", "forced_gpu": 0, "json_gpu": 1,
 "cuda_launch_blocking": "1", "status": "ok", "elapsed_s": 12.443,
 "num_nodes": 1552119, "num_nets": 1080409, "num_movable_nodes": 829614}
```

No Xid in `dmesg`, no reboot. The CPU-DB / GPU-placer split is a viable
permanent workaround until the GB10 MMU-fault root cause is fixed upstream.

### Q2 — DREAMPlace is built against CUDA 13.0 (matches driver)
No toolkit/driver mismatch:

- `dreamplace_build/CMakeCache.txt`: `CUDA_VERSION:STRING=13.0`
- `ldd .../move_boundary_cuda.cpython-312-aarch64-linux-gnu.so` → `libcudart.so.13`
- Driver 580.142 ships with CUDA 13.0; `nvcc --version` reports `V13.0.88`

This rules out the "kernel built against a different CUDA than the driver"
hypothesis for Xid 31. The fault is not from a stale-toolkit cudart shim.

### Q3 — A newer Spark driver exists, no announced fix
- A June 2026 NVIDIA Developer Forum thread is on driver **580.159.03** (newer
  than our 580.142). Upgrade path on Spark is OTA, not documented for manual
  install in the post.
- A separate forum thread titled *"Recurring Xid 31 (MMU FAULT_PDE) on DGX
  Spark / GB10 / SM121"* (April 2026) shows the same fault class on **the same
  driver we're on (580.142)**. No NVIDIA-supplied fix, no driver release
  identified as resolving it. Reported workarounds (lower mem-fraction,
  smaller chunked-prefill, `--disable-cuda-graph`) are SGLang-specific and
  do not apply to DREAMPlace.

Conclusion: bump to 580.159.03 if/when an OTA arrives, but don't expect it to
fix Xid 31 on GB10 — keep the CPU-DB workaround until NVIDIA ships a fix
that explicitly calls out the GB10 GRAPHICS-engine MMU path.

### Q4 — Crash is size/workload-dependent, not always-on
Same probe, same harness, same driver, opposite ends of the size bracket:

| Benchmark    | `p.gpu` | nodes  | nets    | result           |
|--------------|---------|--------|---------|------------------|
| `superblue1` | 0       | 2.15 M | 0.82 M  | ok, 9.5 s        |
| `superblue1` | 1       | 2.15 M | 0.82 M  | ok, 9.85 s       |
| `superblue15`| 0       | 1.55 M | 1.08 M  | ok, 12.4 s       |
| `superblue15`| 1       | 1.55 M | 1.08 M  | **hangs (prior crash, not re-tested)** |

`adaptec1` is not present on this machine; smallest available is `superblue1`.
`superblue1` passes cleanly on the *stock* `gpu=1` path — the bug is not
"DREAMPlace's CUDA path is universally broken on GB10". It triggers on
`superblue15` specifically.

Note that `superblue15` has FEWER total nodes than `superblue1` but ~30 %
more nets. The threshold likely tracks net count, fixed/movable ratio, or
some other PlaceDB allocation that scales non-linearly with topology — NOT
raw cell count. Worth bisecting `superblue{2,4,5,10,12,14}` next if the
exact trigger becomes important; for now the workaround is enough.

The `superblue15` + `gpu=1` cell is intentionally left untested in this
follow-up: the prior crash already establishes that data point, and
re-triggering it risks another unclean reboot.

## Open questions (remaining)

- What's the exact `PlaceDB` allocation that scales past the GB10 GRAPHICS
  GPC MMU's mapped range? (Would need a smaller repro than a full PlaceDB
  call, or a `cuda-gdb` trace at the moment of fault.)
- Has anyone else hit this with DREAMPlace on GB10 (or is `bench_run` on
  May 21 our own harness)? Worth filing on the DREAMPlace tracker once we
  have a non-superblue minimal repro.

## Change log

- **2026-06-08** — Initial diagnosis after 6+ unclean reboots between Jun 5–7. Identified Xid 31 pattern from May 21 logs, linked to Jun 6 SIGABRT during `superblue15` `PlaceDB` load.
- **2026-06-08** — All four open questions probed. Workaround (`p.gpu=0` before `db(p)`) confirmed on `superblue15` (Q1). DREAMPlace verified built against matching CUDA 13.0 (Q2). Newer driver 580.159.03 exists but no announced Xid 31 fix (Q3). `superblue1` passes cleanly on `gpu=1`, so the crash is size/workload-dependent (Q4). Added `scripts/probe_placedb_crash.py` as the reproducible probe used here.
