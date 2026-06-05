# DGX Spark (GB10) Bring-up Guide

The Spark is **Grace + Blackwell: aarch64, not x86** — everything compiled
must be rebuilt there. Compute capability is **12.1 (sm_121)**; sm_120
binaries are minor-version compatible, so building for 12.0 works too.
DGX OS is Ubuntu-based with the CUDA toolkit usually preinstalled
(`nvcc --version`; expect CUDA 12.8+ or 13.x — our cub/CCCL fix covers
CUDA ≥ 12).

Written 2026-06-05 after the RTX 3060 bring-up; see NOTES.md for the
lessons this encodes.

## 1. Clone (now safe — everything is pushed)

```bash
git clone --recurse-submodules https://github.com/themoddedcube/evoplace
cd evoplace
git -C vendor/dreamplace checkout evoplace-hooks   # branch with hook patches
```

The submodule must be on `evoplace-hooks` (4 patches over upstream:
PlaceObj gamma/lambda hooks + deferred net-weights, cub CCCL fix,
NumPy 2.0 fix). If the submodule SHA ever fails to resolve, the patch
series is archived at `dreamplace_ext/patches/` — apply onto upstream
master with `git am`.

## 2. System deps

```bash
sudo apt-get update && sudo apt-get install -y \
    cmake libboost-all-dev zlib1g-dev libomp-dev bison flex tcl tcl-dev \
    python3-venv python3-dev
```

Do NOT install any driver or the `cuda` metapackage — DGX OS manages those.

## 3. Python venv + PyTorch (aarch64 CUDA wheels)

PyTorch ships aarch64 CUDA wheels from cu128 onward (Blackwell sm_120+
support requires cu128 minimum). Match the wheel to the installed toolkit
major.minor where possible:

```bash
python3 -m venv ~/evoplace_venv
source ~/evoplace_venv/bin/activate
# pick ONE, matching `nvcc --version`:
pip install torch --index-url https://download.pytorch.org/whl/cu128
# pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt -r vendor/dreamplace/requirements.txt
python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# expect: ('GB10' or similar, (12, 1))
```

Alternative: NVIDIA NGC PyTorch container (`nvcr.io/nvidia/pytorch`) —
preferred if native wheel/toolkit mismatches bite; build DREAMPlace inside.

## 4. Build DREAMPlace

```bash
CXX_ABI=$(python3 -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)")
mkdir -p ~/dreamplace_build && cd ~/dreamplace_build
cmake ~/evoplace/vendor/dreamplace \
    -DCMAKE_INSTALL_PREFIX=~/evoplace/vendor/dreamplace/install \
    -DCMAKE_CXX_ABI=$CXX_ABI \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_CUDA_ARCHITECTURES=12.0
make -j$(nproc) && make install
```

Gotchas from the 3060 bring-up that may recur:
- `cuda_select_nvcc_arch_flags` may not recognize `12.0`/`12.1` by name.
  If configure fails on it, OMIT `-DCMAKE_CUDA_ARCHITECTURES` entirely —
  DREAMPlace's CMakeLists manually appends `sm_120` gencode for CUDA ≥ 12.8.
- If `CUDA_CUDA_LIBRARY NOTFOUND`: add
  `-DCUDA_CUDA_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so`.
- `-DPYTHON_EXECUTABLE` is the all-caps legacy name; `Python_EXECUTABLE`
  is silently ignored.
- The PlaceObj/PlaceDB patches are committed in the submodule, so
  `make install` propagates them automatically. No manual copy step.

## 5. Benchmarks

```bash
cd /tmp && wget http://www.cerc.utexas.edu/~zixuan/ispd2015dp.tar.xz
tar -xf ispd2015dp.tar.xz
# flat layout: benchmarks/<short_name>/ <- ispd2015/mgc_<short_name>/
for d in /tmp/ispd2015/mgc_*; do
  short=$(basename "$d" | sed 's/^mgc_//')
  mv "$d" ~/evoplace/benchmarks/"$short"
done
```

JSON configs for all 11 circuits are already committed in `benchmarks/*/`.

## 6. Verify + re-baseline (REQUIRED before any evolution run)

```bash
cd ~/evoplace && source ~/evoplace_venv/bin/activate
python3 -m pytest tests/ -q                      # expect 114/114
python3 experiments/exp00_baseline/run.py --suite small
```

Baselines are hardware-specific. After exp00, re-measure the stage-matched
cascade baselines (50/300 iters, default schedule, no hooks) and update
BOTH `BASELINE_HPWL` and `BASELINE_HPWL_STAGES` in
`evolve/evaluator_wrapper.py`:

```bash
python3 - <<'EOF'
from pathlib import Path
from evaluator.run_placement import run_placement
for bench in ['fft_1', 'fft_2']:
    for iters in [50, 300]:
        r = run_placement(Path(f'benchmarks/{bench}'), Path(f'/tmp/sb/{bench}_{iters}'),
                          max_iterations=iters)
        print(bench, iters, f"{r.metrics['hpwl']:.6e}")
EOF
```

Sanity gate (the hook-liveness check that was missing for the entire CPU
campaign): evaluate the seed through the real cascade and confirm
`norm_hpwl ≈ 1.0 ± 0.01`:

```bash
python3 -c "
from evolve.evaluator_wrapper import evaluate
r = evaluate('evolve/initial_program.py')
print(r['metrics']['normalized_hpwl'])"
```

If it's ~1.0 the hooks are live and calibrated. If it's wildly off (or all
candidates later score identically), STOP — something in the chain is dead.

## 7. RTX 3060 reference numbers (for cross-machine comparison)

| Measurement | Value |
|---|---|
| fft_1 converged (seed 42) | HPWL 2.180e6, ovf 0.082, 29.4 s |
| fft_2 converged | HPWL 1.921e6, ovf 0.070, 12.2 s |
| fft_1 stage baselines (50/300) | 4.960e6 / 5.420e6 |
| fft_2 stage baselines (50/300) | 3.248e6 / 3.906e6 |
| Cascade cost per candidate | ~5 s stage-0 cull, ~70 s full pass |
| Exp 1 fitness noise floor (single seed) | σ ≈ 0.15% norm_hpwl |

## 8. What the Spark unlocks (next experiments)

- Full `ispd2015_no_region` suite incl. superblue (128 GB unified memory)
- Multi-seed fitness averaging — addresses the 0.15% noise floor that
  capped Exp 1 signal on the 3060
- Multi-design fitness (Klein-4 overfit mitigation)
- Parallel candidate evaluation (multiple placements resident at once)
- Exp 3 (GNN warm-init) / Exp 4 (timing surrogate) training
