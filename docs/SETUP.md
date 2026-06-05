# Setup

> For **DGX Spark (GB10 / aarch64 / CUDA 13)** bring-up, follow
> [SPARK_SETUP.md](SPARK_SETUP.md) instead — it covers the arch-specific
> CMake fixes and dependency differences.

## Requirements

- Linux or WSL2 Ubuntu 22.04+ (tested on Ubuntu 24.04 in WSL2 and DGX OS)
- Python 3.10+
- CMake ≥ 3.14, GCC ≥ 11, Boost ≥ 1.55
- PyTorch ≥ 2.0 (CPU sufficient for development; GPU required for convergent runs)
- CUDA 12.x/13.x toolkit for GPU placement (tested: CUDA 12.6 on RTX 3060 sm_86; CUDA 13.0 on GB10 sm_121)
- [Claude Code CLI](https://claude.ai/code) for LLM-guided evolution (no API key needed)

## GPU Setup (WSL2 / Ubuntu 24.04)

The NVIDIA *driver* lives on the Windows side (exposed via `/usr/lib/wsl/lib`) — never install a Linux driver inside WSL. Only the CUDA *toolkit* is needed:

```bash
# 1. System packages
sudo apt-get update && sudo apt-get install -y \
    cmake libboost-all-dev zlib1g-dev libomp-dev bison flex libfl-dev \
    tcl tcl-dev python3.12-venv python3-dev

# 2. CUDA toolkit 12.6 (toolkit only — NOT the `cuda` metapackage, which pulls a driver)
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-6
export PATH=/usr/local/cuda-12.6/bin:$PATH

# 3. Python venv + PyTorch (wheel CUDA version must match the toolkit: cu126 ↔ 12.6)
python3 -m venv ~/evoplace_venv
source ~/evoplace_venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

> **Note on paths with spaces**: DREAMPlace's C++ parser splits file paths on whitespace. If your project directory contains spaces, create a symlink (e.g. `ln -s "/path/with spaces/evoplace" ~/evoplace`) and work through it.

## Manual Build (any Linux)

```bash
# 1. Install system deps
sudo apt-get install -y cmake libboost-all-dev zlib1g-dev libomp-dev bison flex libfl-dev

# 2. Init DREAMPlace submodules (or clone evoplace with --recurse-submodules)
cd vendor/dreamplace
git submodule update --init --recursive
git checkout evoplace-hooks   # branch carrying the hook + build patches

# 3. Build (detect PyTorch ABI automatically)
CXX_ABI=$(python3 -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)")
mkdir -p ~/dreamplace_build && cd ~/dreamplace_build
cmake /path/to/evoplace/vendor/dreamplace \
    -DCMAKE_INSTALL_PREFIX=/path/to/evoplace/vendor/dreamplace/install \
    -DCMAKE_CXX_ABI=$CXX_ABI \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_CUDA_ARCHITECTURES=8.6   # match your GPU; omit on CUDA ≥ 13 (handled by CMakeLists)
make -j$(nproc) && make install
```

The hook patches (`PlaceObj.py`, `PlaceDB.py`, CUDA 13 build fixes) are
committed on the submodule's `evoplace-hooks` branch, so `make install`
propagates them automatically — no manual copy step. Verify after any
rebuild with the sanity gate in [RUNNING.md](RUNNING.md).

## Benchmarks

Benchmark files are not committed (too large). Download:

```bash
# ISPD 2015 (no registration required) — downloads ~170 MB
cd /tmp && wget http://www.cerc.utexas.edu/~zixuan/ispd2015dp.tar.xz && tar -xf ispd2015dp.tar.xz
for d in /tmp/ispd2015/mgc_*; do
  short=$(basename "$d" | sed 's/^mgc_//')
  mv "$d" /path/to/evoplace/benchmarks/"$short"
done

# ICCAD 2015 (timing-driven, free academic registration)
# Register at http://iccad-contest.org/2015/ → download to benchmarks/iccad2015/
```

JSON configs for all circuits are already committed in `benchmarks/*/`.

**Circuits used:**

| Suite | Circuits | Primary use |
|-------|----------|-------------|
| `small` | `fft_1`, `fft_2` | Fast iteration during evolution |
| `ispd2015_no_region` | 11 circuits (fft, matrix_mult, superblue, ...) | Baselines |
| `iccad2015` | 6 circuits with timing constraints | Timing evaluation |
