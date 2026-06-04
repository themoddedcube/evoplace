#!/usr/bin/env bash
# WSL2 + DreamPlace build script for EvoPlace
# Run from the evoplace repo root:
#   bash scripts/setup_wsl.sh
#
# Two-phase: first run with sudo for apt packages, then without for Python/CMake.
# The script detects which phase is needed and skips completed steps.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$HOME/evoplace_venv"
BUILD_DIR="$HOME/dreamplace_build"          # native Linux fs — much faster than /mnt/c
INSTALL_PREFIX="$REPO_ROOT/vendor/dreamplace/install"
DREAMPLACE_DIR="$REPO_ROOT/vendor/dreamplace"
BENCHMARKS_DIR="$REPO_ROOT/benchmarks"

TORCH_VERSION="2.5.1"
PYTHON_BIN="$VENV_DIR/bin/python"

log() { echo -e "\033[1;32m[setup_wsl]\033[0m $*"; }
warn() { echo -e "\033[1;33m[setup_wsl]\033[0m $*"; }
die() { echo -e "\033[1;31m[setup_wsl]\033[0m $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Phase 1: apt packages (needs sudo)
# ---------------------------------------------------------------------------
phase_apt() {
    log "Phase 1: installing system packages..."
    sudo apt-get update -qq
    sudo apt-get install -y \
        cmake \
        build-essential \
        libboost-all-dev \
        zlib1g-dev \
        libomp-dev \
        libcairo2-dev \
        python3-dev \
        python3-venv \
        python3-pip \
        bison \
        flex \
        patool \
        git
    log "Phase 1 complete."
}

# ---------------------------------------------------------------------------
# Phase 2: Python venv + packages
# ---------------------------------------------------------------------------
phase_python() {
    log "Phase 2: setting up Python venv at $VENV_DIR ..."

    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"

    pip install --upgrade pip wheel setuptools

    # CPU-only PyTorch — matches DGX workflow (GPU build is identical, just different index URL)
    log "Installing PyTorch $TORCH_VERSION (CPU)..."
    pip install torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cpu

    # DreamPlace Python requirements (minus torch which we just installed)
    pip install \
        numpy \
        scipy \
        matplotlib \
        "cairocffi>=0.9.0" \
        "pkgconfig>=1.4.0" \
        "shapely>=1.7.0" \
        "pyunpack>=0.1.2" \
        "patool>=1.12" \
        "torch_optimizer==0.3.0" \
        "ncg_optimizer==0.2.2"

    # EvoPlace project dependencies
    if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
        pip install -r "$REPO_ROOT/requirements.txt"
    fi

    log "Phase 2 complete. Activate venv with: source $VENV_DIR/bin/activate"
}

# ---------------------------------------------------------------------------
# Phase 3: Build DreamPlace with CMake
# ---------------------------------------------------------------------------
phase_build() {
    log "Phase 3: building DreamPlace..."

    source "$VENV_DIR/bin/activate"

    # Detect PyTorch ABI (torch 2.5.x cpu wheels use ABI=0; GPU wheels may differ)
    CXX_ABI=$(python3 -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)" 2>/dev/null || echo "0")
    log "Detected _GLIBCXX_USE_CXX11_ABI=$CXX_ABI"

    # Initialize DreamPlace's own git submodules (pybind11, Limbo, etc.)
    # Note: --init without --recursive to skip Limbo's broken OpenBLAS sub-submodule
    log "Initializing DreamPlace submodules..."
    cd "$DREAMPLACE_DIR"
    git submodule update --init thirdparty/pybind11 thirdparty/munkres-cpp \
        thirdparty/cub thirdparty/OpenTimer thirdparty/Limbo thirdparty/HeteroSTA \
        2>&1 || true   # Limbo's OpenBLAS nested sub may fail — that's OK for CPU build

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake "$DREAMPLACE_DIR" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DCMAKE_CXX_ABI=$CXX_ABI \
        -DPython_EXECUTABLE="$PYTHON_BIN" \
        -DCMAKE_BUILD_TYPE=Release \
        2>&1 | tee cmake_configure.log

    log "CMake configure done. Starting build (this takes 5-15 min)..."
    make -j"$(nproc)" 2>&1 | tee make.log
    make install 2>&1 | tee make_install.log

    log "Phase 3 complete. DreamPlace installed to $INSTALL_PREFIX"
}

# ---------------------------------------------------------------------------
# Phase 4: Download ISPD 2015 benchmarks + create JSON configs
# ---------------------------------------------------------------------------
phase_benchmarks() {
    log "Phase 4: downloading ISPD 2005/2015 benchmarks..."

    source "$VENV_DIR/bin/activate"

    # DreamPlace's own downloader — pulls from UT Austin server
    cd "$DREAMPLACE_DIR"
    python benchmarks/ispd2005_2015.py || warn "Benchmark download failed — check network access to cerc.utexas.edu"

    # Create JSON configs in our benchmarks/ dir with absolute paths.
    # DreamPlace test configs use paths relative to DREAMPLACE_DIR; we resolve them.
    ISPD2015_SRC="$DREAMPLACE_DIR/benchmarks/ispd2015"
    TEMPLATE_DIR="$DREAMPLACE_DIR/test/ispd2015/lefdef"

    if [[ -d "$ISPD2015_SRC" ]]; then
        for circuit in fft_1 fft_2 fft_a fft_b matrix_mult_1 matrix_mult_2 matrix_mult_a superblue12 superblue14 superblue19; do
            TEMPLATE_JSON="$TEMPLATE_DIR/mgc_${circuit}.json"
            DST_DIR="$BENCHMARKS_DIR/$circuit"
            DST_JSON="$DST_DIR/${circuit}.json"

            if [[ ! -f "$TEMPLATE_JSON" ]]; then
                warn "  No template JSON for $circuit — skipping"
                continue
            fi

            mkdir -p "$DST_DIR"
            # Rewrite relative paths to absolute using python json edit
            python3 - "$TEMPLATE_JSON" "$DST_JSON" "$DREAMPLACE_DIR" <<'PYEOF'
import json, sys, os
src, dst, base = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    cfg = json.load(f)

def abs_path(v):
    return os.path.join(base, v) if not os.path.isabs(v) else v

for key in ("def_input", "verilog_input", "aux_input", "sdc_input", "detailed_place_engine"):
    if key in cfg:
        cfg[key] = abs_path(cfg[key])
for key in ("lef_input",):
    if key in cfg:
        cfg[key] = [abs_path(v) for v in (cfg[key] if isinstance(cfg[key], list) else [cfg[key]])]

with open(dst, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"  Created {dst}")
PYEOF
        done
    else
        warn "ISPD 2015 benchmarks not found at $ISPD2015_SRC"
        warn "Download manually: http://www.cerc.utexas.edu/~zixuan/ispd2015dp.tar.xz"
        warn "Then re-run: bash scripts/setup_wsl.sh benchmarks"
    fi

    log "Phase 4 done."
}

# ---------------------------------------------------------------------------
# Phase 5: Smoke test
# ---------------------------------------------------------------------------
phase_test() {
    log "Phase 5: smoke-testing the build..."

    source "$VENV_DIR/bin/activate"
    cd "$REPO_ROOT"

    python3 -c "
import sys
sys.path.insert(0, '$INSTALL_PREFIX')
import dreamplace.Params as Params
import dreamplace.PlaceDB as PlaceDB
import dreamplace.NonLinearPlace as NonLinearPlace
print('DreamPlace import: OK (Params, PlaceDB, NonLinearPlace)')
"
    log "Build is working. Run exp00 baseline:"
    log "  source $VENV_DIR/bin/activate"
    log "  cd $REPO_ROOT"
    log "  python experiments/exp00_baseline/run.py --suite small"
}

# ---------------------------------------------------------------------------
# Main: run all phases or a specific one
# ---------------------------------------------------------------------------
PHASE="${1:-all}"
case "$PHASE" in
    apt)        phase_apt ;;
    python)     phase_python ;;
    build)      phase_build ;;
    benchmarks) phase_benchmarks ;;
    test)       phase_test ;;
    all)
        phase_apt
        phase_python
        phase_build
        phase_benchmarks
        phase_test
        ;;
    *)
        echo "Usage: $0 [apt|python|build|benchmarks|test|all]"
        exit 1
        ;;
esac
