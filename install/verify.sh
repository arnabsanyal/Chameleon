#!/bin/bash

# Chameleon Environment Verifier
# Checks that the 'chameleon' conda environment has all required packages
# and dependencies correctly installed.
#
# Usage: bash verify.sh
# Exit code: 0 if all checks pass, 1 if any critical check fails.

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'   # No colour

PASS="${GREEN}[PASS]${NC}"
FAIL="${RED}[FAIL]${NC}"
WARN="${YELLOW}[WARN]${NC}"
INFO="${CYAN}[INFO]${NC}"

# Counters
n_pass=0
n_fail=0
n_warn=0

pass()  { echo -e "  $PASS  $1"; ((n_pass++)); }
fail()  { echo -e "  $FAIL  $1"; ((n_fail++)); }
warn()  { echo -e "  $WARN  $1"; ((n_warn++)); }
info()  { echo -e "  $INFO  $1"; }
header(){ echo -e "\n${BOLD}${CYAN}── $1 ──${NC}"; }

# Helper: check a Python import in the chameleon env
py_import() {
    local pkg="$1"
    local import_name="${2:-$1}"
    if conda run --name chameleon python -c "import ${import_name}" 2>/dev/null; then
        local ver
        ver=$(conda run --name chameleon python -c \
            "import ${import_name}; print(getattr(${import_name}, '__version__', 'installed'))" 2>/dev/null)
        pass "${pkg}  (${ver})"
    else
        fail "${pkg}  — import failed"
    fi
}

# Helper: check a Python import and print a warning (non-critical)
py_import_warn() {
    local pkg="$1"
    local import_name="${2:-$1}"
    if conda run --name chameleon python -c "import ${import_name}" 2>/dev/null; then
        local ver
        ver=$(conda run --name chameleon python -c \
            "import ${import_name}; print(getattr(${import_name}, '__version__', 'installed'))" 2>/dev/null)
        pass "${pkg}  (${ver})"
    else
        warn "${pkg}  — not found (optional or may need manual build)"
    fi
}

# ── Banner ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}#############################################################${NC}"
echo -e "${BOLD}#         Chameleon Environment Verification                #${NC}"
echo -e "${BOLD}#############################################################${NC}"
echo ""
echo "Verifying conda environment: 'chameleon'"
echo "Date: $(date)"
echo ""

# ── 1. Conda environment ───────────────────────────────────────────────────────
header "1. Conda Environment"

if conda info --envs 2>/dev/null | grep -q '\bchameleon\b'; then
    pass "Conda environment 'chameleon' exists"
    ENV_PATH=$(conda info --envs 2>/dev/null | grep '\bchameleon\b' | awk '{print $NF}')
    info "Path: ${ENV_PATH}"
else
    fail "Conda environment 'chameleon' NOT found — run install.sh first"
    echo ""
    echo -e "${RED}Cannot continue without the conda environment. Exiting.${NC}"
    exit 1
fi

# ── 2. Python version ──────────────────────────────────────────────────────────
header "2. Python Version"

PY_VER=$(conda run --name chameleon python --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [[ "$PY_MAJOR" == "3" && "$PY_MINOR" == "12" ]]; then
    pass "Python ${PY_VER}  (expected 3.12.x)"
else
    fail "Python ${PY_VER}  (expected 3.12.x)"
fi

# ── 3. CUDA & GPU ──────────────────────────────────────────────────────────────
header "3. CUDA & GPU"

# nvidia-smi
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    pass "nvidia-smi available"
    info "GPU   : ${GPU_NAME}"
    info "Driver: ${DRIVER_VER}"
else
    warn "nvidia-smi not found — GPU drivers may not be installed"
fi

# PyTorch CUDA
CUDA_AVAIL=$(conda run --name chameleon python -c \
    "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [[ "$CUDA_AVAIL" == "True" ]]; then
    CUDA_VER=$(conda run --name chameleon python -c \
        "import torch; print(torch.version.cuda)" 2>/dev/null)
    GPU_COUNT=$(conda run --name chameleon python -c \
        "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    pass "torch.cuda.is_available() = True"
    info "CUDA version : ${CUDA_VER}"
    info "GPU count    : ${GPU_COUNT}"
else
    fail "torch.cuda.is_available() = False — CUDA not accessible from PyTorch"
fi

# CUDA toolkit
if conda run --name chameleon nvcc --version &>/dev/null; then
    NVCC_VER=$(conda run --name chameleon nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
    pass "nvcc (CUDA toolkit)  (${NVCC_VER})"
else
    warn "nvcc not found — cuda-toolkit may not be on PATH in this env"
fi

# ── 4. Core deep learning packages ────────────────────────────────────────────
header "4. Core Deep Learning Packages"

py_import "torch"
py_import "diffusers"
py_import "transformers"
py_import "accelerate"
py_import "safetensors"

# ── 5. Requirements.txt packages ──────────────────────────────────────────────
header "5. Requirements.txt Packages"

py_import "Pillow" "PIL"
py_import "tqdm"
py_import "numpy"
py_import "clean-fid" "cleanfid"
py_import "imageio"
py_import "imageio-ffmpeg" "imageio_ffmpeg"
py_import "opencv-python" "cv2"
py_import "tomli"
py_import "tomli_w"
py_import_warn "xformers"    # Optional

# ── 6. Quantization & MX libraries ───────────────────────────────────────────
header "6. Quantization & MX Libraries"

py_import "bitsandbytes"
py_import "transformer-engine" "transformer_engine"
py_import "torchao"


# ── 7. MixDQ ──────────────────────────────────────────────────────────────────
header "7. MixDQ (CUDA Kernels)"

py_import_warn "mixdq-extension" "mixdq_extension"
py_import "ortools"

# ── 8. Additional packages ────────────────────────────────────────────────────
header "8. Additional Packages"

py_import "tiktoken"
py_import "sentencepiece"
py_import "beautifulsoup4" "bs4"
py_import "ftfy"

# ── 9. Diffusers patch verification ───────────────────────────────────────────
header "9. Diffusers torchao Patch"

SITE_PACKAGES=$(conda run --name chameleon python -c \
    "import site; print(site.getsitepackages()[0])" 2>/dev/null)
PATCH_FILE="${SITE_PACKAGES}/diffusers/quantizers/torchao/torchao_quantizer.py"

if [ -f "$PATCH_FILE" ]; then
    # Check that logger.warning and logger.debug have been replaced
    if grep -q "logger\.warning\|logger\.debug" "$PATCH_FILE"; then
        fail "torchao_quantizer.py — logger.warning / logger.debug still present (patch not applied)"
    else
        pass "torchao_quantizer.py — logger calls patched to print()"
    fi
else
    warn "torchao_quantizer.py not found at expected path — diffusers version may differ"
    info "Expected: ${PATCH_FILE}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── 10. Codebase import checks ────────────────────────────────────────────────
header "10. Chameleon Codebase Imports"

CHAMELEON_MODULES=(
    "src/sdxl-chameleon/generation:chameleon_quant:ChameleonQuantizer"
    "src/chameleon-dit/generation:chameleon_dit_quant:ChameleonDiTQuantizer"
)

for entry in "${CHAMELEON_MODULES[@]}"; do
    dir=$(echo "$entry" | cut -d: -f1)
    mod=$(echo "$entry" | cut -d: -f2)
    cls=$(echo "$entry" | cut -d: -f3)
    full_path="${REPO_ROOT}/${dir}"
    if conda run --name chameleon python -c \
        "import sys; sys.path.insert(0,'${full_path}'); from ${mod} import ${cls}; print('OK')" \
        2>/dev/null | grep -q OK; then
        pass "${mod}.${cls}"
    else
        fail "${mod}.${cls}  (${dir})"
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}#############################################################${NC}"
echo -e "${BOLD}#                   VERIFICATION SUMMARY                   #${NC}"
echo -e "${BOLD}#############################################################${NC}"
echo ""
echo -e "  ${GREEN}Passed : ${n_pass}${NC}"
echo -e "  ${RED}Failed : ${n_fail}${NC}"
echo -e "  ${YELLOW}Warnings: ${n_warn}${NC}"
echo ""

if [ "$n_fail" -eq 0 ] && [ "$n_warn" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  All checks passed. Environment is ready.${NC}"
elif [ "$n_fail" -eq 0 ]; then
    echo -e "${YELLOW}${BOLD}  All critical checks passed. Review warnings above.${NC}"
else
    echo -e "${RED}${BOLD}  ${n_fail} check(s) failed. Review failures above and re-run install.sh.${NC}"
fi
echo ""

[ "$n_fail" -eq 0 ]   # exit 0 on full pass, exit 1 if any failures
