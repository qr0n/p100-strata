#!/usr/bin/env bash
# Build the bundled, patched Strata engine for the Tesla P100 (sm_60).
#
#   ./build.sh                 build strata/build-p100/strata for the GPUs in this machine
#   CUDA_ARCH=60 ./build.sh    pick the architecture yourself (P100 = 60)
#   JOBS=8 ./build.sh          limit parallel compile jobs (default: all cores)
#
# Needs CUDA 12.x (CUDA 13 dropped Pascal) and cmake. CUDA 12.4's nvcc rejects
# gcc newer than 13, so gcc-13 is used as the host compiler when it is installed.
# Only sm_60 is measured here; other Pascal/Volta cards build, but are untested.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
src=$here/strata
jobs=${JOBS:-$(nproc)}

die() { echo "build.sh: $*" >&2; exit 1; }

if [[ -z ${CUDA_ARCH:-} ]] && command -v nvidia-smi >/dev/null; then
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | tr -d . | sort -u | sed 's/$/-real/' | paste -sd';' || true)
fi
arch=${CUDA_ARCH:-60-real}
echo "build.sh: CUDA_ARCH=$arch" >&2

command -v cmake >/dev/null || die "cmake not found"
command -v nvcc  >/dev/null || die "nvcc not found (install CUDA 12.x and put its bin/ on PATH)"

cuda_major=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
[[ $cuda_major == 12 ]] || die "CUDA $cuda_major found; Pascal needs CUDA 12.x (13 removed sm_60/sm_61)"

compilers=()
if command -v gcc-13 >/dev/null && command -v g++-13 >/dev/null; then
    compilers=(-DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13
               -DCMAKE_CUDA_HOST_COMPILER=g++-13)
else
    echo "build.sh: gcc-13 not found, using the default compiler (CUDA 12.4 needs gcc <= 13)" >&2
fi

cmake -S "$src" -B "$src/build-p100" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_EXPERIMENTAL_SM60=ON \
    -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
    -DCMAKE_CUDA_ARCHITECTURES="$arch" \
    "${compilers[@]}"
cmake --build "$src/build-p100" -j "$jobs" --target strata

echo
echo "Built: $src/build-p100/strata"
echo "Serve: see README.md (configs/strata-atomic.json)"
