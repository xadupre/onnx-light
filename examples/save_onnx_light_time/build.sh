#!/usr/bin/env bash
# build.sh -- installs onnx_light locally and builds the standalone
# save_onnx_light_time example against that install.
#
# Usage (run from the repository root or from this directory):
#   bash examples/save_onnx_light_time/build.sh [install-prefix] [lib-build-dir] [example-build-dir]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INSTALL_PREFIX="${1:-${REPO_ROOT}/build/install-save-onnx-light-time}"
LIB_BUILD_DIR="${2:-${REPO_ROOT}/build/save-onnx-light-time-lib}"
EXAMPLE_BUILD_DIR="${3:-${REPO_ROOT}/build/save-onnx-light-time-example}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
PARALLEL_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)}"

echo "=== Step 1: configure and build onnx_light (${BUILD_TYPE}) ==="
cmake -S "${REPO_ROOT}" -B "${LIB_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DONNX_LIGHT_BUILD_PYTHON=OFF \
    -DONNX_LIGHT_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
cmake --build "${LIB_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${PARALLEL_JOBS}"
cmake --install "${LIB_BUILD_DIR}" --config "${BUILD_TYPE}"

echo "=== Step 2: configure and build save_onnx_light_time (${BUILD_TYPE}) ==="
cmake -S "${SCRIPT_DIR}" -B "${EXAMPLE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}"
cmake --build "${EXAMPLE_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${PARALLEL_JOBS}"

echo
echo "Example binary:"
echo "  ${EXAMPLE_BUILD_DIR}/save_onnx_light_time"
