#!/usr/bin/env bash
# build.sh -- installs onnx_light locally and builds the standalone
# aidge_onnx_light example against that install.
#
# Eclipse Aidge is treated as optional: if its CMake packages
# (``aidge_core`` and ``aidge_onnx``) are available on the prefix path
# pointed at by ``AIDGE_PREFIX`` (or ``CMAKE_PREFIX_PATH``), the example
# is built with the Aidge integration enabled; otherwise the example is
# built in the onnx-light-only mode.
#
# Usage (run from the repository root or from this directory):
#   bash examples/aidge_onnx_light/build.sh \
#        [install-prefix] [lib-build-dir] [example-build-dir]
#
# Environment overrides:
#   AIDGE_PREFIX        Extra prefix appended to CMAKE_PREFIX_PATH so the
#                       example can find a manually installed Aidge.
#   CMAKE_BUILD_TYPE    Build type (default: Release).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INSTALL_PREFIX="${1:-${REPO_ROOT}/build/install-aidge-onnx-light}"
LIB_BUILD_DIR="${2:-${REPO_ROOT}/build/aidge-onnx-light-lib}"
EXAMPLE_BUILD_DIR="${3:-${REPO_ROOT}/build/aidge-onnx-light-example}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
PARALLEL_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)}"

PREFIX_PATH="${INSTALL_PREFIX}"
if [[ -n "${AIDGE_PREFIX:-}" ]]; then
    PREFIX_PATH="${PREFIX_PATH};${AIDGE_PREFIX}"
fi

echo "=== Step 1: configure and build onnx_light (${BUILD_TYPE}) ==="
cmake -S "${REPO_ROOT}" -B "${LIB_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DONNX_LIGHT_BUILD_PYTHON=OFF \
    -DONNX_LIGHT_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
cmake --build "${LIB_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${PARALLEL_JOBS}"
cmake --install "${LIB_BUILD_DIR}" --config "${BUILD_TYPE}"

echo "=== Step 2: configure and build aidge_onnx_light (${BUILD_TYPE}) ==="
cmake -S "${SCRIPT_DIR}" -B "${EXAMPLE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_PREFIX_PATH="${PREFIX_PATH}"
cmake --build "${EXAMPLE_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${PARALLEL_JOBS}"

echo
echo "Example binary:"
echo "  ${EXAMPLE_BUILD_DIR}/aidge_onnx_light"
