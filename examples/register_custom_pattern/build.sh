#!/usr/bin/env bash
# Installs onnx_light locally and builds the standalone custom-pattern example.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INSTALL_PREFIX="${1:-${REPO_ROOT}/build/install-register-custom-pattern}"
LIB_BUILD_DIR="${2:-${REPO_ROOT}/build/register-custom-pattern-lib}"
EXAMPLE_BUILD_DIR="${3:-${REPO_ROOT}/build/register-custom-pattern-example}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
PARALLEL_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)}"

cmake -S "${REPO_ROOT}" -B "${LIB_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DONNX_LIGHT_BUILD_PYTHON=OFF \
    -DONNX_LIGHT_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
cmake --build "${LIB_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${PARALLEL_JOBS}"
cmake --install "${LIB_BUILD_DIR}" --config "${BUILD_TYPE}"

cmake -S "${SCRIPT_DIR}" -B "${EXAMPLE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}"
cmake --build "${EXAMPLE_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${PARALLEL_JOBS}"

echo "Example binary: ${EXAMPLE_BUILD_DIR}/register_custom_pattern"
