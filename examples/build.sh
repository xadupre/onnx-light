#!/usr/bin/env bash
# build.sh -- installs onnx_light locally and builds all standalone examples
# in the examples directory against that install.
#
# Usage (run from the repository root or from examples/):
#   bash examples/build.sh [install-prefix] [lib-build-dir] [examples-build-root]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INSTALL_PREFIX="${1:-${REPO_ROOT}/build/install-examples}"
LIB_BUILD_DIR="${2:-${REPO_ROOT}/build/examples-lib}"
EXAMPLES_BUILD_ROOT="${3:-${REPO_ROOT}/build/examples}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

echo "=== Step 1: configure and build onnx_light (${BUILD_TYPE}) ==="
cmake -S "${REPO_ROOT}" -B "${LIB_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DONNX_LIGHT_BUILD_PYTHON=OFF \
    -DONNX_LIGHT_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
cmake --build "${LIB_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel
cmake --install "${LIB_BUILD_DIR}" --config "${BUILD_TYPE}"

mkdir -p "${EXAMPLES_BUILD_ROOT}"

echo "=== Step 2: configure and build examples (${BUILD_TYPE}) ==="
for example_dir in "${SCRIPT_DIR}"/*; do
    if [ ! -d "${example_dir}" ] || [ ! -f "${example_dir}/CMakeLists.txt" ]; then
        continue
    fi

    example_name="$(basename "${example_dir}")"
    example_build_dir="${EXAMPLES_BUILD_ROOT}/${example_name}"

    echo "--- Building ${example_name} ---"
    cmake -S "${example_dir}" -B "${example_build_dir}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}"
    cmake --build "${example_build_dir}" --config "${BUILD_TYPE}" --parallel

done
echo "Built examples in: ${EXAMPLES_BUILD_ROOT}"
