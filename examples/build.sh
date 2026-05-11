#!/usr/bin/env bash
# build.sh -- Installs onnx_light locally and builds all standalone examples
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

mkdir -p "${EXAMPLES_BUILD_ROOT}"

echo "=== Building examples (${BUILD_TYPE}) ==="
for example_dir in "${SCRIPT_DIR}"/*; do
    if [ ! -d "${example_dir}" ] || [ ! -f "${example_dir}/build.sh" ]; then
        continue
    fi

    example_name="$(basename "${example_dir}")"
    example_build_dir="${EXAMPLES_BUILD_ROOT}/${example_name}"

    echo "--- Building ${example_name} ---"
    CMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        bash "${example_dir}/build.sh" \
        "${INSTALL_PREFIX}" \
        "${LIB_BUILD_DIR}" \
        "${example_build_dir}"

done
echo "Built examples in: ${EXAMPLES_BUILD_ROOT}"
