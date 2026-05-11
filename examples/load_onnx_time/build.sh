#!/usr/bin/env bash
# build.sh -- builds the standalone load_onnx_time example against the
# standard onnx C++ library.
#
# By default the script expects the system onnx C++ library to be installed.
# On Ubuntu/Debian:
#   sudo apt-get install -y libonnx-dev libprotobuf-dev
#
# Alternatively, set ONNX_GIT_TAG to a git tag or branch to clone and build
# onnx from source (protobuf-dev must still be installed separately):
#   ONNX_GIT_TAG=v1.17.0 bash examples/load_onnx_time/build.sh
#
# Usage (run from the repository root or from this directory):
#   bash examples/load_onnx_time/build.sh [install-prefix] [lib-build-dir] [example-build-dir]
#
# Arguments:
#   install-prefix    onnx install prefix when ONNX_GIT_TAG is set
#                     (default: build/install-load-onnx-time)
#   lib-build-dir     onnx build directory when ONNX_GIT_TAG is set
#                     (default: build/load-onnx-time-lib)
#   example-build-dir load_onnx_time build directory
#                     (default: build/load-onnx-time-example)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INSTALL_PREFIX="${1:-${REPO_ROOT}/build/install-load-onnx-time}"
LIB_BUILD_DIR="${2:-${REPO_ROOT}/build/load-onnx-time-lib}"
EXAMPLE_BUILD_DIR="${3:-${REPO_ROOT}/build/load-onnx-time-example}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

# Optional: set ONNX_GIT_TAG to a git tag or branch (e.g. v1.17.0) to clone
# and build onnx from source instead of using a system-installed package.
ONNX_GIT_TAG="${ONNX_GIT_TAG:-}"
ONNX_GIT_URL="${ONNX_GIT_URL:-https://github.com/onnx/onnx.git}"

CMAKE_PREFIX_PATH_ARG=""

if [ -n "${ONNX_GIT_TAG}" ]; then
    ONNX_SRC_DIR="${LIB_BUILD_DIR}/onnx-src"
    ONNX_BUILD_DIR="${LIB_BUILD_DIR}/onnx-build"

    echo "=== Step 1: clone onnx ${ONNX_GIT_TAG} ==="
    if [ ! -d "${ONNX_SRC_DIR}/.git" ]; then
        git clone --depth 1 --branch "${ONNX_GIT_TAG}" "${ONNX_GIT_URL}" "${ONNX_SRC_DIR}"
        git -C "${ONNX_SRC_DIR}" submodule update --init --recursive
    else
        echo "Source directory ${ONNX_SRC_DIR} already exists, skipping clone."
    fi

    echo "=== Step 2: build and install onnx ==="
    cmake -S "${ONNX_SRC_DIR}" -B "${ONNX_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DONNX_ML=ON \
        -DONNX_BUILD_TESTS=OFF \
        -DONNX_BUILD_BENCHMARKS=OFF \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
    cmake --build "${ONNX_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel
    cmake --install "${ONNX_BUILD_DIR}" --config "${BUILD_TYPE}"

    CMAKE_PREFIX_PATH_ARG="-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}"
fi

echo "=== Configure and build load_onnx_time (${BUILD_TYPE}) ==="
cmake -S "${SCRIPT_DIR}" -B "${EXAMPLE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    ${CMAKE_PREFIX_PATH_ARG:+"${CMAKE_PREFIX_PATH_ARG}"}
cmake --build "${EXAMPLE_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel

echo
echo "Example binary:"
echo "  ${EXAMPLE_BUILD_DIR}/load_onnx_time"
