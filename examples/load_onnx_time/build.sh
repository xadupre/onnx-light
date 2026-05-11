#!/usr/bin/env bash
# build.sh -- builds the standalone load_onnx_time example against the
# standard onnx C++ library.
#
# The standard onnx C++ development library must be installed before running
# this script.  On Ubuntu/Debian:
#   sudo apt-get install -y libonnx-dev libprotobuf-dev
#
# Usage (run from the repository root or from this directory):
#   bash examples/load_onnx_time/build.sh [install-prefix] [lib-build-dir] [example-build-dir]
#
# The first two arguments are accepted for interface compatibility with the
# top-level examples/build.sh but are not used by this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# First two args are unused (kept for compatibility with examples/build.sh)
EXAMPLE_BUILD_DIR="${3:-${REPO_ROOT}/build/load-onnx-time-example}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

echo "=== Configure and build load_onnx_time (${BUILD_TYPE}) ==="
cmake -S "${SCRIPT_DIR}" -B "${EXAMPLE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
cmake --build "${EXAMPLE_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel

echo
echo "Example binary:"
echo "  ${EXAMPLE_BUILD_DIR}/load_onnx_time"
