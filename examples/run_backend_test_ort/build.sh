#!/usr/bin/env bash
# build.sh -- downloads a pre-built onnxruntime CPU release, installs
# onnx_light locally, and builds the standalone run_backend_test_ort
# example against both.
#
# Usage (run from the repository root or from this directory):
#   bash examples/run_backend_test_ort/build.sh \
#        [install-prefix] [lib-build-dir] [example-build-dir] [ort-root]
#
# Environment overrides:
#   ONNXRUNTIME_VERSION   onnxruntime release tag without the leading 'v'
#                         (default: 1.19.2)
#   ONNXRUNTIME_ROOT_DIR  Skip the download; use this existing extracted
#                         onnxruntime release directory instead.
#   CMAKE_BUILD_TYPE      Build type (default: Release).
#
# Note: onnxruntime itself cannot reasonably be built from source as part of
# this example -- the official pre-built CPU release archive is downloaded.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INSTALL_PREFIX="${1:-${REPO_ROOT}/build/install-run-backend-test-ort}"
LIB_BUILD_DIR="${2:-${REPO_ROOT}/build/run-backend-test-ort-lib}"
EXAMPLE_BUILD_DIR="${3:-${REPO_ROOT}/build/run-backend-test-ort-example}"
ORT_ROOT_OVERRIDE="${4:-${ONNXRUNTIME_ROOT_DIR:-}}"

ORT_VERSION="${ONNXRUNTIME_VERSION:-1.19.2}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
PARALLEL_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)}"

uname_s="$(uname -s)"
uname_m="$(uname -m)"
case "${uname_s}-${uname_m}" in
    Linux-x86_64)   ORT_PLATFORM="linux-x64" ;;
    Linux-aarch64)  ORT_PLATFORM="linux-aarch64" ;;
    Darwin-x86_64)  ORT_PLATFORM="osx-x86_64" ;;
    Darwin-arm64)   ORT_PLATFORM="osx-arm64" ;;
    *)
        echo "Unsupported platform ${uname_s}-${uname_m}. Set ONNXRUNTIME_ROOT_DIR=/path to a pre-extracted release and re-run." >&2
        exit 1
        ;;
esac

if [[ -n "${ORT_ROOT_OVERRIDE}" ]]; then
    ORT_ROOT="${ORT_ROOT_OVERRIDE}"
    echo "=== Step 0: using existing onnxruntime release: ${ORT_ROOT} ==="
else
    ORT_DOWNLOAD_DIR="${REPO_ROOT}/build/onnxruntime-downloads"
    ORT_ARCHIVE_NAME="onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.tgz"
    ORT_ARCHIVE_PATH="${ORT_DOWNLOAD_DIR}/${ORT_ARCHIVE_NAME}"
    ORT_ROOT="${ORT_DOWNLOAD_DIR}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}"
    ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE_NAME}"

    mkdir -p "${ORT_DOWNLOAD_DIR}"

    if [[ ! -d "${ORT_ROOT}" ]]; then
        echo "=== Step 0: downloading onnxruntime ${ORT_VERSION} for ${ORT_PLATFORM} ==="
        echo "    URL: ${ORT_URL}"
        if command -v curl >/dev/null 2>&1; then
            curl -fL --retry 3 -o "${ORT_ARCHIVE_PATH}" "${ORT_URL}"
        elif command -v wget >/dev/null 2>&1; then
            wget -O "${ORT_ARCHIVE_PATH}" "${ORT_URL}"
        else
            echo "Neither curl nor wget is available. Install one of them, or set ONNXRUNTIME_ROOT_DIR=/path to a pre-extracted release." >&2
            exit 1
        fi
        tar -xzf "${ORT_ARCHIVE_PATH}" -C "${ORT_DOWNLOAD_DIR}"
    else
        echo "=== Step 0: reusing cached onnxruntime release: ${ORT_ROOT} ==="
    fi
fi

if [[ ! -f "${ORT_ROOT}/include/onnxruntime_cxx_api.h" ]]; then
    echo "ERROR: ${ORT_ROOT}/include/onnxruntime_cxx_api.h not found." >&2
    echo "       Pass the correct release directory as the 4th positional argument or via ONNXRUNTIME_ROOT_DIR." >&2
    exit 1
fi

echo "=== Step 1: configure and build onnx_light (${BUILD_TYPE}) ==="
cmake -S "${REPO_ROOT}" -B "${LIB_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DONNX_LIGHT_BUILD_PYTHON=OFF \
    -DONNX_LIGHT_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
cmake --build "${LIB_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${PARALLEL_JOBS}"
cmake --install "${LIB_BUILD_DIR}" --config "${BUILD_TYPE}"

echo "=== Step 2: configure and build run_backend_test_ort (${BUILD_TYPE}) ==="
cmake -S "${SCRIPT_DIR}" -B "${EXAMPLE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}" \
    -DONNXRUNTIME_ROOT_DIR="${ORT_ROOT}"
cmake --build "${EXAMPLE_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${PARALLEL_JOBS}"

echo
echo "Example binary:"
echo "  ${EXAMPLE_BUILD_DIR}/run_backend_test_ort"
echo
echo "Run with:"
echo "  LD_LIBRARY_PATH=${ORT_ROOT}/lib ${EXAMPLE_BUILD_DIR}/run_backend_test_ort"
