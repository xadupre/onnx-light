#!/usr/bin/env bash
# build.sh -- builds the standalone load_onnx_time example against the
# standard onnx C++ library.
#
# The script has three operating modes:
#
# 1. System install (default when onnx CMake package is detected):
#      bash examples/load_onnx_time/build.sh
#    Requires: sudo apt-get install -y libonnx-dev libprotobuf-dev
#
# 2. Explicit from-source build (set ONNX_GIT_TAG):
#      ONNX_GIT_TAG=v1.17.0 bash examples/load_onnx_time/build.sh
#    Set PROTOBUF_GIT_TAG as well if protobuf is not installed:
#      ONNX_GIT_TAG=v1.17.0 PROTOBUF_GIT_TAG=v3.21.12 bash examples/load_onnx_time/build.sh
#
# 3. Automatic from-source build (no env vars needed):
#    When neither ONNX_GIT_TAG nor the onnx CMake package is found, the script
#    automatically switches to a from-source build.  The onnx git tag is read
#    from the Python onnx package in site-packages (import onnx) when
#    available, or falls back to ONNX_DEFAULT_GIT_TAG.  Protobuf is also built
#    from source when it is not detected as a CMake package.
#
# Usage (run from the repository root or from this directory):
#   bash examples/load_onnx_time/build.sh [install-prefix] [lib-build-dir] [example-build-dir]
#
# Arguments:
#   install-prefix    install prefix for onnx (and protobuf) when building
#                     from source (default: build/install-load-onnx-time)
#   lib-build-dir     directory for library source trees and build trees
#                     (default: build/load-onnx-time-lib)
#   example-build-dir load_onnx_time build directory
#                     (default: build/load-onnx-time-example)
#
# Environment variables:
#   ONNX_GIT_TAG             git tag/branch for onnx (e.g. v1.17.0)
#   ONNX_GIT_URL             onnx git URL (default: https://github.com/onnx/onnx.git)
#   ONNX_DEFAULT_GIT_TAG     fallback onnx tag when Python onnx is not found
#                            (default: v1.17.0)
#   PROTOBUF_GIT_TAG         git tag/branch for protobuf (e.g. v3.21.12)
#   PROTOBUF_GIT_URL         protobuf git URL
#   PROTOBUF_DEFAULT_GIT_TAG fallback protobuf tag (default: v3.21.12)
#   CMAKE_BUILD_TYPE         build type (default: Release)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INSTALL_PREFIX="${1:-${REPO_ROOT}/build/install-load-onnx-time}"
LIB_BUILD_DIR="${2:-${REPO_ROOT}/build/load-onnx-time-lib}"
EXAMPLE_BUILD_DIR="${3:-${REPO_ROOT}/build/load-onnx-time-example}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

ONNX_GIT_TAG="${ONNX_GIT_TAG:-}"
ONNX_GIT_URL="${ONNX_GIT_URL:-https://github.com/onnx/onnx.git}"
ONNX_DEFAULT_GIT_TAG="${ONNX_DEFAULT_GIT_TAG:-v1.17.0}"

PROTOBUF_GIT_TAG="${PROTOBUF_GIT_TAG:-}"
PROTOBUF_GIT_URL="${PROTOBUF_GIT_URL:-https://github.com/protocolbuffers/protobuf.git}"
PROTOBUF_DEFAULT_GIT_TAG="${PROTOBUF_DEFAULT_GIT_TAG:-v3.21.12}"

# ---- cmake package probe ---------------------------------------------------
# Returns 0 (true) if the named CMake package is findable, 1 otherwise.
_cmake_pkg_found() {
    local pkg="$1" tmpdir rc
    tmpdir=$(mktemp -d)
    {
        printf 'cmake_minimum_required(VERSION 3.15)\n'
        printf 'project(probe)\n'
        printf 'find_package(%s QUIET)\n' "${pkg}"
        printf 'if(NOT %s_FOUND)\n' "${pkg}"
        printf '    message(FATAL_ERROR "not found")\n'
        printf 'endif()\n'
    } > "${tmpdir}/CMakeLists.txt"
    cmake -S "${tmpdir}" -B "${tmpdir}/build" -DCMAKE_BUILD_TYPE=Release \
        > /dev/null 2>&1
    rc=$?
    rm -rf "${tmpdir}"
    return ${rc}
}

# ---- auto-detect: switch to from-source if onnx CMake package is absent ----
if [ -z "${ONNX_GIT_TAG}" ] && ! _cmake_pkg_found ONNX; then
    # Derive the onnx version from the Python onnx package in site-packages.
    _py=""
    for _cmd in python3 python; do
        if command -v "${_cmd}" > /dev/null 2>&1 \
           && "${_cmd}" -c "import onnx" > /dev/null 2>&1; then
            _py="${_cmd}"
            break
        fi
    done

    if [ -n "${_py}" ]; then
        ONNX_GIT_TAG="v$(${_py} -c "import onnx; print(onnx.__version__)")"
    else
        ONNX_GIT_TAG="${ONNX_DEFAULT_GIT_TAG}"
    fi
    echo "onnx CMake package not found; will build onnx from source (${ONNX_GIT_TAG})."

    # Also build protobuf from source when it is not present.
    if [ -z "${PROTOBUF_GIT_TAG}" ] && ! _cmake_pkg_found Protobuf; then
        PROTOBUF_GIT_TAG="${PROTOBUF_DEFAULT_GIT_TAG}"
        echo "Protobuf CMake package not found; will build Protobuf from source (${PROTOBUF_GIT_TAG})."
    fi
fi

CMAKE_EXTRA=()
STEP=1

# ---- optionally build protobuf from source ---------------------------------
if [ -n "${PROTOBUF_GIT_TAG}" ]; then
    PROTO_SRC_DIR="${LIB_BUILD_DIR}/protobuf-src"
    PROTO_BUILD_DIR="${LIB_BUILD_DIR}/protobuf-build"

    echo "=== Step ${STEP}: clone protobuf ${PROTOBUF_GIT_TAG} ==="
    STEP=$((STEP + 1))
    if [ ! -d "${PROTO_SRC_DIR}/.git" ]; then
        git clone --depth 1 --branch "${PROTOBUF_GIT_TAG}" \
            "${PROTOBUF_GIT_URL}" "${PROTO_SRC_DIR}"
        git -C "${PROTO_SRC_DIR}" submodule update --init --recursive
    else
        echo "Source directory ${PROTO_SRC_DIR} already exists, skipping clone."
    fi

    echo "=== Step ${STEP}: build and install protobuf ==="
    STEP=$((STEP + 1))
    cmake -S "${PROTO_SRC_DIR}" -B "${PROTO_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -Dprotobuf_BUILD_TESTS=OFF \
        -Dprotobuf_BUILD_SHARED_LIBS=OFF \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
    cmake --build "${PROTO_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel
    cmake --install "${PROTO_BUILD_DIR}" --config "${BUILD_TYPE}"

    CMAKE_EXTRA+=(-DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}")
fi

# ---- optionally build onnx from source -------------------------------------
if [ -n "${ONNX_GIT_TAG}" ]; then
    ONNX_SRC_DIR="${LIB_BUILD_DIR}/onnx-src"
    ONNX_BUILD_DIR="${LIB_BUILD_DIR}/onnx-build"

    echo "=== Step ${STEP}: clone onnx ${ONNX_GIT_TAG} ==="
    STEP=$((STEP + 1))
    if [ ! -d "${ONNX_SRC_DIR}/.git" ]; then
        git clone --depth 1 --branch "${ONNX_GIT_TAG}" \
            "${ONNX_GIT_URL}" "${ONNX_SRC_DIR}"
        git -C "${ONNX_SRC_DIR}" submodule update --init --recursive
    else
        echo "Source directory ${ONNX_SRC_DIR} already exists, skipping clone."
    fi

    echo "=== Step ${STEP}: build and install onnx ==="
    STEP=$((STEP + 1))
    cmake -S "${ONNX_SRC_DIR}" -B "${ONNX_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DONNX_ML=ON \
        -DONNX_BUILD_TESTS=OFF \
        -DONNX_BUILD_BENCHMARKS=OFF \
        ${CMAKE_EXTRA[@]+"${CMAKE_EXTRA[@]}"} \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
    cmake --build "${ONNX_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel
    cmake --install "${ONNX_BUILD_DIR}" --config "${BUILD_TYPE}"

    # cmake --install may not copy the protobuf-generated headers (e.g.
    # onnx-ml.pb.h, onnx-operators-ml.pb.h) that live only in the build
    # directory.  Copy them explicitly so that the example can compile.
    find "${ONNX_BUILD_DIR}" -name "*.pb.h" \
        -exec cp {} "${INSTALL_PREFIX}/include/onnx/" \;

    CMAKE_EXTRA=(-DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}")
fi

# ---- build the example -----------------------------------------------------
echo "=== Step ${STEP}: configure and build load_onnx_time (${BUILD_TYPE}) ==="
cmake -S "${SCRIPT_DIR}" -B "${EXAMPLE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    ${CMAKE_EXTRA[@]+"${CMAKE_EXTRA[@]}"}
cmake --build "${EXAMPLE_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel

echo
echo "Example binary:"
echo "  ${EXAMPLE_BUILD_DIR}/load_onnx_time"
