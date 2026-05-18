#!/usr/bin/env bash
# build.sh -- builds the standalone load_onnx_time example against the
# standard onnx C++ library.
#
# The script has two operating modes:
#
# 1. System install (default when onnx CMake package is detected):
#      bash examples/load_onnx_time/build.sh
#    Requires: sudo apt-get install -y libonnx-dev libprotobuf-dev
#
# 2. From-source build (automatic when system onnx is absent, or explicit):
#      ONNX_GIT_TAG=v1.17.0 bash examples/load_onnx_time/build.sh
#    The script clones onnx from git and passes FETCHCONTENT_SOURCE_DIR_ONNX
#    to cmake so that onnx (and all its transitive dependencies: protobuf,
#    abseil, utf8_range, …) is built inline inside the example's cmake build.
#    No separate protobuf install is needed – onnx's own CMakeLists handles it.
#
# Usage (run from the repository root or from this directory):
#   bash examples/load_onnx_time/build.sh [install-prefix] [lib-build-dir] [example-build-dir]
#
# Arguments:
#   install-prefix    unused (kept for backward compatibility with examples/build.sh)
#   lib-build-dir     directory for library source trees
#                     (default: build/load-onnx-time-lib)
#   example-build-dir load_onnx_time build directory
#                     (default: build/load-onnx-time-example)
#
# Environment variables:
#   ONNX_GIT_TAG         git tag/branch for onnx (e.g. v1.17.0).  When unset
#                        the script probes for the system onnx CMake package;
#                        if absent it derives the tag from Python onnx or falls
#                        back to ONNX_DEFAULT_GIT_TAG.
#   ONNX_GIT_URL         onnx git URL (default: https://github.com/onnx/onnx.git)
#   ONNX_DEFAULT_GIT_TAG fallback onnx tag when Python onnx is not found
#                        (default: v1.17.0)
#   CMAKE_BUILD_TYPE     build type (default: Release)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# install-prefix is accepted for backward compat but not used in from-source mode
LIB_BUILD_DIR="${2:-${REPO_ROOT}/build/load-onnx-time-lib}"
EXAMPLE_BUILD_DIR="${3:-${REPO_ROOT}/build/load-onnx-time-example}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
PARALLEL_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)}"

ONNX_GIT_TAG="${ONNX_GIT_TAG:-}"
ONNX_GIT_URL="${ONNX_GIT_URL:-https://github.com/onnx/onnx.git}"
ONNX_DEFAULT_GIT_TAG="${ONNX_DEFAULT_GIT_TAG:-v1.17.0}"

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
fi

CMAKE_EXTRA=()
STEP=1

# ---- optionally pre-clone onnx for FetchContent ----------------------------
# cmake's FetchContent can download onnx on its own, but pre-cloning here lets
# subsequent cmake reconfigures skip the download and work offline.
if [ -n "${ONNX_GIT_TAG}" ]; then
    ONNX_SRC_DIR="${LIB_BUILD_DIR}/onnx-src"

    echo "=== Step ${STEP}: clone onnx ${ONNX_GIT_TAG} ==="
    STEP=$((STEP + 1))
    if [ ! -d "${ONNX_SRC_DIR}/.git" ]; then
        git clone --depth 1 --branch "${ONNX_GIT_TAG}" \
            "${ONNX_GIT_URL}" "${ONNX_SRC_DIR}"
        git -C "${ONNX_SRC_DIR}" submodule update --init --recursive
    else
        echo "Source directory ${ONNX_SRC_DIR} already exists, skipping clone."
    fi

    # Tell cmake FetchContent to use the local source instead of downloading.
    CMAKE_EXTRA+=(
        "-DFETCHCONTENT_SOURCE_DIR_ONNX=${ONNX_SRC_DIR}"
        "-DONNX_GIT_TAG=${ONNX_GIT_TAG}"
        "-DONNX_GIT_URL=${ONNX_GIT_URL}"
    )
fi

# ---- build the example -----------------------------------------------------
# cmake builds onnx (and all its transitive deps: protobuf, abseil, utf8_range)
# inline via FetchContent – no manual install step or separate cmake project.
echo "=== Step ${STEP}: configure and build load_onnx_time (${BUILD_TYPE}) ==="
cmake -S "${SCRIPT_DIR}" -B "${EXAMPLE_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    ${CMAKE_EXTRA[@]+"${CMAKE_EXTRA[@]}"}
cmake --build "${EXAMPLE_BUILD_DIR}" --config "${BUILD_TYPE}" --parallel "${PARALLEL_JOBS}"

echo
echo "Example binary:"
echo "  ${EXAMPLE_BUILD_DIR}/load_onnx_time"
