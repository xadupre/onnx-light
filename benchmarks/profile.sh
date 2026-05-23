#!/usr/bin/env bash
# profile.sh — build bench_parse_serialize, bench_load_file, or bench_save_file
# with RelWithDebInfo and run a
# gprof/perf/valgrind profile.
#
# Usage (run from the repository root):
#   bash benchmarks/profile.sh [gprof|perf|valgrind] [bench_parse_serialize|bench_load_file|bench_save_file] [--with-upstream-onnx] [extra bench flags]
#
# Examples:
#   bash benchmarks/profile.sh gprof    -n 20 -t 1
#   bash benchmarks/profile.sh perf     -n 20 -t 1
#   bash benchmarks/profile.sh valgrind -n 20  -t 1
#   bash benchmarks/profile.sh perf bench_load_file -n 20 -t 1
#   bash benchmarks/profile.sh perf bench_save_file -n 20 -t 1
#   bash benchmarks/profile.sh perf bench_load_file --with-upstream-onnx -n 20
#
# The default tool is gprof.
#
# --with-upstream-onnx (bench_load_file only) sets
# -DONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX=ON so CMake fetches and builds the
# upstream onnx (protobuf-based) C++ library via FetchContent, enabling the
# side-by-side onnx vs onnx_light comparison block (BENCH_HAS_UPSTREAM_ONNX).
#
# Requirements:
#   cmake, make/ninja, g++ (for gprof), perf (for perf), valgrind (for valgrind)

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
TOOL="${1:-gprof}"
shift || true          # no-op when no extra bench flags are given; suppresses set -e failure

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_BASE="${REPO_ROOT}/build"
BENCH_TARGET="bench_parse_serialize"

if [[ "${1:-}" == "bench_parse_serialize" || "${1:-}" == "bench_load_file" || "${1:-}" == "bench_save_file" ]]; then
    BENCH_TARGET="${1}"
    shift || true
fi

WITH_UPSTREAM_ONNX=0
if [[ "${1:-}" == "--with-upstream-onnx" ]]; then
    WITH_UPSTREAM_ONNX=1
    shift || true
fi

EXTRA_CMAKE_FLAGS=()
if [[ "${WITH_UPSTREAM_ONNX}" == "1" ]]; then
    ONNX_PREFIX="$(python -c 'import onnx, os; print(os.path.dirname(onnx.__file__))' 2>/dev/null || true)"
    if [[ -n "${ONNX_PREFIX}" ]]; then
        echo "pip-installed onnx detected at: ${ONNX_PREFIX} (informational only)"
    fi
    echo "--with-upstream-onnx: CMake will fetch and build upstream onnx via FetchContent"
    EXTRA_CMAKE_FLAGS+=("-DONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX=ON")
fi

case "${TOOL}" in
    gprof)
        BUILD_DIR="${BUILD_BASE}/bench_gprof"
        GPROF_FLAG="-DONNX_LIGHT_BENCH_GPROF=ON"
        ;;
    perf|valgrind)
        BUILD_DIR="${BUILD_BASE}/bench_rdi"
        GPROF_FLAG=""
        ;;
    *)
        echo "Unknown tool '${TOOL}'.  Choose: gprof | perf | valgrind" >&2
        exit 1
        ;;
esac

BENCH="${BUILD_DIR}/${BENCH_TARGET}"

# ---------------------------------------------------------------------------
# Step 1 — build
# ---------------------------------------------------------------------------
echo "=== Step 1: cmake configure (${TOOL}) ==="
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DONNX_LIGHT_BUILD_BENCHMARKS=ON \
    -DONNX_LIGHT_BUILD_PYTHON=OFF \
    ${GPROF_FLAG} \
    ${EXTRA_CMAKE_FLAGS[@]+"${EXTRA_CMAKE_FLAGS[@]}"}

echo "=== Step 1: cmake build ==="
cmake --build "${BUILD_DIR}" --target "${BENCH_TARGET}" -j"$(nproc)"

echo ""
echo "Binary: ${BENCH}"
echo ""

# ---------------------------------------------------------------------------
# Step 2 — profile
# ---------------------------------------------------------------------------
case "${TOOL}" in
    gprof)
        echo "=== Step 2: run with gprof instrumentation ==="
        (cd "${BUILD_DIR}" && "${BENCH}" "$@")

        echo ""
        echo "=== Step 3: gprof report (top 40 lines) ==="
        gprof -b "${BENCH}" "${BUILD_DIR}/gmon.out" | head -40
        ;;

    perf)
        echo "=== Step 2: perf stat ==="
        perf stat "${BENCH}" "$@"

        echo ""
        echo "=== Step 3: perf record + report (top 30 functions) ==="
        perf record -g -o "${BUILD_DIR}/perf.data" "${BENCH}" "$@"
        perf report --stdio --no-children -n -i "${BUILD_DIR}/perf.data" | head -60
        ;;

    valgrind)
        CALLGRIND_OUT="${BUILD_DIR}/callgrind.out"
        echo "=== Step 2: valgrind/callgrind ==="
        valgrind --tool=callgrind \
                 --callgrind-out-file="${CALLGRIND_OUT}" \
                 "${BENCH}" "$@"

        echo ""
        echo "=== Step 3: callgrind_annotate (top 80 lines) ==="
        callgrind_annotate "${CALLGRIND_OUT}" | head -80
        ;;
esac
