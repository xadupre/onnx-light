/**
 * bench_graph_builder.cc
 *
 * Standalone C++ benchmark for the onnx-light GraphBuilder "create a model"
 * path, i.e. constructing a ``core::builder::GraphBuilder`` from an existing
 * ``ModelProto`` (which replays every graph and function node through
 * ``MakeNode`` and runs the incremental compute analyses).
 *
 * The model is loaded from a ``.onnx`` file passed with ``-f``. Point it at a
 * realistic multi-layer transformer (optionally with its local functions
 * inlined via the default, or left as-is with ``--no-inline``) to exercise the
 * per-node cost of the builder.
 *
 * GraphBuilder can execute operator kernels while building, but it must assume
 * kernels may be missing, so this benchmark intentionally does NOT link against
 * ``lib_onnx_kernels`` (nor ``lib_onnx_backend_test``, which pulls it in). It
 * only needs the model file loader, the operator schemas and the incremental
 * shape functions.
 *
 * Designed to be compiled with RelWithDebInfo (-O2 -g) so that Linux profiling
 * tools (perf, gprof, valgrind/callgrind) can attribute wall-clock or
 * instruction samples back to named C++ functions.
 *
 * Build (see CMakeLists.txt ONNX_LIGHT_BUILD_BENCHMARKS option):
 *
 *   cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
 *                  -DONNX_LIGHT_BUILD_BENCHMARKS=ON \
 *                  -DONNX_LIGHT_BUILD_PYTHON=OFF
 *   cmake --build build --target bench_graph_builder -j
 *
 * Usage:
 *   ./build/bench_graph_builder -f <model.onnx> [OPTIONS]
 *     -f <file>     Path to the .onnx model to build (required)
 *     -n <iters>    Number of GraphBuilder construction iterations (default: 5)
 *     --no-inline   Do not inline model-local functions before building
 *
 * Typical profiling workflow:
 *
 *   perf record -g ./build/bench_graph_builder -f model.onnx -n 5
 *   perf report --stdio --no-children -n | head -60
 */

#include "onnx.h"
#include "onnx_core/builder/graph_builder.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_lib/inliner/inliner.h"
#include "stream.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

int64_t CountNodes(const GraphProto &graph) {
  int64_t total = graph.node().size();
  for (const auto &node : graph.node()) {
    for (const auto &attribute : node.attribute()) {
      if (attribute.has_g()) {
        total += CountNodes(attribute.g());
      }
      for (const auto &sub : attribute.graphs()) {
        total += CountNodes(sub);
      }
    }
  }
  return total;
}

} // namespace

int main(int argc, char **argv) {
  int iters = 5;
  std::string model_path;
  bool inline_functions = true;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--no-inline") == 0) {
      inline_functions = false;
    } else {
      std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
      return 1;
    }
  }

  if (model_path.empty()) {
    std::cerr << "Missing required -f <model.onnx> argument\n";
    return 1;
  }

  // Shape functions are looked up by GraphBuilder's incremental shape
  // inference; register the built-ins before building.
  onnx_shapes::RegisterShapeFunctions();

  ModelProto model;
  {
    utils::MmapFileStream stream(model_path);
    ParseOptions opts;
    if (!model.ParseFromStream(stream, opts)) {
      std::cerr << "Failed to parse model from: " << model_path << "\n";
      return 1;
    }
  }
  if (inline_functions) {
    inliner::InlineLocalFunctions(model);
  }

  const int64_t node_count = CountNodes(model.graph());
  std::cout << "model:  " << model_path << "\n";
  std::cout << "inline: " << (inline_functions ? "yes" : "no") << "\n";
  std::cout << "nodes:  " << node_count << " (top-level " << model.graph().node().size() << ")\n";
  std::cout << "functions: " << model.functions().size() << "\n";

  // Warm-up build so lazily-initialized caches (schema table, dispatch table)
  // are populated before timing.
  {
    core::builder::GraphBuilder warmup(model);
  }

  double total_ms = 0.0;
  for (int i = 0; i < iters; ++i) {
    const auto start = std::chrono::steady_clock::now();
    core::builder::GraphBuilder builder(model);
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    total_ms += ms;
    std::cout << "iter " << i << ": " << ms << " ms\n";
  }

  std::cout << "GraphBuilder construction: " << (total_ms / iters) << " ms/build (avg over "
            << iters << " iters)\n";
  return 0;
}
