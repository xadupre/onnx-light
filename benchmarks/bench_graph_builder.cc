/**
 * bench_graph_builder.cc
 *
 * Standalone C++ benchmark for the onnx-light GraphBuilder "create a model"
 * path, i.e. constructing a ``core::builder::GraphBuilder`` from an existing
 * ``ModelProto`` (which replays every graph and function node through
 * ``MakeNode`` and runs the incremental compute analyses).
 *
 * By default it profiles the Qwen3-like model retrieved from the backend test
 * cases (``test_cc_shape_inference_big_qwen3_4_layers_like``), optionally with
 * its local functions inlined so the flattened graph exercises the per-node
 * cost of the builder on a realistic multi-layer transformer.
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
 *   ./build/bench_graph_builder [OPTIONS]
 *     -n <iters>    Number of GraphBuilder construction iterations (default: 5)
 *     -m <regex>    Backend test case name regex to load
 *                   (default: test_cc_shape_inference_big_qwen3_4_layers_like)
 *     --no-inline   Do not inline model-local functions before building
 *
 * Typical profiling workflow:
 *
 *   perf record -g ./build/bench_graph_builder -n 5
 *   perf report --stdio --no-children -n | head -60
 */

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/builder/graph_builder.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_lib/inliner/inliner.h"

#include <chrono>
#include <cstring>
#include <iostream>
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
  std::string name_regex = "test_cc_shape_inference_big_qwen3_4_layers_like";
  bool inline_functions = true;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
      name_regex = argv[++i];
    } else if (std::strcmp(argv[i], "--no-inline") == 0) {
      inline_functions = false;
    } else {
      std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
      return 1;
    }
  }

  // Shape functions are looked up by GraphBuilder's incremental shape
  // inference; register the built-ins before building.
  onnx_shapes::RegisterShapeFunctions();

  std::vector<core::backend_test::TestCase> cases =
      core::backend_test::CollectTestCasesByName(name_regex, /*include_big=*/true);
  if (cases.empty()) {
    std::cerr << "No backend test case matched regex: " << name_regex << "\n";
    return 1;
  }

  ModelProto model = cases.front().model();
  if (inline_functions) {
    inliner::InlineLocalFunctions(model);
  }

  const int64_t node_count = CountNodes(model.graph());
  std::cout << "case:   " << cases.front().name << "\n";
  std::cout << "inline: " << (inline_functions ? "yes" : "no") << "\n";
  std::cout << "nodes:  " << node_count << " (top-level " << model.graph().node().size() << ")\n";
  std::cout << "functions: " << model.functions().size() << "\n";

  // Warm-up build so lazily-initialised caches (schema table, dispatch table)
  // are populated before timing.
  { core::builder::GraphBuilder warmup(model); }

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
