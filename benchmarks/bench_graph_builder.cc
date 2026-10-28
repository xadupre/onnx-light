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
 *     -f <file>     Path to the .onnx model (unless --synthetic-layers is used)
 *     -n <iters>    Number of GraphBuilder construction iterations (default: 5)
 *     --no-inline   Do not inline model-local functions before building
 *     --synthetic-layers <count>  Uses metadata-only transformed weights and late outputs
 *     --dependency-first  Interleaves synthetic weight transforms with their consumers
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

#include <charconv>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ModelProto SyntheticModel(int layers, bool dependency_first) {
  ModelProto model;
  model.set_ir_version(10);
  auto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(23);
  opset = model.add_opset_import();
  opset->set_domain("benchmark");
  opset->set_version(1);
  auto &graph = *model.mutable_graph();
  graph.set_name("deferred_weights");
  const auto declare = [](ValueInfoProto &value, const std::string &name) {
    value.set_name(name);
    auto *tensor = value.mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(TensorProto::FLOAT);
    auto *shape = tensor->mutable_shape();
    shape->add_dim()->set_dim_param("batch");
    shape->add_dim()->set_dim_param("sequence");
    shape->add_dim()->set_dim_value(64);
  };
  declare(*graph.add_input(), "x");
  const auto node = [&](const std::string &op, const std::vector<std::string> &inputs,
                        const std::string &output, const std::string &domain = "") {
    auto *result = graph.add_node();
    result->set_op_type(op);
    result->set_domain(domain);
    result->set_name(output);
    for (const auto &input : inputs)
      result->add_input(input);
    result->add_output(output);
    return result;
  };
  const auto weight = [&](const std::string &name) {
    auto *tensor = graph.add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(TensorProto::BFLOAT16);
    tensor->add_dims(64);
    tensor->add_dims(64);
    tensor->set_data_location(TensorProto::EXTERNAL);
    auto *location = tensor->add_external_data();
    location->set_key("location");
    location->set_value("#metadata-only");
    auto *cast = node("Cast", {name}, name + "_float");
    auto *to = cast->add_attribute();
    to->set_name("to");
    to->set_type(AttributeProto::INT);
    to->set_i(TensorProto::FLOAT);
    node("Transpose", {name + "_float"}, name + "_transposed");
  };
  weight("tied");
  if (!dependency_first)
    for (int layer = 0; layer < layers; ++layer)
      for (int projection = 0; projection < 4; ++projection)
        weight("w" + std::to_string(layer) + "_" + std::to_string(projection));
  node("MatMul", {"x", "tied_transposed"}, "embedding");
  std::string activation = "embedding";
  for (int layer = 0; layer < layers; ++layer) {
    for (int projection = 0; projection < 4; ++projection) {
      const std::string name = "w" + std::to_string(layer) + "_" + std::to_string(projection);
      if (dependency_first)
        weight(name);
      node("MatMul", {activation, name + "_transposed"}, name + "_product");
      activation = name + "_activation";
      node("Abs", {name + "_product"}, activation);
    }
    const std::string cache = "cache" + std::to_string(layer);
    node("Snapshot", {activation}, cache, "benchmark");
    declare(*graph.add_output(), cache);
  }
  node("MatMul", {activation, "tied_transposed"}, "logits");
  declare(*graph.add_output(), "logits");
  return model;
}

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
  int synthetic_layers = 0;
  bool dependency_first = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--no-inline") == 0) {
      inline_functions = false;
    } else if (std::strcmp(argv[i], "--synthetic-layers") == 0 && i + 1 < argc) {
      const std::string value = argv[++i];
      const auto parsed =
          std::from_chars(value.data(), value.data() + value.size(), synthetic_layers);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
          synthetic_layers <= 0 || synthetic_layers > 4096) {
        std::cerr << "Synthetic layer count must be between 1 and 4096\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--dependency-first") == 0) {
      dependency_first = true;
    } else {
      std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
      return 1;
    }
  }

  if (model_path.empty() == (synthetic_layers == 0) || iters <= 0 ||
      (dependency_first && synthetic_layers == 0)) {
    std::cerr << "Specify either -f <model.onnx> or --synthetic-layers <count>, and positive -n\n";
    return 1;
  }

  // Shape functions are looked up by GraphBuilder's incremental shape
  // inference; register the built-ins before building.
  onnx_shapes::RegisterShapeFunctions();

  ModelProto model;
  if (synthetic_layers != 0) {
    model = SyntheticModel(synthetic_layers, dependency_first);
  } else {
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
  std::cout << "model:  " << (synthetic_layers ? "metadata-only synthetic" : model_path) << "\n";
  if (synthetic_layers)
    std::cout << "order: " << (dependency_first ? "dependency-first" : "hoisted") << "\n";
  std::cout << "inline: " << (inline_functions ? "yes" : "no") << "\n";
  std::cout << "nodes:  " << node_count << " (top-level " << model.graph().node().size() << ")\n";
  std::cout << "outputs: " << model.graph().output_size() << "\n";
  std::cout << "functions: " << model.functions().size() << "\n";

  // Warm-up build so lazily-initialized caches (schema table, dispatch table)
  // are populated before timing.
  {
    core::builder::GraphBuilder warmup(model);
    if (synthetic_layers)
      warmup.ToModel();
  }

  double total_ms = 0.0;
  double export_ms = 0.0;
  for (int i = 0; i < iters; ++i) {
    const auto start = std::chrono::steady_clock::now();
    core::builder::GraphBuilder builder(model);
    const auto stop = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
    total_ms += ms;
    std::cout << "iter " << i << ": " << ms << " ms\n";
    if (synthetic_layers) {
      const auto export_start = std::chrono::steady_clock::now();
      builder.ToModel();
      const auto exported = std::chrono::steady_clock::now();
      const double elapsed =
          std::chrono::duration<double, std::milli>(exported - export_start).count();
      export_ms += elapsed;
      std::cout << "export " << i << ": " << elapsed << " ms\n";
    }
  }

  std::cout << "GraphBuilder construction: " << (total_ms / iters) << " ms/build (avg over "
            << iters << " iters)\n";
  if (synthetic_layers) {
    std::cout << "Export: " << export_ms / iters << " ms/export\n";
    std::cout << "Construction + export: " << (total_ms + export_ms) / iters << " ms/model\n";
  }
  return 0;
}
