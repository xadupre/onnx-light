// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"

#include <algorithm>
#include <charconv>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

class SelectivePattern final : public core::builder::PatternOptimization {
public:
  explicit SelectivePattern(int index)
      : PatternOptimization(1, "Selective" + std::to_string(index)),
        type_("Op" + std::to_string(index % 128)) {}

  std::set<std::string> FastOpType() const override { return {type_}; }
  core::builder::MatchResult Match(core::builder::GraphGraph &, const NodeProto &) const override {
    return {};
  }
  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &) const override {
    throw core::builder::BuilderError("SelectivePattern never matches.");
  }

private:
  std::string type_;
};

int PositiveInteger(const char *text) {
  const std::string value(text);
  int result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result <= 0)
    throw std::invalid_argument("Benchmark arguments must be positive integers.");
  return result;
}

} // namespace

// Usage: bench_pattern_candidates [nodes=20000] [patterns=512] [iterations=7]
int main(int argc, char **argv) {
  if (argc > 4)
    throw std::invalid_argument("Expected at most nodes, patterns and iterations.");
  const int nodes = argc > 1 ? PositiveInteger(argv[1]) : 20000;
  const int patterns = argc > 2 ? PositiveInteger(argv[2]) : 512;
  const int iterations = argc > 3 ? PositiveInteger(argv[3]) : 7;
  core::builder::GraphBuilder builder("heterogeneous", {}, {{"benchmark", 1}});
  core::symbolic::SymShape shape;
  shape.PushBack(core::symbolic::SymDim(1));
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, shape);
  std::string input = "x";
  for (int index = 0; index < nodes; ++index) {
    const std::string output = "y" + std::to_string(index);
    builder.MakeNode("Op" + std::to_string(index % 128), {input}, {output}, "benchmark");
    input = output;
  }
  builder.MakeOutput(input);
  std::vector<std::shared_ptr<core::builder::PatternOptimization>> active;
  for (int index = 0; index < patterns; ++index)
    active.push_back(std::make_shared<SelectivePattern>(index));
  core::builder::GraphGraph graph(builder, active);
  core::builder::ConstantFoldingOptions folding;
  folding.enabled = false;
  std::vector<int64_t> times;
  core::builder::OptimizationReport warmup;
  graph.Optimize(1, &warmup, folding, false);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    core::builder::OptimizationReport report;
    graph.Optimize(1, &report, folding, false);
    times.push_back(report.matching_time_ns);
  }
  std::sort(times.begin(), times.end());
  std::cout << "nodes=" << nodes << " patterns=" << patterns
            << " operator_types=128 median_matching_us=" << times[times.size() / 2] / 1000.0
            << "\n";
}
