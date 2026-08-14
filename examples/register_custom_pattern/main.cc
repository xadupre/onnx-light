// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"
#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_optimization.h"
#include "onnx_core/builder/pattern_registry.h"
#include "onnx_proto/onnx_helper.h"

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace builder = ONNX_LIGHT_NAMESPACE::core::builder;
using ONNX_LIGHT_NAMESPACE::NodeProto;

namespace example {

class CollapseIdentityChainPattern final : public builder::PatternOptimization {
public:
  CollapseIdentityChainPattern() : PatternOptimization(1, "example.CollapseIdentityChain") {}

  std::set<std::string> FastOpType() const override { return {"Identity"}; }

  builder::MatchResult Match(builder::GraphGraph &graph,
                             const NodeProto &candidate) const override {
    if (candidate.op_type().value() != "Identity" || candidate.input_size() != 1 ||
        candidate.output_size() != 1) {
      return {};
    }
    const NodeProto *inner = graph.NodeBefore(candidate.input()[0].value());
    if (inner == nullptr || inner->op_type().value() != "Identity" || inner->input_size() != 1 ||
        inner->output_size() != 1 || graph.IsUsedMoreThanOnce(inner->output()[0].value())) {
      return {};
    }
    return builder::MatchResult{this, {inner, &candidate}, &candidate};
  }

  ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto>
  Apply(builder::GraphGraph &, const std::vector<const NodeProto *> &nodes) const override {
    ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(ONNX_LIGHT_NAMESPACE::MakeNode(
        "Identity", {nodes[0]->input()[0].value()}, {nodes[1]->output()[0].value()}));
    return replacements;
  }
};

builder::GraphBuilder MakeBuilder() {
  builder::GraphBuilder graph("identity_chain");
  graph.SetOpsetVersion("", 18);
  graph.MakeInput("x", ONNX_LIGHT_NAMESPACE::core::symbolic::TensorType::kFloat, {});
  graph.MakeNode("Identity", {"x"}, {"middle"});
  graph.MakeNode("Identity", {"middle"}, {"y"});
  graph.MakeOutput("y");
  return graph;
}

} // namespace example

int main() {
  // Direct selection keeps the custom pattern local to this optimizer.
  builder::GraphBuilder direct_builder = example::MakeBuilder();
  std::vector<std::unique_ptr<builder::PatternOptimization>> selected_patterns;
  selected_patterns.push_back(std::make_unique<example::CollapseIdentityChainPattern>());
  builder::GraphGraph direct_optimizer(direct_builder, std::move(selected_patterns));
  direct_optimizer.Optimize();

  if (direct_builder.Nodes().size() != 1 || direct_builder.Nodes()[0].input()[0].value() != "x") {
    std::cerr << "Direct custom-pattern selection failed.\n";
    return 1;
  }

  // Registry selection makes a factory available to default GraphGraph instances.
  builder::RegisterPattern("example.CollapseIdentityChain", []() {
    return std::make_unique<example::CollapseIdentityChainPattern>();
  });
  builder::GraphBuilder registered_builder = example::MakeBuilder();
  builder::GraphGraph registered_optimizer(registered_builder);
  registered_optimizer.Optimize();

  if (registered_builder.Nodes().size() != 1 ||
      registered_builder.Nodes()[0].input()[0].value() != "x") {
    std::cerr << "Registered custom-pattern selection failed.\n";
    return 1;
  }

  std::cout << "PASS: direct and registered custom patterns collapsed the Identity chain.\n";
  return 0;
}
