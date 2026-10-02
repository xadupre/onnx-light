// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include <gtest/gtest.h>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/patterns/canonicalization/initializer_unsqueeze_cast_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

struct GraphOptions {
  bool shared_unsqueeze = false;
  bool shared_cast = false;
  bool cast_output = false;
  bool overridable_source = false;
  bool overridable_axes = false;
  bool cast_first = false;
  bool external_source = false;
  bool skip_schema = false;
  int custom_domain_node = -1;
  std::vector<int64_t> axes = {0};
  std::vector<int64_t> axes_shape = {1};
};

core::builder::GraphBuilder MakeGraph(const GraphOptions &options = {}) {
  core::builder::GraphBuilder::SchemaLookupFn schema_lookup;
  if (!options.skip_schema) {
    schema_lookup = [](const std::string &op) {
      return onnx_op::GetAllOnnxOpSchemasWithHistory(op, false);
    };
  }
  core::builder::GraphBuilder builder("g", std::move(schema_lookup));
  builder.SetOpsetVersion("", 18);
  if (options.custom_domain_node >= 0) {
    builder.SetOpsetVersion("custom.domain", 1);
  }
  if (options.overridable_source) {
    core::symbolic::SymShape shape;
    shape.PushBack(core::symbolic::SymDim(2));
    builder.MakeInput("weight", core::symbolic::TensorType::kFloat, shape);
  }
  if (options.overridable_axes) {
    core::symbolic::SymShape shape;
    shape.PushBack(core::symbolic::SymDim(static_cast<int64_t>(options.axes.size())));
    builder.MakeInput("axes", core::symbolic::TensorType::kInt64, shape);
  }
  builder.MakeInput("x", core::symbolic::TensorType::kDouble, [] {
    core::symbolic::SymShape shape;
    shape.PushBack(core::symbolic::SymDim(1));
    shape.PushBack(core::symbolic::SymDim(2));
    return shape;
  }());
  if (options.external_source) {
    TensorProto weight;
    weight.set_name("weight");
    weight.set_data_type(TensorProto::DataType::FLOAT);
    weight.add_dims(2);
    weight.set_data_location(TensorProto::DataLocation::EXTERNAL);
    auto *location = weight.add_external_data();
    location->set_key("location");
    location->set_value("weights.bin");
    builder.MakeInitializer(weight);
  } else {
    builder.MakeInitializer(MakeInitializer<float>("weight", {2}, {1.25f, 2.5f}));
  }
  builder.MakeInitializer(MakeInitializer<int64_t>("axes", options.axes_shape, options.axes));
  builder.MakeNode("Unsqueeze", {"weight", "axes"}, {"expanded"},
                   options.custom_domain_node == 0 ? "custom.domain" : "");
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &to = attributes.add();
  to.set_name("to");
  to.set_type(AttributeProto::AttributeType::INT);
  to.set_i(static_cast<int64_t>(TensorProto::DataType::DOUBLE));
  builder.MakeNode("Cast", {"expanded"}, {"converted"},
                   options.custom_domain_node == 1 ? "custom.domain" : "", "", attributes);
  builder.MakeNode("Add",
                   options.cast_first ? std::vector<std::string>{"converted", "x"}
                                      : std::vector<std::string>{"x", "converted"},
                   {"y"}, options.custom_domain_node == 2 ? "custom.domain" : "");
  if (options.shared_unsqueeze) {
    builder.MakeNode("Identity", {"expanded"}, {"extra"});
    builder.MakeOutput("extra");
  }
  if (options.shared_cast) {
    builder.MakeNode("Identity", {"converted"}, {"extra"});
    builder.MakeOutput("extra");
  }
  if (options.cast_output) {
    builder.MakeOutput("converted");
  }
  builder.MakeOutput("y");
  return builder;
}

TEST(InitializerUnsqueezeCastPattern, FoldsShapeAndDtypeBeforeAdd) {
  onnx_kernels::RegisterKernelFunctions();
  for (bool cast_first : {false, true}) {
    GraphOptions options;
    options.cast_first = cast_first;
    options.axes = {-2};
    auto builder = MakeGraph(options);
    std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
    patterns.push_back(std::make_unique<onnx_patterns::InitializerUnsqueezeCastPattern>());
    core::builder::GraphGraph graph(builder, std::move(patterns));
    graph.Optimize();

    ASSERT_EQ(builder.Nodes().size(), 1u);
    const NodeProto &add = builder.Nodes()[0];
    EXPECT_EQ(add.op_type().value(), "Add");
    EXPECT_EQ(add.input()[cast_first ? 1 : 0].value(), "x");
    EXPECT_EQ(add.output()[0].value(), "y");
    const TensorProto *folded = graph.GetComputedConstant(add.input()[cast_first ? 0 : 1].value());
    ASSERT_NE(folded, nullptr);
    EXPECT_EQ(folded->data_type(), TensorProto::DataType::DOUBLE);
    ASSERT_EQ(folded->dims_size(), 2);
    EXPECT_EQ(folded->dims(0), 1);
    EXPECT_EQ(folded->dims(1), 2);
    auto tensor = core::runtime::TensorFromProto(*folded);
    EXPECT_DOUBLE_EQ(tensor.As<double>()[0], 1.25);
    EXPECT_DOUBLE_EQ(tensor.As<double>()[1], 2.5);
  }
}

TEST(InitializerUnsqueezeCastPattern, FoldsEmptyAxesWithoutChangingShape) {
  onnx_kernels::RegisterKernelFunctions();
  GraphOptions options;
  options.axes.clear();
  options.axes_shape = {0};
  auto builder = MakeGraph(options);
  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::InitializerUnsqueezeCastPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 1u);
  const TensorProto *folded = graph.GetComputedConstant(builder.Nodes()[0].input()[1].value());
  ASSERT_NE(folded, nullptr);
  EXPECT_EQ(folded->data_type(), TensorProto::DataType::DOUBLE);
  ASSERT_EQ(folded->dims_size(), 1);
  EXPECT_EQ(folded->dims(0), 2);
  auto tensor = core::runtime::TensorFromProto(*folded);
  EXPECT_DOUBLE_EQ(tensor.As<double>()[0], 1.25);
  EXPECT_DOUBLE_EQ(tensor.As<double>()[1], 2.5);
}

TEST(InitializerUnsqueezeCastPattern, RejectsSharedAndOverridableInputs) {
  onnx_kernels::RegisterKernelFunctions();
  onnx_patterns::InitializerUnsqueezeCastPattern pattern;
  GraphOptions options;
  options.shared_unsqueeze = true;
  auto shared = MakeGraph(options);
  core::builder::GraphGraph shared_graph(shared);
  EXPECT_EQ(pattern.Match(shared_graph, shared.Nodes()[1]).pattern, nullptr);

  options = {};
  options.overridable_source = true;
  auto overridable_source = MakeGraph(options);
  core::builder::GraphGraph overridable_source_graph(overridable_source);
  EXPECT_EQ(pattern.Match(overridable_source_graph, overridable_source.Nodes()[1]).pattern,
            nullptr);

  options = {};
  options.overridable_axes = true;
  auto overridable_axes = MakeGraph(options);
  core::builder::GraphGraph overridable_axes_graph(overridable_axes);
  EXPECT_EQ(pattern.Match(overridable_axes_graph, overridable_axes.Nodes()[1]).pattern, nullptr);
  EXPECT_THROW(
      pattern.Apply(shared_graph, {&shared.Nodes()[0], &shared.Nodes()[1], &shared.Nodes()[2]}),
      core::builder::BuilderError);
}

TEST(InitializerUnsqueezeCastPattern, RejectsSharedAndGraphOutputCast) {
  onnx_kernels::RegisterKernelFunctions();
  onnx_patterns::InitializerUnsqueezeCastPattern pattern;
  for (bool graph_output : {false, true}) {
    GraphOptions options;
    options.shared_cast = !graph_output;
    options.cast_output = graph_output;
    auto builder = MakeGraph(options);
    core::builder::GraphGraph graph(builder);
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
  }
}

TEST(InitializerUnsqueezeCastPattern, RejectsNonDefaultDomains) {
  onnx_kernels::RegisterKernelFunctions();
  onnx_patterns::InitializerUnsqueezeCastPattern pattern;
  for (int node_index = 0; node_index < 3; ++node_index) {
    GraphOptions options;
    options.skip_schema = true;
    options.custom_domain_node = node_index;
    auto builder = MakeGraph(options);
    core::builder::GraphGraph graph(builder);
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr) << node_index;
  }
}

TEST(InitializerUnsqueezeCastPattern, RejectsInvalidAxes) {
  onnx_kernels::RegisterKernelFunctions();
  onnx_patterns::InitializerUnsqueezeCastPattern pattern;
  const std::vector<TensorProto> cases = {
      MakeInitializer<int64_t>("axes", {2, 1}, {0, 1}),
      MakeInitializer<int64_t>("axes", {2}, {0, 0}),
      MakeInitializer<int64_t>("axes", {1}, {2}),
  };
  for (const TensorProto &axes : cases) {
    auto builder = MakeGraph();
    auto &initializers =
        const_cast<utils::RepeatedProtoField<TensorProto> &>(builder.Initializers());
    initializers[1] = axes;
    core::builder::GraphGraph graph(builder);
    EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
  }
}

TEST(InitializerUnsqueezeCastPattern, RejectsUnloadedExternalInitializer) {
  onnx_kernels::RegisterKernelFunctions();
  GraphOptions options;
  options.external_source = true;
  auto builder = MakeGraph(options);
  core::builder::GraphGraph graph(builder);
  onnx_patterns::InitializerUnsqueezeCastPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

} // namespace
