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

core::builder::GraphBuilder MakeGraph(bool shared = false, bool overridable = false,
                                      bool cast_first = false, int64_t axis = 0) {
  core::builder::GraphBuilder builder("g", [](const std::string &op) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op, false);
  });
  builder.SetOpsetVersion("", 18);
  if (overridable) {
    core::symbolic::SymShape shape;
    shape.PushBack(core::symbolic::SymDim(2));
    builder.MakeInput("weight", core::symbolic::TensorType::kFloat, shape);
  }
  builder.MakeInput("x", core::symbolic::TensorType::kDouble, [] {
    core::symbolic::SymShape shape;
    shape.PushBack(core::symbolic::SymDim(1));
    shape.PushBack(core::symbolic::SymDim(2));
    return shape;
  }());
  builder.MakeInitializer(MakeInitializer<float>("weight", {2}, {1.25f, 2.5f}));
  builder.MakeInitializer(MakeInitializer<int64_t>("axes", {1}, {axis}));
  builder.MakeNode("Unsqueeze", {"weight", "axes"}, {"expanded"});
  utils::RepeatedProtoField<AttributeProto> attributes;
  AttributeProto &to = attributes.add();
  to.set_name("to");
  to.set_type(AttributeProto::AttributeType::INT);
  to.set_i(static_cast<int64_t>(TensorProto::DataType::DOUBLE));
  builder.MakeNode("Cast", {"expanded"}, {"converted"}, "", "", attributes);
  builder.MakeNode("Add",
                   cast_first ? std::vector<std::string>{"converted", "x"}
                              : std::vector<std::string>{"x", "converted"},
                   {"y"});
  if (shared) {
    builder.MakeNode("Identity", {"expanded"}, {"extra"});
    builder.MakeOutput("extra");
  }
  builder.MakeOutput("y");
  return builder;
}

TEST(InitializerUnsqueezeCastPattern, FoldsShapeAndDtypeBeforeAdd) {
  onnx_kernels::RegisterKernelFunctions();
  for (bool cast_first : {false, true}) {
    auto builder = MakeGraph(false, false, cast_first, -2);
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

TEST(InitializerUnsqueezeCastPattern, RejectsSharedAndOverridableInputs) {
  onnx_kernels::RegisterKernelFunctions();
  onnx_patterns::InitializerUnsqueezeCastPattern pattern;
  auto shared = MakeGraph(true);
  core::builder::GraphGraph shared_graph(shared);
  EXPECT_EQ(pattern.Match(shared_graph, shared.Nodes()[1]).pattern, nullptr);

  auto overridable = MakeGraph(false, true);
  core::builder::GraphGraph overridable_graph(overridable);
  EXPECT_EQ(pattern.Match(overridable_graph, overridable.Nodes()[1]).pattern, nullptr);
  EXPECT_THROW(
      pattern.Apply(shared_graph, {&shared.Nodes()[0], &shared.Nodes()[1], &shared.Nodes()[2]}),
      core::builder::BuilderError);
}

} // namespace
