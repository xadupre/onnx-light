// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/algebra/common_pattern.h"
#include "onnx_extensions/patterns/algebra/mul_pattern.h"
#include "onnx_extensions/patterns/algebra/range_pattern.h"
#include "onnx_extensions/patterns/algebra/reduce_pattern.h"
#include "onnx_extensions/patterns/algebra/shape_pattern.h"
#include "onnx_extensions/patterns/algebra/sub_pattern.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, false);
  };
}

core::symbolic::SymShape Shape(const std::vector<int64_t> &dims) {
  core::symbolic::SymShape shape;
  for (int64_t dim : dims) {
    shape.PushBack(core::symbolic::SymDim(dim));
  }
  return shape;
}

void AddFloatInitializer(core::builder::GraphBuilder &builder, const std::string &name,
                         const std::vector<int64_t> &dims, const std::vector<float> &values) {
  builder.MakeInitializer(MakeInitializer<float>(name.c_str(), dims, values));
}

void AddIntInitializer(core::builder::GraphBuilder &builder, const std::string &name,
                       const std::vector<int64_t> &dims, const std::vector<int64_t> &values) {
  builder.MakeInitializer(MakeInitializer<int64_t>(name.c_str(), dims, values));
}

void AddFloat16Initializer(core::builder::GraphBuilder &builder, const std::string &name,
                           const std::vector<int64_t> &dims, const std::vector<uint16_t> &values) {
  builder.MakeInitializer(MakeInitializer<uint16_t>(name.c_str(), dims, values));
}

void AddConstantOfShape(core::builder::GraphBuilder &builder, const std::string &shape,
                        const std::string &output, float value) {
  NodeProto node = MakeNode("ConstantOfShape", {shape}, {output});
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name("value");
  attribute->set_type(AttributeProto::AttributeType::TENSOR);
  *attribute->mutable_t() = MakeInitializer<float>("", {1}, {value});
  builder.MakeNode("ConstantOfShape", {shape}, {output}, "", "", node.attribute());
}

NodeProto NodeWithIntAttribute(const char *op_type, const std::vector<std::string> &inputs,
                               const std::vector<std::string> &outputs, const char *attribute,
                               int64_t value) {
  NodeProto node = MakeNode(op_type, inputs, outputs);
  AddAttribute<int64_t>(node, attribute, value);
  return node;
}

TEST(MulMulMulScalarPattern, CombinesTwoDivisions) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2}));
  AddFloatInitializer(builder, "two", {}, {2.0F});
  AddFloatInitializer(builder, "three", {}, {3.0F});
  builder.MakeNode("Div", {"x", "two"}, {"left"});
  builder.MakeNode("Div", {"y", "three"}, {"right"});
  builder.MakeNode("Mul", {"left", "right"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::MulMulMulScalarPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Mul");
  EXPECT_EQ(replacements[1].op_type().value(), "Div");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
  ASSERT_EQ(builder.Initializers().size(), 3u);
  EXPECT_EQ(builder.Initializers()[2].float_data()[0], 6.0F);
}

TEST(MulMulMulScalarPattern, RejectsSharedInnerResult) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2}));
  AddFloatInitializer(builder, "two", {}, {2.0F});
  AddFloatInitializer(builder, "three", {}, {3.0F});
  builder.MakeNode("Div", {"x", "two"}, {"left"});
  builder.MakeNode("Div", {"y", "three"}, {"right"});
  builder.MakeNode("Mul", {"left", "right"}, {"out"});
  builder.MakeNode("Identity", {"left"}, {"shared"});
  builder.MakeOutput("out");
  builder.MakeOutput("shared");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::MulMulMulScalarPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[2]).pattern, nullptr);
}

TEST(MulMulMulScalarPattern, CombinesFloat16DivisionConstants) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat16, Shape({2}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat16, Shape({2}));
  AddFloat16Initializer(builder, "two", {}, {0x4000});
  AddFloat16Initializer(builder, "three", {}, {0x4200});
  builder.MakeNode("Div", {"x", "two"}, {"left"});
  builder.MakeNode("Div", {"y", "three"}, {"right"});
  builder.MakeNode("Mul", {"left", "right"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::MulMulMulScalarPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  ASSERT_EQ(match.pattern, &pattern);
  (void)pattern.Apply(graph, match.nodes);
  ASSERT_EQ(builder.Initializers().size(), 3u);
  EXPECT_EQ(builder.Initializers()[2].data_type(), TensorProto::DataType::FLOAT16);
  EXPECT_EQ(builder.Initializers()[2].int32_data()[0], 0x4600);
}

TEST(SwitchOrderBinaryPattern, MovesTheSmallBroadcastBeforeTheLargeOne) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, Shape({2, 2, 3, 4}));
  builder.MakeInput("c", core::symbolic::TensorType::kFloat, Shape({2, 1, 3, 4}));
  builder.MakeInput("a", core::symbolic::TensorType::kFloat, Shape({2, 1, 3, 4}));
  builder.MakeNode("Add", {"b", "c"}, {"bc"});
  builder.MakeNode("Add", {"bc", "a"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwitchOrderBinaryPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].input()[0].value(), "c");
  EXPECT_EQ(replacements[0].input()[1].value(), "a");
  EXPECT_EQ(replacements[1].input()[1].value(), "b");
}

TEST(SwitchOrderBinaryPattern, RejectsEqualBroadcastCosts) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("b", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeInput("c", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeInput("a", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeNode("Add", {"b", "c"}, {"bc"});
  builder.MakeNode("Add", {"bc", "a"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwitchOrderBinaryPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

TEST(SwapRangeAddScalarPattern, MovesTheScalarIntoRangeBounds) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  AddIntInitializer(builder, "start", {}, {0});
  AddIntInitializer(builder, "end", {}, {5});
  AddIntInitializer(builder, "delta", {}, {1});
  AddIntInitializer(builder, "offset", {1}, {2});
  builder.MakeNode("Range", {"start", "end", "delta"}, {"r"});
  builder.MakeNode("Add", {"r", "offset"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwapRangeAddScalarPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 3u);
  EXPECT_EQ(replacements[0].op_type().value(), "Squeeze");
  EXPECT_EQ(replacements[1].op_type().value(), "Add");
  EXPECT_EQ(replacements[2].op_type().value(), "Range");
  EXPECT_EQ(replacements[2].input()[0].value(), replacements[0].output()[0].value());
}

TEST(SwapRangeAddScalarPattern, RejectsScalarInsteadOfOneElementVector) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  AddIntInitializer(builder, "start", {}, {0});
  AddIntInitializer(builder, "end", {}, {5});
  AddIntInitializer(builder, "delta", {}, {1});
  AddIntInitializer(builder, "offset", {}, {2});
  builder.MakeNode("Range", {"start", "end", "delta"}, {"r"});
  builder.MakeNode("Add", {"r", "offset"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwapRangeAddScalarPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(Sub1MulPattern, ReassociatesOneMinusProduct) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2}));
  AddFloatInitializer(builder, "one", {1}, {1.0F});
  builder.MakeNode("Sub", {"one", "x"}, {"one_minus_x"});
  builder.MakeNode("Mul", {"one_minus_x", "y"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::Sub1MulPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Mul");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].op_type().value(), "Sub");
  EXPECT_EQ(replacements[1].input()[0].value(), "y");
}

TEST(Sub1MulPattern, RejectsAConstantOtherThanOne) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2}));
  AddFloatInitializer(builder, "two", {1}, {2.0F});
  builder.MakeNode("Sub", {"two", "x"}, {"two_minus_x"});
  builder.MakeNode("Mul", {"two_minus_x", "y"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::Sub1MulPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

TEST(Sub1MulPattern, ReassociatesConstantOfShapeOnBothSides) {
  for (bool sub_on_left : {true, false}) {
    core::builder::GraphBuilder builder(sub_on_left ? "left" : "right", SchemaLookup());
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
    builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2, 3}));
    AddIntInitializer(builder, "shape", {2}, {1, 3});
    AddConstantOfShape(builder, "shape", "one", 1.0F);
    builder.MakeNode("Sub", {"one", sub_on_left ? "x" : "y"}, {"one_minus"});
    builder.MakeNode("Mul",
                     sub_on_left ? std::vector<std::string>{"one_minus", "y"}
                                 : std::vector<std::string>{"x", "one_minus"},
                     {"out"});
    builder.MakeOutput("out");

    core::builder::GraphGraph graph(builder);
    onnx_patterns::Sub1MulPattern pattern;
    const auto match = pattern.Match(graph, builder.Nodes()[2]);
    ASSERT_EQ(match.pattern, &pattern) << (sub_on_left ? "left" : "right");
    const auto replacements = pattern.Apply(graph, match.nodes);
    ASSERT_EQ(replacements.size(), 2u);
    EXPECT_EQ(replacements[0].op_type().value(), "Mul");
    EXPECT_EQ(replacements[1].op_type().value(), "Sub");
    EXPECT_EQ(replacements[1].output()[0].value(), "out");
  }
}

TEST(Sub1MulPattern, RejectsNonOneConstantOfShape) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddIntInitializer(builder, "shape", {2}, {1, 3});
  AddConstantOfShape(builder, "shape", "two", 2.0F);
  builder.MakeNode("Sub", {"two", "x"}, {"two_minus"});
  builder.MakeNode("Mul", {"two_minus", "y"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::Sub1MulPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[2]).pattern, nullptr);
}

TEST(Sub1MulPattern, RejectsConstantOfShapeChangingBroadcastRank) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({3}));
  AddIntInitializer(builder, "shape", {3}, {2, 1, 3});
  AddConstantOfShape(builder, "shape", "one", 1.0F);
  builder.MakeNode("Sub", {"one", "x"}, {"one_minus"});
  builder.MakeNode("Mul", {"one_minus", "y"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::Sub1MulPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason,
            "removing ConstantOfShape would change or cannot prove the output shape");
}

TEST(Sub1MulPattern, PreservesExternalConstantOfShapeOutput) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddIntInitializer(builder, "shape", {2}, {1, 3});
  AddConstantOfShape(builder, "shape", "one", 1.0F);
  builder.MakeNode("Sub", {"one", "x"}, {"one_minus"});
  builder.MakeNode("Mul", {"one_minus", "y"}, {"out"});
  builder.MakeOutput("out");
  builder.MakeOutput("one");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns;
  patterns.push_back(std::make_unique<onnx_patterns::Sub1MulPattern>());
  core::builder::GraphGraph graph(builder, std::move(patterns));
  graph.Optimize();

  ASSERT_EQ(builder.Nodes().size(), 3u);
  EXPECT_EQ(builder.Nodes()[0].op_type().value(), "ConstantOfShape");
  EXPECT_EQ(builder.Nodes()[1].op_type().value(), "Mul");
  EXPECT_EQ(builder.Nodes()[2].op_type().value(), "Sub");
  EXPECT_EQ(builder.Nodes()[0].output()[0].value(), "one");
}

TEST(ReduceArgTopKPattern, ReplacesSiblingReduceAndArgMax) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddIntInitializer(builder, "axes", {1}, {1});
  builder.MakeNode("ReduceMax", {"x", "axes"}, {"values"});
  NodeProto arg = NodeWithIntAttribute("ArgMax", {"x"}, {"indices"}, "axis", 1);
  builder.MakeNode("ArgMax", {"x"}, {"indices"}, "", "", arg.attribute());
  builder.MakeOutput("values");
  builder.MakeOutput("indices");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReduceArgTopKPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "TopK");
  EXPECT_EQ(GetAttributeOr<int64_t>(replacements[0], "largest", 0), 1);
  EXPECT_EQ(replacements[0].output()[0].value(), "values");
  EXPECT_EQ(replacements[0].output()[1].value(), "indices");
}

TEST(ReduceArgTopKPattern, RejectsPre18Opset) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 17);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeNode("ArgMax", {"x"}, {"indices"});
  builder.MakeOutput("indices");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReduceArgTopKPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ReduceSumNormalizePattern, MovesReductionBeforeTheInputCast) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddIntInitializer(builder, "axes", {1}, {1});
  AddFloatInitializer(builder, "scale", {}, {0.5F});
  NodeProto cast = NodeWithIntAttribute("Cast", {"x"}, {"xc"}, "to",
                                        static_cast<int64_t>(TensorProto::DataType::DOUBLE));
  builder.MakeNode("Cast", {"x"}, {"xc"}, "", "", cast.attribute());
  builder.MakeNode("ReduceSum", {"xc", "axes"}, {"sum"});
  builder.MakeNode("Mul", {"sum", "scale"}, {"scaled"});
  builder.MakeNode("Sub", {"xc", "scaled"}, {"normalised"});
  NodeProto cast2 = NodeWithIntAttribute("Cast", {"normalised"}, {"out"}, "to",
                                         static_cast<int64_t>(TensorProto::DataType::FLOAT));
  builder.MakeNode("Cast", {"normalised"}, {"out"}, "", "", cast2.attribute());
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReduceSumNormalizePattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[1]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 4u);
  EXPECT_EQ(replacements[0].op_type().value(), "ReduceSum");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[3].op_type().value(), "Sub");
  EXPECT_EQ(replacements[3].output()[0].value(), "out");
}

TEST(ReduceSumNormalizePattern, RejectsDifferentInputAndOutputTypes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.SetOpsetVersion("", 18);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddIntInitializer(builder, "axes", {1}, {1});
  AddFloatInitializer(builder, "scale", {}, {0.5F});
  NodeProto cast = NodeWithIntAttribute("Cast", {"x"}, {"xc"}, "to",
                                        static_cast<int64_t>(TensorProto::DataType::DOUBLE));
  builder.MakeNode("Cast", {"x"}, {"xc"}, "", "", cast.attribute());
  builder.MakeNode("ReduceSum", {"xc", "axes"}, {"sum"});
  builder.MakeNode("Mul", {"sum", "scale"}, {"scaled"});
  builder.MakeNode("Sub", {"xc", "scaled"}, {"normalised"});
  NodeProto cast2 = NodeWithIntAttribute("Cast", {"normalised"}, {"out"}, "to",
                                         static_cast<int64_t>(TensorProto::DataType::DOUBLE));
  builder.MakeNode("Cast", {"normalised"}, {"out"}, "", "", cast2.attribute());
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ReduceSumNormalizePattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

TEST(SameChildrenPattern, ReplacesOneDuplicateChildWithIdentity) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2}));
  builder.MakeNode("Identity", {"x"}, {"shared"});
  builder.MakeNode("Neg", {"shared"}, {"left"});
  builder.MakeNode("Neg", {"shared"}, {"right"});
  builder.MakeOutput("left");
  builder.MakeOutput("right");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SameChildrenPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Neg");
  EXPECT_EQ(replacements[1].op_type().value(), "Identity");
  EXPECT_EQ(replacements[1].input()[0].value(), "left");
  EXPECT_EQ(replacements[1].output()[0].value(), "right");
}

TEST(SameChildrenPattern, RejectsDifferentChildren) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2}));
  builder.MakeNode("Identity", {"x"}, {"shared"});
  builder.MakeNode("Neg", {"shared"}, {"left"});
  builder.MakeNode("Add", {"shared", "y"}, {"right"});
  builder.MakeOutput("left");
  builder.MakeOutput("right");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SameChildrenPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(SameChildrenFromInputPattern, ReplacesDuplicateInputConsumers) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2}));
  NodeProto first = NodeWithIntAttribute("Cast", {"x"}, {"left"}, "to",
                                         static_cast<int64_t>(TensorProto::DataType::DOUBLE));
  NodeProto second = NodeWithIntAttribute("Cast", {"x"}, {"right"}, "to",
                                          static_cast<int64_t>(TensorProto::DataType::DOUBLE));
  builder.MakeNode("Cast", {"x"}, {"left"}, "", "", first.attribute());
  builder.MakeNode("Cast", {"x"}, {"right"}, "", "", second.attribute());
  builder.MakeOutput("left");
  builder.MakeOutput("right");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SameChildrenFromInputPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[1].op_type().value(), "Identity");
}

TEST(SameChildrenFromInputPattern, RejectsIntermediateInputs) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2}));
  builder.MakeNode("Identity", {"x"}, {"shared"});
  builder.MakeNode("Neg", {"shared"}, {"left"});
  builder.MakeNode("Neg", {"shared"}, {"right"});
  builder.MakeOutput("left");
  builder.MakeOutput("right");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SameChildrenFromInputPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[1]).pattern, nullptr);
}

TEST(ShapeBasedSameChildrenPattern, UsesTheFirstEquivalentExpand) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 3}));
  AddIntInitializer(builder, "shape1", {2}, {2, 3});
  AddIntInitializer(builder, "shape2", {2}, {2, 3});
  builder.MakeNode("Expand", {"x", "shape1"}, {"left"});
  builder.MakeNode("Expand", {"x", "shape2"}, {"right"});
  builder.MakeOutput("left");
  builder.MakeOutput("right");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedSameChildrenPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[1].op_type().value(), "Identity");
  EXPECT_EQ(replacements[1].input()[0].value(), "left");
}

TEST(ShapeBasedSameChildrenPattern, RejectsDifferentOutputShapes) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({1, 3}));
  AddIntInitializer(builder, "shape1", {2}, {2, 3});
  AddIntInitializer(builder, "shape2", {2}, {3, 3});
  builder.MakeNode("Expand", {"x", "shape1"}, {"left"});
  builder.MakeNode("Expand", {"x", "shape2"}, {"right"});
  builder.MakeOutput("left");
  builder.MakeOutput("right");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedSameChildrenPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ShapeBasedIdentityPattern, ReplacesFullSliceWithIdentity) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddIntInitializer(builder, "starts", {1}, {0});
  AddIntInitializer(builder, "ends", {1}, {2});
  AddIntInitializer(builder, "axes", {1}, {0});
  AddIntInitializer(builder, "steps", {1}, {1});
  builder.MakeNode("Slice", {"x", "starts", "ends", "axes", "steps"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedIdentityPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 1u);
  EXPECT_EQ(replacements[0].op_type().value(), "Identity");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
}

TEST(ShapeBasedIdentityPattern, RejectsNonUnitSteps) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddIntInitializer(builder, "starts", {1}, {0});
  AddIntInitializer(builder, "ends", {1}, {2});
  AddIntInitializer(builder, "axes", {1}, {0});
  AddIntInitializer(builder, "steps", {1}, {-1});
  builder.MakeNode("Slice", {"x", "starts", "ends", "axes", "steps"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedIdentityPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(SwapUnaryPattern, MovesExpBeforeTranspose) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  NodeProto transpose = MakeNode("Transpose", {"x"}, {"xt"});
  AddAttribute<std::vector<int64_t>>(transpose, "perm", {1, 0});
  builder.MakeNode("Transpose", {"x"}, {"xt"}, "", "", transpose.attribute());
  builder.MakeNode("Exp", {"xt"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwapUnaryPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[0]);
  ASSERT_EQ(match.pattern, &pattern);
  const auto replacements = pattern.Apply(graph, match.nodes);
  ASSERT_EQ(replacements.size(), 2u);
  EXPECT_EQ(replacements[0].op_type().value(), "Exp");
  EXPECT_EQ(replacements[0].input()[0].value(), "x");
  EXPECT_EQ(replacements[1].op_type().value(), "Transpose");
  EXPECT_EQ(replacements[1].output()[0].value(), "out");
}

TEST(SwapUnaryPattern, RejectsScalarBinaryConstant) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  AddFloatInitializer(builder, "scalar", {}, {2.0F});
  NodeProto transpose = MakeNode("Transpose", {"x"}, {"xt"});
  AddAttribute<std::vector<int64_t>>(transpose, "perm", {1, 0});
  builder.MakeNode("Transpose", {"x"}, {"xt"}, "", "", transpose.attribute());
  builder.MakeNode("Mul", {"xt", "scalar"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::SwapUnaryPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

TEST(ShapeBasedShapeShapeAddPattern, PreservesTheUpstreamNoMatch) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeInput("y", core::symbolic::TensorType::kFloat, Shape({2, 3}));
  builder.MakeNode("Shape", {"x"}, {"sx"});
  builder.MakeNode("Shape", {"y"}, {"sy"});
  builder.MakeNode("Add", {"sx", "sy"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedShapeShapeAddPattern pattern;
  const auto match = pattern.Match(graph, builder.Nodes()[2]);
  EXPECT_EQ(match.pattern, nullptr);
  ASSERT_TRUE(match.no_match.has_value());
  EXPECT_EQ(match.no_match->reason, "the upstream Shape plus Shape rewrite is not implemented");
}

TEST(ShapeBasedShapeShapeAddPattern, RejectsNonShapeInputs) {
  core::builder::GraphBuilder builder("g", SchemaLookup());
  builder.MakeInput("x", core::symbolic::TensorType::kInt64, Shape({2}));
  builder.MakeInput("y", core::symbolic::TensorType::kInt64, Shape({2}));
  builder.MakeNode("Add", {"x", "y"}, {"out"});
  builder.MakeOutput("out");

  core::builder::GraphGraph graph(builder);
  onnx_patterns::ShapeBasedShapeShapeAddPattern pattern;
  EXPECT_EQ(pattern.Match(graph, builder.Nodes()[0]).pattern, nullptr);
}

} // namespace
} // namespace Test
