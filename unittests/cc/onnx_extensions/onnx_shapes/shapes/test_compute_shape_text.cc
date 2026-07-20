// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/onnx_shapes/shapes/text/shape_text.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeStringSplitNode(const std::string &in = "X", const std::string &substrings = "Y",
                              const std::string &length = "Z") {
  NodeProto node;
  node.set_op_type("StringSplit");
  node.add_input(in);
  node.add_output(substrings);
  node.add_output(length);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTextStringSplit, PropagatesInputShapeToBothOutputs) {
  NodeProto node = MakeStringSplitNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape input_shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("X",
          core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, input_shape));

  onnx_shapes::shapes::text::ComputeShapeStringSplit(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_TRUE(ctx.Has("Z"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kString);
  EXPECT_EQ(ctx.Get("Z").Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("Y").Shape()[0], core::symbolic::SymDim(2));
  EXPECT_EQ(ctx.Get("Y").Shape()[1], core::symbolic::SymDim(3));
  EXPECT_EQ(ctx.Get("Y").Shape()[2], core::symbolic::SymDim("StringSplit(X)"));
  EXPECT_EQ(ctx.Get("Z").Shape(), input_shape);
}

TEST(OnnxOptimShapesTextStringSplit, ScalarInputProducesRankOneSubstrings) {
  NodeProto node = MakeStringSplitNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, {}));

  onnx_shapes::shapes::text::ComputeShapeStringSplit(ctx, node, "X");

  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 1u);
  EXPECT_EQ(ctx.Get("Y").Shape()[0], core::symbolic::SymDim("StringSplit(X)"));
  EXPECT_TRUE(ctx.Get("Z").Shape().Empty());
}

TEST(OnnxOptimShapesTextStringSplit, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("And");
  node.add_input("X");
  node.add_output("Y");
  node.add_output("Z");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::text::ComputeShapeStringSplit(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTextStringSplit, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeStringSplitNode();
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::text::ComputeShapeStringSplit(ctx, node, "X"),
               std::out_of_range);
}

namespace {

NodeProto MakeStringConcatNode(const std::string &a = "A", const std::string &b = "B",
                               const std::string &out = "C") {
  NodeProto node;
  node.set_op_type("StringConcat");
  node.add_input(a);
  node.add_input(b);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTextStringConcat, PropagatesEqualShapesWithStringDtype) {
  NodeProto node = MakeStringConcatNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, shape));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, shape));

  onnx_shapes::shapes::text::ComputeShapeStringConcat(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kString);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesTextStringConcat, BroadcastsShapes) {
  NodeProto node = MakeStringConcatNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(1)};
  core::symbolic::SymShape shape_b{core::symbolic::SymDim(1), core::symbolic::SymDim(3)};
  core::symbolic::SymShape expected{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, shape_b));

  onnx_shapes::shapes::text::ComputeShapeStringConcat(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kString);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesTextStringConcat, ScalarBroadcast) {
  NodeProto node = MakeStringConcatNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape_a{core::symbolic::SymDim(2), core::symbolic::SymDim(2)};
  core::symbolic::SymShape shape_b{};
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, shape_a));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, shape_b));

  onnx_shapes::shapes::text::ComputeShapeStringConcat(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), core::symbolic::TensorType::kString);
  EXPECT_EQ(ctx.Get("C").Shape(), shape_a);
}

TEST(OnnxOptimShapesTextStringConcat, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("And");
  node.add_input("A");
  node.add_input("B");
  node.add_output("C");
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, {}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, {}));
  EXPECT_THROW(onnx_shapes::shapes::text::ComputeShapeStringConcat(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTextStringConcat, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeStringConcatNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, {}));
  EXPECT_THROW(onnx_shapes::shapes::text::ComputeShapeStringConcat(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesTextStringConcat, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeStringConcatNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("A", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  ctx.Set("B", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString,
                                         core::symbolic::SymShape{core::symbolic::SymDim(4)}));
  EXPECT_THROW(onnx_shapes::shapes::text::ComputeShapeStringConcat(ctx, node, "A", "B"),
               std::invalid_argument);
}

namespace {

NodeProto MakeStringNormalizerNode(const std::string &in = "X", const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("StringNormalizer");
  node.add_input(in);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTextStringNormalizer, OneDimensionalProducesSymbolicLastDim) {
  NodeProto node = MakeStringNormalizerNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString,
                                         core::symbolic::SymShape{core::symbolic::SymDim(5)}));

  onnx_shapes::shapes::text::ComputeShapeStringNormalizer(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kString);
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 1u);
  // The output dimension is symbolic because it depends on the
  // ``stopwords`` attribute at runtime.
  EXPECT_FALSE(ctx.Get("Y").Shape()[0].IsInt());
}

TEST(OnnxOptimShapesTextStringNormalizer, TwoDimensionalKeepsLeadingOne) {
  NodeProto node = MakeStringNormalizerNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kString,
                   core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(5)}));

  onnx_shapes::shapes::text::ComputeShapeStringNormalizer(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kString);
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 2u);
  ASSERT_TRUE(ctx.Get("Y").Shape()[0].IsInt());
  EXPECT_EQ(ctx.Get("Y").Shape()[0].AsInt(), 1);
  EXPECT_FALSE(ctx.Get("Y").Shape()[1].IsInt());
}

TEST(OnnxOptimShapesTextStringNormalizer, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("And");
  node.add_input("X");
  node.add_output("Y");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::text::ComputeShapeStringNormalizer(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTextStringNormalizer, RejectsBadRank) {
  NodeProto node = MakeStringNormalizerNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString,
                                         core::symbolic::SymShape{core::symbolic::SymDim(1),
                                                                  core::symbolic::SymDim(1),
                                                                  core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::text::ComputeShapeStringNormalizer(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTextStringNormalizer, RejectsTwoDimLeadingNotOne) {
  NodeProto node = MakeStringNormalizerNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(
                   nullptr, core::symbolic::TensorType::kString,
                   core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::text::ComputeShapeStringNormalizer(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTextStringNormalizer, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeStringNormalizerNode();
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::text::ComputeShapeStringNormalizer(ctx, node, "X"),
               std::out_of_range);
}

namespace {

NodeProto MakeRegexFullMatchNode(const std::string &in = "X", const std::string &out = "Y") {
  NodeProto node;
  node.set_op_type("RegexFullMatch");
  node.add_input(in);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesTextRegexFullMatch, PropagatesInputShapeAsBoolOutput) {
  NodeProto node = MakeRegexFullMatchNode();
  core::shapes::ShapesContext ctx;
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, shape));

  onnx_shapes::shapes::text::ComputeShapeRegexFullMatch(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_EQ(ctx.Get("Y").Shape(), shape);
}

TEST(OnnxOptimShapesTextRegexFullMatch, ScalarInputProducesScalarBool) {
  NodeProto node = MakeRegexFullMatchNode();
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString, {}));

  onnx_shapes::shapes::text::ComputeShapeRegexFullMatch(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), core::symbolic::TensorType::kBool);
  EXPECT_TRUE(ctx.Get("Y").Shape().Empty());
}

TEST(OnnxOptimShapesTextRegexFullMatch, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("And");
  node.add_input("X");
  node.add_output("Y");
  core::shapes::ShapesContext ctx;
  ctx.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kString,
                                         core::symbolic::SymShape{core::symbolic::SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::text::ComputeShapeRegexFullMatch(ctx, node, "X"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTextRegexFullMatch, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeRegexFullMatchNode();
  core::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_shapes::shapes::text::ComputeShapeRegexFullMatch(ctx, node, "X"),
               std::out_of_range);
}

} // namespace Test
