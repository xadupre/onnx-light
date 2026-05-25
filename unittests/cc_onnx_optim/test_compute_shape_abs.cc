// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/abs.h"
#include "onnx_optim/shapes/shapes_context.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeAbsNode(const std::string &input_name = "X", const std::string &output_name = "Y") {
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input(input_name);
  node.add_output(output_name);
  return node;
}

} // namespace

TEST(OnnxOptimShapesMathAbs, PropagatesFullyKnownShape) {
  NodeProto node = MakeAbsNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimTensor input(nullptr, onnx_optim::TensorType::kFloat, shape);
  ctx.Set("X", input);

  onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), input);
}

TEST(OnnxOptimShapesMathAbs, PropagatesSymbolicShape) {
  NodeProto node = MakeAbsNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  onnx_optim::OptimTensor input(nullptr, onnx_optim::TensorType::kInt64, shape);
  ctx.Set("X", input);

  onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), input);
}

TEST(OnnxOptimShapesMathAbs, UsesNodeOutputNameAsKey) {
  NodeProto node = MakeAbsNode("input0", "result");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimTensor input(nullptr, onnx_optim::TensorType::kDouble,
                                onnx_optim::OptimShape{onnx_optim::OptimDim(5)});
  ctx.Set("input0", input);

  onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "input0");

  EXPECT_TRUE(ctx.Has("result"));
  EXPECT_FALSE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("result"), input);
}

TEST(OnnxOptimShapesMathAbs, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Neg");
  node.add_input("X");
  node.add_output("Y");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAbs, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAbs, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeAbsNode();
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesContext, OpsetVersionDefaultsToUnknown) {
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_FALSE(ctx.HasOpsetVersion("ai.onnx"));
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx"), onnx_optim::shapes::kUnknownOpsetVersion);
  EXPECT_EQ(ctx.OpsetVersion(""), onnx_optim::shapes::kUnknownOpsetVersion);
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx.ml"), onnx_optim::shapes::kUnknownOpsetVersion);
}

TEST(OnnxOptimShapesContext, OpsetVersionStoreAndRetrieve) {
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 13);
  ctx.SetOpsetVersion("ai.onnx.ml", 3);

  EXPECT_TRUE(ctx.HasOpsetVersion("ai.onnx"));
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx"), 13);
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx.ml"), 3);
  EXPECT_EQ(ctx.Opsets().size(), 2u);

  // Empty domain is normalised to ai.onnx.
  EXPECT_TRUE(ctx.HasOpsetVersion(""));
  EXPECT_EQ(ctx.OpsetVersion(""), 13);

  // Replacing an existing entry overwrites the recorded version.
  ctx.SetOpsetVersion("ai.onnx", 22);
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx"), 22);
  EXPECT_EQ(ctx.Opsets().size(), 2u);
}

TEST(OnnxOptimShapesContext, ClearAlsoClearsOpsets) {
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.SetOpsetVersion("ai.onnx", 13);
  ASSERT_FALSE(ctx.Empty());
  ASSERT_TRUE(ctx.HasOpsetVersion("ai.onnx"));

  ctx.Clear();
  EXPECT_TRUE(ctx.Empty());
  EXPECT_FALSE(ctx.HasOpsetVersion("ai.onnx"));
  EXPECT_TRUE(ctx.Opsets().empty());
}

TEST(OnnxOptimShapesMathAbs, WorksWithOpsetVersionRecordedInContext) {
  NodeProto node = MakeAbsNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("ai.onnx", 13);
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor input(nullptr, onnx_optim::TensorType::kFloat, shape);
  ctx.Set("X", input);

  onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), input);
  // The opset entry must be left intact by ComputeShape*.
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx"), 13);
}

} // namespace Test
