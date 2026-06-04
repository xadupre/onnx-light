// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"
#include "onnx_optim/shapes/shape_broadcast.h"
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
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathAbs, PropagatesSymbolicShape) {
  NodeProto node = MakeAbsNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape));

  onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape));
}

TEST(OnnxOptimShapesMathAbs, UsesNodeOutputNameAsKey) {
  NodeProto node = MakeAbsNode("input0", "result");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(5)};
  ctx.Set("input0", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "input0");

  EXPECT_TRUE(ctx.Has("result"));
  EXPECT_FALSE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("result"),
            onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
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

namespace {

NodeProto MakeNegNode(const std::string &input_name = "X", const std::string &output_name = "Y") {
  NodeProto node;
  node.set_op_type("Neg");
  node.add_input(input_name);
  node.add_output(output_name);
  return node;
}

} // namespace

TEST(OnnxOptimShapesMathNeg, PropagatesFullyKnownShape) {
  NodeProto node = MakeNegNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeNeg(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathNeg, PropagatesSymbolicShape) {
  NodeProto node = MakeNegNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape));

  onnx_optim::shapes::math::ComputeShapeNeg(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape));
}

TEST(OnnxOptimShapesMathNeg, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input("X");
  node.add_output("Y");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeNeg(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathNeg, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeNegNode();
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeNeg(ctx, node, "X"), std::out_of_range);
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
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeAbs(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  // The opset entry must be left intact by ComputeShape*.
  EXPECT_EQ(ctx.OpsetVersion("ai.onnx"), 13);
}

namespace {

NodeProto MakeUnaryNode(const std::string &op_type, const std::string &input_name = "X",
                        const std::string &output_name = "Y") {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input(input_name);
  node.add_output(output_name);
  return node;
}

} // namespace

TEST(OnnxOptimShapesMathAcos, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Acos");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeAcos(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathAcos, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Acos");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeAcos(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathAcos, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Cos");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAcos(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAcos, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Acos");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAcos(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAcos, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Acos");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAcos(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathAcosh, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Acosh");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeAcosh(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathAcosh, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Acosh");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeAcosh(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathAcosh, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Cosh");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAcosh(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAcosh, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Acosh");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAcosh(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAcosh, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Acosh");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAcosh(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathAsin, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Asin");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeAsin(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathAsin, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Asin");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeAsin(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathAsin, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Sin");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAsin(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAsin, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Asin");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAsin(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAsin, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Asin");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAsin(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathAsinh, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Asinh");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeAsinh(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathAsinh, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Asinh");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeAsinh(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathAsinh, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Sinh");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAsinh(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAsinh, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Asinh");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAsinh(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAsinh, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Asinh");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAsinh(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathAtan, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Atan");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeAtan(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathAtan, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Atan");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeAtan(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathAtan, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Tan");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAtan(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAtan, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Atan");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAtan(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAtan, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Atan");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAtan(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathAtanh, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Atanh");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeAtanh(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathAtanh, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Atanh");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeAtanh(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathAtanh, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Tanh");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAtanh(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAtanh, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Atanh");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAtanh(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathAtanh, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Atanh");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAtanh(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathCos, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Cos");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeCos(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathCos, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Cos");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeCos(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathCos, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Acos");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeCos(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathCos, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Cos");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeCos(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathCos, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Cos");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeCos(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathCosh, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Cosh");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeCosh(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathCosh, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Cosh");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeCosh(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathCosh, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Acosh");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeCosh(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathCosh, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Cosh");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeCosh(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathCosh, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Cosh");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeCosh(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathExp, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Exp");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeExp(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathExp, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Exp");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeExp(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathExp, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Log");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeExp(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathExp, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Exp");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeExp(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathExp, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Exp");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeExp(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathErf, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Erf");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeErf(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathErf, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Erf");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeErf(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathErf, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Exp");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeErf(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathErf, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Erf");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeErf(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathErf, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Erf");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeErf(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathLog, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Log");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeLog(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathLog, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Log");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeLog(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathLog, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Exp");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeLog(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathLog, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Log");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeLog(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathLog, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Log");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeLog(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathSqrt, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Sqrt");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeSqrt(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathSqrt, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Sqrt");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeSqrt(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathSqrt, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Log");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeSqrt(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathSqrt, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Sqrt");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeSqrt(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathSqrt, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Sqrt");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeSqrt(ctx, node, "X"), std::out_of_range);
}

namespace {

NodeProto MakeAddNode(const std::string &a = "A", const std::string &b = "B",
                      const std::string &out = "C") {
  NodeProto node;
  node.set_op_type("Add");
  node.add_input(a);
  node.add_input(b);
  node.add_output(out);
  return node;
}

} // namespace

TEST(OnnxOptimShapesBroadcast, EqualConcreteShapes) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape out = onnx_optim::shapes::BroadcastShapes(a, b);
  EXPECT_EQ(out, a);
}

TEST(OnnxOptimShapesBroadcast, OneIsPrefix) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                           onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape out = onnx_optim::shapes::BroadcastShapes(a, b);
  EXPECT_EQ(out, a);
  // Commutativity.
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(b, a), a);
}

TEST(OnnxOptimShapesBroadcast, OneDimIsOne) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1),
                           onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3),
                           onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                  onnx_optim::OptimDim(4)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), expected);
}

TEST(OnnxOptimShapesBroadcast, ScalarBroadcastsAgainstEverything) {
  onnx_optim::OptimShape scalar{};
  onnx_optim::OptimShape a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(scalar, a), a);
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, scalar), a);
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(scalar, scalar), scalar);
}

TEST(OnnxOptimShapesBroadcast, SymbolicEqualDims) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), a);
}

TEST(OnnxOptimShapesBroadcast, SymbolicVsOnePicksSymbolic) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(1), onnx_optim::OptimDim("M")};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim("N"), onnx_optim::OptimDim("M")};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), expected);
}

TEST(OnnxOptimShapesBroadcast, SymbolicVsConcreteNonOnePicksConcrete) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim("N")};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(5)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(5)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), expected);
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(b, a), expected);
}

TEST(OnnxOptimShapesBroadcast, TwoDifferentSymbolicProducesSynthesisedExpr) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim("N")};
  onnx_optim::OptimShape b{onnx_optim::OptimDim("M")};
  onnx_optim::OptimShape out = onnx_optim::shapes::BroadcastShapes(a, b);
  ASSERT_EQ(out.Rank(), 1u);
  ASSERT_TRUE(out[0].IsExpr());
  EXPECT_EQ(out[0].AsExpr(), "broadcast(N, M)");
}

TEST(OnnxOptimShapesBroadcast, IncompatibleConcreteThrows) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(2), onnx_optim::OptimDim(4)};
  EXPECT_THROW(onnx_optim::shapes::BroadcastShapes(a, b), std::invalid_argument);
}

// The following tests mirror the broadcast_shape tests from
// yet-another-onnx-builder/unittests/xshape/test_shape_type_compute.py
// (test_broadcast_shape_*) so that the equivalent C++ behaviour is
// covered by ``BroadcastShapes``.

TEST(OnnxOptimShapesBroadcast, EmptyFirstRank2) {
  // Python: broadcast_shape((), (3, 4)) == (3, 4)
  onnx_optim::OptimShape empty{};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(empty, b), b);
}

TEST(OnnxOptimShapesBroadcast, EmptySecondRank2) {
  // Python: broadcast_shape((3, 4), ()) == (3, 4)
  onnx_optim::OptimShape a{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape empty{};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, empty), a);
}

TEST(OnnxOptimShapesBroadcast, ScalarRank1FirstBroadcastsToRank2) {
  // Python: broadcast_shape((1,), (3, 4)) == (3, 4)
  onnx_optim::OptimShape a{onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), b);
}

TEST(OnnxOptimShapesBroadcast, ScalarRank1SecondBroadcastsToRank2) {
  // Python: broadcast_shape((3, 4), (1,)) == (3, 4)
  onnx_optim::OptimShape a{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(1)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), a);
}

TEST(OnnxOptimShapesBroadcast, ExtendRank) {
  // Python: broadcast_shape((4,), (3, 4)) == (3, 4)
  onnx_optim::OptimShape a{onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), b);
}

TEST(OnnxOptimShapesBroadcast, WithOnesRank2) {
  // Python: broadcast_shape((1, 4), (3, 1)) == (3, 4)
  onnx_optim::OptimShape a{onnx_optim::OptimDim(1), onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(3), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), expected);
}

TEST(OnnxOptimShapesBroadcast, DynamicOneVsSymbolic) {
  // Python: broadcast_shape((1, "seq"), ("batch", "seq")) == ("batch", "seq")
  onnx_optim::OptimShape a{onnx_optim::OptimDim(1), onnx_optim::OptimDim("seq")};
  onnx_optim::OptimShape b{onnx_optim::OptimDim("batch"), onnx_optim::OptimDim("seq")};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(a, b), b);
}

TEST(OnnxOptimShapesBroadcast, IntOverridesOneBothDirections) {
  // Python: broadcast_shape((5,), (1,)) == (5,) and broadcast_shape((1,), (5,)) == (5,)
  onnx_optim::OptimShape five{onnx_optim::OptimDim(5)};
  onnx_optim::OptimShape one{onnx_optim::OptimDim(1)};
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(five, one), five);
  EXPECT_EQ(onnx_optim::shapes::BroadcastShapes(one, five), five);
}

TEST(OnnxOptimShapesMathAdd, PropagatesEqualShapesAndDtype) {
  NodeProto node = MakeAddNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeAdd(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesMathAdd, BroadcastsShapes) {
  NodeProto node = MakeAddNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1),
                                 onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                  onnx_optim::OptimDim(4)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape_b));

  onnx_optim::shapes::math::ComputeShapeAdd(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesMathAdd, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Sub");
  node.add_input("A");
  node.add_input("B");
  node.add_output("C");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAdd(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathAdd, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Add");
  node.add_input("A");
  node.add_input("B");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAdd(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathAdd, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeAddNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAdd(ctx, node, "A", "B"), std::out_of_range);
}

TEST(OnnxOptimShapesMathAdd, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeAddNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeAdd(ctx, node, "A", "B"),
               std::invalid_argument);
}

namespace {

NodeProto MakeBinaryNode(const std::string &op_type, const std::string &a = "A",
                         const std::string &b = "B", const std::string &out = "C") {
  NodeProto node;
  node.set_op_type(op_type);
  node.add_input(a);
  node.add_input(b);
  node.add_output(out);
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// Sub
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesMathSub, PropagatesEqualShapesAndDtype) {
  NodeProto node = MakeBinaryNode("Sub");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeSub(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesMathSub, BroadcastsShapes) {
  NodeProto node = MakeBinaryNode("Sub");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(3), onnx_optim::OptimDim(1),
                                 onnx_optim::OptimDim(5)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(4), onnx_optim::OptimDim(5)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4),
                                  onnx_optim::OptimDim(5)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape_b));

  onnx_optim::shapes::math::ComputeShapeSub(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesMathSub, RejectsWrongOpType) {
  NodeProto node = MakeBinaryNode("Add");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeSub(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathSub, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryNode("Sub");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeSub(ctx, node, "A", "B"), std::out_of_range);
}

TEST(OnnxOptimShapesMathSub, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryNode("Sub");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeSub(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Mul
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesMathMul, PropagatesEqualShapesAndDtype) {
  NodeProto node = MakeBinaryNode("Mul");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeMul(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesMathMul, BroadcastsShapes) {
  NodeProto node = MakeBinaryNode("Mul");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1), onnx_optim::OptimDim("M")};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim("N"), onnx_optim::OptimDim("M")};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape_b));

  onnx_optim::shapes::math::ComputeShapeMul(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesMathMul, RejectsWrongOpType) {
  NodeProto node = MakeBinaryNode("Add");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeMul(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathMul, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryNode("Mul");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeMul(ctx, node, "A", "B"), std::out_of_range);
}

TEST(OnnxOptimShapesMathMul, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryNode("Mul");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeMul(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// PRelu
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesMathPRelu, PropagatesEqualShapesAndDtype) {
  NodeProto node = MakeBinaryNode("PRelu");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapePRelu(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesMathPRelu, BroadcastsSlopeShape) {
  NodeProto node = MakeBinaryNode("PRelu");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_x{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                 onnx_optim::OptimDim(5)};
  onnx_optim::OptimShape shape_slope{onnx_optim::OptimDim(5)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_x));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_slope));

  onnx_optim::shapes::math::ComputeShapePRelu(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(), shape_x);
}

TEST(OnnxOptimShapesMathPRelu, RejectsWrongOpType) {
  NodeProto node = MakeBinaryNode("Mul");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapePRelu(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathPRelu, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryNode("PRelu");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapePRelu(ctx, node, "A", "B"), std::out_of_range);
}

// ---------------------------------------------------------------------------
// Div
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesMathDiv, PropagatesEqualShapesAndDtype) {
  NodeProto node = MakeBinaryNode("Div");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeDiv(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesMathDiv, BroadcastsShapes) {
  NodeProto node = MakeBinaryNode("Div");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4),
                                 onnx_optim::OptimDim(5)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(5)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4),
                                  onnx_optim::OptimDim(5)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_b));

  onnx_optim::shapes::math::ComputeShapeDiv(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesMathDiv, RejectsWrongOpType) {
  NodeProto node = MakeBinaryNode("Add");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeDiv(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathDiv, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryNode("Div");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeDiv(ctx, node, "A", "B"), std::out_of_range);
}

TEST(OnnxOptimShapesMathDiv, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryNode("Div");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeDiv(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Mod
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesMathMod, PropagatesEqualShapesAndDtype) {
  NodeProto node = MakeBinaryNode("Mod");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, shape));

  onnx_optim::shapes::math::ComputeShapeMod(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesMathMod, BroadcastsShapes) {
  NodeProto node = MakeBinaryNode("Mod");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(3), onnx_optim::OptimDim(2),
                                 onnx_optim::OptimDim(5)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(3), onnx_optim::OptimDim(2),
                                  onnx_optim::OptimDim(5)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, shape_b));

  onnx_optim::shapes::math::ComputeShapeMod(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kInt32);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesMathMod, RejectsWrongOpType) {
  NodeProto node = MakeBinaryNode("Add");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeMod(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathMod, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryNode("Mod");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeMod(ctx, node, "A", "B"), std::out_of_range);
}

TEST(OnnxOptimShapesMathMod, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeBinaryNode("Mod");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeMod(ctx, node, "A", "B"),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// MatMul
// ---------------------------------------------------------------------------
TEST(OnnxOptimShapesMathMatMul, PropagatesRank2xRank2ShapeAndDtype) {
  NodeProto node = MakeBinaryNode("MatMul");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)}));

  onnx_optim::shapes::math::ComputeShapeMatMul(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("C").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapesMathMatMul, HandlesRank1xRank2VectorMatrix) {
  NodeProto node = MakeBinaryNode("MatMul");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt32,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kInt32,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(5)}));

  onnx_optim::shapes::math::ComputeShapeMatMul(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Shape(), (onnx_optim::OptimShape{onnx_optim::OptimDim(5)}));
}

TEST(OnnxOptimShapesMathMatMul, BroadcastsBatchPrefixes) {
  NodeProto node = MakeBinaryNode("MatMul");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kDouble,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1),
                                          onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kDouble,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(1), onnx_optim::OptimDim(5),
                                          onnx_optim::OptimDim(4), onnx_optim::OptimDim(6)}));

  onnx_optim::shapes::math::ComputeShapeMatMul(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(5),
                                    onnx_optim::OptimDim(3), onnx_optim::OptimDim(6)}));
}

TEST(OnnxOptimShapesMathMatMul, RejectsWrongOpType) {
  NodeProto node = MakeBinaryNode("Mul");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeMatMul(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathMatMul, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeBinaryNode("MatMul");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeMatMul(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesMathMatMul, RejectsIncompatibleInnerDimensions) {
  NodeProto node = MakeBinaryNode("MatMul");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(5), onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeMatMul(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesMathTan, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Tan");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeTan(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathTan, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Tan");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeTan(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathTan, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Atan");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeTan(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathTan, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Tan");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeTan(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathTan, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Tan");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeTan(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathTanh, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Tanh");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeTanh(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathTanh, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Tanh");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeTanh(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathTanh, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Atanh");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeTanh(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathTanh, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Tanh");
  node.add_input("X");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeTanh(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathTanh, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Tanh");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeTanh(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathFloor, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Floor");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeFloor(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathFloor, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Ceil");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeFloor(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathFloor, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Floor");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeFloor(ctx, node, "X"), std::out_of_range);
}

TEST(OnnxOptimShapesMathCeil, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Ceil");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeCeil(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathCeil, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Floor");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeCeil(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathClip, PropagatesFullyKnownShape) {
  NodeProto node = MakeUnaryNode("Clip");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeClip(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathClip, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Clip");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));

  onnx_optim::shapes::math::ComputeShapeClip(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble, shape));
}

TEST(OnnxOptimShapesMathClip, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Ceil");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeClip(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathRound, PropagatesSymbolicShape) {
  NodeProto node = MakeUnaryNode("Round");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::math::ComputeShapeRound(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y"), onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
}

TEST(OnnxOptimShapesMathRound, RejectsWrongOpType) {
  NodeProto node = MakeUnaryNode("Floor");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, {}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeRound(ctx, node, "X"), std::invalid_argument);
}

TEST(OnnxOptimShapesMathRound, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeUnaryNode("Round");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeRound(ctx, node, "X"), std::out_of_range);
}

// ---------------------------------------------------------------------------
// Einsum
// ---------------------------------------------------------------------------
namespace {

NodeProto MakeEinsumNode(const std::string &equation, const std::vector<std::string> &inputs,
                         const std::string &output = "Y") {
  NodeProto node;
  node.set_op_type("Einsum");
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  node.add_output(output);
  AttributeProto *attr = node.add_attribute();
  attr->set_name("equation");
  attr->set_type(AttributeProto::STRING);
  attr->set_s(equation);
  return node;
}

} // namespace

TEST(OnnxOptimShapesMathEinsum, ExplicitMatMulShape) {
  NodeProto node = MakeEinsumNode("ij,jk->ik", {"A", "B"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(4)}));

  onnx_optim::shapes::math::ComputeShapeEinsum(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(4)}));
}

TEST(OnnxOptimShapesMathEinsum, ImplicitOutputSortsLabels) {
  // Implicit output: labels appearing once across all input terms, sorted.
  NodeProto node = MakeEinsumNode("ji", {"X"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::math::ComputeShapeEinsum(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  // Sorted labels are "ij", so output dim order corresponds to (i=3, j=2).
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(2)}));
}

TEST(OnnxOptimShapesMathEinsum, EllipsisBatchPropagates) {
  NodeProto node = MakeEinsumNode("...ij,...jk->...ik", {"A", "B"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(3),
                                                              onnx_optim::OptimDim(4)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kDouble,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(2),
                                                              onnx_optim::OptimDim(4),
                                                              onnx_optim::OptimDim(5)}));

  onnx_optim::shapes::math::ComputeShapeEinsum(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kDouble);
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                                    onnx_optim::OptimDim(5)}));
}

TEST(OnnxOptimShapesMathEinsum, PropagatesSymbolicDims) {
  NodeProto node = MakeEinsumNode("ij,jk->ik", {"A", "B"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim("K")}));
  ctx.Set("B", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim("K"), onnx_optim::OptimDim("M")}));

  onnx_optim::shapes::math::ComputeShapeEinsum(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_EQ(ctx.Get("Y").Shape(),
            (onnx_optim::OptimShape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim("M")}));
}

TEST(OnnxOptimShapesMathEinsum, TraceProducesScalar) {
  NodeProto node = MakeEinsumNode("ii->", {"X"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(3), onnx_optim::OptimDim(3)}));

  onnx_optim::shapes::math::ComputeShapeEinsum(ctx, node);

  ASSERT_TRUE(ctx.Has("Y"));
  EXPECT_TRUE(ctx.Get("Y").Shape().Empty());
}

TEST(OnnxOptimShapesMathEinsum, RejectsRankMismatch) {
  NodeProto node = MakeEinsumNode("i->i", {"X"});
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(
                   nullptr, onnx_optim::TensorType::kFloat,
                   onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeEinsum(ctx, node), std::invalid_argument);
}

TEST(OnnxOptimShapesMathEinsum, RejectsWrongOpType) {
  NodeProto node = MakeEinsumNode("i->i", {"X"});
  node.set_op_type("Mul");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeEinsum(ctx, node), std::invalid_argument);
}

namespace {

NodeProto MakeTopKNode(bool with_k_attribute, int64_t axis = -1) {
  NodeProto node;
  node.set_op_type("TopK");
  node.add_input("X");
  if (!with_k_attribute) {
    node.add_input("K");
  }
  node.add_output("Values");
  node.add_output("Indices");
  if (with_k_attribute) {
    AttributeProto *k_attr = node.add_attribute();
    k_attr->set_name("k");
    k_attr->set_type(AttributeProto::INT);
    k_attr->set_i(3);
  }
  AttributeProto *axis_attr = node.add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(AttributeProto::INT);
  axis_attr->set_i(axis);
  return node;
}

} // namespace

TEST(OnnxOptimShapesMathTopK, ResolvesKFromAttribute) {
  NodeProto node = MakeTopKNode(/*with_k_attribute=*/true, /*axis=*/-1);
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape in_shape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(5)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, in_shape));

  onnx_optim::shapes::math::ComputeShapeTopK(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Values"));
  ASSERT_TRUE(ctx.Has("Indices"));
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(4), onnx_optim::OptimDim(3)};
  EXPECT_EQ(ctx.Get("Values"),
            onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, expected));
  EXPECT_EQ(ctx.Get("Indices"),
            onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, expected));
}

TEST(OnnxOptimShapesMathTopK, EmitsSymbolicDimWhenKIsTensorInput) {
  NodeProto node = MakeTopKNode(/*with_k_attribute=*/false, /*axis=*/0);
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape in_shape{onnx_optim::OptimDim(4), onnx_optim::OptimDim(5)};
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, in_shape));
  ctx.Set("K", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(1)}));

  onnx_optim::shapes::math::ComputeShapeTopK(ctx, node, "X");

  ASSERT_TRUE(ctx.Has("Values"));
  const onnx_optim::OptimTensor &values = ctx.Get("Values");
  ASSERT_EQ(values.Shape().Rank(), 2u);
  EXPECT_TRUE(values.Shape()[0].IsExpr());
  EXPECT_EQ(values.Shape()[1], onnx_optim::OptimDim(5));
  EXPECT_EQ(values.Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.Get("Indices").Dtype(), onnx_optim::TensorType::kInt64);
}

TEST(OnnxOptimShapesMathTopK, RejectsWrongOpType) {
  NodeProto node = MakeTopKNode(/*with_k_attribute=*/true);
  node.set_op_type("Mul");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("X", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::math::ComputeShapeTopK(ctx, node, "X"), std::invalid_argument);
}

} // namespace Test
