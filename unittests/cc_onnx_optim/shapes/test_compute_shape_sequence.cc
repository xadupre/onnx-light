// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/sequence/shape_sequence.h"

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

NodeProto MakeSequenceConstructNode(const std::vector<std::string> &inputs,
                                    const std::string &output) {
  NodeProto node;
  node.set_op_type("SequenceConstruct");
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  node.add_output(output);
  return node;
}

} // namespace

TEST(OnnxOptimShapeSequenceConstruct, ThreeInputsCommonShape) {
  NodeProto node = MakeSequenceConstructNode({"a", "b", "c"}, "y");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("c", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  const onnx_optim::OptimSequence &seq = ctx.GetSequence("y");
  EXPECT_TRUE(seq.HasElemDtype());
  EXPECT_EQ(seq.ElemDtype(), onnx_optim::TensorType::kFloat);
  EXPECT_TRUE(seq.HasElemShape());
  EXPECT_EQ(seq.ElemShape(), shape);
  ASSERT_TRUE(seq.Length().IsInt());
  EXPECT_EQ(seq.Length().AsInt(), 3);
}

TEST(OnnxOptimShapeSequenceConstruct, SymbolicDimsAreMergedToPlaceholder) {
  NodeProto node = MakeSequenceConstructNode({"a", "b"}, "y");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(2), onnx_optim::OptimDim(5)};
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape_a));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape_b));

  onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  const onnx_optim::OptimSequence &seq = ctx.GetSequence("y");
  EXPECT_EQ(seq.ElemDtype(), onnx_optim::TensorType::kInt64);
  ASSERT_TRUE(seq.HasElemShape());
  ASSERT_EQ(seq.ElemShape().Rank(), 2u);
  EXPECT_EQ(seq.ElemShape()[0], onnx_optim::OptimDim(2));
  EXPECT_TRUE(seq.ElemShape()[1].IsExpr());
  EXPECT_EQ(seq.ElemShape()[1].AsExpr(), "?");
  ASSERT_TRUE(seq.Length().IsInt());
  EXPECT_EQ(seq.Length().AsInt(), 2);
}

TEST(OnnxOptimShapeSequenceConstruct, RankMismatchDropsElementShape) {
  NodeProto node = MakeSequenceConstructNode({"a", "b"}, "y");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(4)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(2), onnx_optim::OptimDim(2)};
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_a));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape_b));

  onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  const onnx_optim::OptimSequence &seq = ctx.GetSequence("y");
  EXPECT_FALSE(seq.HasElemShape());
  EXPECT_EQ(seq.Length().AsInt(), 2);
}

TEST(OnnxOptimShapeSequenceConstruct, DtypeMismatchThrows) {
  NodeProto node = MakeSequenceConstructNode({"a", "b"}, "y");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(3)};
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kInt64, shape));

  EXPECT_THROW(onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeSequenceConstruct, ZeroInputsProducesEmptySequence) {
  NodeProto node = MakeSequenceConstructNode({}, "y");
  onnx_optim::shapes::ShapesContext ctx;

  onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  const onnx_optim::OptimSequence &seq = ctx.GetSequence("y");
  EXPECT_FALSE(seq.HasElemDtype());
  EXPECT_FALSE(seq.HasElemShape());
  ASSERT_TRUE(seq.Length().IsInt());
  EXPECT_EQ(seq.Length().AsInt(), 0);
}

TEST(OnnxOptimShapeSequenceConstruct, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("NotSequenceConstruct");
  node.add_output("y");
  onnx_optim::shapes::ShapesContext ctx;
  EXPECT_THROW(onnx_optim::shapes::sequence::ComputeShapeSequenceConstruct(ctx, node),
               std::invalid_argument);
}

TEST(OnnxOptimShapeInference, DispatchesSequenceConstruct) {
  NodeProto node = MakeSequenceConstructNode({"a", "b"}, "y");
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(4)};
  ctx.Set("a", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.Set("b", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));

  onnx_optim::shapes::ComputeShapeNode(ctx, node);

  ASSERT_TRUE(ctx.HasSequence("y"));
  EXPECT_EQ(ctx.GetSequence("y").Length().AsInt(), 2);
  EXPECT_EQ(ctx.GetSequence("y").ElemDtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(ctx.GetSequence("y").ElemShape(), shape);
}

TEST(OnnxOptimOptimSequence, DefaultsAndSetters) {
  onnx_optim::OptimSequence seq;
  EXPECT_FALSE(seq.HasElemDtype());
  EXPECT_FALSE(seq.HasElemShape());
  EXPECT_TRUE(seq.Length().IsInt());
  EXPECT_EQ(seq.Length().AsInt(), 0);

  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim("N")};
  seq.SetElemDtype(onnx_optim::TensorType::kInt64);
  seq.SetElemShape(shape);
  seq.SetLength(onnx_optim::OptimDim("L"));
  EXPECT_TRUE(seq.HasElemDtype());
  EXPECT_TRUE(seq.HasElemShape());
  EXPECT_EQ(seq.ElemDtype(), onnx_optim::TensorType::kInt64);
  EXPECT_EQ(seq.ElemShape(), shape);
  EXPECT_TRUE(seq.Length().IsExpr());
  EXPECT_EQ(seq.Length().AsExpr(), "L");

  seq.ClearElemShape();
  EXPECT_FALSE(seq.HasElemShape());
  seq.SetElemDtype(onnx_optim::TensorType::kUndefined);
  EXPECT_FALSE(seq.HasElemDtype());
}

TEST(OnnxOptimOptimSequence, Equality) {
  onnx_optim::OptimSequence a(onnx_optim::TensorType::kFloat,
                              onnx_optim::OptimShape{onnx_optim::OptimDim(3)},
                              onnx_optim::OptimDim(static_cast<int64_t>(2)));
  onnx_optim::OptimSequence b(onnx_optim::TensorType::kFloat,
                              onnx_optim::OptimShape{onnx_optim::OptimDim(3)},
                              onnx_optim::OptimDim(static_cast<int64_t>(2)));
  EXPECT_EQ(a, b);
  b.SetLength(onnx_optim::OptimDim(static_cast<int64_t>(3)));
  EXPECT_NE(a, b);
}

TEST(OnnxOptimShapesContext, SequencesMapIsIndependentFromTensorsMap) {
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  ctx.Set("t", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kFloat, shape));
  ctx.SetSequence("s", onnx_optim::OptimSequence(onnx_optim::TensorType::kFloat, shape,
                                                 onnx_optim::OptimDim(static_cast<int64_t>(4))));
  EXPECT_TRUE(ctx.Has("t"));
  EXPECT_FALSE(ctx.HasSequence("t"));
  EXPECT_TRUE(ctx.HasSequence("s"));
  EXPECT_FALSE(ctx.Has("s"));
  EXPECT_EQ(ctx.SequencesSize(), 1u);
  EXPECT_EQ(ctx.Size(), 1u);

  ctx.Clear();
  EXPECT_EQ(ctx.SequencesSize(), 0u);
  EXPECT_EQ(ctx.Size(), 0u);
}

} // namespace Test
