// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_optim/shapes/text/shape_text.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

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
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kString, shape));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kString, shape));

  onnx_optim::shapes::text::ComputeShapeStringConcat(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kString);
  EXPECT_EQ(ctx.Get("C").Shape(), shape);
}

TEST(OnnxOptimShapesTextStringConcat, BroadcastsShapes) {
  NodeProto node = MakeStringConcatNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(1)};
  onnx_optim::OptimShape shape_b{onnx_optim::OptimDim(1), onnx_optim::OptimDim(3)};
  onnx_optim::OptimShape expected{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kString, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kString, shape_b));

  onnx_optim::shapes::text::ComputeShapeStringConcat(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kString);
  EXPECT_EQ(ctx.Get("C").Shape(), expected);
}

TEST(OnnxOptimShapesTextStringConcat, ScalarBroadcast) {
  NodeProto node = MakeStringConcatNode();
  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::OptimShape shape_a{onnx_optim::OptimDim(2), onnx_optim::OptimDim(2)};
  onnx_optim::OptimShape shape_b{};
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kString, shape_a));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kString, shape_b));

  onnx_optim::shapes::text::ComputeShapeStringConcat(ctx, node, "A", "B");

  ASSERT_TRUE(ctx.Has("C"));
  EXPECT_EQ(ctx.Get("C").Dtype(), onnx_optim::TensorType::kString);
  EXPECT_EQ(ctx.Get("C").Shape(), shape_a);
}

TEST(OnnxOptimShapesTextStringConcat, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("And");
  node.add_input("A");
  node.add_input("B");
  node.add_output("C");
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kString, {}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kString, {}));
  EXPECT_THROW(onnx_optim::shapes::text::ComputeShapeStringConcat(ctx, node, "A", "B"),
               std::invalid_argument);
}

TEST(OnnxOptimShapesTextStringConcat, ThrowsWhenInputMissingFromContext) {
  NodeProto node = MakeStringConcatNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kString, {}));
  EXPECT_THROW(onnx_optim::shapes::text::ComputeShapeStringConcat(ctx, node, "A", "B"),
               std::out_of_range);
}

TEST(OnnxOptimShapesTextStringConcat, ThrowsOnIncompatibleShapes) {
  NodeProto node = MakeStringConcatNode();
  onnx_optim::shapes::ShapesContext ctx;
  ctx.Set("A", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kString,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(3)}));
  ctx.Set("B", onnx_optim::OptimTensor(nullptr, onnx_optim::TensorType::kString,
                                       onnx_optim::OptimShape{onnx_optim::OptimDim(4)}));
  EXPECT_THROW(onnx_optim::shapes::text::ComputeShapeStringConcat(ctx, node, "A", "B"),
               std::invalid_argument);
}

} // namespace Test
