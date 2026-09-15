// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace core::symbolic;

namespace Test {

TEST(PoolingShapeCompatibility, MaxUnpoolExplicitShapeAndUnknownValues) {
  core::shapes::ShapesContext ctx;
  NodeProto node;
  node.set_op_type("MaxUnpool");
  node.add_output("Y");
  AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2});
  AddAttribute<std::vector<int64_t>>(node, "strides", {2});
  ctx.Set("X", SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(1), SymDim(1), SymDim(2)}));
  SymTensor shape(nullptr, TensorType::kInt64, SymShape{SymDim(3)});
  const SymShape expected{SymDim(2), SymDim(1), SymDim(3)};
  shape.SetValueAsShape(expected);
  ctx.Set("S", SymTensor(shape));
  onnx_shapes::shapes::nn::ComputeShapeMaxUnpool(ctx, node, "X", "I", "S");
  EXPECT_EQ(ctx.Get("Y").Shape(), expected);
  shape.ClearValueAsShape();
  ctx.Set("S", SymTensor(shape));
  onnx_shapes::shapes::nn::ComputeShapeMaxUnpool(ctx, node, "X", "I", "S");
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_FALSE(ctx.Get("Y").Shape()[i].IsInt());
  }
  shape.SetValueAsShape(SymShape{SymDim(1), SymDim(1), SymDim(-1)});
  ctx.Set("S", SymTensor(shape));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeMaxUnpool(ctx, node, "X", "I", "S"),
               std::invalid_argument);
  ctx.Set("S", SymTensor(nullptr, TensorType::kInt64, SymShape{SymDim(2)}));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeMaxUnpool(ctx, node, "X", "I", "S"),
               std::invalid_argument);
  ctx.Set("S", SymTensor(nullptr, TensorType::kFloat, SymShape{SymDim(3)}));
  EXPECT_THROW(onnx_shapes::shapes::nn::ComputeShapeMaxUnpool(ctx, node, "X", "I", "S"),
               std::invalid_argument);
}

} // namespace Test
