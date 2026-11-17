// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/shapes/shapes/math/shape_math.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace core::symbolic;

namespace Test {

TEST(ShapeGeGLU, DispatchPreservesSupportedScalarEmptyAndSymbolicInputs) {
  const NodeProto node = MakeNode("GeGLU", {"A", "B"}, {"Y"});
  for (TensorType dtype :
       {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16}) {
    for (const SymShape &shape :
         {SymShape{}, SymShape{SymDim(2), SymDim(int64_t{0})}, SymShape{SymDim("N"), SymDim(3)}}) {
      core::shapes::ShapesContext ctx;
      ctx.SetOpsetVersion("", 29);
      ctx.Set("A", SymTensor(nullptr, dtype, shape));
      ctx.Set("B", SymTensor(nullptr, dtype, shape));
      ctx.ComputeShapeNode(node);
      ASSERT_TRUE(ctx.Has("Y"));
      EXPECT_EQ(ctx.Get("Y"), SymTensor(nullptr, dtype, shape));
    }
  }
}

TEST(ShapeGeGLU, MergesKnownAndSymbolicDimensionsWithoutBroadcasting) {
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 29);
  ctx.Set("A", SymTensor(nullptr, TensorType::kFloat,
                         {SymDim("N"), SymDim(""), SymDim("K"), SymDim(4)}));
  ctx.Set("B",
          SymTensor(nullptr, TensorType::kFloat, {SymDim(2), SymDim("M"), SymDim("L"), SymDim(4)}));
  const NodeProto node = MakeNode("GeGLU", {"A", "B"}, {"Y"});
  ctx.ComputeShapeNode(node);
  EXPECT_EQ(ctx.Get("Y").Shape(), (SymShape{SymDim(2), SymDim("M"), SymDim("K"), SymDim(4)}));
}

TEST(ShapeGeGLU, RejectsShapeTypeAttributeAndNodeErrors) {
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 29);
  ctx.Set("A", SymTensor(nullptr, TensorType::kFloat, {SymDim(2), SymDim(3)}));
  NodeProto node = MakeNode("GeGLU", {"A", "B"}, {"Y"});
  for (const SymShape &shape : {SymShape{SymDim(3)}, SymShape{SymDim(1), SymDim(3)},
                                SymShape{SymDim(2), SymDim(1)}, SymShape{SymDim(2), SymDim(4)}}) {
    ctx.Set("B", SymTensor(nullptr, TensorType::kFloat, shape));
    EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
  }
  ctx.Set("B", SymTensor(nullptr, TensorType::kDouble, {SymDim(2), SymDim(3)}));
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
  ctx.Set("A", SymTensor(nullptr, TensorType::kInt32, {}));
  ctx.Set("B", SymTensor(nullptr, TensorType::kInt32, {}));
  EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
  for (const SymShape &shape : {SymShape{}, SymShape{SymDim(int64_t{0})}, SymShape{SymDim("N")}}) {
    ctx.Set("A", SymTensor(nullptr, TensorType::kFloat, shape));
    ctx.Set("B", SymTensor(nullptr, TensorType::kFloat, shape));
    auto *attr = node.add_attribute();
    attr->set_name("approximate");
    attr->set_type(AttributeProto::STRING);
    attr->set_s("invalid");
    EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
    attr->set_s("tanh");
    EXPECT_NO_THROW(ctx.ComputeShapeNode(node));
    attr->set_type(AttributeProto::INT);
    EXPECT_THROW(ctx.ComputeShapeNode(node), std::invalid_argument);
    node.clear_attribute();
  }
  const NodeProto wrong_op = MakeNode("Gelu", {"A", "B"}, {"Y"});
  EXPECT_THROW(onnx_shapes::shapes::math::ComputeShapeGeGLU(ctx, wrong_op, "A", "B"),
               std::invalid_argument);
  const NodeProto no_output = MakeNode("GeGLU", {"A", "B"}, {});
  EXPECT_THROW(onnx_shapes::shapes::math::ComputeShapeGeGLU(ctx, no_output, "A", "B"),
               std::invalid_argument);
  const NodeProto no_input = MakeNode("GeGLU", {"A"}, {"Y"});
  EXPECT_THROW(ctx.ComputeShapeNode(no_input), std::invalid_argument);
  EXPECT_THROW(onnx_shapes::shapes::math::ComputeShapeGeGLU(ctx, node, "A", "missing"),
               std::out_of_range);
}
} // namespace Test
