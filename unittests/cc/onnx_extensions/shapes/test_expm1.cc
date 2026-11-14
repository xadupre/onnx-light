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

TEST(ShapeExpm1, DispatchPreservesSymbolicScalarAndEmptyShapesForAllTypes) {
  const NodeProto node = MakeNode("Expm1", {"X"}, {"Y"});
  for (TensorType dtype :
       {TensorType::kFloat, TensorType::kDouble, TensorType::kFloat16, TensorType::kBfloat16}) {
    for (const SymShape &shape :
         {SymShape{}, SymShape{SymDim(2), SymDim(int64_t{0})}, SymShape{SymDim("N"), SymDim(3)}}) {
      core::shapes::ShapesContext ctx;
      ctx.SetOpsetVersion("", 29);
      ctx.Set("X", SymTensor(nullptr, dtype, shape));
      ctx.ComputeShapeNode(node);
      ASSERT_TRUE(ctx.Has("Y"));
      EXPECT_EQ(ctx.Get("Y"), SymTensor(nullptr, dtype, shape));
    }
  }
}

TEST(ShapeExpm1, RejectsInvalidNodesAndMissingInput) {
  core::shapes::ShapesContext ctx;
  ctx.SetOpsetVersion("", 29);
  ctx.Set("X", SymTensor(nullptr, TensorType::kFloat, {}));
  const NodeProto wrong_op = MakeNode("Exp", {"X"}, {"Y"});
  EXPECT_THROW(onnx_shapes::shapes::math::ComputeShapeExpm1(ctx, wrong_op, "X"),
               std::invalid_argument);
  const NodeProto no_output = MakeNode("Expm1", {"X"}, {});
  EXPECT_THROW(onnx_shapes::shapes::math::ComputeShapeExpm1(ctx, no_output, "X"),
               std::invalid_argument);
  const NodeProto node = MakeNode("Expm1", {"missing"}, {"Y"});
  EXPECT_THROW(onnx_shapes::shapes::math::ComputeShapeExpm1(ctx, node, "missing"),
               std::out_of_range);
  const NodeProto no_input = MakeNode("Expm1", {}, {"Y"});
  EXPECT_THROW(ctx.ComputeShapeNode(no_input), std::invalid_argument);
}

} // namespace Test
