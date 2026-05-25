// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/optim_tensor.h"

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

TEST(OnnxOptimDim, IntegerDim) {
  onnx_optim::OptimDim d(42);
  EXPECT_TRUE(d.IsInt());
  EXPECT_FALSE(d.IsExpr());
  EXPECT_EQ(d.AsInt(), 42);
  EXPECT_THROW(d.AsExpr(), std::bad_variant_access);
}

TEST(OnnxOptimDim, SymbolicDim) {
  onnx_optim::OptimDim d("N");
  EXPECT_FALSE(d.IsInt());
  EXPECT_TRUE(d.IsExpr());
  EXPECT_EQ(d.AsExpr(), "N");
  EXPECT_THROW(d.AsInt(), std::bad_variant_access);
}

TEST(OnnxOptimDim, Equality) {
  EXPECT_EQ(onnx_optim::OptimDim(3), onnx_optim::OptimDim(3));
  EXPECT_NE(onnx_optim::OptimDim(3), onnx_optim::OptimDim(4));
  EXPECT_EQ(onnx_optim::OptimDim("N"), onnx_optim::OptimDim(std::string("N")));
  EXPECT_NE(onnx_optim::OptimDim("N"), onnx_optim::OptimDim(2));
}

TEST(OnnxOptimShape, ConstructAndAccess) {
  onnx_optim::OptimShape s{onnx_optim::OptimDim(1), onnx_optim::OptimDim("N"),
                           onnx_optim::OptimDim(3)};
  EXPECT_EQ(s.Rank(), 3u);
  EXPECT_FALSE(s.Empty());
  EXPECT_EQ(s[0].AsInt(), 1);
  EXPECT_EQ(s[1].AsExpr(), "N");
  EXPECT_EQ(s[2].AsInt(), 3);
  EXPECT_FALSE(s.IsFullyKnown());
  EXPECT_THROW(s.NumElements(), std::runtime_error);
  EXPECT_THROW((void)s[5], std::out_of_range);
}

TEST(OnnxOptimShape, FullyKnown) {
  onnx_optim::OptimShape s{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                           onnx_optim::OptimDim(4)};
  EXPECT_TRUE(s.IsFullyKnown());
  EXPECT_EQ(s.NumElements(), 24);
}

TEST(OnnxOptimShape, PushBackAndRankLimit) {
  onnx_optim::OptimShape s;
  for (std::size_t i = 0; i < onnx_optim::kMaxOptimRank; ++i) {
    s.PushBack(onnx_optim::OptimDim(static_cast<int64_t>(i + 1)));
  }
  EXPECT_EQ(s.Rank(), onnx_optim::kMaxOptimRank);
  EXPECT_THROW(s.PushBack(onnx_optim::OptimDim(1)), std::length_error);
}

TEST(OnnxOptimShape, RejectsOversizedInitializerList) {
  EXPECT_THROW((onnx_optim::OptimShape{
                   onnx_optim::OptimDim(1), onnx_optim::OptimDim(2), onnx_optim::OptimDim(3),
                   onnx_optim::OptimDim(4), onnx_optim::OptimDim(5), onnx_optim::OptimDim(6),
                   onnx_optim::OptimDim(7), onnx_optim::OptimDim(8), onnx_optim::OptimDim(9)}),
               std::length_error);
}

TEST(OnnxOptimTensor, WrapsExternalBufferWithoutAllocation) {
  std::array<float, 6> buffer = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimTensor t(buffer.data(), onnx_optim::TensorType::kFloat, shape);

  EXPECT_FALSE(t.IsNull());
  // The tensor must reference the exact same buffer (no copy / no allocation).
  EXPECT_EQ(t.Data(), static_cast<void *>(buffer.data()));
  EXPECT_EQ(t.Dtype(), onnx_optim::TensorType::kFloat);
  EXPECT_EQ(t.Shape().Rank(), 2u);
  EXPECT_EQ(t.Shape().NumElements(), 6);

  // Mutating through the tensor reflects in the original buffer.
  static_cast<float *>(t.Data())[0] = 99.0f;
  EXPECT_FLOAT_EQ(buffer[0], 99.0f);
}

TEST(OnnxOptimTensor, DefaultConstructedIsNull) {
  onnx_optim::OptimTensor t;
  EXPECT_TRUE(t.IsNull());
  EXPECT_EQ(t.Data(), nullptr);
  EXPECT_TRUE(t.Shape().Empty());
  EXPECT_EQ(t.Dtype(), onnx_optim::TensorType::kUndefined);
}

TEST(OnnxOptimTensor, SymbolicShapeIsAllowed) {
  std::vector<int64_t> buffer(4, 0);
  onnx_optim::OptimShape shape{onnx_optim::OptimDim("N"), onnx_optim::OptimDim(4)};
  onnx_optim::OptimTensor t(buffer.data(), onnx_optim::TensorType::kInt64, shape);
  EXPECT_FALSE(t.Shape().IsFullyKnown());
  EXPECT_EQ(t.Shape()[0].AsExpr(), "N");
  EXPECT_EQ(t.Shape()[1].AsInt(), 4);
}

TEST(OnnxOptimTensor, ValueAsShapeDefaultsToAbsent) {
  onnx_optim::OptimTensor t;
  EXPECT_FALSE(t.HasValueAsShape());
  EXPECT_THROW(t.ValueAsShape(), std::bad_optional_access);
}

TEST(OnnxOptimTensor, SetValueAsShapeStoresNonEmptyShape) {
  std::array<int64_t, 3> buffer = {2, 3, 4};
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(static_cast<int64_t>(3))};
  onnx_optim::OptimTensor t(buffer.data(), onnx_optim::TensorType::kInt64, shape);

  t.SetValueAsShape(onnx_optim::OptimShape{onnx_optim::OptimDim(static_cast<int64_t>(2)),
                                           onnx_optim::OptimDim(static_cast<int64_t>(3)),
                                           onnx_optim::OptimDim(static_cast<int64_t>(4))});
  EXPECT_TRUE(t.HasValueAsShape());
  ASSERT_EQ(t.ValueAsShape().Rank(), 3u);
  EXPECT_EQ(t.ValueAsShape()[0].AsInt(), 2);
  EXPECT_EQ(t.ValueAsShape()[2].AsInt(), 4);
}

TEST(OnnxOptimTensor, SetValueAsShapeAcceptsEmptyShape) {
  onnx_optim::OptimTensor t;
  t.SetValueAsShape(onnx_optim::OptimShape{});
  EXPECT_TRUE(t.HasValueAsShape());
  EXPECT_TRUE(t.ValueAsShape().Empty());
}

TEST(OnnxOptimTensor, ClearValueAsShapeResetsFlag) {
  onnx_optim::OptimTensor t;
  t.SetValueAsShape(onnx_optim::OptimShape{onnx_optim::OptimDim("N")});
  EXPECT_TRUE(t.HasValueAsShape());
  t.ClearValueAsShape();
  EXPECT_FALSE(t.HasValueAsShape());
}

TEST(OnnxOptimShape, Equality) {
  onnx_optim::OptimShape a{onnx_optim::OptimDim(2), onnx_optim::OptimDim("N")};
  onnx_optim::OptimShape b{onnx_optim::OptimDim(2), onnx_optim::OptimDim("N")};
  onnx_optim::OptimShape c{onnx_optim::OptimDim(2), onnx_optim::OptimDim("M")};
  onnx_optim::OptimShape d{onnx_optim::OptimDim(2)};
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);
}

TEST(OnnxOptimTensor, Equality) {
  std::array<float, 6> buf = {1, 2, 3, 4, 5, 6};
  std::array<float, 6> other = {1, 2, 3, 4, 5, 6};
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};

  onnx_optim::OptimTensor a(buf.data(), onnx_optim::TensorType::kFloat, shape);
  onnx_optim::OptimTensor b(buf.data(), onnx_optim::TensorType::kFloat, shape);
  EXPECT_EQ(a, b);

  // Different data pointer.
  onnx_optim::OptimTensor different_buffer(other.data(), onnx_optim::TensorType::kFloat, shape);
  EXPECT_NE(a, different_buffer);

  // Different dtype.
  onnx_optim::OptimTensor different_dtype(buf.data(), onnx_optim::TensorType::kDouble, shape);
  EXPECT_NE(a, different_dtype);

  // Different shape.
  onnx_optim::OptimTensor different_shape(buf.data(), onnx_optim::TensorType::kFloat,
                                          onnx_optim::OptimShape{onnx_optim::OptimDim(6)});
  EXPECT_NE(a, different_shape);

  // Differing value-as-shape annotation.
  onnx_optim::OptimTensor with_vas = a;
  with_vas.SetValueAsShape(onnx_optim::OptimShape{onnx_optim::OptimDim(2)});
  EXPECT_NE(a, with_vas);
}

} // namespace Test
