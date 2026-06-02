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

TEST(OnnxOptimDim, ToString) {
  EXPECT_EQ(onnx_optim::OptimDim(42).ToString(), "42");
  EXPECT_EQ(onnx_optim::OptimDim(-1).ToString(), "-1");
  EXPECT_EQ(onnx_optim::OptimDim("N").ToString(), "N");
}

TEST(OnnxOptimShape, ToString) {
  EXPECT_EQ(onnx_optim::OptimShape{}.ToString(), "[]");
  EXPECT_EQ(onnx_optim::OptimShape{onnx_optim::OptimDim(7)}.ToString(), "[7]");
  onnx_optim::OptimShape s{onnx_optim::OptimDim(2), onnx_optim::OptimDim("N"),
                           onnx_optim::OptimDim(3)};
  EXPECT_EQ(s.ToString(), "[2,N,3]");
}

TEST(OnnxOptimTensor, ToString) {
  // Null tensor (no data, undefined dtype, empty shape).
  onnx_optim::OptimTensor null_tensor;
  EXPECT_EQ(null_tensor.ToString(), "OptimTensor(dtype=Undefined, shape=[])");

  // Tensor with data, dtype, and shape: data pointer is included.
  std::array<float, 6> buf = {1, 2, 3, 4, 5, 6};
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimTensor t(buf.data(), onnx_optim::TensorType::kFloat, shape);
  const std::string s = t.ToString();
  EXPECT_NE(s.find("OptimTensor(dtype=Float, shape=[2,3]"), std::string::npos);
  EXPECT_NE(s.find(", data="), std::string::npos);

  // Tensor with a value-as-shape annotation.
  t.SetValueAsShape(onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim("N")});
  const std::string s2 = t.ToString();
  EXPECT_NE(s2.find("value_as_shape=[2,N]"), std::string::npos);
}

TEST(OnnxOptimTensor, CmpEqualIsMorePrecise) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)};
  onnx_optim::OptimTensor a(nullptr, onnx_optim::TensorType::kFloat, shape);
  onnx_optim::OptimTensor b(nullptr, onnx_optim::TensorType::kFloat, shape);
  EXPECT_EQ(a.Cmp(b), onnx_optim::OptimCmpResult::kMorePrecise);
  EXPECT_EQ(b.Cmp(a), onnx_optim::OptimCmpResult::kMorePrecise);
}

TEST(OnnxOptimTensor, CmpConflictDtype) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor a(nullptr, onnx_optim::TensorType::kFloat, shape);
  onnx_optim::OptimTensor b(nullptr, onnx_optim::TensorType::kDouble, shape);
  EXPECT_EQ(a.Cmp(b), onnx_optim::OptimCmpResult::kConflict);
  EXPECT_EQ(b.Cmp(a), onnx_optim::OptimCmpResult::kConflict);
}

TEST(OnnxOptimTensor, CmpConflictRank) {
  onnx_optim::OptimTensor a(nullptr, onnx_optim::TensorType::kFloat,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(2)});
  onnx_optim::OptimTensor b(
      nullptr, onnx_optim::TensorType::kFloat,
      onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});
  EXPECT_EQ(a.Cmp(b), onnx_optim::OptimCmpResult::kConflict);
}

TEST(OnnxOptimTensor, CmpConflictDim) {
  onnx_optim::OptimTensor a(nullptr, onnx_optim::TensorType::kFloat,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(2)});
  onnx_optim::OptimTensor b(nullptr, onnx_optim::TensorType::kFloat,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(3)});
  EXPECT_EQ(a.Cmp(b), onnx_optim::OptimCmpResult::kConflict);

  onnx_optim::OptimTensor c(nullptr, onnx_optim::TensorType::kFloat,
                            onnx_optim::OptimShape{onnx_optim::OptimDim("N")});
  onnx_optim::OptimTensor d(nullptr, onnx_optim::TensorType::kFloat,
                            onnx_optim::OptimShape{onnx_optim::OptimDim("M")});
  EXPECT_EQ(c.Cmp(d), onnx_optim::OptimCmpResult::kConflict);
}

TEST(OnnxOptimTensor, CmpMorePreciseDtype) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor known(nullptr, onnx_optim::TensorType::kFloat, shape);
  onnx_optim::OptimTensor unknown(nullptr, onnx_optim::TensorType::kUndefined, shape);
  EXPECT_EQ(known.Cmp(unknown), onnx_optim::OptimCmpResult::kMorePrecise);
  EXPECT_EQ(unknown.Cmp(known), onnx_optim::OptimCmpResult::kLessPrecise);
}

TEST(OnnxOptimTensor, CmpMorePreciseDim) {
  onnx_optim::OptimTensor concrete(
      nullptr, onnx_optim::TensorType::kFloat,
      onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim(3)});
  onnx_optim::OptimTensor symbolic(
      nullptr, onnx_optim::TensorType::kFloat,
      onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim("N")});
  EXPECT_EQ(concrete.Cmp(symbolic), onnx_optim::OptimCmpResult::kMorePrecise);
  EXPECT_EQ(symbolic.Cmp(concrete), onnx_optim::OptimCmpResult::kLessPrecise);
}

TEST(OnnxOptimTensor, CmpComplementaryDims) {
  // lhs is more precise on dim 0, rhs is more precise on dim 1.
  onnx_optim::OptimTensor lhs(
      nullptr, onnx_optim::TensorType::kFloat,
      onnx_optim::OptimShape{onnx_optim::OptimDim(2), onnx_optim::OptimDim("N")});
  onnx_optim::OptimTensor rhs(
      nullptr, onnx_optim::TensorType::kFloat,
      onnx_optim::OptimShape{onnx_optim::OptimDim("M"), onnx_optim::OptimDim(3)});
  EXPECT_EQ(lhs.Cmp(rhs), onnx_optim::OptimCmpResult::kComplementary);
  EXPECT_EQ(rhs.Cmp(lhs), onnx_optim::OptimCmpResult::kComplementary);
}

TEST(OnnxOptimTensor, CmpComplementaryDtypeAndShape) {
  // lhs knows dtype only; rhs knows a more concrete dim only.
  onnx_optim::OptimTensor lhs(nullptr, onnx_optim::TensorType::kFloat,
                              onnx_optim::OptimShape{onnx_optim::OptimDim("N")});
  onnx_optim::OptimTensor rhs(nullptr, onnx_optim::TensorType::kUndefined,
                              onnx_optim::OptimShape{onnx_optim::OptimDim(5)});
  EXPECT_EQ(lhs.Cmp(rhs), onnx_optim::OptimCmpResult::kComplementary);
  EXPECT_EQ(rhs.Cmp(lhs), onnx_optim::OptimCmpResult::kComplementary);
}

TEST(OnnxOptimTensor, CmpValueAsShape) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor with_vas(nullptr, onnx_optim::TensorType::kInt64, shape);
  with_vas.SetValueAsShape(onnx_optim::OptimShape{onnx_optim::OptimDim(4)});
  onnx_optim::OptimTensor without_vas(nullptr, onnx_optim::TensorType::kInt64, shape);
  EXPECT_EQ(with_vas.Cmp(without_vas), onnx_optim::OptimCmpResult::kMorePrecise);
  EXPECT_EQ(without_vas.Cmp(with_vas), onnx_optim::OptimCmpResult::kLessPrecise);

  // Conflicting value-as-shape.
  onnx_optim::OptimTensor other_vas(nullptr, onnx_optim::TensorType::kInt64, shape);
  other_vas.SetValueAsShape(onnx_optim::OptimShape{onnx_optim::OptimDim(5)});
  EXPECT_EQ(with_vas.Cmp(other_vas), onnx_optim::OptimCmpResult::kConflict);
}

TEST(OnnxOptimTensor, CmpDataPresence) {
  std::array<float, 2> buf = {1.0f, 2.0f};
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor with_data(buf.data(), onnx_optim::TensorType::kFloat, shape);
  onnx_optim::OptimTensor without_data(nullptr, onnx_optim::TensorType::kFloat, shape);
  EXPECT_EQ(with_data.Cmp(without_data), onnx_optim::OptimCmpResult::kMorePrecise);
  EXPECT_EQ(without_data.Cmp(with_data), onnx_optim::OptimCmpResult::kLessPrecise);

  // Two distinct non-null pointers carry no precision signal.
  std::array<float, 2> buf2 = {1.0f, 2.0f};
  onnx_optim::OptimTensor with_other_data(buf2.data(), onnx_optim::TensorType::kFloat, shape);
  EXPECT_EQ(with_data.Cmp(with_other_data), onnx_optim::OptimCmpResult::kMorePrecise);
}

TEST(OnnxOptimDevice, MakeGPUDeviceAndHelpers) {
  EXPECT_FALSE(onnx_optim::IsGPU(onnx_optim::Device::kUndefined));
  EXPECT_FALSE(onnx_optim::IsGPU(onnx_optim::Device::kCPU));
  EXPECT_TRUE(onnx_optim::IsGPU(onnx_optim::Device::kGPU0));
  EXPECT_TRUE(onnx_optim::IsGPU(onnx_optim::Device::kGPU8191));

  EXPECT_EQ(onnx_optim::GPUIndex(onnx_optim::Device::kUndefined), -1);
  EXPECT_EQ(onnx_optim::GPUIndex(onnx_optim::Device::kCPU), -1);
  EXPECT_EQ(onnx_optim::GPUIndex(onnx_optim::Device::kGPU0), 0);
  EXPECT_EQ(onnx_optim::GPUIndex(onnx_optim::Device::kGPU8191), onnx_optim::kMaxGPUIndex);

  EXPECT_EQ(onnx_optim::MakeGPUDevice(0), onnx_optim::Device::kGPU0);
  EXPECT_EQ(onnx_optim::MakeGPUDevice(7),
            static_cast<onnx_optim::Device>(static_cast<int32_t>(onnx_optim::Device::kGPU0) + 7));
  EXPECT_EQ(onnx_optim::MakeGPUDevice(onnx_optim::kMaxGPUIndex), onnx_optim::Device::kGPU8191);
  EXPECT_THROW(onnx_optim::MakeGPUDevice(-1), std::out_of_range);
  EXPECT_THROW(onnx_optim::MakeGPUDevice(onnx_optim::kMaxGPUIndex + 1), std::out_of_range);
}

TEST(OnnxOptimDevice, DeviceName) {
  EXPECT_EQ(onnx_optim::DeviceName(onnx_optim::Device::kUndefined), "Undefined");
  EXPECT_EQ(onnx_optim::DeviceName(onnx_optim::Device::kCPU), "CPU");
  EXPECT_EQ(onnx_optim::DeviceName(onnx_optim::Device::kGPU0), "GPU0");
  EXPECT_EQ(onnx_optim::DeviceName(onnx_optim::MakeGPUDevice(42)), "GPU42");
  EXPECT_EQ(onnx_optim::DeviceName(onnx_optim::Device::kGPU8191), "GPU8191");
}

TEST(OnnxOptimTensor, DeviceDefaultsToUndefined) {
  onnx_optim::OptimTensor t;
  EXPECT_EQ(t.GetDevice(), onnx_optim::Device::kUndefined);
}

TEST(OnnxOptimTensor, SetDeviceRoundTrip) {
  onnx_optim::OptimTensor t;
  t.SetDevice(onnx_optim::Device::kCPU);
  EXPECT_EQ(t.GetDevice(), onnx_optim::Device::kCPU);
  t.SetDevice(onnx_optim::MakeGPUDevice(3));
  EXPECT_EQ(onnx_optim::GPUIndex(t.GetDevice()), 3);
}

TEST(OnnxOptimTensor, EqualityIncludesDevice) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor a(nullptr, onnx_optim::TensorType::kFloat, shape);
  onnx_optim::OptimTensor b(nullptr, onnx_optim::TensorType::kFloat, shape);
  EXPECT_EQ(a, b);
  b.SetDevice(onnx_optim::Device::kCPU);
  EXPECT_NE(a, b);
  a.SetDevice(onnx_optim::Device::kCPU);
  EXPECT_EQ(a, b);
}

TEST(OnnxOptimTensor, ToStringIncludesDevice) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kFloat, shape);
  // Undefined device is omitted from the string.
  EXPECT_EQ(t.ToString().find("device="), std::string::npos);
  t.SetDevice(onnx_optim::Device::kCPU);
  EXPECT_NE(t.ToString().find("device=CPU"), std::string::npos);
  t.SetDevice(onnx_optim::MakeGPUDevice(5));
  EXPECT_NE(t.ToString().find("device=GPU5"), std::string::npos);
}

TEST(OnnxOptimTensor, CmpDevice) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor known(nullptr, onnx_optim::TensorType::kFloat, shape);
  known.SetDevice(onnx_optim::Device::kCPU);
  onnx_optim::OptimTensor unknown(nullptr, onnx_optim::TensorType::kFloat, shape);
  EXPECT_EQ(known.Cmp(unknown), onnx_optim::OptimCmpResult::kMorePrecise);
  EXPECT_EQ(unknown.Cmp(known), onnx_optim::OptimCmpResult::kLessPrecise);

  onnx_optim::OptimTensor cpu(nullptr, onnx_optim::TensorType::kFloat, shape);
  cpu.SetDevice(onnx_optim::Device::kCPU);
  onnx_optim::OptimTensor gpu(nullptr, onnx_optim::TensorType::kFloat, shape);
  gpu.SetDevice(onnx_optim::Device::kGPU0);
  EXPECT_EQ(cpu.Cmp(gpu), onnx_optim::OptimCmpResult::kConflict);

  onnx_optim::OptimTensor cpu2(nullptr, onnx_optim::TensorType::kFloat, shape);
  cpu2.SetDevice(onnx_optim::Device::kCPU);
  EXPECT_EQ(cpu.Cmp(cpu2), onnx_optim::OptimCmpResult::kMorePrecise);
}

TEST(OnnxOptimDevice, DeviceFromName) {
  EXPECT_EQ(onnx_optim::DeviceFromName("CPU"), onnx_optim::Device::kCPU);
  EXPECT_EQ(onnx_optim::DeviceFromName("Undefined"), onnx_optim::Device::kUndefined);
  EXPECT_EQ(onnx_optim::DeviceFromName("GPU0"), onnx_optim::Device::kGPU0);
  EXPECT_EQ(onnx_optim::DeviceFromName("GPU42"), onnx_optim::MakeGPUDevice(42));
  EXPECT_EQ(onnx_optim::DeviceFromName("GPU8191"), onnx_optim::Device::kGPU8191);
  // Round-trip every well-formed name.
  for (int i = 0; i <= onnx_optim::kMaxGPUIndex; i += 2731) {
    const onnx_optim::Device d = onnx_optim::MakeGPUDevice(i);
    EXPECT_EQ(onnx_optim::DeviceFromName(onnx_optim::DeviceName(d)), d);
  }
  // Malformed / out-of-range names map back to Undefined.
  EXPECT_EQ(onnx_optim::DeviceFromName(""), onnx_optim::Device::kUndefined);
  EXPECT_EQ(onnx_optim::DeviceFromName("cpu"), onnx_optim::Device::kUndefined);
  EXPECT_EQ(onnx_optim::DeviceFromName("GPU"), onnx_optim::Device::kUndefined);
  EXPECT_EQ(onnx_optim::DeviceFromName("GPU-1"), onnx_optim::Device::kUndefined);
  EXPECT_EQ(onnx_optim::DeviceFromName("GPU+1"), onnx_optim::Device::kUndefined);
  EXPECT_EQ(onnx_optim::DeviceFromName("GPU 1"), onnx_optim::Device::kUndefined);
  EXPECT_EQ(onnx_optim::DeviceFromName("GPU8192"), onnx_optim::Device::kUndefined);
  EXPECT_EQ(onnx_optim::DeviceFromName("Unknown"), onnx_optim::Device::kUndefined);
}

namespace {

ValueInfoProto MakeTensorValueInfo(const std::string &name, TensorProto::DataType dtype,
                                   const std::vector<onnx_optim::OptimDim> &dims) {
  ValueInfoProto vi;
  vi.set_name(name);
  TypeProto *tp = vi.add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(static_cast<int>(dtype));
  TensorShapeProto *sp = tt->add_shape();
  for (const auto &d : dims) {
    auto *dim = sp->add_dim();
    if (d.IsInt()) {
      dim->set_dim_value(d.AsInt());
    } else {
      dim->set_dim_param(d.AsExpr());
    }
  }
  return vi;
}

} // namespace

TEST(OnnxOptimValueInfo, FromValueInfoTensorTypeAndShape) {
  ValueInfoProto vi = MakeTensorValueInfo(
      "x", TensorProto::DataType::FLOAT,
      {onnx_optim::OptimDim(2), onnx_optim::OptimDim("N"), onnx_optim::OptimDim(3)});
  onnx_optim::OptimTensor t;
  EXPECT_TRUE(onnx_optim::OptimTensorFromValueInfo(vi, t));
  EXPECT_EQ(t.Dtype(), onnx_optim::TensorType::kFloat);
  ASSERT_EQ(t.Shape().Rank(), 3u);
  EXPECT_TRUE(t.Shape()[0].IsInt());
  EXPECT_EQ(t.Shape()[0].AsInt(), 2);
  EXPECT_TRUE(t.Shape()[1].IsExpr());
  EXPECT_EQ(t.Shape()[1].AsExpr(), "N");
  EXPECT_EQ(t.Shape()[2].AsInt(), 3);
  EXPECT_EQ(t.GetDevice(), onnx_optim::Device::kUndefined);
}

TEST(OnnxOptimValueInfo, FromValueInfoMissingTypeReturnsFalse) {
  ValueInfoProto vi;
  vi.set_name("x");
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kInt32, onnx_optim::OptimShape{});
  EXPECT_FALSE(onnx_optim::OptimTensorFromValueInfo(vi, t));
  // ``out`` must be left untouched on failure.
  EXPECT_EQ(t.Dtype(), onnx_optim::TensorType::kInt32);
}

TEST(OnnxOptimValueInfo, FromValueInfoReadsDeviceMetadata) {
  ValueInfoProto vi =
      MakeTensorValueInfo("x", TensorProto::DataType::FLOAT, {onnx_optim::OptimDim(4)});
  auto *entry = vi.add_metadata_props();
  entry->set_key(onnx_optim::kValueInfoDeviceMetadataKey);
  entry->set_value("GPU3");
  onnx_optim::OptimTensor t;
  ASSERT_TRUE(onnx_optim::OptimTensorFromValueInfo(vi, t));
  EXPECT_EQ(t.GetDevice(), onnx_optim::MakeGPUDevice(3));
}

TEST(OnnxOptimValueInfo, FromValueInfoIgnoresUnknownDeviceMetadata) {
  ValueInfoProto vi =
      MakeTensorValueInfo("x", TensorProto::DataType::FLOAT, {onnx_optim::OptimDim(4)});
  auto *entry = vi.add_metadata_props();
  entry->set_key(onnx_optim::kValueInfoDeviceMetadataKey);
  entry->set_value("Mars");
  onnx_optim::OptimTensor t;
  ASSERT_TRUE(onnx_optim::OptimTensorFromValueInfo(vi, t));
  EXPECT_EQ(t.GetDevice(), onnx_optim::Device::kUndefined);
}

TEST(OnnxOptimValueInfo, ToValueInfoWritesTypeShape) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim("M")};
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kInt64, shape);
  ValueInfoProto vi;
  vi.set_name("y");
  ASSERT_TRUE(onnx_optim::OptimTensorToValueInfo(t, vi));
  ASSERT_TRUE(vi.has_type());
  ASSERT_TRUE(vi.type().has_tensor_type());
  const auto &tt = vi.type().tensor_type();
  EXPECT_EQ(tt.elem_type(), static_cast<int>(TensorProto::DataType::INT64));
  ASSERT_TRUE(tt.has_shape());
  ASSERT_EQ(tt.shape().dim().size(), 2);
  EXPECT_EQ(tt.shape().dim()[0].dim_value(), 2);
  EXPECT_EQ(tt.shape().dim()[1].dim_param().as_string(), "M");
  // No device set => no metadata_props entry added.
  EXPECT_EQ(vi.metadata_props().size(), 0u);
  // The original name is preserved.
  EXPECT_EQ(vi.name().as_string(), "y");
}

TEST(OnnxOptimValueInfo, ToValueInfoUndefinedDtypeReturnsFalse) {
  onnx_optim::OptimTensor t;
  ValueInfoProto vi;
  vi.set_name("y");
  EXPECT_FALSE(onnx_optim::OptimTensorToValueInfo(t, vi));
  EXPECT_FALSE(vi.has_type());
}

TEST(OnnxOptimValueInfo, ToValueInfoWritesDeviceMetadata) {
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kFloat,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  t.SetDevice(onnx_optim::MakeGPUDevice(7));
  ValueInfoProto vi;
  ASSERT_TRUE(onnx_optim::OptimTensorToValueInfo(t, vi));
  ASSERT_EQ(vi.metadata_props().size(), 1u);
  EXPECT_EQ(vi.metadata_props()[0].key().as_string(), onnx_optim::kValueInfoDeviceMetadataKey);
  EXPECT_EQ(vi.metadata_props()[0].value().as_string(), "GPU7");
}

TEST(OnnxOptimValueInfo, ToValueInfoUpdatesExistingDeviceMetadataInPlace) {
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kFloat,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  t.SetDevice(onnx_optim::Device::kCPU);
  ValueInfoProto vi;
  // Pre-existing unrelated metadata + stale device entry.
  auto *misc = vi.add_metadata_props();
  misc->set_key("author");
  misc->set_value("test");
  auto *dev = vi.add_metadata_props();
  dev->set_key(onnx_optim::kValueInfoDeviceMetadataKey);
  dev->set_value("GPU0");
  ASSERT_TRUE(onnx_optim::OptimTensorToValueInfo(t, vi));
  ASSERT_EQ(vi.metadata_props().size(), 2u);
  // The unrelated entry survives untouched.
  EXPECT_EQ(vi.metadata_props()[0].key().as_string(), "author");
  EXPECT_EQ(vi.metadata_props()[0].value().as_string(), "test");
  // The device entry is updated in place.
  EXPECT_EQ(vi.metadata_props()[1].key().as_string(), onnx_optim::kValueInfoDeviceMetadataKey);
  EXPECT_EQ(vi.metadata_props()[1].value().as_string(), "CPU");
}

TEST(OnnxOptimValueInfo, ToValueInfoRemovesStaleDeviceMetadata) {
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kFloat,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  // device left undefined on purpose.
  ValueInfoProto vi;
  auto *dev = vi.add_metadata_props();
  dev->set_key(onnx_optim::kValueInfoDeviceMetadataKey);
  dev->set_value("GPU0");
  auto *misc = vi.add_metadata_props();
  misc->set_key("author");
  misc->set_value("test");
  ASSERT_TRUE(onnx_optim::OptimTensorToValueInfo(t, vi));
  ASSERT_EQ(vi.metadata_props().size(), 1u);
  // The remaining entry is the unrelated one (swap-and-pop moved it
  // from position 1 to position 0).
  EXPECT_EQ(vi.metadata_props()[0].key().as_string(), "author");
  EXPECT_EQ(vi.metadata_props()[0].value().as_string(), "test");
}

TEST(OnnxOptimValueInfo, RoundTripPreservesDtypeShapeAndDevice) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2), onnx_optim::OptimDim("N"),
                               onnx_optim::OptimDim(5)};
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kDouble, shape);
  t.SetDevice(onnx_optim::MakeGPUDevice(42));
  ValueInfoProto vi;
  vi.set_name("rt");
  ASSERT_TRUE(onnx_optim::OptimTensorToValueInfo(t, vi));
  onnx_optim::OptimTensor back;
  ASSERT_TRUE(onnx_optim::OptimTensorFromValueInfo(vi, back));
  EXPECT_EQ(back.Dtype(), t.Dtype());
  EXPECT_EQ(back.GetDevice(), t.GetDevice());
  ASSERT_EQ(back.Shape().Rank(), shape.Rank());
  EXPECT_EQ(back.Shape()[0].AsInt(), 2);
  EXPECT_EQ(back.Shape()[1].AsExpr(), "N");
  EXPECT_EQ(back.Shape()[2].AsInt(), 5);
}

TEST(OnnxOptimTensor, MinMaxDefaultsToAbsent) {
  onnx_optim::OptimTensor t;
  EXPECT_FALSE(t.HasMin());
  EXPECT_FALSE(t.HasMax());
  EXPECT_FALSE(t.IsNullConstant());
  EXPECT_THROW((void)t.Min(), std::bad_optional_access);
  EXPECT_THROW((void)t.Max(), std::bad_optional_access);
}

TEST(OnnxOptimTensor, SetMinMaxStoresBounds) {
  onnx_optim::OptimTensor t;
  t.SetMin(-1.5);
  t.SetMax(2.5);
  EXPECT_TRUE(t.HasMin());
  EXPECT_TRUE(t.HasMax());
  EXPECT_EQ(t.Min(), -1.5);
  EXPECT_EQ(t.Max(), 2.5);
}

TEST(OnnxOptimTensor, SetMinMaxAtomicRejectsInvertedRange) {
  onnx_optim::OptimTensor t;
  EXPECT_THROW(t.SetMinMax(1.0, 0.0), std::invalid_argument);
  EXPECT_FALSE(t.HasMin());
  EXPECT_FALSE(t.HasMax());
  t.SetMinMax(-2.0, 3.0);
  EXPECT_EQ(t.Min(), -2.0);
  EXPECT_EQ(t.Max(), 3.0);
  // Degenerate range (single value) is allowed.
  t.SetMinMax(4.0, 4.0);
  EXPECT_EQ(t.Min(), 4.0);
  EXPECT_EQ(t.Max(), 4.0);
}

TEST(OnnxOptimTensor, ClearMinMax) {
  onnx_optim::OptimTensor t;
  t.SetMinMax(0.0, 1.0);
  t.ClearMin();
  EXPECT_FALSE(t.HasMin());
  EXPECT_TRUE(t.HasMax());
  t.ClearMax();
  EXPECT_FALSE(t.HasMax());
  t.SetMinMax(-1.0, 1.0);
  t.ClearMinMax();
  EXPECT_FALSE(t.HasMin());
  EXPECT_FALSE(t.HasMax());
}

TEST(OnnxOptimTensor, IsNullConstantDetection) {
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kFloat,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(3)});
  // No bounds => not detected.
  EXPECT_FALSE(t.IsNullConstant());
  // Only min set => not detected (cannot prove max == 0).
  t.SetMin(0.0);
  EXPECT_FALSE(t.IsNullConstant());
  // Both bounds at 0 => null constant.
  t.SetMax(0.0);
  EXPECT_TRUE(t.IsNullConstant());
  // Any non-zero bound breaks the detection.
  t.SetMax(1.0);
  EXPECT_FALSE(t.IsNullConstant());
  t.SetMinMax(-1.0, 0.0);
  EXPECT_FALSE(t.IsNullConstant());
}

TEST(OnnxOptimTensor, EqualityIncludesMinMax) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor a(nullptr, onnx_optim::TensorType::kFloat, shape);
  onnx_optim::OptimTensor b(nullptr, onnx_optim::TensorType::kFloat, shape);
  EXPECT_EQ(a, b);
  a.SetMin(0.0);
  EXPECT_NE(a, b);
  b.SetMin(0.0);
  EXPECT_EQ(a, b);
  a.SetMax(1.0);
  b.SetMax(2.0);
  EXPECT_NE(a, b);
  b.SetMax(1.0);
  EXPECT_EQ(a, b);
}

TEST(OnnxOptimTensor, ToStringIncludesMinMax) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kFloat, shape);
  // Absent bounds are omitted from the string.
  const std::string empty = t.ToString();
  EXPECT_EQ(empty.find("min="), std::string::npos);
  EXPECT_EQ(empty.find("max="), std::string::npos);
  t.SetMinMax(-1.0, 1.0);
  const std::string s = t.ToString();
  EXPECT_NE(s.find("min=-1"), std::string::npos);
  EXPECT_NE(s.find("max=1"), std::string::npos);
}

TEST(OnnxOptimTensor, CmpMinMaxPresence) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor known(nullptr, onnx_optim::TensorType::kFloat, shape);
  known.SetMinMax(-1.0, 1.0);
  onnx_optim::OptimTensor unknown(nullptr, onnx_optim::TensorType::kFloat, shape);
  EXPECT_EQ(known.Cmp(unknown), onnx_optim::OptimCmpResult::kMorePrecise);
  EXPECT_EQ(unknown.Cmp(known), onnx_optim::OptimCmpResult::kLessPrecise);
}

TEST(OnnxOptimTensor, CmpMinMaxTighterIsMorePrecise) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor tight(nullptr, onnx_optim::TensorType::kFloat, shape);
  tight.SetMinMax(0.0, 1.0);
  onnx_optim::OptimTensor loose(nullptr, onnx_optim::TensorType::kFloat, shape);
  loose.SetMinMax(-1.0, 2.0);
  EXPECT_EQ(tight.Cmp(loose), onnx_optim::OptimCmpResult::kMorePrecise);
  EXPECT_EQ(loose.Cmp(tight), onnx_optim::OptimCmpResult::kLessPrecise);
}

TEST(OnnxOptimTensor, CmpMinMaxComplementaryBounds) {
  // One side has a tighter min, the other a tighter max.
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor a(nullptr, onnx_optim::TensorType::kFloat, shape);
  a.SetMinMax(0.0, 2.0);
  onnx_optim::OptimTensor b(nullptr, onnx_optim::TensorType::kFloat, shape);
  b.SetMinMax(-1.0, 1.0);
  EXPECT_EQ(a.Cmp(b), onnx_optim::OptimCmpResult::kComplementary);
  EXPECT_EQ(b.Cmp(a), onnx_optim::OptimCmpResult::kComplementary);
}

TEST(OnnxOptimTensor, CmpMinMaxDisjointConflict) {
  onnx_optim::OptimShape shape{onnx_optim::OptimDim(2)};
  onnx_optim::OptimTensor a(nullptr, onnx_optim::TensorType::kFloat, shape);
  a.SetMinMax(0.0, 1.0);
  onnx_optim::OptimTensor b(nullptr, onnx_optim::TensorType::kFloat, shape);
  b.SetMinMax(2.0, 3.0);
  EXPECT_EQ(a.Cmp(b), onnx_optim::OptimCmpResult::kConflict);
  EXPECT_EQ(b.Cmp(a), onnx_optim::OptimCmpResult::kConflict);
}

TEST(OnnxOptimValueInfo, ToValueInfoWritesMinMaxMetadata) {
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kFloat,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  t.SetMinMax(-2.5, 3.5);
  ValueInfoProto vi;
  ASSERT_TRUE(onnx_optim::OptimTensorToValueInfo(t, vi));
  ASSERT_EQ(vi.metadata_props().size(), 2u);
  bool seen_min = false;
  bool seen_max = false;
  for (int i = 0; i < vi.metadata_props().size(); ++i) {
    const std::string key = vi.metadata_props()[i].key().as_string();
    const std::string value = vi.metadata_props()[i].value().as_string();
    if (key == onnx_optim::kValueInfoMinMetadataKey) {
      seen_min = true;
      EXPECT_EQ(std::stod(value), -2.5);
    } else if (key == onnx_optim::kValueInfoMaxMetadataKey) {
      seen_max = true;
      EXPECT_EQ(std::stod(value), 3.5);
    }
  }
  EXPECT_TRUE(seen_min);
  EXPECT_TRUE(seen_max);
}

TEST(OnnxOptimValueInfo, ToValueInfoRemovesStaleMinMaxMetadata) {
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kFloat,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(1)});
  // No min/max set; ensure pre-existing entries are removed.
  ValueInfoProto vi;
  auto *min_entry = vi.add_metadata_props();
  min_entry->set_key(onnx_optim::kValueInfoMinMetadataKey);
  min_entry->set_value("-1");
  auto *max_entry = vi.add_metadata_props();
  max_entry->set_key(onnx_optim::kValueInfoMaxMetadataKey);
  max_entry->set_value("1");
  auto *misc = vi.add_metadata_props();
  misc->set_key("author");
  misc->set_value("test");
  ASSERT_TRUE(onnx_optim::OptimTensorToValueInfo(t, vi));
  ASSERT_EQ(vi.metadata_props().size(), 1u);
  EXPECT_EQ(vi.metadata_props()[0].key().as_string(), "author");
}

TEST(OnnxOptimValueInfo, FromValueInfoReadsMinMaxMetadata) {
  ValueInfoProto vi =
      MakeTensorValueInfo("x", TensorProto::DataType::FLOAT, {onnx_optim::OptimDim(4)});
  auto *min_entry = vi.add_metadata_props();
  min_entry->set_key(onnx_optim::kValueInfoMinMetadataKey);
  min_entry->set_value("-3.25");
  auto *max_entry = vi.add_metadata_props();
  max_entry->set_key(onnx_optim::kValueInfoMaxMetadataKey);
  max_entry->set_value("4.75");
  onnx_optim::OptimTensor t;
  ASSERT_TRUE(onnx_optim::OptimTensorFromValueInfo(vi, t));
  ASSERT_TRUE(t.HasMin());
  ASSERT_TRUE(t.HasMax());
  EXPECT_EQ(t.Min(), -3.25);
  EXPECT_EQ(t.Max(), 4.75);
}

TEST(OnnxOptimValueInfo, FromValueInfoIgnoresUnparseableMinMax) {
  ValueInfoProto vi =
      MakeTensorValueInfo("x", TensorProto::DataType::FLOAT, {onnx_optim::OptimDim(4)});
  auto *min_entry = vi.add_metadata_props();
  min_entry->set_key(onnx_optim::kValueInfoMinMetadataKey);
  min_entry->set_value("not-a-number");
  onnx_optim::OptimTensor t;
  ASSERT_TRUE(onnx_optim::OptimTensorFromValueInfo(vi, t));
  EXPECT_FALSE(t.HasMin());
  EXPECT_FALSE(t.HasMax());
}

TEST(OnnxOptimValueInfo, RoundTripPreservesMinMax) {
  onnx_optim::OptimTensor t(nullptr, onnx_optim::TensorType::kFloat,
                            onnx_optim::OptimShape{onnx_optim::OptimDim(3)});
  t.SetMinMax(0.0, 0.0);
  ValueInfoProto vi;
  ASSERT_TRUE(onnx_optim::OptimTensorToValueInfo(t, vi));
  onnx_optim::OptimTensor back;
  ASSERT_TRUE(onnx_optim::OptimTensorFromValueInfo(vi, back));
  ASSERT_TRUE(back.HasMin());
  ASSERT_TRUE(back.HasMax());
  EXPECT_EQ(back.Min(), 0.0);
  EXPECT_EQ(back.Max(), 0.0);
  EXPECT_TRUE(back.IsNullConstant());
}

} // namespace Test
