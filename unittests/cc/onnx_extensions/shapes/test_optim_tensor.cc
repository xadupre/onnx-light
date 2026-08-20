// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/symbolic/sym_tensor.h"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

TEST(OnnxOptimDim, IntegerDim) {
  core::symbolic::SymDim d(42);
  EXPECT_TRUE(d.IsInt());
  EXPECT_FALSE(d.IsExpr());
  EXPECT_EQ(d.AsInt(), 42);
  EXPECT_THROW(d.AsExpr(), std::bad_variant_access);
}

TEST(OnnxOptimDim, SymbolicDim) {
  core::symbolic::SymDim d("N");
  EXPECT_FALSE(d.IsInt());
  EXPECT_TRUE(d.IsExpr());
  EXPECT_EQ(d.AsExpr(), "N");
  EXPECT_THROW(d.AsInt(), std::bad_variant_access);
}

TEST(OnnxOptimDim, Equality) {
  EXPECT_EQ(core::symbolic::SymDim(3), core::symbolic::SymDim(3));
  EXPECT_NE(core::symbolic::SymDim(3), core::symbolic::SymDim(4));
  EXPECT_EQ(core::symbolic::SymDim("N"), core::symbolic::SymDim(std::string("N")));
  EXPECT_NE(core::symbolic::SymDim("N"), core::symbolic::SymDim(2));
}

TEST(OnnxOptimShape, ConstructAndAccess) {
  core::symbolic::SymShape s{core::symbolic::SymDim(1), core::symbolic::SymDim("N"),
                             core::symbolic::SymDim(3)};
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
  core::symbolic::SymShape s{core::symbolic::SymDim(2), core::symbolic::SymDim(3),
                             core::symbolic::SymDim(4)};
  EXPECT_TRUE(s.IsFullyKnown());
  EXPECT_EQ(s.NumElements(), 24);
}

TEST(OnnxOptimShape, PushBackAndRankLimit) {
  core::symbolic::SymShape s;
  for (std::size_t i = 0; i < core::symbolic::kMaxOptimRank; ++i) {
    s.PushBack(core::symbolic::SymDim(static_cast<int64_t>(i + 1)));
  }
  EXPECT_EQ(s.Rank(), core::symbolic::kMaxOptimRank);
  EXPECT_THROW(s.PushBack(core::symbolic::SymDim(1)), std::length_error);
}

TEST(OnnxOptimShape, RejectsOversizedInitializerList) {
  const auto make_oversized = []<std::size_t... I>(std::index_sequence<I...>) {
    return core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(I + 1))...};
  };
  EXPECT_THROW((make_oversized(std::make_index_sequence<core::symbolic::kMaxOptimRank + 1>{})),
               std::length_error);
}

TEST(OnnxOptimTensor, WrapsExternalBufferWithoutAllocation) {
  std::array<float, 6> buffer = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  core::symbolic::SymTensor t(buffer.data(), core::symbolic::TensorType::kFloat, shape);

  EXPECT_FALSE(t.IsNull());
  // The tensor must reference the exact same buffer (no copy / no allocation).
  EXPECT_EQ(t.Data(), static_cast<void *>(buffer.data()));
  EXPECT_EQ(t.Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(t.Shape().Rank(), 2u);
  EXPECT_EQ(t.Shape().NumElements(), 6);

  // Mutating through the tensor reflects in the original buffer.
  static_cast<float *>(t.Data())[0] = 99.0f;
  EXPECT_FLOAT_EQ(buffer[0], 99.0f);
}

TEST(OnnxOptimTensor, DefaultConstructedIsNull) {
  core::symbolic::SymTensor t;
  EXPECT_TRUE(t.IsNull());
  EXPECT_EQ(t.Data(), nullptr);
  EXPECT_TRUE(t.Shape().Empty());
  EXPECT_EQ(t.Dtype(), core::symbolic::TensorType::kUndefined);
}

TEST(OnnxOptimTensor, SymbolicShapeIsAllowed) {
  std::vector<int64_t> buffer(4, 0);
  core::symbolic::SymShape shape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4)};
  core::symbolic::SymTensor t(buffer.data(), core::symbolic::TensorType::kInt64, shape);
  EXPECT_FALSE(t.Shape().IsFullyKnown());
  EXPECT_EQ(t.Shape()[0].AsExpr(), "N");
  EXPECT_EQ(t.Shape()[1].AsInt(), 4);
}

TEST(OnnxOptimTensor, ValueAsShapeDefaultsToAbsent) {
  core::symbolic::SymTensor t;
  EXPECT_FALSE(t.HasValueAsShape());
  EXPECT_THROW(t.ValueAsShape(), std::bad_optional_access);
}

TEST(OnnxOptimTensor, SetValueAsShapeStoresNonEmptyShape) {
  std::array<int64_t, 3> buffer = {2, 3, 4};
  core::symbolic::SymShape shape{core::symbolic::SymDim(static_cast<int64_t>(3))};
  core::symbolic::SymTensor t(buffer.data(), core::symbolic::TensorType::kInt64, shape);

  t.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(static_cast<int64_t>(2)),
                                             core::symbolic::SymDim(static_cast<int64_t>(3)),
                                             core::symbolic::SymDim(static_cast<int64_t>(4))});
  EXPECT_TRUE(t.HasValueAsShape());
  ASSERT_EQ(t.ValueAsShape().Rank(), 3u);
  EXPECT_EQ(t.ValueAsShape()[0].AsInt(), 2);
  EXPECT_EQ(t.ValueAsShape()[2].AsInt(), 4);
}

TEST(OnnxOptimTensor, SetValueAsShapeAcceptsEmptyShape) {
  core::symbolic::SymTensor t;
  t.SetValueAsShape(core::symbolic::SymShape{});
  EXPECT_TRUE(t.HasValueAsShape());
  EXPECT_TRUE(t.ValueAsShape().Empty());
}

TEST(OnnxOptimTensor, ClearValueAsShapeResetsFlag) {
  core::symbolic::SymTensor t;
  t.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim("N")});
  EXPECT_TRUE(t.HasValueAsShape());
  t.ClearValueAsShape();
  EXPECT_FALSE(t.HasValueAsShape());
}

TEST(OnnxOptimShape, Equality) {
  core::symbolic::SymShape a{core::symbolic::SymDim(2), core::symbolic::SymDim("N")};
  core::symbolic::SymShape b{core::symbolic::SymDim(2), core::symbolic::SymDim("N")};
  core::symbolic::SymShape c{core::symbolic::SymDim(2), core::symbolic::SymDim("M")};
  core::symbolic::SymShape d{core::symbolic::SymDim(2)};
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);
}

TEST(OnnxOptimTensor, Equality) {
  std::array<float, 6> buf = {1, 2, 3, 4, 5, 6};
  std::array<float, 6> other = {1, 2, 3, 4, 5, 6};
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};

  core::symbolic::SymTensor a(buf.data(), core::symbolic::TensorType::kFloat, shape);
  core::symbolic::SymTensor b(buf.data(), core::symbolic::TensorType::kFloat, shape);
  EXPECT_EQ(a, b);

  // Different data pointer.
  core::symbolic::SymTensor different_buffer(other.data(), core::symbolic::TensorType::kFloat,
                                             shape);
  EXPECT_NE(a, different_buffer);

  // Different dtype.
  core::symbolic::SymTensor different_dtype(buf.data(), core::symbolic::TensorType::kDouble, shape);
  EXPECT_NE(a, different_dtype);

  // Different shape.
  core::symbolic::SymTensor different_shape(buf.data(), core::symbolic::TensorType::kFloat,
                                            core::symbolic::SymShape{core::symbolic::SymDim(6)});
  EXPECT_NE(a, different_shape);

  // Differing value-as-shape annotation.
  core::symbolic::SymTensor with_vas = a;
  with_vas.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(2)});
  EXPECT_NE(a, with_vas);
}

TEST(OnnxOptimDim, ToString) {
  EXPECT_EQ(core::symbolic::SymDim(42).ToString(), "42");
  EXPECT_EQ(core::symbolic::SymDim(-1).ToString(), "-1");
  EXPECT_EQ(core::symbolic::SymDim("N").ToString(), "N");
}

TEST(OnnxOptimShape, ToString) {
  EXPECT_EQ(core::symbolic::SymShape{}.ToString(), "[]");
  EXPECT_EQ(core::symbolic::SymShape{core::symbolic::SymDim(7)}.ToString(), "[7]");
  core::symbolic::SymShape s{core::symbolic::SymDim(2), core::symbolic::SymDim("N"),
                             core::symbolic::SymDim(3)};
  EXPECT_EQ(s.ToString(), "[2,N,3]");
}

TEST(OnnxOptimTensor, ToString) {
  // Null tensor (no data, undefined dtype, empty shape).
  core::symbolic::SymTensor null_tensor;
  EXPECT_EQ(null_tensor.ToString(), "SymTensor(dtype=Undefined, shape=[])");

  // Tensor with data, dtype, and shape: data pointer is included.
  std::array<float, 6> buf = {1, 2, 3, 4, 5, 6};
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  core::symbolic::SymTensor t(buf.data(), core::symbolic::TensorType::kFloat, shape);
  const std::string s = t.ToString();
  EXPECT_NE(s.find("SymTensor(dtype=Float, shape=[2,3]"), std::string::npos);
  EXPECT_NE(s.find(", data="), std::string::npos);

  // Tensor with a value-as-shape annotation.
  t.SetValueAsShape(
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim("N")});
  const std::string s2 = t.ToString();
  EXPECT_NE(s2.find("value_as_shape=[2,N]"), std::string::npos);
}

TEST(OnnxOptimTensor, CmpEqualIsMorePrecise) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)};
  core::symbolic::SymTensor a(nullptr, core::symbolic::TensorType::kFloat, shape);
  core::symbolic::SymTensor b(nullptr, core::symbolic::TensorType::kFloat, shape);
  EXPECT_EQ(a.Cmp(b), core::symbolic::SymCmpResult::kMorePrecise);
  EXPECT_EQ(b.Cmp(a), core::symbolic::SymCmpResult::kMorePrecise);
}

TEST(OnnxOptimTensor, CmpConflictDtype) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor a(nullptr, core::symbolic::TensorType::kFloat, shape);
  core::symbolic::SymTensor b(nullptr, core::symbolic::TensorType::kDouble, shape);
  EXPECT_EQ(a.Cmp(b), core::symbolic::SymCmpResult::kConflict);
  EXPECT_EQ(b.Cmp(a), core::symbolic::SymCmpResult::kConflict);
}

TEST(OnnxOptimTensor, CmpConflictRank) {
  core::symbolic::SymTensor a(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(2)});
  core::symbolic::SymTensor b(
      nullptr, core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});
  EXPECT_EQ(a.Cmp(b), core::symbolic::SymCmpResult::kConflict);
}

TEST(OnnxOptimTensor, CmpConflictDim) {
  core::symbolic::SymTensor a(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(2)});
  core::symbolic::SymTensor b(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(3)});
  EXPECT_EQ(a.Cmp(b), core::symbolic::SymCmpResult::kConflict);

  core::symbolic::SymTensor c(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim("N")});
  core::symbolic::SymTensor d(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim("M")});
  EXPECT_EQ(c.Cmp(d), core::symbolic::SymCmpResult::kConflict);
}

TEST(OnnxOptimTensor, CmpMorePreciseDtype) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor known(nullptr, core::symbolic::TensorType::kFloat, shape);
  core::symbolic::SymTensor unknown(nullptr, core::symbolic::TensorType::kUndefined, shape);
  EXPECT_EQ(known.Cmp(unknown), core::symbolic::SymCmpResult::kMorePrecise);
  EXPECT_EQ(unknown.Cmp(known), core::symbolic::SymCmpResult::kLessPrecise);
}

TEST(OnnxOptimTensor, CmpMorePreciseDim) {
  core::symbolic::SymTensor concrete(
      nullptr, core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(3)});
  core::symbolic::SymTensor symbolic(
      nullptr, core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim("N")});
  EXPECT_EQ(concrete.Cmp(symbolic), core::symbolic::SymCmpResult::kMorePrecise);
  EXPECT_EQ(symbolic.Cmp(concrete), core::symbolic::SymCmpResult::kLessPrecise);
}

TEST(OnnxOptimTensor, CmpComplementaryDims) {
  // lhs is more precise on dim 0, rhs is more precise on dim 1.
  core::symbolic::SymTensor lhs(
      nullptr, core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim("N")});
  core::symbolic::SymTensor rhs(
      nullptr, core::symbolic::TensorType::kFloat,
      core::symbolic::SymShape{core::symbolic::SymDim("M"), core::symbolic::SymDim(3)});
  EXPECT_EQ(lhs.Cmp(rhs), core::symbolic::SymCmpResult::kComplementary);
  EXPECT_EQ(rhs.Cmp(lhs), core::symbolic::SymCmpResult::kComplementary);
}

TEST(OnnxOptimTensor, CmpComplementaryDtypeAndShape) {
  // lhs knows dtype only; rhs knows a more concrete dim only.
  core::symbolic::SymTensor lhs(nullptr, core::symbolic::TensorType::kFloat,
                                core::symbolic::SymShape{core::symbolic::SymDim("N")});
  core::symbolic::SymTensor rhs(nullptr, core::symbolic::TensorType::kUndefined,
                                core::symbolic::SymShape{core::symbolic::SymDim(5)});
  EXPECT_EQ(lhs.Cmp(rhs), core::symbolic::SymCmpResult::kComplementary);
  EXPECT_EQ(rhs.Cmp(lhs), core::symbolic::SymCmpResult::kComplementary);
}

TEST(OnnxOptimTensor, CmpValueAsShape) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor with_vas(nullptr, core::symbolic::TensorType::kInt64, shape);
  with_vas.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(4)});
  core::symbolic::SymTensor without_vas(nullptr, core::symbolic::TensorType::kInt64, shape);
  EXPECT_EQ(with_vas.Cmp(without_vas), core::symbolic::SymCmpResult::kMorePrecise);
  EXPECT_EQ(without_vas.Cmp(with_vas), core::symbolic::SymCmpResult::kLessPrecise);

  // Conflicting value-as-shape.
  core::symbolic::SymTensor other_vas(nullptr, core::symbolic::TensorType::kInt64, shape);
  other_vas.SetValueAsShape(core::symbolic::SymShape{core::symbolic::SymDim(5)});
  EXPECT_EQ(with_vas.Cmp(other_vas), core::symbolic::SymCmpResult::kConflict);
}

TEST(OnnxOptimTensor, CmpDataPresence) {
  std::array<float, 2> buf = {1.0f, 2.0f};
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor with_data(buf.data(), core::symbolic::TensorType::kFloat, shape);
  core::symbolic::SymTensor without_data(nullptr, core::symbolic::TensorType::kFloat, shape);
  EXPECT_EQ(with_data.Cmp(without_data), core::symbolic::SymCmpResult::kMorePrecise);
  EXPECT_EQ(without_data.Cmp(with_data), core::symbolic::SymCmpResult::kLessPrecise);

  // Two distinct non-null pointers carry no precision signal.
  std::array<float, 2> buf2 = {1.0f, 2.0f};
  core::symbolic::SymTensor with_other_data(buf2.data(), core::symbolic::TensorType::kFloat, shape);
  EXPECT_EQ(with_data.Cmp(with_other_data), core::symbolic::SymCmpResult::kMorePrecise);
}

TEST(OnnxOptimDevice, MakeGPUDeviceAndHelpers) {
  EXPECT_FALSE(core::symbolic::IsGPU(core::symbolic::Device::kUndefined));
  EXPECT_FALSE(core::symbolic::IsGPU(core::symbolic::Device::kCPU));
  EXPECT_TRUE(core::symbolic::IsGPU(core::symbolic::Device::kGPU0));
  EXPECT_TRUE(core::symbolic::IsGPU(core::symbolic::Device::kGPU8191));

  EXPECT_EQ(core::symbolic::GPUIndex(core::symbolic::Device::kUndefined), -1);
  EXPECT_EQ(core::symbolic::GPUIndex(core::symbolic::Device::kCPU), -1);
  EXPECT_EQ(core::symbolic::GPUIndex(core::symbolic::Device::kGPU0), 0);
  EXPECT_EQ(core::symbolic::GPUIndex(core::symbolic::Device::kGPU8191),
            core::symbolic::kMaxGPUIndex);

  EXPECT_EQ(core::symbolic::MakeGPUDevice(0), core::symbolic::Device::kGPU0);
  EXPECT_EQ(
      core::symbolic::MakeGPUDevice(7),
      static_cast<core::symbolic::Device>(static_cast<int32_t>(core::symbolic::Device::kGPU0) + 7));
  EXPECT_EQ(core::symbolic::MakeGPUDevice(core::symbolic::kMaxGPUIndex),
            core::symbolic::Device::kGPU8191);
  EXPECT_THROW(core::symbolic::MakeGPUDevice(-1), std::out_of_range);
  EXPECT_THROW(core::symbolic::MakeGPUDevice(core::symbolic::kMaxGPUIndex + 1), std::out_of_range);
}

TEST(OnnxOptimDevice, DeviceName) {
  EXPECT_EQ(core::symbolic::DeviceName(core::symbolic::Device::kUndefined), "Undefined");
  EXPECT_EQ(core::symbolic::DeviceName(core::symbolic::Device::kCPU), "CPU");
  EXPECT_EQ(core::symbolic::DeviceName(core::symbolic::Device::kGPU0), "GPU0");
  EXPECT_EQ(core::symbolic::DeviceName(core::symbolic::MakeGPUDevice(42)), "GPU42");
  EXPECT_EQ(core::symbolic::DeviceName(core::symbolic::Device::kGPU8191), "GPU8191");
}

TEST(OnnxOptimDevice, DeviceKeySuffix) {
  // The default host devices contribute no suffix so dispatch keys keep their
  // plain "<domain>:<op_type>" form.
  EXPECT_EQ(core::symbolic::DeviceKeySuffix(core::symbolic::Device::kUndefined), "");
  EXPECT_EQ(core::symbolic::DeviceKeySuffix(core::symbolic::Device::kCPU), "");
  // Any other device appends ":<device>" (the integer enumerator value).
  EXPECT_EQ(core::symbolic::DeviceKeySuffix(core::symbolic::Device::kGPU0),
            ":" + std::to_string(static_cast<int32_t>(core::symbolic::Device::kGPU0)));
  EXPECT_EQ(core::symbolic::DeviceKeySuffix(core::symbolic::MakeGPUDevice(42)),
            ":" + std::to_string(static_cast<int32_t>(core::symbolic::MakeGPUDevice(42))));
}

TEST(OnnxOptimTensor, DeviceDefaultsToUndefined) {
  core::symbolic::SymTensor t;
  EXPECT_EQ(t.GetDevice(), core::symbolic::Device::kUndefined);
}

TEST(OnnxOptimTensor, SetDeviceRoundTrip) {
  core::symbolic::SymTensor t;
  t.SetDevice(core::symbolic::Device::kCPU);
  EXPECT_EQ(t.GetDevice(), core::symbolic::Device::kCPU);
  t.SetDevice(core::symbolic::MakeGPUDevice(3));
  EXPECT_EQ(core::symbolic::GPUIndex(t.GetDevice()), 3);
}

TEST(OnnxOptimTensor, EqualityIncludesDevice) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor a(nullptr, core::symbolic::TensorType::kFloat, shape);
  core::symbolic::SymTensor b(nullptr, core::symbolic::TensorType::kFloat, shape);
  EXPECT_EQ(a, b);
  b.SetDevice(core::symbolic::Device::kCPU);
  EXPECT_NE(a, b);
  a.SetDevice(core::symbolic::Device::kCPU);
  EXPECT_EQ(a, b);
}

TEST(OnnxOptimTensor, ToStringIncludesDevice) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kFloat, shape);
  // Undefined device is omitted from the string.
  EXPECT_EQ(t.ToString().find("device="), std::string::npos);
  t.SetDevice(core::symbolic::Device::kCPU);
  EXPECT_NE(t.ToString().find("device=CPU"), std::string::npos);
  t.SetDevice(core::symbolic::MakeGPUDevice(5));
  EXPECT_NE(t.ToString().find("device=GPU5"), std::string::npos);
}

TEST(OnnxOptimTensor, CmpDevice) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor known(nullptr, core::symbolic::TensorType::kFloat, shape);
  known.SetDevice(core::symbolic::Device::kCPU);
  core::symbolic::SymTensor unknown(nullptr, core::symbolic::TensorType::kFloat, shape);
  EXPECT_EQ(known.Cmp(unknown), core::symbolic::SymCmpResult::kMorePrecise);
  EXPECT_EQ(unknown.Cmp(known), core::symbolic::SymCmpResult::kLessPrecise);

  core::symbolic::SymTensor cpu(nullptr, core::symbolic::TensorType::kFloat, shape);
  cpu.SetDevice(core::symbolic::Device::kCPU);
  core::symbolic::SymTensor gpu(nullptr, core::symbolic::TensorType::kFloat, shape);
  gpu.SetDevice(core::symbolic::Device::kGPU0);
  EXPECT_EQ(cpu.Cmp(gpu), core::symbolic::SymCmpResult::kConflict);

  core::symbolic::SymTensor cpu2(nullptr, core::symbolic::TensorType::kFloat, shape);
  cpu2.SetDevice(core::symbolic::Device::kCPU);
  EXPECT_EQ(cpu.Cmp(cpu2), core::symbolic::SymCmpResult::kMorePrecise);
}

TEST(OnnxOptimDevice, DeviceFromName) {
  EXPECT_EQ(core::symbolic::DeviceFromName("CPU"), core::symbolic::Device::kCPU);
  EXPECT_EQ(core::symbolic::DeviceFromName("Undefined"), core::symbolic::Device::kUndefined);
  EXPECT_EQ(core::symbolic::DeviceFromName("GPU0"), core::symbolic::Device::kGPU0);
  EXPECT_EQ(core::symbolic::DeviceFromName("GPU42"), core::symbolic::MakeGPUDevice(42));
  EXPECT_EQ(core::symbolic::DeviceFromName("GPU8191"), core::symbolic::Device::kGPU8191);
  // Round-trip every well-formed name.
  for (int i = 0; i <= core::symbolic::kMaxGPUIndex; i += 2731) {
    const core::symbolic::Device d = core::symbolic::MakeGPUDevice(i);
    EXPECT_EQ(core::symbolic::DeviceFromName(core::symbolic::DeviceName(d)), d);
  }
  // Malformed / out-of-range names map back to Undefined.
  EXPECT_EQ(core::symbolic::DeviceFromName(""), core::symbolic::Device::kUndefined);
  EXPECT_EQ(core::symbolic::DeviceFromName("cpu"), core::symbolic::Device::kUndefined);
  EXPECT_EQ(core::symbolic::DeviceFromName("GPU"), core::symbolic::Device::kUndefined);
  EXPECT_EQ(core::symbolic::DeviceFromName("GPU-1"), core::symbolic::Device::kUndefined);
  EXPECT_EQ(core::symbolic::DeviceFromName("GPU+1"), core::symbolic::Device::kUndefined);
  EXPECT_EQ(core::symbolic::DeviceFromName("GPU 1"), core::symbolic::Device::kUndefined);
  EXPECT_EQ(core::symbolic::DeviceFromName("GPU8192"), core::symbolic::Device::kUndefined);
  EXPECT_EQ(core::symbolic::DeviceFromName("GPU18446744073709551616"),
            core::symbolic::Device::kUndefined);
  EXPECT_EQ(core::symbolic::DeviceFromName("Unknown"), core::symbolic::Device::kUndefined);
}

namespace {

ValueInfoProto MakeTensorValueInfo(const std::string &name, TensorProto::DataType dtype,
                                   const std::vector<core::symbolic::SymDim> &dims) {
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
      {core::symbolic::SymDim(2), core::symbolic::SymDim("N"), core::symbolic::SymDim(3)});
  core::symbolic::SymTensor t;
  EXPECT_TRUE(core::symbolic::SymTensorFromValueInfo(vi, t));
  EXPECT_EQ(t.Dtype(), core::symbolic::TensorType::kFloat);
  ASSERT_EQ(t.Shape().Rank(), 3u);
  EXPECT_TRUE(t.Shape()[0].IsInt());
  EXPECT_EQ(t.Shape()[0].AsInt(), 2);
  EXPECT_TRUE(t.Shape()[1].IsExpr());
  EXPECT_EQ(t.Shape()[1].AsExpr(), "N");
  EXPECT_EQ(t.Shape()[2].AsInt(), 3);
  EXPECT_EQ(t.GetDevice(), core::symbolic::Device::kUndefined);
}

TEST(OnnxOptimValueInfo, FromValueInfoMissingTypeReturnsFalse) {
  ValueInfoProto vi;
  vi.set_name("x");
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kInt32,
                              core::symbolic::SymShape{});
  EXPECT_FALSE(core::symbolic::SymTensorFromValueInfo(vi, t));
  // ``out`` must be left untouched on failure.
  EXPECT_EQ(t.Dtype(), core::symbolic::TensorType::kInt32);
}

TEST(OnnxOptimValueInfo, FromValueInfoReadsDeviceMetadata) {
  ValueInfoProto vi =
      MakeTensorValueInfo("x", TensorProto::DataType::FLOAT, {core::symbolic::SymDim(4)});
  auto *entry = vi.add_metadata_props();
  entry->set_key(core::symbolic::kValueInfoDeviceMetadataKey);
  entry->set_value("GPU3");
  core::symbolic::SymTensor t;
  ASSERT_TRUE(core::symbolic::SymTensorFromValueInfo(vi, t));
  EXPECT_EQ(t.GetDevice(), core::symbolic::MakeGPUDevice(3));
}

TEST(OnnxOptimValueInfo, FromValueInfoIgnoresUnknownDeviceMetadata) {
  ValueInfoProto vi =
      MakeTensorValueInfo("x", TensorProto::DataType::FLOAT, {core::symbolic::SymDim(4)});
  auto *entry = vi.add_metadata_props();
  entry->set_key(core::symbolic::kValueInfoDeviceMetadataKey);
  entry->set_value("Mars");
  core::symbolic::SymTensor t;
  ASSERT_TRUE(core::symbolic::SymTensorFromValueInfo(vi, t));
  EXPECT_EQ(t.GetDevice(), core::symbolic::Device::kUndefined);
}

TEST(OnnxOptimValueInfo, ToValueInfoWritesTypeShape) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim("M")};
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kInt64, shape);
  ValueInfoProto vi;
  vi.set_name("y");
  ASSERT_TRUE(core::symbolic::SymTensorToValueInfo(t, vi));
  ASSERT_TRUE(vi.has_type());
  ASSERT_TRUE(vi.type().has_tensor_type());
  const auto &tt = vi.type().tensor_type();
  EXPECT_EQ(tt.elem_type(), static_cast<int>(TensorProto::DataType::INT64));
  ASSERT_TRUE(tt.has_shape());
  ASSERT_EQ(tt.shape().dim().size(), 2);
  EXPECT_EQ(tt.shape().dim()[0].dim_value(), 2);
  EXPECT_EQ(std::string(tt.shape().dim()[1].dim_param()), "M");
  // No device set => no metadata_props entry added.
  EXPECT_EQ(vi.metadata_props().size(), 0u);
  // The original name is preserved.
  EXPECT_EQ(vi.name(), "y");
}

TEST(OnnxOptimValueInfo, ToValueInfoUndefinedDtypeReturnsFalse) {
  core::symbolic::SymTensor t;
  ValueInfoProto vi;
  vi.set_name("y");
  EXPECT_FALSE(core::symbolic::SymTensorToValueInfo(t, vi));
  EXPECT_FALSE(vi.has_type());
}

TEST(OnnxOptimValueInfo, ToValueInfoWritesDeviceMetadata) {
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(1)});
  t.SetDevice(core::symbolic::MakeGPUDevice(7));
  ValueInfoProto vi;
  ASSERT_TRUE(core::symbolic::SymTensorToValueInfo(t, vi));
  ASSERT_EQ(vi.metadata_props().size(), 1u);
  EXPECT_EQ(std::string(vi.metadata_props()[0].key()), core::symbolic::kValueInfoDeviceMetadataKey);
  EXPECT_EQ(std::string(vi.metadata_props()[0].value()), "GPU7");
}

TEST(OnnxOptimValueInfo, ToValueInfoUpdatesExistingDeviceMetadataInPlace) {
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(1)});
  t.SetDevice(core::symbolic::Device::kCPU);
  ValueInfoProto vi;
  // Pre-existing unrelated metadata + stale device entry.
  auto *misc = vi.add_metadata_props();
  misc->set_key("author");
  misc->set_value("test");
  auto *dev = vi.add_metadata_props();
  dev->set_key(core::symbolic::kValueInfoDeviceMetadataKey);
  dev->set_value("GPU0");
  ASSERT_TRUE(core::symbolic::SymTensorToValueInfo(t, vi));
  ASSERT_EQ(vi.metadata_props().size(), 2u);
  // The unrelated entry survives untouched.
  EXPECT_EQ(std::string(vi.metadata_props()[0].key()), "author");
  EXPECT_EQ(std::string(vi.metadata_props()[0].value()), "test");
  // The device entry is updated in place.
  EXPECT_EQ(std::string(vi.metadata_props()[1].key()), core::symbolic::kValueInfoDeviceMetadataKey);
  EXPECT_EQ(std::string(vi.metadata_props()[1].value()), "CPU");
}

TEST(OnnxOptimValueInfo, ToValueInfoRemovesStaleDeviceMetadata) {
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(1)});
  // device left undefined on purpose.
  ValueInfoProto vi;
  auto *dev = vi.add_metadata_props();
  dev->set_key(core::symbolic::kValueInfoDeviceMetadataKey);
  dev->set_value("GPU0");
  auto *misc = vi.add_metadata_props();
  misc->set_key("author");
  misc->set_value("test");
  ASSERT_TRUE(core::symbolic::SymTensorToValueInfo(t, vi));
  ASSERT_EQ(vi.metadata_props().size(), 1u);
  // The remaining entry is the unrelated one (swap-and-pop moved it
  // from position 1 to position 0).
  EXPECT_EQ(std::string(vi.metadata_props()[0].key()), "author");
  EXPECT_EQ(std::string(vi.metadata_props()[0].value()), "test");
}

TEST(OnnxOptimValueInfo, RoundTripPreservesDtypeShapeAndDevice) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(2), core::symbolic::SymDim("N"),
                                 core::symbolic::SymDim(5)};
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kDouble, shape);
  t.SetDevice(core::symbolic::MakeGPUDevice(42));
  ValueInfoProto vi;
  vi.set_name("rt");
  ASSERT_TRUE(core::symbolic::SymTensorToValueInfo(t, vi));
  core::symbolic::SymTensor back;
  ASSERT_TRUE(core::symbolic::SymTensorFromValueInfo(vi, back));
  EXPECT_EQ(back.Dtype(), t.Dtype());
  EXPECT_EQ(back.GetDevice(), t.GetDevice());
  ASSERT_EQ(back.Shape().Rank(), shape.Rank());
  EXPECT_EQ(back.Shape()[0].AsInt(), 2);
  EXPECT_EQ(back.Shape()[1].AsExpr(), "N");
  EXPECT_EQ(back.Shape()[2].AsInt(), 5);
}

TEST(OnnxOptimTensor, MinMaxDefaultsToAbsent) {
  core::symbolic::SymTensor t;
  EXPECT_FALSE(t.HasMin());
  EXPECT_FALSE(t.HasMax());
  EXPECT_FALSE(t.IsNullConstant());
  EXPECT_THROW((void)t.Min(), std::bad_optional_access);
  EXPECT_THROW((void)t.Max(), std::bad_optional_access);
}

TEST(OnnxOptimTensor, SetMinMaxStoresBounds) {
  core::symbolic::SymTensor t;
  t.SetMin(-1.5);
  t.SetMax(2.5);
  EXPECT_TRUE(t.HasMin());
  EXPECT_TRUE(t.HasMax());
  EXPECT_EQ(t.Min(), -1.5);
  EXPECT_EQ(t.Max(), 2.5);
}

TEST(OnnxOptimTensor, SetMinMaxAtomicRejectsInvertedRange) {
  core::symbolic::SymTensor t;
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
  core::symbolic::SymTensor t;
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
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(3)});
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
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor a(nullptr, core::symbolic::TensorType::kFloat, shape);
  core::symbolic::SymTensor b(nullptr, core::symbolic::TensorType::kFloat, shape);
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
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kFloat, shape);
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
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor known(nullptr, core::symbolic::TensorType::kFloat, shape);
  known.SetMinMax(-1.0, 1.0);
  core::symbolic::SymTensor unknown(nullptr, core::symbolic::TensorType::kFloat, shape);
  EXPECT_EQ(known.Cmp(unknown), core::symbolic::SymCmpResult::kMorePrecise);
  EXPECT_EQ(unknown.Cmp(known), core::symbolic::SymCmpResult::kLessPrecise);
}

TEST(OnnxOptimTensor, CmpMinMaxTighterIsMorePrecise) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor tight(nullptr, core::symbolic::TensorType::kFloat, shape);
  tight.SetMinMax(0.0, 1.0);
  core::symbolic::SymTensor loose(nullptr, core::symbolic::TensorType::kFloat, shape);
  loose.SetMinMax(-1.0, 2.0);
  EXPECT_EQ(tight.Cmp(loose), core::symbolic::SymCmpResult::kMorePrecise);
  EXPECT_EQ(loose.Cmp(tight), core::symbolic::SymCmpResult::kLessPrecise);
}

TEST(OnnxOptimTensor, CmpMinMaxComplementaryBounds) {
  // One side has a tighter min, the other a tighter max.
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor a(nullptr, core::symbolic::TensorType::kFloat, shape);
  a.SetMinMax(0.0, 2.0);
  core::symbolic::SymTensor b(nullptr, core::symbolic::TensorType::kFloat, shape);
  b.SetMinMax(-1.0, 1.0);
  EXPECT_EQ(a.Cmp(b), core::symbolic::SymCmpResult::kComplementary);
  EXPECT_EQ(b.Cmp(a), core::symbolic::SymCmpResult::kComplementary);
}

TEST(OnnxOptimTensor, CmpMinMaxDisjointConflict) {
  core::symbolic::SymShape shape{core::symbolic::SymDim(2)};
  core::symbolic::SymTensor a(nullptr, core::symbolic::TensorType::kFloat, shape);
  a.SetMinMax(0.0, 1.0);
  core::symbolic::SymTensor b(nullptr, core::symbolic::TensorType::kFloat, shape);
  b.SetMinMax(2.0, 3.0);
  EXPECT_EQ(a.Cmp(b), core::symbolic::SymCmpResult::kConflict);
  EXPECT_EQ(b.Cmp(a), core::symbolic::SymCmpResult::kConflict);
}

TEST(OnnxOptimValueInfo, ToValueInfoWritesMinMaxMetadata) {
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(1)});
  t.SetMinMax(-2.5, 3.5);
  ValueInfoProto vi;
  ASSERT_TRUE(core::symbolic::SymTensorToValueInfo(t, vi));
  ASSERT_EQ(vi.metadata_props().size(), 2u);
  bool seen_min = false;
  bool seen_max = false;
  for (int i = 0; i < static_cast<int>(vi.metadata_props().size()); ++i) {
    const std::string key = vi.metadata_props()[i].key();
    const std::string value = vi.metadata_props()[i].value();
    if (key == core::symbolic::kValueInfoMinMetadataKey) {
      seen_min = true;
      EXPECT_EQ(std::stod(value), -2.5);
    } else if (key == core::symbolic::kValueInfoMaxMetadataKey) {
      seen_max = true;
      EXPECT_EQ(std::stod(value), 3.5);
    }
  }
  EXPECT_TRUE(seen_min);
  EXPECT_TRUE(seen_max);
}

TEST(OnnxOptimValueInfo, ToValueInfoRemovesStaleMinMaxMetadata) {
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(1)});
  // No min/max set; ensure pre-existing entries are removed.
  ValueInfoProto vi;
  auto *min_entry = vi.add_metadata_props();
  min_entry->set_key(core::symbolic::kValueInfoMinMetadataKey);
  min_entry->set_value("-1");
  auto *max_entry = vi.add_metadata_props();
  max_entry->set_key(core::symbolic::kValueInfoMaxMetadataKey);
  max_entry->set_value("1");
  auto *misc = vi.add_metadata_props();
  misc->set_key("author");
  misc->set_value("test");
  ASSERT_TRUE(core::symbolic::SymTensorToValueInfo(t, vi));
  ASSERT_EQ(vi.metadata_props().size(), 1u);
  EXPECT_EQ(std::string(vi.metadata_props()[0].key()), "author");
}

TEST(OnnxOptimValueInfo, FromValueInfoReadsMinMaxMetadata) {
  ValueInfoProto vi =
      MakeTensorValueInfo("x", TensorProto::DataType::FLOAT, {core::symbolic::SymDim(4)});
  auto *min_entry = vi.add_metadata_props();
  min_entry->set_key(core::symbolic::kValueInfoMinMetadataKey);
  min_entry->set_value("-3.25");
  auto *max_entry = vi.add_metadata_props();
  max_entry->set_key(core::symbolic::kValueInfoMaxMetadataKey);
  max_entry->set_value("4.75");
  core::symbolic::SymTensor t;
  ASSERT_TRUE(core::symbolic::SymTensorFromValueInfo(vi, t));
  ASSERT_TRUE(t.HasMin());
  ASSERT_TRUE(t.HasMax());
  EXPECT_EQ(t.Min(), -3.25);
  EXPECT_EQ(t.Max(), 4.75);
}

TEST(OnnxOptimValueInfo, FromValueInfoIgnoresUnparseableMinMax) {
  ValueInfoProto vi =
      MakeTensorValueInfo("x", TensorProto::DataType::FLOAT, {core::symbolic::SymDim(4)});
  auto *min_entry = vi.add_metadata_props();
  min_entry->set_key(core::symbolic::kValueInfoMinMetadataKey);
  min_entry->set_value("not-a-number");
  core::symbolic::SymTensor t;
  ASSERT_TRUE(core::symbolic::SymTensorFromValueInfo(vi, t));
  EXPECT_FALSE(t.HasMin());
  EXPECT_FALSE(t.HasMax());
}

TEST(OnnxOptimValueInfo, FromValueInfoIgnoresOutOfRangeMinMetadata) {
  ValueInfoProto vi =
      MakeTensorValueInfo("x", TensorProto::DataType::FLOAT, {core::symbolic::SymDim(4)});
  auto *min_entry = vi.add_metadata_props();
  min_entry->set_key(core::symbolic::kValueInfoMinMetadataKey);
  min_entry->set_value("1e309");
  core::symbolic::SymTensor t;
  ASSERT_TRUE(core::symbolic::SymTensorFromValueInfo(vi, t));
  EXPECT_FALSE(t.HasMin());
  EXPECT_FALSE(t.HasMax());
}

TEST(OnnxOptimValueInfo, RoundTripPreservesMinMax) {
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kFloat,
                              core::symbolic::SymShape{core::symbolic::SymDim(3)});
  t.SetMinMax(0.0, 0.0);
  ValueInfoProto vi;
  ASSERT_TRUE(core::symbolic::SymTensorToValueInfo(t, vi));
  core::symbolic::SymTensor back;
  ASSERT_TRUE(core::symbolic::SymTensorFromValueInfo(vi, back));
  ASSERT_TRUE(back.HasMin());
  ASSERT_TRUE(back.HasMax());
  EXPECT_EQ(back.Min(), 0.0);
  EXPECT_EQ(back.Max(), 0.0);
  EXPECT_TRUE(back.IsNullConstant());
}

namespace {

TensorProto MakeTensorProto(const std::string &name, TensorProto::DataType dtype,
                            const std::vector<int64_t> &dims) {
  TensorProto tp;
  tp.set_name(name);
  tp.set_data_type(static_cast<int>(dtype));
  for (int64_t d : dims) {
    tp.add_dims(d);
  }
  return tp;
}

} // namespace

TEST(OnnxOptimTensorProto, FromTensorProtoDtypeAndShape) {
  // FLOAT initializer without any payload: dtype and shape are still
  // populated but min/max are absent (nothing to read).
  TensorProto tp = MakeTensorProto("w", TensorProto::DataType::FLOAT, {2, 3, 4});
  core::symbolic::SymTensor t;
  EXPECT_TRUE(core::symbolic::SymTensorFromTensorProto(tp, t));
  EXPECT_EQ(t.Dtype(), core::symbolic::TensorType::kFloat);
  ASSERT_EQ(t.Shape().Rank(), 3u);
  EXPECT_EQ(t.Shape()[0].AsInt(), 2);
  EXPECT_EQ(t.Shape()[1].AsInt(), 3);
  EXPECT_EQ(t.Shape()[2].AsInt(), 4);
  EXPECT_EQ(t.Data(), nullptr);
  EXPECT_EQ(t.GetDevice(), core::symbolic::Device::kUndefined);
  EXPECT_FALSE(t.HasMin());
  EXPECT_FALSE(t.HasMax());
  EXPECT_FALSE(t.HasValueAsShape());
}

TEST(OnnxOptimTensorProto, FromTensorProtoScalarHasEmptyShape) {
  TensorProto tp = MakeTensorProto("s", TensorProto::DataType::INT64, {});
  core::symbolic::SymTensor t;
  EXPECT_TRUE(core::symbolic::SymTensorFromTensorProto(tp, t));
  EXPECT_EQ(t.Dtype(), core::symbolic::TensorType::kInt64);
  EXPECT_EQ(t.Shape().Rank(), 0u);
}

TEST(OnnxOptimTensorProto, FromTensorProtoUndefinedDtypeReturnsFalse) {
  TensorProto tp = MakeTensorProto("u", TensorProto::DataType::UNDEFINED, {1, 2});
  core::symbolic::SymTensor t(nullptr, core::symbolic::TensorType::kInt32,
                              core::symbolic::SymShape{});
  EXPECT_FALSE(core::symbolic::SymTensorFromTensorProto(tp, t));
  // ``out`` must be left untouched on failure.
  EXPECT_EQ(t.Dtype(), core::symbolic::TensorType::kInt32);
  EXPECT_EQ(t.Shape().Rank(), 0u);
}

TEST(OnnxOptimTensorProto, FromTensorProtoFloatPopulatesMinMax) {
  TensorProto tp = MakeTensorProto("w", TensorProto::DataType::FLOAT, {4});
  tp.add_float_data(-1.5f);
  tp.add_float_data(0.0f);
  tp.add_float_data(2.25f);
  tp.add_float_data(-3.5f);
  core::symbolic::SymTensor t;
  ASSERT_TRUE(core::symbolic::SymTensorFromTensorProto(tp, t));
  ASSERT_TRUE(t.HasMin());
  ASSERT_TRUE(t.HasMax());
  EXPECT_DOUBLE_EQ(t.Min(), -3.5);
  EXPECT_DOUBLE_EQ(t.Max(), 2.25);
  // Float initializers never get a value-as-shape annotation.
  EXPECT_FALSE(t.HasValueAsShape());
}

TEST(OnnxOptimTensorProto, FromTensorProtoDoublePopulatesMinMax) {
  TensorProto tp = MakeTensorProto("w", TensorProto::DataType::DOUBLE, {3});
  tp.add_double_data(7.0);
  tp.add_double_data(-2.5);
  tp.add_double_data(4.0);
  core::symbolic::SymTensor t;
  ASSERT_TRUE(core::symbolic::SymTensorFromTensorProto(tp, t));
  ASSERT_TRUE(t.HasMin());
  ASSERT_TRUE(t.HasMax());
  EXPECT_DOUBLE_EQ(t.Min(), -2.5);
  EXPECT_DOUBLE_EQ(t.Max(), 7.0);
}

TEST(OnnxOptimTensorProto, FromTensorProtoIntegerPopulatesMinMaxAndValueAsShape) {
  TensorProto tp = MakeTensorProto("dims", TensorProto::DataType::INT64, {3});
  tp.add_int64_data(2);
  tp.add_int64_data(5);
  tp.add_int64_data(-1);
  core::symbolic::SymTensor t;
  ASSERT_TRUE(core::symbolic::SymTensorFromTensorProto(tp, t));
  ASSERT_TRUE(t.HasMin());
  ASSERT_TRUE(t.HasMax());
  EXPECT_DOUBLE_EQ(t.Min(), -1.0);
  EXPECT_DOUBLE_EQ(t.Max(), 5.0);
  // Small 1-D integer initializers also get the value-as-shape annotation.
  ASSERT_TRUE(t.HasValueAsShape());
  ASSERT_EQ(t.ValueAsShape().Rank(), 3u);
  EXPECT_EQ(t.ValueAsShape()[0].AsInt(), 2);
  EXPECT_EQ(t.ValueAsShape()[1].AsInt(), 5);
  EXPECT_EQ(t.ValueAsShape()[2].AsInt(), -1);
}

TEST(OnnxOptimTensorProto, FromTensorProtoLargeIntegerSkipsValueAsShapeButKeepsMinMax) {
  TensorProto tp = MakeTensorProto("dims", TensorProto::DataType::INT64,
                                   {core::symbolic::kOptimValueAsShapeMaxElements});
  for (int64_t i = 0; i < core::symbolic::kOptimValueAsShapeMaxElements; ++i) {
    tp.add_int64_data(i);
  }
  core::symbolic::SymTensor t;
  ASSERT_TRUE(core::symbolic::SymTensorFromTensorProto(tp, t));
  ASSERT_TRUE(t.HasMin());
  ASSERT_TRUE(t.HasMax());
  EXPECT_DOUBLE_EQ(t.Min(), 0.0);
  EXPECT_DOUBLE_EQ(t.Max(), static_cast<double>(core::symbolic::kOptimValueAsShapeMaxElements - 1));
  EXPECT_FALSE(t.HasValueAsShape());
}

} // namespace Test
