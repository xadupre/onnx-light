// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/memory/simple_tensor.h"

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::AllocationHandle;
using core::runtime::DataType;
using core::runtime::ElementSize;
using core::runtime::FillValueInfo;
using core::runtime::MakeOutputTensor;
using core::runtime::PackedByteSize;
using core::runtime::RawByteBuffer;
using core::runtime::SimpleRawBufferAllocator;
using core::runtime::Tensor;
using core::runtime::TensorFromProto;

namespace Test {

// ---------------------------------------------------------------------------
// ElementSize
// ---------------------------------------------------------------------------

TEST(SimpleTensorElementSize, KnownTypes) {
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::FLOAT)), sizeof(float));
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::DOUBLE)), sizeof(double));
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::INT32)), sizeof(int32_t));
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::INT64)), sizeof(int64_t));
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::UINT8)), 1u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::INT8)), 1u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::BOOL)), 1u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::FLOAT8E4M3FN)), 1u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::FLOAT8E4M3FNUZ)), 1u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::FLOAT8E5M2)), 1u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::FLOAT8E5M2FNUZ)), 1u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::FLOAT8E8M0)), 1u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::UINT16)), 2u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::INT16)), 2u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::FLOAT16)), 2u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::BFLOAT16)), 2u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::UINT32)), 4u);
  EXPECT_EQ(ElementSize(static_cast<int32_t>(DataType::UINT64)), 8u);
}

TEST(SimpleTensorElementSize, UnsupportedThrows) {
  EXPECT_THROW(ElementSize(static_cast<int32_t>(DataType::UNDEFINED)), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// PackedByteSize
// ---------------------------------------------------------------------------

TEST(SimpleTensorPackedByteSize, SubByteTypes) {
  EXPECT_EQ(PackedByteSize(static_cast<int32_t>(DataType::INT4), 0), 0u);
  EXPECT_EQ(PackedByteSize(static_cast<int32_t>(DataType::INT4), 1), 1u);
  EXPECT_EQ(PackedByteSize(static_cast<int32_t>(DataType::UINT4), 3), 2u);
  EXPECT_EQ(PackedByteSize(static_cast<int32_t>(DataType::FLOAT4E2M1), 4), 2u);
  EXPECT_EQ(PackedByteSize(static_cast<int32_t>(DataType::INT2), 4), 1u);
  EXPECT_EQ(PackedByteSize(static_cast<int32_t>(DataType::UINT2), 5), 2u);
}

TEST(SimpleTensorPackedByteSize, RegularTypes) {
  EXPECT_EQ(PackedByteSize(static_cast<int32_t>(DataType::FLOAT), 4), 16u);
  EXPECT_EQ(PackedByteSize(static_cast<int32_t>(DataType::INT64), 3), 24u);
}

TEST(SimpleTensorPackedByteSize, NegativeCountThrows) {
  EXPECT_THROW(PackedByteSize(static_cast<int32_t>(DataType::FLOAT), -1), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Typed factories
// ---------------------------------------------------------------------------

TEST(SimpleTensorFactories, FromFloat) {
  Tensor t = Tensor::FromFloat("x", {2}, {1.5f, -2.5f});
  EXPECT_EQ(t.name, "x");
  EXPECT_EQ(t.data_type, static_cast<int32_t>(DataType::FLOAT));
  ASSERT_EQ(t.element_count(), 2);
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 1.5f);
  EXPECT_FLOAT_EQ(t.AsFloat()[1], -2.5f);
}

TEST(SimpleTensorFactories, FromDouble) {
  Tensor t = Tensor::FromDouble("d", {1}, {3.25});
  EXPECT_EQ(t.data_type, static_cast<int32_t>(DataType::DOUBLE));
  EXPECT_DOUBLE_EQ(t.AsDouble()[0], 3.25);
}

TEST(SimpleTensorFactories, FromInt32AndInt64) {
  Tensor i32 = Tensor::FromInt32("i32", {2}, {7, -3});
  EXPECT_EQ(i32.AsInt32()[0], 7);
  EXPECT_EQ(i32.AsInt32()[1], -3);
  Tensor i64 = Tensor::FromInt64("i64", {2}, {100, -200});
  EXPECT_EQ(i64.AsInt64()[0], 100);
  EXPECT_EQ(i64.AsInt64()[1], -200);
}

TEST(SimpleTensorFactories, FromInt8AndUint8) {
  Tensor i8 = Tensor::FromInt8("i8", {2}, {-1, 2});
  EXPECT_EQ(i8.AsInt8()[0], -1);
  EXPECT_EQ(i8.AsInt8()[1], 2);
  Tensor u8 = Tensor::FromUint8("u8", {2}, {200, 5});
  EXPECT_EQ(u8.AsUint8()[0], 200);
  EXPECT_EQ(u8.AsUint8()[1], 5);
}

TEST(SimpleTensorFactories, FromInt16AndUint16) {
  Tensor i16 = Tensor::FromInt16("i16", {2}, {-30000, 30000});
  EXPECT_EQ(i16.AsInt16()[0], -30000);
  EXPECT_EQ(i16.AsInt16()[1], 30000);
  Tensor u16 = Tensor::FromUint16("u16", {2}, {60000, 1});
  EXPECT_EQ(u16.AsUint16()[0], 60000);
  EXPECT_EQ(u16.AsUint16()[1], 1);
}

TEST(SimpleTensorFactories, FromUint32AndUint64) {
  Tensor u32 = Tensor::FromUint32("u32", {2}, {4000000000u, 1u});
  EXPECT_EQ(u32.AsUint32()[0], 4000000000u);
  EXPECT_EQ(u32.AsUint32()[1], 1u);
  Tensor u64 = Tensor::FromUint64("u64", {1}, {18000000000000000000ull});
  EXPECT_EQ(u64.AsUint64()[0], 18000000000000000000ull);
}

TEST(SimpleTensorFactories, FromBool) {
  Tensor b = Tensor::FromBool("b", {3}, {1, 0, 5});
  EXPECT_EQ(b.data_type, static_cast<int32_t>(DataType::BOOL));
  const uint8_t *v = b.AsBool();
  EXPECT_EQ(v[0], 1);
  EXPECT_EQ(v[1], 0);
  EXPECT_EQ(v[2], 1); // Any non-zero maps to 1.
}

TEST(SimpleTensorFactories, FromBoolWithAllocator) {
  SimpleRawBufferAllocator alloc(2);
  Tensor b = Tensor::FromBool("b", {2}, {0, 7}, &alloc);
  EXPECT_TRUE(b.has_allocation());
  EXPECT_EQ(b.AsBool()[0], 0);
  EXPECT_EQ(b.AsBool()[1], 1);
}

TEST(SimpleTensorFactories, FromStrings) {
  Tensor s = Tensor::FromStrings("s", {2}, {"foo", "bar"});
  EXPECT_EQ(s.data_type, static_cast<int32_t>(DataType::STRING));
  ASSERT_EQ(s.AsStrings().size(), 2u);
  EXPECT_EQ(s.AsStrings()[0], "foo");
  EXPECT_EQ(s.AsStrings()[1], "bar");
}

TEST(SimpleTensorFactories, ShapeMismatchThrows) {
  EXPECT_THROW(Tensor::FromFloat("x", {3}, {1.0f, 2.0f}), std::invalid_argument);
  EXPECT_THROW(Tensor::FromBool("b", {3}, {1, 0}), std::invalid_argument);
  EXPECT_THROW(Tensor::FromStrings("s", {3}, {"a", "b"}), std::invalid_argument);
}

TEST(SimpleTensorFactories, WithAllocator) {
  SimpleRawBufferAllocator alloc(2);
  Tensor t = Tensor::FromFloat("x", {2}, {1.0f, 2.0f}, &alloc);
  EXPECT_TRUE(t.has_allocation());
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(t.AsFloat()[1], 2.0f);
}

// ---------------------------------------------------------------------------
// element_count / element_size
// ---------------------------------------------------------------------------

TEST(SimpleTensorMeta, ElementCountAndSize) {
  Tensor t = Tensor::FromInt64("x", {2, 3}, {1, 2, 3, 4, 5, 6});
  EXPECT_EQ(t.element_count(), 6);
  EXPECT_EQ(t.element_size(), sizeof(int64_t));
}

// ---------------------------------------------------------------------------
// Typed accessors mismatch throws
// ---------------------------------------------------------------------------

TEST(SimpleTensorAccessors, WrongTypeThrows) {
  Tensor t = Tensor::FromFloat("x", {1}, {1.0f});
  EXPECT_THROW(t.AsInt64(), std::invalid_argument);
  const Tensor &ct = t;
  EXPECT_THROW(ct.AsInt64(), std::invalid_argument);
}

TEST(SimpleTensorAccessors, AsBoolWrongTypeThrows) {
  Tensor t = Tensor::FromFloat("x", {1}, {1.0f});
  EXPECT_THROW(t.AsBool(), std::invalid_argument);
  const Tensor &ct = t;
  EXPECT_THROW(ct.AsBool(), std::invalid_argument);
}

TEST(SimpleTensorAccessors, AsStringsWrongTypeThrows) {
  Tensor t = Tensor::FromFloat("x", {1}, {1.0f});
  EXPECT_THROW(t.AsStrings(), std::invalid_argument);
  const Tensor &ct = t;
  EXPECT_THROW(ct.AsStrings(), std::invalid_argument);
}

TEST(SimpleTensorAccessors, MutableAsFloatWrites) {
  Tensor t = Tensor::FromFloat("x", {2}, {1.0f, 2.0f});
  t.AsFloat()[0] = 9.0f;
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 9.0f);
}

// ---------------------------------------------------------------------------
// Copy / move semantics
// ---------------------------------------------------------------------------

TEST(SimpleTensorCopyMove, InlineCopyIsIndependent) {
  Tensor a = Tensor::FromFloat("a", {2}, {1.0f, 2.0f});
  Tensor b = a; // copy ctor
  b.AsFloat()[0] = 42.0f;
  EXPECT_FLOAT_EQ(a.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(b.AsFloat()[0], 42.0f);
}

TEST(SimpleTensorCopyMove, CopyAssign) {
  Tensor a = Tensor::FromInt32("a", {2}, {1, 2});
  Tensor b = Tensor::FromInt32("b", {1}, {9});
  b = a; // copy assignment
  ASSERT_EQ(b.element_count(), 2);
  EXPECT_EQ(b.AsInt32()[0], 1);
  EXPECT_EQ(b.AsInt32()[1], 2);
}

TEST(SimpleTensorCopyMove, SelfCopyAssign) {
  Tensor a = Tensor::FromInt32("a", {2}, {1, 2});
  const Tensor &ref = a;
  a = ref; // self assignment
  EXPECT_EQ(a.AsInt32()[0], 1);
  EXPECT_EQ(a.AsInt32()[1], 2);
}

TEST(SimpleTensorCopyMove, AllocatorBackedCopyIsDeep) {
  SimpleRawBufferAllocator alloc(4);
  Tensor a = Tensor::FromFloat("a", {2}, {1.0f, 2.0f}, &alloc);
  Tensor b = a; // deep copy from same allocator
  ASSERT_TRUE(b.has_allocation());
  EXPECT_NE(a.allocation(), b.allocation());
  b.AsFloat()[0] = 7.0f;
  EXPECT_FLOAT_EQ(a.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(b.AsFloat()[0], 7.0f);
}

TEST(SimpleTensorCopyMove, AllocatorBackedCopyAssignIsDeep) {
  SimpleRawBufferAllocator alloc(4);
  Tensor a = Tensor::FromFloat("a", {2}, {1.0f, 2.0f}, &alloc);
  Tensor b = Tensor::FromFloat("b", {1}, {5.0f});
  b = a;
  ASSERT_TRUE(b.has_allocation());
  EXPECT_NE(a.allocation(), b.allocation());
  EXPECT_FLOAT_EQ(b.AsFloat()[1], 2.0f);
}

TEST(SimpleTensorCopyMove, MoveCtorTransfersAllocation) {
  SimpleRawBufferAllocator alloc(2);
  Tensor a = Tensor::FromFloat("a", {2}, {1.0f, 2.0f}, &alloc);
  Tensor b = std::move(a);
  EXPECT_TRUE(b.has_allocation());
  EXPECT_FALSE(a.has_allocation());
  EXPECT_FLOAT_EQ(b.AsFloat()[0], 1.0f);
}

TEST(SimpleTensorCopyMove, MoveAssignTransfersAllocation) {
  SimpleRawBufferAllocator alloc(2);
  Tensor a = Tensor::FromFloat("a", {2}, {1.0f, 2.0f}, &alloc);
  Tensor b = Tensor::FromFloat("b", {1}, {5.0f});
  b = std::move(a);
  EXPECT_TRUE(b.has_allocation());
  EXPECT_FALSE(a.has_allocation());
  EXPECT_FLOAT_EQ(b.AsFloat()[1], 2.0f);
}

TEST(SimpleTensorCopyMove, SelfMoveAssign) {
  Tensor a = Tensor::FromInt32("a", {1}, {5});
  Tensor &ref = a;
  a = std::move(ref); // self move assignment is a no-op
  EXPECT_EQ(a.AsInt32()[0], 5);
}

TEST(SimpleTensorCopyMove, RawByteBufferMoveKeepsStorage) {
  RawByteBuffer bytes(2 * sizeof(float));
  uint8_t *original = bytes.data();
  Tensor tensor = Tensor::FromRawBytes("moved", DataType::FLOAT, {2}, std::move(bytes));
  EXPECT_EQ(tensor.data.data(), original);
}

TEST(AllocationHandle, MoveTransfersAndReturnsAllocationOnce) {
  SimpleRawBufferAllocator allocator(2);
  auto *buffer = allocator.Allocate(16);
  ASSERT_NE(buffer, nullptr);
  const auto *data = buffer->data();

  {
    AllocationHandle first(&allocator, buffer);
    EXPECT_EQ(first.buffer()->data(), data);
    EXPECT_EQ(first.owner(), &allocator);
    EXPECT_EQ(first.logical_size(), 16u);
    EXPECT_GE(first.retained_capacity(), first.logical_size());

    AllocationHandle second(std::move(first));
    EXPECT_FALSE(first);
    EXPECT_TRUE(second);
    EXPECT_EQ(allocator.allocated_count(), 1u);

    AllocationHandle third;
    third = std::move(second);
    EXPECT_FALSE(second);
    EXPECT_TRUE(third);
    EXPECT_EQ(allocator.allocated_count(), 1u);
  }

  EXPECT_EQ(allocator.allocated_count(), 0u);
}

TEST(AllocationHandle, TensorReleaseOutlivesTensorAndReturnsOnLastOwner) {
  SimpleRawBufferAllocator allocator(2);
  AllocationHandle external;
  {
    Tensor tensor = Tensor::FromFloat("x", {2}, {1.0f, 2.0f}, &allocator);
    external = tensor.ReleaseAllocation();
    EXPECT_FALSE(tensor.has_allocation());
    EXPECT_TRUE(external);
    EXPECT_EQ(allocator.allocated_count(), 1u);
  }

  EXPECT_EQ(allocator.allocated_count(), 1u);
  external.Reset();
  EXPECT_EQ(allocator.allocated_count(), 0u);
  external.Reset();
  EXPECT_EQ(allocator.allocated_count(), 0u);
}

// ---------------------------------------------------------------------------
// Borrow / BorrowStrings / ToOwned / is_borrowed
// ---------------------------------------------------------------------------

TEST(SimpleTensorBorrow, BorrowBytes) {
  const float vals[] = {1.0f, 2.0f, 3.0f};
  Tensor t = Tensor::Borrow("b", static_cast<int32_t>(DataType::FLOAT), {3},
                            reinterpret_cast<const uint8_t *>(vals), sizeof(vals));
  EXPECT_TRUE(t.is_borrowed());
  EXPECT_EQ(t.size_bytes(), sizeof(vals));
  EXPECT_FLOAT_EQ(t.AsFloat()[2], 3.0f);
}

TEST(SimpleTensorBorrow, RetainsBorrowedOwnerAcrossProtoRelease) {
  TensorProto proto;
  proto.set_name("mapped");
  proto.set_data_type(TensorProto::DataType::FLOAT);
  proto.add_dims(1);
  auto storage = std::make_shared<std::vector<float>>(std::initializer_list<float>{4.0f});
  std::shared_ptr<void> owner = storage;
  proto.ref_raw_data().assign_borrowed(reinterpret_cast<const uint8_t *>(storage->data()),
                                       sizeof(float), owner);

  Tensor tensor = TensorFromProto(proto);
  const uint8_t *payload = tensor.bytes();
  proto.clear_raw_data();
  storage.reset();
  owner.reset();

  EXPECT_EQ(tensor.bytes(), payload);
  EXPECT_TRUE(tensor.borrowed_owner());
  EXPECT_FLOAT_EQ(tensor.AsFloat()[0], 4.0f);
}

TEST(SimpleTensorBorrow, BorrowViewReusesPayloadAndOwner) {
  auto storage = std::make_shared<std::vector<float>>(std::initializer_list<float>{2.0f, 3.0f});
  Tensor source = Tensor::Borrow("source", static_cast<int32_t>(DataType::FLOAT), {2},
                                 reinterpret_cast<const uint8_t *>(storage->data()),
                                 2 * sizeof(float), storage);

  Tensor view = source.BorrowView();

  EXPECT_EQ(view.bytes(), source.bytes());
  EXPECT_EQ(view.borrowed_owner(), source.borrowed_owner());
  EXPECT_EQ(view.name, source.name);
  EXPECT_EQ(view.shape, source.shape);
}

TEST(SimpleTensorBorrow, ToOwnedDetachesBytes) {
  const float vals[] = {4.0f, 5.0f};
  Tensor borrowed = Tensor::Borrow("b", static_cast<int32_t>(DataType::FLOAT), {2},
                                   reinterpret_cast<const uint8_t *>(vals), sizeof(vals));
  Tensor owned = borrowed.ToOwned();
  EXPECT_FALSE(owned.is_borrowed());
  EXPECT_FLOAT_EQ(owned.AsFloat()[0], 4.0f);
  EXPECT_FLOAT_EQ(owned.AsFloat()[1], 5.0f);
}

TEST(SimpleTensorBorrow, BorrowStringsAndToOwned) {
  std::vector<std::string> strings{"a", "b"};
  Tensor borrowed = Tensor::BorrowStrings("s", {2}, strings);
  EXPECT_TRUE(borrowed.is_borrowed());
  const Tensor &cborrowed = borrowed;
  ASSERT_EQ(cborrowed.AsStrings().size(), 2u);
  EXPECT_EQ(cborrowed.AsStrings()[1], "b");
  Tensor owned = borrowed.ToOwned();
  EXPECT_FALSE(owned.is_borrowed());
  EXPECT_EQ(owned.AsStrings()[0], "a");
}

TEST(SimpleTensorBorrow, BorrowedStringsMutableAsStringsThrows) {
  std::vector<std::string> strings{"a"};
  Tensor borrowed = Tensor::BorrowStrings("s", {1}, strings);
  EXPECT_THROW(borrowed.AsStrings(), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// MakeOutputTensor
// ---------------------------------------------------------------------------

TEST(SimpleTensorMakeOutput, WithoutAllocator) {
  Tensor t =
      MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), {2}, 2 * sizeof(float), nullptr);
  EXPECT_FALSE(t.has_allocation());
  EXPECT_EQ(t.size_bytes(), 2 * sizeof(float));
  // The buffer is left uninitialised; callers fully overwrite it. Verify it is
  // writable and reads back the written values.
  t.AsFloat()[0] = 1.5f;
  t.AsFloat()[1] = -2.5f;
  EXPECT_FLOAT_EQ(t.AsFloat()[0], 1.5f);
  EXPECT_FLOAT_EQ(t.AsFloat()[1], -2.5f);
}

TEST(SimpleTensorMakeOutput, WithAllocator) {
  SimpleRawBufferAllocator alloc(2);
  Tensor t =
      MakeOutputTensor(static_cast<int32_t>(DataType::FLOAT), {2}, 2 * sizeof(float), &alloc);
  EXPECT_TRUE(t.has_allocation());
  EXPECT_EQ(t.size_bytes(), 2 * sizeof(float));
}

// ---------------------------------------------------------------------------
// FillValueInfo
// ---------------------------------------------------------------------------

TEST(SimpleTensorFillValueInfo, PopulatesTypeAndShape) {
  Tensor t = Tensor::FromInt64("t", {2, 3}, {1, 2, 3, 4, 5, 6});
  ValueInfoProto vi;
  FillValueInfo(t, vi);
  EXPECT_EQ(vi.name(), "t");
  ASSERT_TRUE(vi.type().has_tensor_type());
  EXPECT_EQ(vi.type().tensor_type().elem_type(), static_cast<int32_t>(DataType::INT64));
  ASSERT_TRUE(vi.type().tensor_type().has_shape());
  const auto &shape = vi.type().tensor_type().shape();
  ASSERT_EQ(shape.dim().size(), 2);
  EXPECT_EQ(shape.dim()[0].dim_value(), 2);
  EXPECT_EQ(shape.dim()[1].dim_value(), 3);
}

// ---------------------------------------------------------------------------
// TensorFromProto — additional dtype paths
// ---------------------------------------------------------------------------

TEST(SimpleTensorFromProto, DoubleTypedField) {
  TensorProto tp;
  tp.set_name("d");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::DOUBLE);
  tp.add_double_data(1.5);
  tp.add_double_data(2.5);
  Tensor t = TensorFromProto(tp);
  EXPECT_DOUBLE_EQ(t.AsDouble()[0], 1.5);
  EXPECT_DOUBLE_EQ(t.AsDouble()[1], 2.5);
}

TEST(SimpleTensorFromProto, Uint64TypedField) {
  TensorProto tp;
  tp.set_name("u64");
  tp.ref_dims().push_back(1);
  tp.set_data_type(TensorProto::DataType::UINT64);
  tp.add_uint64_data(123456789ull);
  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.AsUint64()[0], 123456789ull);
}

TEST(SimpleTensorFromProto, Uint32TypedField) {
  TensorProto tp;
  tp.set_name("u32");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::UINT32);
  tp.add_uint64_data(7u);
  tp.add_uint64_data(4000000000u);
  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.AsUint32()[0], 7u);
  EXPECT_EQ(t.AsUint32()[1], 4000000000u);
}

TEST(SimpleTensorFromProto, Int16FromInt32Field) {
  TensorProto tp;
  tp.set_name("i16");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::INT16);
  tp.add_int32_data(-3);
  tp.add_int32_data(5);
  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.AsInt16()[0], -3);
  EXPECT_EQ(t.AsInt16()[1], 5);
}

TEST(SimpleTensorFromProto, Int8FromInt32Field) {
  TensorProto tp;
  tp.set_name("i8");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::INT8);
  tp.add_int32_data(-1);
  tp.add_int32_data(2);
  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.AsInt8()[0], -1);
  EXPECT_EQ(t.AsInt8()[1], 2);
}

TEST(SimpleTensorFromProto, SubByteInt4FromInt32Field) {
  TensorProto tp;
  tp.set_name("i4");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::INT4);
  // Two 4-bit elements packed into one byte.
  tp.add_int32_data(0x21);
  Tensor t = TensorFromProto(tp);
  EXPECT_EQ(t.size_bytes(), 1u);
  EXPECT_EQ(t.bytes()[0], 0x21);
}

TEST(SimpleTensorFromProto, StringField) {
  TensorProto tp;
  tp.set_name("s");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::STRING);
  tp.add_string_data("hello");
  tp.add_string_data("world");
  Tensor t = TensorFromProto(tp);
  ASSERT_EQ(t.AsStrings().size(), 2u);
  EXPECT_EQ(t.AsStrings()[0], "hello");
  EXPECT_EQ(t.AsStrings()[1], "world");
}

TEST(SimpleTensorFromProto, RawDataIsBorrowed) {
  TensorProto tp;
  tp.set_name("r");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::INT32);
  const int32_t vals[] = {11, 22};
  const auto *raw_ptr = reinterpret_cast<const uint8_t *>(vals);
  tp.ref_raw_data() = std::vector<uint8_t>(raw_ptr, raw_ptr + sizeof(vals));
  Tensor t = TensorFromProto(tp);
  EXPECT_TRUE(t.is_borrowed());
  EXPECT_EQ(t.AsInt32()[0], 11);
  EXPECT_EQ(t.AsInt32()[1], 22);
}

TEST(SimpleTensorFromProto, WithAllocator) {
  SimpleRawBufferAllocator alloc(2);
  TensorProto tp;
  tp.set_name("f");
  tp.ref_dims().push_back(2);
  tp.set_data_type(TensorProto::DataType::FLOAT);
  tp.add_float_data(1.0f);
  tp.add_float_data(2.0f);
  Tensor t = TensorFromProto(tp, &alloc);
  EXPECT_TRUE(t.has_allocation());
  EXPECT_FLOAT_EQ(t.AsFloat()[1], 2.0f);
}

} // namespace Test
