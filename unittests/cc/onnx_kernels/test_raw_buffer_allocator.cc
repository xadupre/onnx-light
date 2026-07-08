// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/raw_buffer_allocator.h"
#include "onnx_kernels/runtime_context.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_kernels::RawBuffer;
using onnx_kernels::RawBufferAllocator;
using onnx_kernels::RuntimeContext;
using onnx_kernels::SimpleRawBufferAllocator;

namespace Test {

// ---------------------------------------------------------------------------
// SimpleRawBufferAllocator tests
// ---------------------------------------------------------------------------

TEST(SimpleRawBufferAllocator, CapacityReturnsConstructedValue) {
  SimpleRawBufferAllocator alloc(4);
  EXPECT_EQ(alloc.capacity(), 4u);
}

TEST(SimpleRawBufferAllocator, InitialStateIsEmpty) {
  SimpleRawBufferAllocator alloc(3);
  EXPECT_EQ(alloc.allocated_count(), 0u);
  EXPECT_EQ(alloc.TotalAllocatedSize(), 0u);
}

TEST(SimpleRawBufferAllocator, AllocateReturnsDifferentPointers) {
  SimpleRawBufferAllocator alloc(2);
  RawBuffer *a = alloc.Allocate(8);
  RawBuffer *b = alloc.Allocate(16);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a, b);
}

TEST(SimpleRawBufferAllocator, AllocateSetsCorrectSize) {
  SimpleRawBufferAllocator alloc(2);
  RawBuffer *buf = alloc.Allocate(32);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(buf->size(), 32u);
}

TEST(SimpleRawBufferAllocator, AllocatedCountIncrements) {
  SimpleRawBufferAllocator alloc(3);
  alloc.Allocate(4);
  EXPECT_EQ(alloc.allocated_count(), 1u);
  alloc.Allocate(8);
  EXPECT_EQ(alloc.allocated_count(), 2u);
}

TEST(SimpleRawBufferAllocator, TotalAllocatedSizeSumsAllocatedBuffers) {
  SimpleRawBufferAllocator alloc(3);
  alloc.Allocate(10);
  alloc.Allocate(20);
  EXPECT_EQ(alloc.TotalAllocatedSize(), 30u);
}

TEST(SimpleRawBufferAllocator, FreeReleasesSlot) {
  SimpleRawBufferAllocator alloc(2);
  RawBuffer *buf = alloc.Allocate(16);
  EXPECT_EQ(alloc.allocated_count(), 1u);
  alloc.Free(buf);
  EXPECT_EQ(alloc.allocated_count(), 0u);
  EXPECT_EQ(alloc.TotalAllocatedSize(), 0u);
}

TEST(SimpleRawBufferAllocator, FreedSlotCanBeReallocated) {
  SimpleRawBufferAllocator alloc(1);
  RawBuffer *first = alloc.Allocate(8);
  alloc.Free(first);
  // After freeing, the same slot should be available again.
  RawBuffer *second = alloc.Allocate(16);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second->size(), 16u);
  EXPECT_EQ(alloc.allocated_count(), 1u);
}

TEST(SimpleRawBufferAllocator, AllocateThrowsWhenFull) {
  SimpleRawBufferAllocator alloc(2);
  alloc.Allocate(4);
  alloc.Allocate(4);
  EXPECT_THROW(alloc.Allocate(4), std::bad_alloc);
}

TEST(SimpleRawBufferAllocator, FreeThrowsForUnknownPointer) {
  SimpleRawBufferAllocator alloc(2);
  RawBuffer external;
  EXPECT_THROW(alloc.Free(&external), std::invalid_argument);
}

TEST(SimpleRawBufferAllocator, ZeroCapacity) {
  SimpleRawBufferAllocator alloc(0);
  EXPECT_EQ(alloc.capacity(), 0u);
  EXPECT_THROW(alloc.Allocate(1), std::bad_alloc);
}

// ---------------------------------------------------------------------------
// RuntimeContext allocator accessor tests
// ---------------------------------------------------------------------------

TEST(RuntimeContextAllocator, DefaultAllocatorIsNull) {
  RuntimeContext ctx;
  EXPECT_EQ(ctx.allocator(), nullptr);
}

TEST(RuntimeContextAllocator, SetAndGetAllocator) {
  SimpleRawBufferAllocator alloc(4);
  RuntimeContext ctx;
  ctx.set_allocator(&alloc);
  EXPECT_EQ(ctx.allocator(), &alloc);
}

TEST(RuntimeContextAllocator, ConstContextExposesAllocator) {
  SimpleRawBufferAllocator alloc(2);
  RuntimeContext ctx;
  ctx.set_allocator(&alloc);
  const RuntimeContext &cref = ctx;
  EXPECT_EQ(cref.allocator(), &alloc);
}

TEST(RuntimeContextAllocator, AllocatorCanBeReset) {
  SimpleRawBufferAllocator alloc(2);
  RuntimeContext ctx;
  ctx.set_allocator(&alloc);
  ctx.set_allocator(nullptr);
  EXPECT_EQ(ctx.allocator(), nullptr);
}

TEST(RuntimeContextAllocator, SetStoresAllocatorBackedTensorData) {
  SimpleRawBufferAllocator alloc(2);
  RuntimeContext ctx;
  ctx.set_allocator(&alloc);

  ctx.Set("x", onnx_kernels::Tensor::FromInt32("", {2}, {1, 2}));
  const auto &stored = ctx.Get("x");
  EXPECT_TRUE(stored.has_allocation());
  EXPECT_EQ(stored.data.size(), 0u);
  EXPECT_EQ(stored.size_bytes(), 2u * sizeof(int32_t));
  EXPECT_EQ(alloc.allocated_count(), 1u);
}

TEST(RuntimeContextAllocator, PutReplacesAndReleasesPreviousAllocation) {
  SimpleRawBufferAllocator alloc(2);
  RuntimeContext ctx;
  ctx.set_allocator(&alloc);

  ctx.Put("x", onnx_kernels::Tensor::FromInt32("", {2}, {1, 2}));
  const uint8_t *first = ctx.Get("x").bytes();
  ctx.Put("x", onnx_kernels::Tensor::FromInt32("", {2}, {3, 4}));
  const uint8_t *second = ctx.Get("x").bytes();

  EXPECT_NE(first, second);
  EXPECT_EQ(alloc.allocated_count(), 1u);
}

TEST(RuntimeContextAllocator, RemoveReleasesAllocatorBackedTensorData) {
  SimpleRawBufferAllocator alloc(1);
  RuntimeContext ctx;
  ctx.set_allocator(&alloc);

  ctx.Set("x", onnx_kernels::Tensor::FromInt32("", {1}, {7}));
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_TRUE(ctx.Remove("x"));
  EXPECT_EQ(alloc.allocated_count(), 0u);
}

TEST(RuntimeContextAllocator, DestroyingContextReleasesAllocatorBackedTensorData) {
  SimpleRawBufferAllocator alloc(1);
  {
    RuntimeContext ctx;
    ctx.set_allocator(&alloc);
    ctx.Set("x", onnx_kernels::Tensor::FromInt32("", {1}, {7}));
    EXPECT_EQ(alloc.allocated_count(), 1u);
  }
  EXPECT_EQ(alloc.allocated_count(), 0u);
}

// ---------------------------------------------------------------------------
// Polymorphism: use RawBufferAllocator* interface
// ---------------------------------------------------------------------------

TEST(RawBufferAllocatorPolymorphism, AllocateAndFreeViaBasePointer) {
  SimpleRawBufferAllocator concrete(3);
  RawBufferAllocator *alloc = &concrete;

  RawBuffer *buf = alloc->Allocate(64);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(buf->size(), 64u);
  EXPECT_EQ(alloc->TotalAllocatedSize(), 64u);

  alloc->Free(buf);
  EXPECT_EQ(alloc->TotalAllocatedSize(), 0u);
}

} // namespace Test
