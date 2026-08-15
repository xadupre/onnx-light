// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/runtime_context.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::ExecutionArena;
using core::runtime::RawBuffer;
using core::runtime::RawBufferAllocator;
using core::runtime::RuntimeContext;
using core::runtime::RuntimeContextOptions;
using core::runtime::SimpleRawBufferAllocator;

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
// Memory peak tests
// ---------------------------------------------------------------------------

TEST(SimpleRawBufferAllocator, InitialPeakIsZero) {
  SimpleRawBufferAllocator alloc(3);
  EXPECT_EQ(alloc.PeakAllocatedSize(), 0u);
}

TEST(SimpleRawBufferAllocator, PeakTracksMaximumTotalAllocatedSize) {
  SimpleRawBufferAllocator alloc(3);
  RawBuffer *a = alloc.Allocate(10);
  RawBuffer *b = alloc.Allocate(20);
  // Peak reaches 30 while both buffers are alive.
  EXPECT_EQ(alloc.TotalAllocatedSize(), 30u);
  EXPECT_EQ(alloc.PeakAllocatedSize(), 30u);

  // Freeing lowers the total but the peak is retained.
  alloc.Free(a);
  EXPECT_EQ(alloc.TotalAllocatedSize(), 20u);
  EXPECT_EQ(alloc.PeakAllocatedSize(), 30u);

  alloc.Free(b);
  EXPECT_EQ(alloc.TotalAllocatedSize(), 0u);
  EXPECT_EQ(alloc.PeakAllocatedSize(), 30u);
}

TEST(SimpleRawBufferAllocator, PeakGrowsAfterReallocation) {
  SimpleRawBufferAllocator alloc(2);
  RawBuffer *a = alloc.Allocate(16);
  EXPECT_EQ(alloc.PeakAllocatedSize(), 16u);
  alloc.Free(a);
  // A larger single allocation raises the peak above the previous value.
  RawBuffer *b = alloc.Allocate(40);
  EXPECT_EQ(alloc.PeakAllocatedSize(), 40u);
  alloc.Free(b);
}

TEST(SimpleRawBufferAllocator, ResetPeakSetsPeakToCurrentTotal) {
  SimpleRawBufferAllocator alloc(3);
  RawBuffer *a = alloc.Allocate(10);
  alloc.Allocate(20);
  EXPECT_EQ(alloc.PeakAllocatedSize(), 30u);

  alloc.Free(a);
  EXPECT_EQ(alloc.TotalAllocatedSize(), 20u);
  // Reset lowers the peak to the current total.
  alloc.ResetPeak();
  EXPECT_EQ(alloc.PeakAllocatedSize(), 20u);

  // Subsequent allocations grow the peak again from the reset baseline.
  alloc.Allocate(5);
  EXPECT_EQ(alloc.PeakAllocatedSize(), 25u);
}

TEST(SimpleRawBufferAllocator, ResetPeakOnEmptyAllocatorIsZero) {
  SimpleRawBufferAllocator alloc(2);
  alloc.Free(alloc.Allocate(64));
  EXPECT_EQ(alloc.PeakAllocatedSize(), 64u);
  alloc.ResetPeak();
  EXPECT_EQ(alloc.PeakAllocatedSize(), 0u);
}

TEST(RawBufferAllocatorPolymorphism, PeakAccessibleViaBasePointer) {
  SimpleRawBufferAllocator concrete(3);
  RawBufferAllocator *alloc = &concrete;

  RawBuffer *buf = alloc->Allocate(64);
  EXPECT_EQ(alloc->PeakAllocatedSize(), 64u);
  alloc->Free(buf);
  EXPECT_EQ(alloc->PeakAllocatedSize(), 64u);
  alloc->ResetPeak();
  EXPECT_EQ(alloc->PeakAllocatedSize(), 0u);
}

TEST(ExecutionArena, RetainsAndReusesPayloadStorage) {
  ExecutionArena arena(1);
  RawBuffer *first = arena.Allocate(64);
  uint8_t *payload = first->data();
  const size_t retained_capacity = first->capacity();

  arena.Free(first);
  EXPECT_EQ(arena.allocated_count(), 0u);
  EXPECT_EQ(arena.retained_count(), 1u);
  EXPECT_EQ(arena.RetainedSize(), retained_capacity);

  RawBuffer *second = arena.Allocate(32);
  EXPECT_EQ(second, first);
  EXPECT_EQ(second->data(), payload);
  EXPECT_EQ(second->size(), 32u);
  EXPECT_EQ(arena.RetainedSize(), 0u);
}

TEST(ExecutionArena, ChoosesSmallestSufficientCapacity) {
  ExecutionArena arena(2);
  RawBuffer *small = arena.Allocate(64);
  RawBuffer *large = arena.Allocate(128);
  uint8_t *small_payload = small->data();
  uint8_t *large_payload = large->data();
  arena.Free(small);
  arena.Free(large);

  RawBuffer *selected = arena.Allocate(48);
  EXPECT_EQ(selected->data(), small_payload);
  EXPECT_NE(selected->data(), large_payload);
}

TEST(ExecutionArena, GrowsAFreeSlotWhenRetainedBuffersAreTooSmall) {
  ExecutionArena arena(1);
  RawBuffer *first = arena.Allocate(16);
  arena.Free(first);

  RawBuffer *grown = arena.Allocate(256);
  EXPECT_EQ(grown, first);
  EXPECT_EQ(grown->size(), 256u);
  EXPECT_GE(grown->capacity(), 256u);
  EXPECT_EQ(arena.retained_count(), 0u);
}

TEST(ExecutionArena, TracksLivePeakAndEnforcesSlotLimit) {
  ExecutionArena arena(2);
  RawBuffer *first = arena.Allocate(10);
  RawBuffer *second = arena.Allocate(20);
  EXPECT_EQ(arena.TotalAllocatedSize(), 30u);
  EXPECT_EQ(arena.PeakAllocatedSize(), 30u);
  EXPECT_EQ(arena.allocated_count(), 2u);
  EXPECT_THROW(arena.Allocate(1), std::bad_alloc);

  arena.Free(first);
  EXPECT_EQ(arena.TotalAllocatedSize(), 20u);
  EXPECT_EQ(arena.PeakAllocatedSize(), 30u);
  arena.ResetPeak();
  EXPECT_EQ(arena.PeakAllocatedSize(), 20u);
  arena.Free(second);
}

TEST(ExecutionArena, RejectsUnknownAndAlreadyFreedBuffers) {
  ExecutionArena arena(1);
  RawBuffer unknown;
  EXPECT_THROW(arena.Free(&unknown), std::invalid_argument);

  RawBuffer *buffer = arena.Allocate(8);
  arena.Free(buffer);
  EXPECT_THROW(arena.Free(buffer), std::invalid_argument);
}

TEST(ExecutionArena, WorksThroughAllocatorInterfaceAndRuntimeContext) {
  ExecutionArena arena(2);
  RawBufferAllocator *allocator = &arena;
  RuntimeContext ctx(RuntimeContextOptions{.allocator = allocator});

  ctx.Set("x", core::runtime::Tensor::FromInt32("", {2}, {1, 2}));
  EXPECT_EQ(ctx.allocator(), allocator);
  EXPECT_EQ(arena.allocated_count(), 1u);
  EXPECT_TRUE(ctx.Remove("x"));
  EXPECT_EQ(arena.allocated_count(), 0u);
  EXPECT_EQ(arena.retained_count(), 1u);
  EXPECT_GE(arena.RetainedSize(), 2u * sizeof(int32_t));
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
  RuntimeContext ctx(RuntimeContextOptions{.allocator = &alloc});
  EXPECT_EQ(ctx.allocator(), &alloc);
}

TEST(RuntimeContextAllocator, ConstContextExposesAllocator) {
  SimpleRawBufferAllocator alloc(2);
  RuntimeContext ctx(RuntimeContextOptions{.allocator = &alloc});
  const RuntimeContext &cref = ctx;
  EXPECT_EQ(cref.allocator(), &alloc);
}

TEST(RuntimeContextAllocator, DefaultConstructorLeavesAllocatorUnset) {
  RuntimeContext ctx;
  EXPECT_EQ(ctx.allocator(), nullptr);
}

TEST(RuntimeContextAllocator, SetStoresAllocatorBackedTensorData) {
  SimpleRawBufferAllocator alloc(2);
  RuntimeContext ctx(RuntimeContextOptions{.allocator = &alloc});

  ctx.Set("x", core::runtime::Tensor::FromInt32("", {2}, {1, 2}));
  const auto &stored = ctx.Get("x");
  EXPECT_TRUE(stored.has_allocation());
  EXPECT_EQ(stored.data.size(), 0u);
  EXPECT_EQ(stored.size_bytes(), 2u * sizeof(int32_t));
  EXPECT_EQ(alloc.allocated_count(), 1u);
}

TEST(RuntimeContextAllocator, PutReplacesAndReleasesPreviousAllocation) {
  SimpleRawBufferAllocator alloc(2);
  RuntimeContext ctx(RuntimeContextOptions{.allocator = &alloc});

  ctx.Put("x", core::runtime::Tensor::FromInt32("", {2}, {1, 2}));
  const uint8_t *first = ctx.Get("x").bytes();
  ctx.Put("x", core::runtime::Tensor::FromInt32("", {2}, {3, 4}));
  const uint8_t *second = ctx.Get("x").bytes();

  EXPECT_NE(first, second);
  EXPECT_EQ(alloc.allocated_count(), 1u);
}

TEST(RuntimeContextAllocator, RemoveReleasesAllocatorBackedTensorData) {
  SimpleRawBufferAllocator alloc(1);
  RuntimeContext ctx(RuntimeContextOptions{.allocator = &alloc});

  ctx.Set("x", core::runtime::Tensor::FromInt32("", {1}, {7}));
  EXPECT_EQ(alloc.allocated_count(), 1u);
  EXPECT_TRUE(ctx.Remove("x"));
  EXPECT_EQ(alloc.allocated_count(), 0u);
}

TEST(RuntimeContextAllocator, DestroyingContextReleasesAllocatorBackedTensorData) {
  SimpleRawBufferAllocator alloc(1);
  {
    RuntimeContext ctx(RuntimeContextOptions{.allocator = &alloc});
    ctx.Set("x", core::runtime::Tensor::FromInt32("", {1}, {7}));
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
