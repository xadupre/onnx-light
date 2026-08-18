// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/runtime_context.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::AllocationHandle;
using core::runtime::ExecutionArena;
using core::runtime::IOArena;
using core::runtime::IOLease;
using core::runtime::RawBuffer;
using core::runtime::RawBufferAllocator;
using core::runtime::RuntimeContext;
using core::runtime::RuntimeContextOptions;
using core::runtime::SimpleRawBufferAllocator;
using core::runtime::Tensor;

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

TEST(ExecutionArena, TrimReleasesRetainedStorageAndReusesSlots) {
  ExecutionArena arena(2);
  RawBuffer *first = arena.Allocate(64);
  RawBuffer *second = arena.Allocate(128);
  const size_t retained_capacity = first->capacity() + second->capacity();
  arena.Free(first);
  arena.Free(second);
  EXPECT_EQ(arena.retained_count(), 2u);
  EXPECT_EQ(arena.RetainedSize(), retained_capacity);

  EXPECT_EQ(arena.Trim(), retained_capacity);
  EXPECT_EQ(arena.retained_count(), 0u);
  EXPECT_EQ(arena.RetainedSize(), 0u);

  // A second trim releases nothing and stays a no-op.
  EXPECT_EQ(arena.Trim(), 0u);

  // Trimmed slots remain usable for later allocations.
  RawBuffer *reused = arena.Allocate(16);
  EXPECT_EQ(reused->size(), 16u);
  EXPECT_EQ(arena.allocated_count(), 1u);
}

TEST(ExecutionArena, TrimLeavesLiveBuffersUntouched) {
  ExecutionArena arena(2);
  RawBuffer *live = arena.Allocate(32);
  RawBuffer *freed = arena.Allocate(48);
  const size_t retained_capacity = freed->capacity();
  arena.Free(freed);
  EXPECT_EQ(arena.RetainedSize(), retained_capacity);

  EXPECT_EQ(arena.Trim(), retained_capacity);
  EXPECT_EQ(arena.retained_count(), 0u);
  EXPECT_EQ(arena.allocated_count(), 1u);
  EXPECT_EQ(arena.TotalAllocatedSize(), 32u);
  EXPECT_EQ(live->size(), 32u);
  arena.Free(live);
}

TEST(ExecutionArena, DefaultRetentionCapIsUnbounded) {
  ExecutionArena arena(1);
  EXPECT_EQ(arena.retention_cap(), std::numeric_limits<size_t>::max());
}

TEST(ExecutionArena, RetentionCapEvictsLeastRecentlyFreedBuffer) {
  ExecutionArena arena(2);
  RawBuffer *first = arena.Allocate(64);
  RawBuffer *second = arena.Allocate(64);
  const size_t cap_second = second->capacity();

  arena.Free(first); // freed first, so least recently freed
  arena.Free(second);
  EXPECT_EQ(arena.retained_count(), 2u);

  // Lowering the cap keeps only the most-recently-freed buffer.
  arena.SetRetentionCap(cap_second);
  EXPECT_EQ(arena.retention_cap(), cap_second);
  EXPECT_EQ(arena.retained_count(), 1u);
  EXPECT_EQ(arena.RetainedSize(), cap_second);

  // The retained buffer is the most-recently-freed one; reusing its size returns
  // the same slot, confirming the least-recently-freed buffer was evicted.
  RawBuffer *reused = arena.Allocate(64);
  EXPECT_EQ(reused, second);
  EXPECT_EQ(arena.retained_count(), 0u);
  arena.Free(reused);
}

TEST(ExecutionArena, RetentionCapFromConstructorNeverEvictsLiveBuffers) {
  ExecutionArena arena(2, /*retention_cap=*/0);
  EXPECT_EQ(arena.retention_cap(), 0u);
  RawBuffer *live = arena.Allocate(64);
  RawBuffer *temp = arena.Allocate(32);

  // Freeing releases the storage immediately because nothing may be retained.
  arena.Free(temp);
  EXPECT_EQ(arena.retained_count(), 0u);
  EXPECT_EQ(arena.RetainedSize(), 0u);

  // The live buffer is untouched by the cap.
  EXPECT_EQ(arena.allocated_count(), 1u);
  EXPECT_EQ(arena.TotalAllocatedSize(), 64u);
  EXPECT_EQ(live->size(), 64u);
  arena.Free(live);
  EXPECT_EQ(arena.retained_count(), 0u);
}

// ---------------------------------------------------------------------------
// IOArena tests
// ---------------------------------------------------------------------------

TEST(IOArena, RetainsAndReusesPayloadStorage) {
  auto arena = IOArena::Create(1);
  RawBuffer *first = arena->Allocate(64);
  uint8_t *payload = first->data();
  const size_t retained_capacity = first->capacity();

  arena->Free(first);
  EXPECT_EQ(arena->allocated_count(), 0u);
  EXPECT_EQ(arena->retained_count(), 1u);
  EXPECT_EQ(arena->RetainedSize(), retained_capacity);

  RawBuffer *second = arena->Allocate(32);
  EXPECT_EQ(second, first);
  EXPECT_EQ(second->data(), payload);
  EXPECT_EQ(second->size(), 32u);
  EXPECT_EQ(arena->RetainedSize(), 0u);
}

TEST(IOArena, ChoosesSmallestSufficientCapacity) {
  auto arena = IOArena::Create(2);
  RawBuffer *small = arena->Allocate(64);
  RawBuffer *large = arena->Allocate(128);
  uint8_t *small_payload = small->data();
  uint8_t *large_payload = large->data();
  arena->Free(small);
  arena->Free(large);

  RawBuffer *selected = arena->Allocate(48);
  EXPECT_EQ(selected->data(), small_payload);
  EXPECT_NE(selected->data(), large_payload);
}

TEST(IOArena, TracksLivePeakAndEnforcesSlotLimit) {
  auto arena = IOArena::Create(2);
  RawBuffer *first = arena->Allocate(10);
  RawBuffer *second = arena->Allocate(20);
  EXPECT_EQ(arena->TotalAllocatedSize(), 30u);
  EXPECT_EQ(arena->PeakAllocatedSize(), 30u);
  EXPECT_EQ(arena->allocated_count(), 2u);
  EXPECT_THROW(arena->Allocate(1), std::bad_alloc);

  arena->Free(first);
  EXPECT_EQ(arena->TotalAllocatedSize(), 20u);
  EXPECT_EQ(arena->PeakAllocatedSize(), 30u);
  arena->ResetPeak();
  EXPECT_EQ(arena->PeakAllocatedSize(), 20u);
  arena->Free(second);
}

TEST(IOArena, RejectsUnknownAndAlreadyFreedBuffers) {
  auto arena = IOArena::Create(1);
  RawBuffer unknown;
  EXPECT_THROW(arena->Free(&unknown), std::invalid_argument);

  RawBuffer *buffer = arena->Allocate(8);
  arena->Free(buffer);
  EXPECT_THROW(arena->Free(buffer), std::invalid_argument);
}

TEST(IOArena, ExportPinsBufferAndKeepsItLive) {
  auto arena = IOArena::Create(1);
  RawBuffer *buffer = arena->Allocate(64);
  uint8_t *payload = buffer->data();

  IOLease lease = arena->Export(buffer);
  EXPECT_TRUE(static_cast<bool>(lease));
  EXPECT_EQ(lease.buffer(), buffer);
  EXPECT_EQ(lease.buffer()->data(), payload);
  EXPECT_EQ(lease.logical_size(), 64u);

  // The exported buffer is still live but no longer owned directly by the arena.
  EXPECT_EQ(arena->allocated_count(), 0u);
  EXPECT_EQ(arena->leased_count(), 1u);
  EXPECT_EQ(arena->TotalAllocatedSize(), 64u);
  EXPECT_EQ(arena->retained_count(), 0u);

  // A leased buffer cannot be freed through the allocator interface.
  EXPECT_THROW(arena->Free(buffer), std::invalid_argument);
  // Nor exported twice.
  EXPECT_THROW(arena->Export(buffer), std::invalid_argument);
}

TEST(IOArena, ReleasingLeaseReturnsStorageForReuse) {
  auto arena = IOArena::Create(1);
  RawBuffer *buffer = arena->Allocate(64);
  uint8_t *payload = buffer->data();
  const size_t retained_capacity = buffer->capacity();

  {
    IOLease lease = arena->Export(buffer);
    EXPECT_EQ(arena->leased_count(), 1u);
  }

  // Destroying the lease returns the buffer to the retained free list.
  EXPECT_EQ(arena->leased_count(), 0u);
  EXPECT_EQ(arena->TotalAllocatedSize(), 0u);
  EXPECT_EQ(arena->retained_count(), 1u);
  EXPECT_EQ(arena->RetainedSize(), retained_capacity);

  RawBuffer *reused = arena->Allocate(32);
  EXPECT_EQ(reused->data(), payload);
}

TEST(IOArena, ResetReleasesLeaseExactlyOnce) {
  auto arena = IOArena::Create(1);
  RawBuffer *buffer = arena->Allocate(16);
  IOLease lease = arena->Export(buffer);

  lease.Reset();
  EXPECT_FALSE(static_cast<bool>(lease));
  EXPECT_EQ(lease.buffer(), nullptr);
  EXPECT_EQ(arena->leased_count(), 0u);
  EXPECT_EQ(arena->retained_count(), 1u);

  // A second reset is a no-op and does not double-return the buffer.
  lease.Reset();
  EXPECT_EQ(arena->retained_count(), 1u);
}

TEST(IOArena, MovedLeaseTransfersOwnership) {
  auto arena = IOArena::Create(1);
  RawBuffer *buffer = arena->Allocate(16);
  IOLease lease = arena->Export(buffer);

  IOLease moved = std::move(lease);
  EXPECT_FALSE(static_cast<bool>(lease));
  EXPECT_TRUE(static_cast<bool>(moved));
  EXPECT_EQ(moved.buffer(), buffer);
  EXPECT_EQ(arena->leased_count(), 1u);

  moved.Reset();
  EXPECT_EQ(arena->leased_count(), 0u);
}

TEST(IOArena, LeaseKeepsArenaAliveAfterSharedPointerReset) {
  auto arena = IOArena::Create(1);
  RawBuffer *buffer = arena->Allocate(16);
  buffer->data()[0] = 42;
  IOLease lease = arena->Export(buffer);

  // Drop the local strong reference; the lease must keep the arena alive so the
  // exported buffer remains valid (mirrors a NumPy capsule outliving the runtime).
  arena.reset();
  ASSERT_TRUE(static_cast<bool>(lease));
  EXPECT_EQ(lease.buffer()->data()[0], 42);

  // Releasing the lease drops the final reference without dangling.
  lease.Reset();
  EXPECT_FALSE(static_cast<bool>(lease));
}

TEST(IOArena, WorksThroughAllocatorInterface) {
  auto arena = IOArena::Create(2);
  RawBufferAllocator *allocator = arena.get();

  RawBuffer *buf = allocator->Allocate(64);
  EXPECT_EQ(allocator->TotalAllocatedSize(), 64u);
  EXPECT_EQ(allocator->PeakAllocatedSize(), 64u);
  allocator->Free(buf);
  EXPECT_EQ(allocator->TotalAllocatedSize(), 0u);
  EXPECT_EQ(allocator->PeakAllocatedSize(), 64u);
}

TEST(IOArena, TrimReleasesRetainedStorageAndReusesSlots) {
  auto arena = IOArena::Create(2);
  RawBuffer *first = arena->Allocate(64);
  RawBuffer *second = arena->Allocate(128);
  const size_t retained_capacity = first->capacity() + second->capacity();
  arena->Free(first);
  arena->Free(second);
  EXPECT_EQ(arena->retained_count(), 2u);
  EXPECT_EQ(arena->RetainedSize(), retained_capacity);

  EXPECT_EQ(arena->Trim(), retained_capacity);
  EXPECT_EQ(arena->retained_count(), 0u);
  EXPECT_EQ(arena->RetainedSize(), 0u);

  // A second trim releases nothing and stays a no-op.
  EXPECT_EQ(arena->Trim(), 0u);

  // Trimmed slots remain usable for later allocations.
  RawBuffer *reused = arena->Allocate(16);
  EXPECT_EQ(reused->size(), 16u);
  EXPECT_EQ(arena->allocated_count(), 1u);
}

TEST(IOArena, TrimLeavesLiveAndLeasedBuffersUntouched) {
  auto arena = IOArena::Create(3);
  RawBuffer *live = arena->Allocate(32);
  RawBuffer *leased = arena->Allocate(24);
  RawBuffer *freed = arena->Allocate(48);
  const size_t retained_capacity = freed->capacity();
  arena->Free(freed);
  IOLease lease = arena->Export(leased);
  EXPECT_EQ(arena->RetainedSize(), retained_capacity);
  EXPECT_EQ(arena->leased_count(), 1u);

  // Only the retained free buffer is released; the live and leased buffers stay.
  EXPECT_EQ(arena->Trim(), retained_capacity);
  EXPECT_EQ(arena->retained_count(), 0u);
  EXPECT_EQ(arena->allocated_count(), 1u);
  EXPECT_EQ(arena->leased_count(), 1u);
  EXPECT_EQ(live->size(), 32u);
  EXPECT_EQ(lease.buffer(), leased);

  lease.Reset();
  arena->Free(live);
}

TEST(IOArena, DefaultRetentionCapIsUnbounded) {
  auto arena = IOArena::Create(1);
  EXPECT_EQ(arena->retention_cap(), std::numeric_limits<size_t>::max());
}

TEST(IOArena, RetentionCapEvictsLeastRecentlyFreedBuffer) {
  auto arena = IOArena::Create(2);
  RawBuffer *first = arena->Allocate(64);
  RawBuffer *second = arena->Allocate(64);
  const size_t cap_second = second->capacity();

  arena->Free(first); // freed first, so least recently freed
  arena->Free(second);
  EXPECT_EQ(arena->retained_count(), 2u);

  // Lowering the cap keeps only the most-recently-freed buffer.
  arena->SetRetentionCap(cap_second);
  EXPECT_EQ(arena->retention_cap(), cap_second);
  EXPECT_EQ(arena->retained_count(), 1u);
  EXPECT_EQ(arena->RetainedSize(), cap_second);

  RawBuffer *reused = arena->Allocate(64);
  EXPECT_EQ(reused, second);
  EXPECT_EQ(arena->retained_count(), 0u);
  arena->Free(reused);
}

TEST(IOArena, RetentionCapNeverEvictsLiveOrLeasedBuffers) {
  auto arena = IOArena::Create(3, /*retention_cap=*/0);
  EXPECT_EQ(arena->retention_cap(), 0u);
  RawBuffer *live = arena->Allocate(64);
  RawBuffer *leased = arena->Allocate(48);
  IOLease lease = arena->Export(leased);
  RawBuffer *temp = arena->Allocate(32);

  // Freeing releases the storage immediately because nothing may be retained.
  arena->Free(temp);
  EXPECT_EQ(arena->retained_count(), 0u);
  EXPECT_EQ(arena->RetainedSize(), 0u);

  // Neither the live nor the leased buffer is evicted by the cap.
  EXPECT_EQ(arena->allocated_count(), 1u);
  EXPECT_EQ(arena->leased_count(), 1u);
  EXPECT_EQ(lease.buffer(), leased);
  EXPECT_EQ(live->size(), 64u);

  // Releasing the lease returns the buffer, which the zero cap evicts at once.
  lease.Reset();
  EXPECT_EQ(arena->leased_count(), 0u);
  EXPECT_EQ(arena->retained_count(), 0u);
  arena->Free(live);
}

// ---------------------------------------------------------------------------
// IOArena::ExportHandle — self-owning exported allocation handle (step 6 of the
// buffer-reuse arena plan, see docs/next_steps/2026-08_buffer_reuse_arena.rst).
// ---------------------------------------------------------------------------

// ExportHandle turns a live buffer into an AllocationHandle that keeps the arena
// alive on its own and returns the buffer exactly once when destroyed.
TEST(IOArenaExportHandle, PinsBufferAndReturnsItOnce) {
  auto arena = IOArena::Create(1);
  RawBuffer *buffer = arena->Allocate(64);
  uint8_t *payload = buffer->data();
  const size_t retained_capacity = buffer->capacity();

  {
    AllocationHandle handle = arena->ExportHandle(buffer);
    EXPECT_TRUE(static_cast<bool>(handle));
    EXPECT_TRUE(handle.holds_lease());
    EXPECT_EQ(handle.buffer(), buffer);
    EXPECT_EQ(handle.owner(), arena.get());
    EXPECT_EQ(handle.logical_size(), 64u);

    // The buffer is leased (pinned), not owned directly by the arena.
    EXPECT_EQ(arena->allocated_count(), 0u);
    EXPECT_EQ(arena->leased_count(), 1u);
    EXPECT_EQ(arena->TotalAllocatedSize(), 64u);
  }

  // Destroying the handle returns the buffer to the retained free list once.
  EXPECT_EQ(arena->leased_count(), 0u);
  EXPECT_EQ(arena->TotalAllocatedSize(), 0u);
  EXPECT_EQ(arena->retained_count(), 1u);
  EXPECT_EQ(arena->RetainedSize(), retained_capacity);

  RawBuffer *reused = arena->Allocate(32);
  EXPECT_EQ(reused->data(), payload);
}

// A handle exported from the arena keeps the arena alive after every other
// strong reference is dropped, mirroring a NumPy capsule that outlives the
// RuntimeContext that produced its data.
TEST(IOArenaExportHandle, KeepsArenaAliveAfterSharedPointerReset) {
  auto arena = IOArena::Create(1);
  RawBuffer *buffer = arena->Allocate(16);
  buffer->data()[0] = 42;

  AllocationHandle handle = arena->ExportHandle(buffer);

  // Drop the local strong reference; the handle's lease must keep the arena
  // alive so the exported buffer remains valid.
  arena.reset();
  ASSERT_TRUE(static_cast<bool>(handle));
  EXPECT_EQ(handle.buffer()->data()[0], 42);

  // Releasing the handle drops the final reference without dangling.
  handle.Reset();
  EXPECT_FALSE(static_cast<bool>(handle));
  EXPECT_FALSE(handle.holds_lease());
}

// Moving a lease-backed handle transfers both the allocation and the arena
// reference, and the buffer is still returned exactly once.
TEST(IOArenaExportHandle, MovePreservesLeaseOwnership) {
  auto arena = IOArena::Create(1);
  RawBuffer *buffer = arena->Allocate(16);

  AllocationHandle first = arena->ExportHandle(buffer);
  AllocationHandle second(std::move(first));
  EXPECT_FALSE(static_cast<bool>(first));
  EXPECT_FALSE(first.holds_lease());
  EXPECT_TRUE(static_cast<bool>(second));
  EXPECT_TRUE(second.holds_lease());
  EXPECT_EQ(second.buffer(), buffer);
  EXPECT_EQ(arena->leased_count(), 1u);

  AllocationHandle third;
  third = std::move(second);
  EXPECT_FALSE(static_cast<bool>(second));
  EXPECT_TRUE(third.holds_lease());
  EXPECT_EQ(arena->leased_count(), 1u);

  third.Reset();
  EXPECT_EQ(arena->leased_count(), 0u);
  EXPECT_EQ(arena->retained_count(), 1u);
}

// ExportHandle(AllocationHandle&&) rejects an empty or already lease-backed
// handle instead of silently mis-exporting a null buffer.
TEST(IOArenaExportHandle, RejectsEmptyOrLeaseBackedHandle) {
  auto arena = IOArena::Create(1);

  AllocationHandle empty;
  EXPECT_THROW(arena->ExportHandle(std::move(empty)), std::invalid_argument);

  RawBuffer *buffer = arena->Allocate(16);
  AllocationHandle leased = arena->ExportHandle(buffer);
  ASSERT_TRUE(leased.holds_lease());
  EXPECT_THROW(arena->ExportHandle(std::move(leased)), std::invalid_argument);
  // The rejected handle keeps its lease intact.
  EXPECT_TRUE(leased.holds_lease());
  EXPECT_EQ(arena->leased_count(), 1u);
}

// A graph output allocated from an IOArena can have its allocation released
// from the tensor and exported as a self-owning handle. The handle then keeps
// the arena's storage alive even after the arena's other owner is destroyed,
// which is exactly what an exported NumPy output requires.
TEST(IOArenaExportHandle, ExportedTensorOutputOutlivesArenaOwner) {
  auto arena = IOArena::Create(1);
  Tensor tensor = Tensor::FromFloat("y", {2}, {1.0f, 2.0f}, arena.get());
  ASSERT_TRUE(tensor.has_allocation());

  // Transfer the allocation out of the tensor and re-export it as a self-owning
  // handle (the tensor no longer owns the buffer once released).
  AllocationHandle exported = arena->ExportHandle(tensor.ReleaseAllocation());
  EXPECT_TRUE(exported.holds_lease());
  EXPECT_FALSE(tensor.has_allocation());

  const float *values = reinterpret_cast<const float *>(exported.buffer()->data());
  EXPECT_FLOAT_EQ(values[0], 1.0f);
  EXPECT_FLOAT_EQ(values[1], 2.0f);

  // Drop the arena's strong reference; the exported handle keeps it alive.
  arena.reset();
  ASSERT_TRUE(static_cast<bool>(exported));
  EXPECT_FLOAT_EQ(reinterpret_cast<const float *>(exported.buffer()->data())[1], 2.0f);

  exported.Reset();
  EXPECT_FALSE(static_cast<bool>(exported));
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

TEST(RuntimeContextAllocator, DefaultIOAllocatorIsNull) {
  RuntimeContext ctx;
  EXPECT_EQ(ctx.io_allocator(), nullptr);
  EXPECT_EQ(ctx.execution_allocator(), nullptr);
}

TEST(RuntimeContextAllocator, SetAndGetIOAllocator) {
  SimpleRawBufferAllocator execution_alloc(4);
  SimpleRawBufferAllocator io_alloc(4);
  RuntimeContext ctx(
      RuntimeContextOptions{.allocator = &execution_alloc, .io_allocator = &io_alloc});
  // Until SetActiveAllocator switches it, allocator() reports the execution
  // allocator, matching pre-existing single-allocator behaviour.
  EXPECT_EQ(ctx.allocator(), &execution_alloc);
  EXPECT_EQ(ctx.execution_allocator(), &execution_alloc);
  EXPECT_EQ(ctx.io_allocator(), &io_alloc);
}

TEST(RuntimeContextAllocator, SetActiveAllocatorSwitchesAllocatorAndKernelContext) {
  SimpleRawBufferAllocator execution_alloc(4);
  SimpleRawBufferAllocator io_alloc(4);
  RuntimeContext ctx(
      RuntimeContextOptions{.allocator = &execution_alloc, .io_allocator = &io_alloc});

  RawBufferAllocator *previous = ctx.SetActiveAllocator(&io_alloc);
  EXPECT_EQ(previous, &execution_alloc);
  EXPECT_EQ(ctx.allocator(), &io_alloc);
  EXPECT_EQ(ctx.kernel_ctx().allocator, &io_alloc);
  // execution_allocator() is unaffected by SetActiveAllocator.
  EXPECT_EQ(ctx.execution_allocator(), &execution_alloc);

  previous = ctx.SetActiveAllocator(&execution_alloc);
  EXPECT_EQ(previous, &io_alloc);
  EXPECT_EQ(ctx.allocator(), &execution_alloc);
  EXPECT_EQ(ctx.kernel_ctx().allocator, &execution_alloc);
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

TEST(RuntimeContextAllocator, SetPreservesBorrowedInputWithoutAllocation) {
  SimpleRawBufferAllocator alloc(1);
  RuntimeContext ctx(RuntimeContextOptions{.allocator = &alloc});
  const int32_t values[] = {1, 2};
  const auto *bytes = reinterpret_cast<const uint8_t *>(values);

  ctx.Set("x",
          Tensor::Borrow("", static_cast<int32_t>(core::runtime::DataType::INT32), {2}, bytes,
                         sizeof(values)),
          core::runtime::RuntimeEventKind::kInput);

  const Tensor &stored = ctx.Get("x");
  EXPECT_TRUE(stored.is_borrowed());
  EXPECT_EQ(stored.bytes(), bytes);
  EXPECT_EQ(alloc.allocated_count(), 0u);
}

TEST(RuntimeContextAllocator, PutMaterializesBorrowedIntermediate) {
  SimpleRawBufferAllocator alloc(1);
  RuntimeContext ctx(RuntimeContextOptions{.allocator = &alloc});
  const int32_t values[] = {3, 4};
  const auto *bytes = reinterpret_cast<const uint8_t *>(values);

  ctx.Put("x",
          Tensor::Borrow("", static_cast<int32_t>(core::runtime::DataType::INT32), {2}, bytes,
                         sizeof(values)),
          core::runtime::RuntimeEventKind::kIntermediate);

  const Tensor &stored = ctx.Get("x");
  EXPECT_FALSE(stored.is_borrowed());
  EXPECT_TRUE(stored.has_allocation());
  EXPECT_NE(stored.bytes(), bytes);
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
