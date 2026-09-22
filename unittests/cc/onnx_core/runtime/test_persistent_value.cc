// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/persistent_value.h"
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

PersistentTensor Cache() {
  PersistentTensor empty(Tensor::FromFloat("", {0}, {}));
  PersistentStorageCounters counters;
  auto reservation = empty.PrepareAppend().Reserve({2}, 0, 4, nullptr, counters);
  EXT_ENFORCE(reservation.has_value(), "Expected a contiguous append reservation.");
  auto *destination = reinterpret_cast<float *>(reservation->writable_bytes().data());
  destination[0] = 1;
  destination[1] = 2;
  return reservation->Commit(2 * sizeof(float));
}

void WriteTail(PersistentTensor::AppendReservation &reservation,
               std::initializer_list<float> values) {
  const auto destination = reservation.writable_bytes();
  ASSERT_EQ(destination.size(), values.size() * sizeof(float));
  std::copy(values.begin(), values.end(), reinterpret_cast<float *>(destination.data()));
}

} // namespace

TEST(PersistentStorageCounters, AccumulatesIndependentOfRuntimeAndOperator) {
  PersistentStorageCounters counters;
  const auto empty = counters.Snapshot();
  EXPECT_EQ(empty.allocations, 0u);
  EXPECT_EQ(empty.allocated_bytes, 0u);
  EXPECT_EQ(empty.prefix_copied_bytes, 0u);
  EXPECT_EQ(empty.append_copied_bytes, 0u);
  EXPECT_EQ(empty.reuse_count, 0u);
  counters.Accumulate({1, 64, 8, 4, 0});
  const auto first = counters.Snapshot();
  counters.Accumulate({0, 0, 0, 4, 1});
  const auto total = counters.Snapshot();
  EXPECT_EQ(total.allocations, 1u);
  EXPECT_EQ(total.allocated_bytes, 64u);
  EXPECT_EQ(total.prefix_copied_bytes, 8u);
  EXPECT_EQ(total.append_copied_bytes, 8u);
  EXPECT_EQ(total.reuse_count, 1u);
  EXPECT_EQ(first.append_copied_bytes, 4u);
  EXPECT_EQ(first.reuse_count, 0u);
}

TEST(PersistentStorageCounters, ConcurrentUpdatesAndSnapshotsStayConsistent) {
  PersistentStorageCounters counters;
  const auto update = [&] {
    for (int i = 0; i < 100; ++i) {
      counters.Accumulate({1, 64, 8, 4, 2});
      const auto snapshot = counters.Snapshot();
      EXPECT_EQ(snapshot.allocated_bytes, snapshot.allocations * 64);
      EXPECT_EQ(snapshot.prefix_copied_bytes, snapshot.allocations * 8);
      EXPECT_EQ(snapshot.append_copied_bytes, snapshot.allocations * 4);
      EXPECT_EQ(snapshot.reuse_count, snapshot.allocations * 2);
    }
  };
  auto first = std::async(std::launch::async, update);
  auto second = std::async(std::launch::async, update);
  first.get();
  second.get();
  EXPECT_EQ(counters.Snapshot().allocations, 200u);
}

TEST(PersistentTensor, RetainsWithoutCopyAndDoesNotCertifyImportedStorage) {
  Tensor tensor = Tensor::FromFloat("", {2}, {1, 2});
  const uint8_t *bytes = tensor.bytes();
  PersistentTensor retained(std::move(tensor));
  EXPECT_EQ(retained.value().bytes(), bytes);
  EXPECT_EQ(retained.capacity_bytes(), 0u);
  PersistentStorageCounters counters;
  auto reservation = retained.PrepareAppend().Reserve({3}, 0, 4, nullptr, counters);
  ASSERT_TRUE(reservation);
  EXPECT_NE(reservation->writable_bytes().data(), bytes + 2 * sizeof(float));
  WriteTail(*reservation, {3});
  const auto candidate = reservation->Commit(sizeof(float));
  EXPECT_FLOAT_EQ(candidate.value().AsFloat()[0], 1);
  EXPECT_EQ(counters.Snapshot().allocations, 1u);
  EXPECT_EQ(counters.Snapshot().reuse_count, 0u);
}

TEST(PersistentTensor, ExplicitLeaseReservesInPlaceOnceAndPreservesCommittedPrefix) {
  static_assert(!std::is_copy_constructible_v<PersistentTensor>);
  static_assert(!std::is_copy_constructible_v<PersistentTensor::AppendLease>);
  static_assert(!std::is_copy_constructible_v<PersistentTensor::AppendReservation>);
  static_assert(std::is_nothrow_move_constructible_v<PersistentTensor::AppendLease>);
  auto cache = Cache();
  EXPECT_EQ(cache.capacity_bytes(), 4 * sizeof(float));
  PersistentStorageCounters counters;
  auto lease = cache.PrepareAppend();
  auto moved = std::move(lease);
  EXPECT_THROW(lease.Reserve({3}, 0, 4, nullptr, counters), std::invalid_argument);
  auto reservation = moved.Reserve({3}, 0, 4, nullptr, counters);
  ASSERT_TRUE(reservation);
  WriteTail(*reservation, {3});
  auto candidate = reservation->Commit(sizeof(float));
  EXPECT_EQ(candidate.value().bytes(), cache.value().bytes());
  EXPECT_EQ(cache.value().shape, Shape{2});
  EXPECT_EQ(cache.value().size_bytes(), 2 * sizeof(float));
  EXPECT_EQ(candidate.value().shape, Shape{3});
  EXPECT_FLOAT_EQ(candidate.value().AsFloat()[0], 1);
  EXPECT_FLOAT_EQ(candidate.value().AsFloat()[1], 2);
  EXPECT_FLOAT_EQ(candidate.value().AsFloat()[2], 3);
  auto second = moved.Reserve({3}, 0, 4, nullptr, counters);
  ASSERT_TRUE(second);
  WriteTail(*second, {9});
  const auto other = second->Commit(sizeof(float));
  EXPECT_NE(other.value().bytes(), candidate.value().bytes());
  EXPECT_FLOAT_EQ(candidate.value().AsFloat()[2], 3);
  EXPECT_EQ(counters.Snapshot().reuse_count, 1u);
  EXPECT_EQ(counters.Snapshot().allocations, 1u);
}

TEST(PersistentTensor, ViewsBlockInPlaceReservationsAndCannotRestoreCapacity) {
  auto cache = Cache();
  PersistentStorageCounters counters;
  {
    Tensor view = cache.BorrowView();
    Tensor copy = view;
    EXPECT_TRUE(cache.Matches(copy));
    auto reservation = cache.PrepareAppend().Reserve({3}, 0, 4, nullptr, counters);
    ASSERT_TRUE(reservation);
    WriteTail(*reservation, {3});
    const auto candidate = reservation->Commit(sizeof(float));
    EXPECT_NE(candidate.value().bytes(), copy.bytes());
  }
  {
    auto reservation = cache.PrepareAppend().Reserve({3}, 0, 4, nullptr, counters);
    ASSERT_TRUE(reservation);
    WriteTail(*reservation, {3});
    EXPECT_EQ(reservation->Commit(sizeof(float)).value().bytes(), cache.value().bytes());
  }
  Tensor view = cache.BorrowView();
  cache = PersistentTensor(std::move(view));
  EXPECT_EQ(cache.capacity_bytes(), 0u);
  auto reservation = cache.PrepareAppend().Reserve({3}, 0, 4, nullptr, counters);
  ASSERT_TRUE(reservation);
  WriteTail(*reservation, {3});
  EXPECT_NE(reservation->Commit(sizeof(float)).value().bytes(), cache.value().bytes());
  EXPECT_EQ(counters.Snapshot().allocations, 2u);
  EXPECT_EQ(counters.Snapshot().reuse_count, 1u);
}

TEST(PersistentTensor, LeaseMatchesExactOwnerPointerShapeExtentAndType) {
  for (int mode = 0; mode < 6; ++mode) {
    SCOPED_TRACE(mode);
    auto cache = Cache();
    auto lease = cache.PrepareAppend();
    Tensor prefix = cache.BorrowView();
    if (mode == 0)
      prefix = prefix.ToOwned();
    if (mode == 1)
      prefix.shape = {1, 2};
    if (mode == 2)
      prefix = Tensor::Borrow("", DataType::FLOAT, {1}, prefix.bytes(), sizeof(float),
                              prefix.borrowed_owner());
    if (mode == 3)
      prefix = Tensor::Borrow("", DataType::FLOAT, {2}, prefix.bytes(), prefix.size_bytes(),
                              std::make_shared<int>(0));
    if (mode == 4)
      prefix.data_type = DataType::INT32;
    if (mode == 5)
      prefix = Tensor::Borrow("", DataType::FLOAT, {2}, prefix.bytes() + sizeof(float),
                              prefix.size_bytes(), prefix.borrowed_owner());
    EXPECT_FALSE(lease.Matches(prefix));
    EXPECT_TRUE(lease.Matches(cache.value()));
    EXPECT_FLOAT_EQ(cache.value().AsFloat()[1], 2);
  }
}

TEST(PersistentTensor, ConcurrentConsumersShareOneExplicitLease) {
  auto cache = Cache();
  auto lease = cache.PrepareAppend();
  PersistentStorageCounters counters;
  const auto append = [&] {
    auto reservation = lease.Reserve({3}, 0, 4, nullptr, counters);
    EXT_ENFORCE(reservation.has_value(), "Expected a contiguous append reservation.");
    WriteTail(*reservation, {3});
    return reservation->Commit(sizeof(float));
  };
  auto first = std::async(std::launch::async, append);
  auto second = std::async(std::launch::async, append);
  auto a = first.get();
  auto b = second.get();
  EXPECT_NE(a.value().bytes(), b.value().bytes());
  EXPECT_TRUE(a.value().bytes() == cache.value().bytes() ||
              b.value().bytes() == cache.value().bytes());
  EXPECT_FLOAT_EQ(a.value().AsFloat()[2], 3);
  EXPECT_FLOAT_EQ(b.value().AsFloat()[2], 3);
  EXPECT_EQ(counters.Snapshot().reuse_count, 1u);
  EXPECT_EQ(counters.Snapshot().allocations, 1u);
}

TEST(PersistentTensor, FailedAppendDoesNotPublishAndRetryRewritesTail) {
  auto cache = Cache();
  PersistentStorageCounters counters;
  {
    auto abandoned = cache.PrepareAppend().Reserve({3}, 0, 4, nullptr, counters);
    ASSERT_TRUE(abandoned);
    WriteTail(*abandoned, {9});
  }
  EXPECT_EQ(cache.value().shape, Shape{2});
  auto reservation = cache.PrepareAppend().Reserve({3}, 0, 4, nullptr, counters);
  ASSERT_TRUE(reservation);
  WriteTail(*reservation, {3});
  auto candidate = reservation->Commit(sizeof(float));
  EXPECT_FLOAT_EQ(candidate.value().AsFloat()[2], 3);
  EXPECT_EQ(counters.Snapshot().allocations, 0u);
  EXPECT_EQ(counters.Snapshot().reuse_count, 2u);
}

TEST(PersistentTensor, GrowthRelocatesOnlyThePrefixAndWritesDirectlyIntoReservedCapacity) {
  auto cache = Cache();
  PersistentStorageCounters counters;
  auto reservation = cache.PrepareAppend().Reserve({5}, 0, 4, nullptr, counters);
  ASSERT_TRUE(reservation);
  WriteTail(*reservation, {3, 4, 5});
  auto candidate = reservation->Commit(3 * sizeof(float));
  EXPECT_EQ(candidate.capacity_bytes(), 8 * sizeof(float));
  EXPECT_NE(candidate.value().bytes(), cache.value().bytes());
  EXPECT_FLOAT_EQ(candidate.value().AsFloat()[0], 1);
  EXPECT_FLOAT_EQ(candidate.value().AsFloat()[1], 2);
  EXPECT_FLOAT_EQ(candidate.value().AsFloat()[4], 5);
  auto next = candidate.PrepareAppend().Reserve({6}, 0, 4, nullptr, counters);
  ASSERT_TRUE(next);
  WriteTail(*next, {6});
  EXPECT_EQ(next->Commit(sizeof(float)).value().bytes(), candidate.value().bytes());
  const auto statistics = counters.Snapshot();
  EXPECT_EQ(statistics.allocations, 1u);
  EXPECT_EQ(statistics.allocated_bytes, 8 * sizeof(float));
  EXPECT_EQ(statistics.prefix_copied_bytes, 2 * sizeof(float));
  EXPECT_EQ(statistics.append_copied_bytes, 0u);
  EXPECT_EQ(statistics.reuse_count, 1u);
}

TEST(PersistentTensor, ReservationMovesAndRequiresACompleteSingleCommit) {
  auto cache = Cache();
  PersistentStorageCounters counters;
  auto reservation = cache.PrepareAppend().Reserve({3}, 0, 4, nullptr, counters);
  ASSERT_TRUE(reservation);
  auto moved = std::move(*reservation);
  EXPECT_THROW(reservation->writable_bytes(), std::invalid_argument);
  WriteTail(moved, {3});
  EXPECT_THROW(moved.Commit(0), std::invalid_argument);
  EXPECT_FLOAT_EQ(moved.Commit(sizeof(float)).value().AsFloat()[2], 3);
  EXPECT_THROW(moved.writable_bytes(), std::invalid_argument);
  EXPECT_THROW(moved.Commit(sizeof(float)), std::invalid_argument);
}

TEST(PersistentTensor, SupportsOtherElementTypesAxesAndEmptyTails) {
  PersistentTensor prefix(Tensor::FromInt64("", {1, 2}, {10, 20}));
  PersistentStorageCounters counters;
  auto reservation = prefix.PrepareAppend().Reserve({1, 3}, 1, 4, nullptr, counters);
  ASSERT_TRUE(reservation);
  ASSERT_EQ(reservation->writable_bytes().size(), sizeof(int64_t));
  *reinterpret_cast<int64_t *>(reservation->writable_bytes().data()) = 30;
  auto candidate = reservation->Commit(sizeof(int64_t));
  EXPECT_EQ(candidate.capacity_bytes(), 4 * sizeof(int64_t));
  EXPECT_EQ(candidate.value().AsInt64()[0], 10);
  EXPECT_EQ(candidate.value().AsInt64()[1], 20);
  EXPECT_EQ(candidate.value().AsInt64()[2], 30);
  auto empty = candidate.PrepareAppend().Reserve({1, 3}, 1, 4, nullptr, counters);
  ASSERT_TRUE(empty);
  EXPECT_TRUE(empty->writable_bytes().empty());
  EXPECT_EQ(empty->Commit(0).value().bytes(), candidate.value().bytes());
  EXPECT_EQ(counters.Snapshot().prefix_copied_bytes, 2 * sizeof(int64_t));
  EXPECT_EQ(counters.Snapshot().append_copied_bytes, 0u);
  EXPECT_EQ(counters.Snapshot().reuse_count, 1u);
}

TEST(PersistentTensor, RejectsInvalidReservationsAndUnretainableAllocations) {
  auto cache = Cache();
  auto lease = cache.PrepareAppend();
  PersistentStorageCounters counters;
  EXPECT_THROW(lease.Reserve({1}, 0, 4, nullptr, counters), std::invalid_argument);
  EXPECT_THROW(lease.Reserve({3}, 1, 4, nullptr, counters), std::invalid_argument);
  EXPECT_THROW(lease.Reserve({1, 3}, 0, 4, nullptr, counters), std::invalid_argument);
  EXPECT_THROW(lease.Reserve({3}, 0, 0, nullptr, counters), std::invalid_argument);
  EXPECT_THROW(lease.Reserve({-1}, 0, 4, nullptr, counters), std::invalid_argument);
  EXPECT_THROW(lease.Reserve({std::numeric_limits<int64_t>::max()}, 0, 4, nullptr, counters),
               std::invalid_argument);
  EXPECT_THROW(lease.Reserve({5}, 0, std::numeric_limits<size_t>::max(), nullptr, counters),
               std::invalid_argument);
  EXPECT_EQ(counters.Snapshot().allocations, 0u);
  SimpleRawBufferAllocator arena(4);
  EXPECT_THROW(lease.Reserve({5}, 0, 4, &arena, counters), std::invalid_argument);
  EXPECT_EQ(counters.Snapshot().allocations, 1u);
  EXPECT_EQ(counters.Snapshot().prefix_copied_bytes, 0u);
  Tensor invalid = Tensor::FromFloat("", {2}, {1, 2});
  invalid.shape[0] = 3;
  PersistentTensor malformed(std::move(invalid));
  EXPECT_THROW(malformed.PrepareAppend().Reserve({4}, 0, 4, nullptr, counters),
               std::invalid_argument);
}

TEST(PersistentTensor, DeclinesLayoutsRequiringRepacking) {
  PersistentStorageCounters counters;
  PersistentTensor slices(Tensor::FromFloat("", {2, 1}, {1, 2}));
  EXPECT_FALSE(slices.PrepareAppend().Reserve({2, 2}, 1, 4, nullptr, counters));
  PersistentTensor zero_width(Tensor::FromFloat("", {1, 0}, {}));
  EXPECT_FALSE(zero_width.PrepareAppend().Reserve({2, 0}, 0, 4, nullptr, counters));
  PersistentTensor packed(Tensor("", DataType::UINT4, {2}, std::vector<uint8_t>{0x21}));
  EXPECT_FALSE(packed.PrepareAppend().Reserve({4}, 0, 4, nullptr, counters));
  PersistentTensor packed_six(Tensor("", DataType::FLOAT6E2M3, {2}, std::vector<uint8_t>{0, 0}));
  EXPECT_FALSE(packed_six.PrepareAppend().Reserve({4}, 0, 4, nullptr, counters));
  EXPECT_EQ(counters.Snapshot().allocations, 0u);
}

TEST(PersistentTensor, AllocatorExhaustionPreservesTheCommittedPrefix) {
  auto arena = IOArena::Create(1);
  PersistentStorageCounters counters;
  PersistentTensor empty(Tensor::FromFloat("", {0}, {}));
  auto reservation = empty.PrepareAppend().Reserve({2}, 0, 2, arena.get(), counters);
  ASSERT_TRUE(reservation);
  WriteTail(*reservation, {1, 2});
  auto cache = reservation->Commit(2 * sizeof(float));
  const auto *bytes = cache.value().bytes();
  EXPECT_THROW(cache.PrepareAppend().Reserve({3}, 0, 2, arena.get(), counters), std::bad_alloc);
  EXPECT_EQ(cache.value().bytes(), bytes);
  EXPECT_EQ(cache.value().shape, Shape{2});
  EXPECT_FLOAT_EQ(cache.value().AsFloat()[1], 2);
  EXPECT_EQ(counters.Snapshot().allocations, 1u);
  EXPECT_EQ(arena->leased_count(), 1u);
}

TEST(PersistentValue, NestedTensorLeavesKeepOwnersAndLiteralFieldNames) {
  Tensor tensor = Tensor::FromFloat("", {1}, {7});
  const uint8_t *bytes = tensor.bytes();
  RuntimeValueMap leaves;
  leaves.emplace("key.part", RuntimeValue(std::move(tensor)));
  RuntimeValueMap fields;
  fields.emplace("cache", RuntimeValue(std::move(leaves)));
  PersistentValue value(RuntimeValue(std::move(fields)));
  EXPECT_EQ(value.tensor(), nullptr);
  RuntimeValue view = value.BorrowView();
  value = PersistentValue(RuntimeValue(Tensor::FromFloat("", {1}, {9})));
  EXPECT_EQ(view.fields.at("cache").fields.at("key.part").tensor.bytes(), bytes);
  EXPECT_FLOAT_EQ(view.fields.at("cache").fields.at("key.part").tensor.AsFloat()[0], 7);
  ASSERT_NE(value.tensor(), nullptr);
  EXPECT_FLOAT_EQ(value.tensor()->value().AsFloat()[0], 9);
}
