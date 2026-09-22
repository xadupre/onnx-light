// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/persistent_value.h"
#include <future>
#include <gtest/gtest.h>
#include <type_traits>

using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

PersistentTensor Cache() {
  Tensor prefix = Tensor::FromFloat("", {1}, {1});
  Tensor tail = Tensor::FromFloat("", {1}, {2});
  return PersistentTensor::Concatenate(
      MakeOutputTensor(DataType::FLOAT, {2}, 4 * sizeof(float), nullptr), prefix, tail, {2});
}

} // namespace

TEST(PersistentTensor, RetainsWithoutCopyAndDoesNotCertifyImportedStorage) {
  Tensor tensor = Tensor::FromFloat("", {2}, {1, 2});
  const uint8_t *bytes = tensor.bytes();
  PersistentTensor retained(std::move(tensor));
  EXPECT_EQ(retained.value().bytes(), bytes);
  EXPECT_EQ(retained.capacity_bytes(), 0u);
  EXPECT_FALSE(retained.AcquireAppendLease());

  auto cache = Cache();
  Tensor exported = cache.BorrowView();
  EXPECT_FALSE(cache.AcquireAppendLease());
  PersistentTensor imported(std::move(exported));
  EXPECT_EQ(imported.value().bytes(), cache.value().bytes());
  EXPECT_EQ(imported.capacity_bytes(), 0u);
  EXPECT_FALSE(imported.AcquireAppendLease());
}

TEST(PersistentTensor, ExplicitLeaseMovesOnceAndPreservesCommittedPrefix) {
  static_assert(!std::is_copy_constructible_v<PersistentTensor>);
  static_assert(!std::is_copy_constructible_v<PersistentTensor::AppendLease>);
  static_assert(std::is_nothrow_move_constructible_v<PersistentTensor::AppendLease>);
  auto cache = Cache();
  EXPECT_EQ(cache.capacity_bytes(), 4 * sizeof(float));
  auto lease = cache.AcquireAppendLease();
  ASSERT_TRUE(lease);
  EXPECT_FALSE(cache.AcquireAppendLease());
  auto moved = std::move(*lease);
  Tensor tail = Tensor::FromFloat("", {1}, {3});
  EXPECT_FALSE(lease->TryAppend(cache.value(), tail, {3}));
  auto candidate = moved.TryAppend(cache.value(), tail, {3});
  ASSERT_TRUE(candidate);
  EXPECT_FALSE(moved.TryAppend(cache.value(), tail, {3}));
  EXPECT_EQ(candidate->value().bytes(), cache.value().bytes());
  EXPECT_EQ(cache.value().shape, Shape{2});
  EXPECT_EQ(cache.value().size_bytes(), 2 * sizeof(float));
  EXPECT_EQ(candidate->value().shape, Shape{3});
  EXPECT_FLOAT_EQ(candidate->value().AsFloat()[0], 1);
  EXPECT_FLOAT_EQ(candidate->value().AsFloat()[1], 2);
  EXPECT_FLOAT_EQ(candidate->value().AsFloat()[2], 3);
}

TEST(PersistentTensor, ViewsBlockLeasesAndCannotRestoreCapacity) {
  auto cache = Cache();
  {
    Tensor view = cache.BorrowView();
    Tensor copy = view;
    EXPECT_TRUE(cache.Matches(copy));
    EXPECT_FALSE(cache.AcquireAppendLease());
  }
  EXPECT_TRUE(cache.AcquireAppendLease());
  Tensor view = cache.BorrowView();
  cache = PersistentTensor(std::move(view));
  EXPECT_FALSE(cache.AcquireAppendLease());
  EXPECT_EQ(cache.capacity_bytes(), 0u);
}

TEST(PersistentTensor, LeaseRejectsChangedOwnerShapeExtentAndCapacity) {
  for (int mode = 0; mode < 7; ++mode) {
    SCOPED_TRACE(mode);
    auto cache = Cache();
    auto lease = cache.AcquireAppendLease();
    ASSERT_TRUE(lease);
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
    Tensor tail = Tensor::FromFloat("", {3}, {3, 4, 5});
    EXPECT_FALSE(lease->TryAppend(prefix, tail, {5}));
    EXPECT_FALSE(lease->TryAppend(cache.value(), Tensor::FromFloat("", {1}, {3}), {3}));
    EXPECT_FLOAT_EQ(cache.value().AsFloat()[1], 2);
  }
}

TEST(PersistentTensor, ConcurrentConsumersShareOneExplicitLease) {
  auto cache = Cache();
  auto lease = cache.AcquireAppendLease();
  ASSERT_TRUE(lease);
  Tensor tail = Tensor::FromFloat("", {1}, {3});
  const auto append = [&] { return lease->TryAppend(cache.value(), tail, {3}); };
  auto first = std::async(std::launch::async, append);
  auto second = std::async(std::launch::async, append);
  auto a = first.get();
  auto b = second.get();
  EXPECT_NE(a.has_value(), b.has_value());
  EXPECT_FLOAT_EQ((a ? a : b)->value().AsFloat()[2], 3);
}

TEST(PersistentTensor, FailedAppendDoesNotPublishAndRetryRewritesTail) {
  auto cache = Cache();
  {
    auto lease = cache.AcquireAppendLease();
    ASSERT_TRUE(lease);
    auto abandoned = lease->TryAppend(cache.value(), Tensor::FromFloat("", {1}, {9}), {3});
    ASSERT_TRUE(abandoned);
  }
  EXPECT_EQ(cache.value().shape, Shape{2});
  auto lease = cache.AcquireAppendLease();
  ASSERT_TRUE(lease);
  auto candidate = lease->TryAppend(cache.value(), Tensor::FromFloat("", {1}, {3}), {3});
  ASSERT_TRUE(candidate);
  EXPECT_FLOAT_EQ(candidate->value().AsFloat()[2], 3);
}

TEST(PersistentTensor, RejectsBorrowedStorageInvalidExtentsAndUnretainableAllocations) {
  Tensor prefix = Tensor::FromFloat("", {1}, {1});
  Tensor tail = Tensor::FromFloat("", {1}, {2});
  Tensor storage = Tensor::FromFloat("", {4}, {0, 0, 0, 0});
  EXPECT_THROW(PersistentTensor::Concatenate(storage.BorrowView(), prefix, tail, {2}),
               std::invalid_argument);
  EXPECT_THROW(PersistentTensor::Concatenate(storage, prefix, tail, {3}), std::invalid_argument);
  EXPECT_THROW(PersistentTensor::Concatenate(prefix, prefix, tail, {2}), std::invalid_argument);
  SimpleRawBufferAllocator arena(4);
  EXPECT_THROW(
      PersistentTensor::Concatenate(
          MakeOutputTensor(DataType::FLOAT, {2}, 4 * sizeof(float), &arena), prefix, tail, {2}),
      std::invalid_argument);
  auto cache = Cache();
  auto lease = cache.AcquireAppendLease();
  ASSERT_TRUE(lease);
  EXPECT_THROW(lease->TryAppend(cache.value(), tail, {4}), std::invalid_argument);
  EXPECT_FALSE(lease->TryAppend(cache.value(), tail, {3}));
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
