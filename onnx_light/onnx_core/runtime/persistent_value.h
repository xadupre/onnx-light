// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_value.h"
#include <atomic>
#include <functional>
#include <optional>
#include <span>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Measures reported kernel work on persistent-capable storage.
 *
 * Includes allocation/copy fallback and failed attempts, independently of the
 * consuming operator. These counters do not automatically track all runtime work.
 */
struct ONNX_LIGHT_CORE_API PersistentStorageStatistics {
  uint64_t allocations = 0;
  uint64_t allocated_bytes = 0;
  uint64_t prefix_copied_bytes = 0;
  uint64_t append_copied_bytes = 0;
  uint64_t reuse_count = 0;
  /** Adds an event's reported work to an explicitly requested total. */
  PersistentStorageStatistics &operator+=(const PersistentStorageStatistics &statistics) noexcept;
};

/**
 * Retains a tensor value and, for internally allocated append buffers, spare capacity.
 *
 * Ordinary Tensor views carry lifetime ownership, never capacity or write permissions.
 * Importing such a view retains its payload without certifying it for append.
 * The owner serializes access, as FeedbackState does; competing consumers of an
 * prepared AppendLease may atomically claim in-place reuse only once.
 */
class ONNX_LIGHT_CORE_API PersistentTensor {
public:
  class AppendLease;
  class AppendReservation;

  explicit PersistentTensor(Tensor value);
  PersistentTensor(PersistentTensor &&) noexcept = default;
  PersistentTensor &operator=(PersistentTensor &&) noexcept = default;
  PersistentTensor(const PersistentTensor &) = delete;
  PersistentTensor &operator=(const PersistentTensor &) = delete;

  /** Returns the committed logical value, excluding spare capacity. */
  const Tensor &value() const noexcept { return value_; }
  /** Returns a lifetime-retaining ordinary view without append permissions. */
  Tensor BorrowView() const { return value_.BorrowView(); }
  /** Returns certified capacity in bytes, or zero for imported storage. */
  size_t capacity_bytes() const noexcept { return capacity_bytes_; }
  /** Returns whether a view has this exact owner, extent, type and shape. */
  bool Matches(const Tensor &view) const noexcept;
  /**
   * Prepares append reservations before invocation-local views are created.
   *
   * Captures the prefix and its capacity. Only exclusive certified storage grants
   * a one-use in-place permission; other reservations allocate independent storage.
   */
  AppendLease PrepareAppend() const;

private:
  Tensor value_;
  size_t capacity_bytes_ = 0;
};

/**
 * Reserves append destinations using the contiguous tensor's storage policy.
 *
 * Captures the retained prefix and may extend its logical extent in place once.
 * Further reservations allocate independently. Allocation, geometric growth and prefix
 * relocation belong here; producing the new elements belongs to the kernel.
 */
class ONNX_LIGHT_CORE_API PersistentTensor::AppendLease {
public:
  AppendLease(AppendLease &&other) noexcept;
  AppendLease &operator=(AppendLease &&other) noexcept;
  AppendLease(const AppendLease &) = delete;
  AppendLease &operator=(const AppendLease &) = delete;

  /** Returns whether the invocation input still matches the captured prefix. */
  bool Matches(const Tensor &view) const noexcept { return prefix_.Matches(view); }
  /**
   * Reserves an uninitialized tail, declining layouts that cannot append contiguously.
   *
   * Only the append axis may grow, and the product of preceding dimensions must
   * be one. Initial capacity is measured along that axis, not in bytes.
   * Reports allocation, prefix copies and reuse only when a callback is supplied,
   * including work before a failure. The callback is invoked synchronously.
   * The caller supplies the output allocator; no alternate allocator is used.
   */
  std::optional<AppendReservation>
  Reserve(const Shape &shape, size_t axis, size_t initial_capacity, RawBufferAllocator *allocator,
          const std::function<void(const PersistentStorageStatistics &)> &on_storage_event = {});

private:
  friend class PersistentTensor;
  AppendLease(const PersistentTensor &tensor, bool available);
  PersistentTensor prefix_;
  std::atomic<bool> available_;
};

/**
 * Exposes only the writable tail of an unpublished contiguous result.
 *
 * The kernel initializes the entire region directly, then seals the candidate.
 * Destruction without Commit abandons it without changing the retained prefix
 * or logical extent. Written tail bytes need not be restored on abandonment.
 */
class ONNX_LIGHT_CORE_API PersistentTensor::AppendReservation {
public:
  AppendReservation(AppendReservation &&) noexcept = default;
  AppendReservation &operator=(AppendReservation &&) noexcept = default;
  AppendReservation(const AppendReservation &) = delete;
  AppendReservation &operator=(const AppendReservation &) = delete;

  /** Returns the reserved tail, valid until this reservation is committed or destroyed. */
  std::span<uint8_t> writable_bytes();
  /**
   * Seals a fully initialized candidate without publishing the feedback state.
   *
   * Requires initialized_bytes to match the whole tail; the caller guarantees
   * that these bytes were written. No writable span may be used after this call.
   */
  PersistentTensor Commit(size_t initialized_bytes);

private:
  friend class AppendLease;
  AppendReservation(PersistentTensor candidate, size_t prefix_bytes)
      : candidate_(std::move(candidate)), prefix_bytes_(prefix_bytes) {}
  PersistentTensor candidate_;
  size_t prefix_bytes_;
};

/** Retains a whole feedback value with PersistentTensor at its numeric tensor leaves. */
class ONNX_LIGHT_CORE_API PersistentValue {
public:
  explicit PersistentValue(RuntimeValue value, const StructTypeCatalogue &catalogue = {});
  explicit PersistentValue(PersistentTensor tensor);
  PersistentValue(PersistentValue &&) noexcept = default;
  PersistentValue &operator=(PersistentValue &&) noexcept = default;
  PersistentValue(const PersistentValue &) = delete;
  PersistentValue &operator=(const PersistentValue &) = delete;

  /** Returns a root tensor, or null for structured and encoded values. */
  const PersistentTensor *tensor() const noexcept { return tensor_ ? &*tensor_ : nullptr; }
  /** Returns ordinary views without exposing capacity or append permissions. */
  RuntimeValue BorrowView() const;

private:
  PersistentValue() = default;
  static PersistentValue FromRetained(RuntimeValue value);
  RuntimeValue::Kind kind_ = RuntimeValue::Kind::kStruct;
  std::optional<PersistentTensor> tensor_;
  std::unordered_map<std::string, PersistentValue> fields_;
  std::shared_ptr<const EncodedValueProto> encoded_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
