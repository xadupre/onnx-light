// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_value.h"
#include <atomic>
#include <mutex>
#include <optional>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Measures reported kernel work on persistent-capable storage.
 *
 * Includes allocation/copy fallback and failed attempts, independently of the
 * consuming operator. These counters do not automatically track all runtime work.
 */
struct PersistentStorageStatistics {
  uint64_t allocations = 0;
  uint64_t allocated_bytes = 0;
  uint64_t prefix_copied_bytes = 0;
  uint64_t append_copied_bytes = 0;
  uint64_t reuse_count = 0;
};

/** Aggregates operator-independent storage work across an invocation's contexts. */
class ONNX_LIGHT_CORE_API PersistentStorageCounters {
public:
  /** Returns a consistent snapshot of all cumulative counters. */
  PersistentStorageStatistics Snapshot() const;
  /** Adds reported work atomically with respect to other updates and snapshots. */
  void Accumulate(const PersistentStorageStatistics &statistics);

private:
  mutable std::mutex mutex_;
  PersistentStorageStatistics values_;
};

/**
 * Retains a tensor value and, for internally allocated append buffers, spare capacity.
 *
 * Ordinary Tensor views carry lifetime ownership, never capacity or write permissions.
 * Importing such a view retains its payload without certifying it for append.
 * The owner serializes access, as FeedbackState does; competing consumers of an
 * acquired AppendLease may atomically claim that lease only once.
 */
class ONNX_LIGHT_CORE_API PersistentTensor {
public:
  class AppendLease;

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
  /** Acquires a one-use append lease before invocation-local views are created. */
  std::optional<AppendLease> AcquireAppendLease() const;

  /**
   * Copies a prefix and tail into fresh owned storage and retains its spare capacity.
   *
   * The caller supplies a byte-concatenable layout and routes the allocation through
   * the output allocator. Borrowed storage and unretainable allocations are rejected.
   */
  static PersistentTensor Concatenate(Tensor storage, const Tensor &prefix, const Tensor &tail,
                                      const Shape &shape);

private:
  Tensor value_;
  size_t capacity_bytes_ = 0;
};

/**
 * Carries a move-only, one-use permission to write beyond a retained tensor's valid prefix.
 *
 * A successful append produces a candidate; it does not publish a new state value.
 * An abandoned candidate may leave tail bytes written but never changes the old extent.
 */
class ONNX_LIGHT_CORE_API PersistentTensor::AppendLease {
public:
  AppendLease(AppendLease &&other) noexcept;
  AppendLease &operator=(AppendLease &&other) noexcept;
  AppendLease(const AppendLease &) = delete;
  AppendLease &operator=(const AppendLease &) = delete;

  /** Appends once if the original prefix still matches and the candidate fits. */
  std::optional<PersistentTensor> TryAppend(const Tensor &prefix, const Tensor &tail,
                                            const Shape &shape);

private:
  friend class PersistentTensor;
  explicit AppendLease(const PersistentTensor &tensor);
  PersistentTensor prefix_;
  std::atomic<bool> available_{true};
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
