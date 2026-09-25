// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/runtime_value.h"
#include <atomic>
#include <optional>
#include <span>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

class RuntimeContext;

/**
 * Retains a tensor value and, for internally allocated append buffers, spare capacity.
 *
 * Ordinary Tensor views carry lifetime ownership, never capacity or write permissions.
 * Importing such a view retains its payload without certifying it for append.
 * The owner serializes access, as PersistentValueState does; each prepared AppendLease
 * may be consumed only once, even when an append requires a copy or is declined.
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
   * Captures the prefix and its capacity for one append attempt. Only exclusive
   * certified storage grants in-place permission; otherwise the append allocates.
   */
  AppendLease PrepareAppend() const;

private:
  Tensor value_;
  size_t capacity_bytes_ = 0;
};

/**
 * Reserves append destinations using the contiguous tensor's storage policy.
 *
 * Captures the retained prefix for one append attempt. Further reservations are
 * rejected. Allocation, geometric growth and prefix relocation belong here;
 * producing the new elements belongs to the kernel.
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
   * Zero capacity declines the reservation but still consumes this lease.
   * Records allocation, prefix copies and reuse in the supplied runtime context
   * when events are enabled, including work before a failure.
   * The caller supplies the output allocator; no alternate allocator is used.
   */
  std::optional<AppendReservation> Reserve(const Shape &shape, size_t axis, size_t initial_capacity,
                                           RawBufferAllocator *allocator,
                                           RuntimeContext *context = nullptr);

private:
  friend class PersistentTensor;
  AppendLease(const PersistentTensor &tensor, bool available);
  enum class State : uint8_t { kReusable, kCopyRequired, kConsumed };
  PersistentTensor prefix_;
  std::atomic<State> state_;
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

  /** Returns a root tensor, or null for structured, sequence and encoded values. */
  const PersistentTensor *tensor() const noexcept { return tensor_ ? &*tensor_ : nullptr; }
  /** Returns ordinary views without exposing capacity or append permissions. */
  RuntimeValue BorrowView() const;

private:
  PersistentValue() = default;
  static PersistentValue FromRetained(RuntimeValue value, size_t depth = 0);
  RuntimeValue BorrowAtDepth(size_t depth) const;
  RuntimeValue::Kind kind_ = RuntimeValue::Kind::kStruct;
  std::optional<PersistentTensor> tensor_;
  std::unordered_map<std::string, PersistentValue> fields_;
  std::vector<PersistentValue> elements_;
  std::shared_ptr<const EncodedValueProto> encoded_;
  std::shared_ptr<const QuantizationParameterCatalogue> quantization_parameters_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
