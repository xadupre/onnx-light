// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/persistent_value.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstring>
#include <limits>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

PersistentTensor::PersistentTensor(Tensor value) : value_(std::move(value).RetainStorage()) {}

bool PersistentTensor::Matches(const Tensor &view) const noexcept {
  return view.bytes() == value_.bytes() && view.size_bytes() == value_.size_bytes() &&
         view.data_type == value_.data_type && view.shape == value_.shape &&
         !view.borrowed_owner().owner_before(value_.borrowed_owner()) &&
         !value_.borrowed_owner().owner_before(view.borrowed_owner());
}

PersistentTensor::AppendLease PersistentTensor::PrepareAppend() const {
  const bool available = capacity_bytes_ != 0 && value_.borrowed_owner().use_count() == 1;
  return AppendLease(*this, available);
}

PersistentTensor::AppendLease::AppendLease(const PersistentTensor &tensor, bool available)
    : prefix_(tensor.BorrowView()), state_(available ? State::kReusable : State::kCopyRequired) {
  prefix_.capacity_bytes_ = tensor.capacity_bytes_;
}

PersistentTensor::AppendLease::AppendLease(AppendLease &&other) noexcept
    : prefix_(std::move(other.prefix_)), state_(other.state_.exchange(State::kConsumed)) {}

PersistentTensor::AppendLease &
PersistentTensor::AppendLease::operator=(AppendLease &&other) noexcept {
  if (this != &other) {
    prefix_ = std::move(other.prefix_);
    state_ = other.state_.exchange(State::kConsumed);
  }
  return *this;
}

std::optional<PersistentTensor::AppendReservation>
PersistentTensor::AppendLease::Reserve(const Shape &shape, size_t axis, size_t initial_capacity,
                                       RawBufferAllocator *allocator, RuntimeContext *context) {
  const State state = state_.exchange(State::kConsumed);
  EXT_ENFORCE_INVALID(state != State::kConsumed,
                      "PersistentTensor: append lease has already been consumed.");
  const Tensor &prefix = prefix_.value_;
  EXT_ENFORCE_INVALID(prefix.borrowed_owner().use_count() != 0,
                      "PersistentTensor: append lease has been moved.");
  EXT_ENFORCE_INVALID(axis < shape.size() && shape.size() == prefix.shape.size(),
                      "PersistentTensor: invalid append axis or rank.");
  for (size_t i = 0; i < shape.size(); ++i)
    EXT_ENFORCE_INVALID(i == axis ? shape[i] >= prefix.shape[i] : shape[i] == prefix.shape[i],
                        "PersistentTensor: only the append axis may grow.");
  if (initial_capacity == 0)
    return std::nullopt;
  const size_t element_bytes = PackedByteSize(prefix.data_type, 1);
  // Packed sub-byte elements and multiple outer slices need repacking, not a tail write.
  if (shape.product(0, axis, "PersistentTensor") != 1 ||
      element_bytes * 8 != PackedByteSize(prefix.data_type, 8))
    return std::nullopt;
  const auto bytes = [&](const Shape &dimensions) {
    const int64_t count = dimensions.product(0, dimensions.size(), "PersistentTensor");
    EXT_ENFORCE_INVALID(static_cast<uint64_t>(count) <=
                            std::numeric_limits<size_t>::max() / element_bytes,
                        "PersistentTensor: byte size overflow.");
    return static_cast<size_t>(count) * element_bytes;
  };
  const size_t logical = bytes(shape);
  const size_t previous = bytes(prefix.shape);
  EXT_ENFORCE_INVALID(prefix.size_bytes() == previous &&
                          (previous == 0 || prefix.bytes() != nullptr),
                      "PersistentTensor: retained prefix byte extent mismatch.");
  const int64_t row_elements = shape.product(axis + 1, shape.size(), "PersistentTensor");
  const size_t row_bytes = bytes({row_elements});
  if (row_bytes == 0)
    return std::nullopt;
  if (state == State::kReusable && logical <= prefix_.capacity_bytes_) {
    PersistentTensor candidate(Tensor::Borrow(prefix.name, prefix.data_type, shape, prefix.bytes(),
                                              logical, prefix.borrowed_owner()));
    candidate.capacity_bytes_ = prefix_.capacity_bytes_;
    if (context && context->events_enabled())
      context->RecordEvent(
          {.action = RuntimeEventAction::kPersistentStorage, .storage_reuse_count = 1});
    return AppendReservation(std::move(candidate), previous);
  }
  const size_t max_capacity = std::numeric_limits<size_t>::max() / row_bytes;
  size_t capacity = std::max(initial_capacity, prefix_.capacity_bytes_ / row_bytes);
  EXT_ENFORCE_INVALID(capacity <= max_capacity,
                      "PersistentTensor: initial capacity byte size overflow.");
  const size_t required = logical / row_bytes;
  while (capacity < required) {
    capacity = capacity > max_capacity / 2 ? max_capacity : capacity * 2;
    EXT_ENFORCE_INVALID(capacity >= required || capacity < max_capacity,
                        "PersistentTensor: capacity overflow.");
  }
  const size_t allocated = capacity * row_bytes;
  Tensor storage = MakeOutputTensor(prefix.data_type, shape, allocated, allocator);
  if (context && context->events_enabled())
    context->RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                          .storage_allocations = 1,
                          .storage_allocated_bytes = allocated});
  PersistentTensor candidate(std::move(storage));
  if (previous != 0) {
    std::memcpy(candidate.value_.mutable_bytes(), prefix.bytes(), previous);
    if (context && context->events_enabled())
      context->RecordEvent({.action = RuntimeEventAction::kPersistentStorage,
                            .storage_prefix_copied_bytes = previous});
  }
  candidate.value_ = Tensor::Borrow(prefix.name, prefix.data_type, shape, candidate.value_.bytes(),
                                    logical, candidate.value_.borrowed_owner());
  candidate.capacity_bytes_ = allocated;
  return AppendReservation(std::move(candidate), previous);
}

std::span<uint8_t> PersistentTensor::AppendReservation::writable_bytes() {
  EXT_ENFORCE_INVALID(candidate_.value_.borrowed_owner().use_count() != 0,
                      "PersistentTensor: append reservation has been moved or committed.");
  return {candidate_.value_.mutable_bytes() + prefix_bytes_,
          candidate_.value_.size_bytes() - prefix_bytes_};
}

PersistentTensor PersistentTensor::AppendReservation::Commit(size_t initialized_bytes) {
  EXT_ENFORCE_INVALID(initialized_bytes == writable_bytes().size(),
                      "PersistentTensor: the entire reserved tail must be initialized.");
  return std::move(candidate_);
}

PersistentValue::PersistentValue(RuntimeValue value, const StructTypeCatalogue &catalogue)
    : PersistentValue(FromRetained(std::move(value).Retain(catalogue))) {}

PersistentValue::PersistentValue(PersistentTensor tensor)
    : kind_(RuntimeValue::Kind::kTensor), tensor_(std::move(tensor)) {}

PersistentValue PersistentValue::FromRetained(RuntimeValue value) {
  PersistentValue result;
  result.kind_ = value.kind;
  if (value.kind == RuntimeValue::Kind::kTensor)
    result.tensor_.emplace(std::move(value.tensor));
  else if (value.kind == RuntimeValue::Kind::kEncoded)
    result.encoded_ = std::move(value.encoded);
  else
    for (auto &[name, field] : value.fields)
      result.fields_.emplace(name, FromRetained(std::move(field)));
  return result;
}

RuntimeValue PersistentValue::BorrowView() const {
  if (tensor_)
    return RuntimeValue(tensor_->BorrowView());
  RuntimeValue result;
  result.kind = kind_;
  result.encoded = encoded_;
  for (const auto &[name, field] : fields_)
    result.fields.emplace(name, field.BorrowView());
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
