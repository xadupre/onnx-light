// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/persistent_value.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

size_t AppendSize(const Tensor &prefix, const Tensor &tail, const Shape &shape) {
  EXT_ENFORCE_INVALID(prefix.data_type != DataType::STRING && prefix.data_type == tail.data_type,
                      "PersistentTensor: append requires matching numeric element types.");
  const auto bytes = [&](const Shape &dimensions) {
    return PackedByteSize(prefix.data_type,
                          dimensions.product(0, dimensions.size(), "PersistentTensor"));
  };
  const size_t logical = bytes(shape);
  EXT_ENFORCE_INVALID(
      prefix.size_bytes() == bytes(prefix.shape) && tail.size_bytes() == bytes(tail.shape) &&
          prefix.size_bytes() <= logical && tail.size_bytes() == logical - prefix.size_bytes(),
      "PersistentTensor: append byte extent mismatch.");
  // Packed sub-byte values require bit-level concatenation, not a byte append.
  EXT_ENFORCE_INVALID(bytes({1}) * 2 == bytes({2}),
                      "PersistentTensor: append requires byte-aligned elements.");
  EXT_ENFORCE_INVALID((prefix.size_bytes() == 0 || prefix.bytes() != nullptr) &&
                          (tail.size_bytes() == 0 || tail.bytes() != nullptr),
                      "PersistentTensor: append input has a null data pointer.");
  return logical;
}

} // namespace

PersistentStorageStatistics PersistentStorageCounters::Snapshot() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return values_;
}

void PersistentStorageCounters::Accumulate(const PersistentStorageStatistics &statistics) {
  const std::lock_guard<std::mutex> lock(mutex_);
  values_.allocations += statistics.allocations;
  values_.allocated_bytes += statistics.allocated_bytes;
  values_.prefix_copied_bytes += statistics.prefix_copied_bytes;
  values_.append_copied_bytes += statistics.append_copied_bytes;
  values_.reuse_count += statistics.reuse_count;
}

PersistentTensor::PersistentTensor(Tensor value) : value_(std::move(value).RetainStorage()) {}

bool PersistentTensor::Matches(const Tensor &view) const noexcept {
  return view.bytes() == value_.bytes() && view.size_bytes() == value_.size_bytes() &&
         view.data_type == value_.data_type && view.shape == value_.shape &&
         !view.borrowed_owner().owner_before(value_.borrowed_owner()) &&
         !value_.borrowed_owner().owner_before(view.borrowed_owner());
}

std::optional<PersistentTensor::AppendLease> PersistentTensor::AcquireAppendLease() const {
  if (capacity_bytes_ == 0 || value_.borrowed_owner().use_count() != 1)
    return std::nullopt;
  return AppendLease(*this);
}

PersistentTensor PersistentTensor::Concatenate(Tensor storage, const Tensor &prefix,
                                               const Tensor &tail, const Shape &shape) {
  const size_t logical = AppendSize(prefix, tail, shape);
  EXT_ENFORCE_INVALID(!storage.is_borrowed() && storage.borrowed_owner().use_count() == 0 &&
                          storage.data_type == prefix.data_type &&
                          storage.size_bytes() >= logical &&
                          (storage.size_bytes() == 0 || storage.bytes() != nullptr),
                      "PersistentTensor: append storage must be a fresh owned allocation.");
  const size_t capacity = storage.size_bytes();
  PersistentTensor result(std::move(storage));
  if (prefix.size_bytes() != 0)
    std::memcpy(result.value_.mutable_bytes(), prefix.bytes(), prefix.size_bytes());
  if (tail.size_bytes() != 0)
    std::memcpy(result.value_.mutable_bytes() + prefix.size_bytes(), tail.bytes(),
                tail.size_bytes());
  result.value_ = Tensor::Borrow(result.value_.name, result.value_.data_type, shape,
                                 result.value_.bytes(), logical, result.value_.borrowed_owner());
  result.capacity_bytes_ = capacity;
  return result;
}

PersistentTensor::AppendLease::AppendLease(const PersistentTensor &tensor)
    : prefix_(tensor.BorrowView()) {
  prefix_.capacity_bytes_ = tensor.capacity_bytes_;
}

PersistentTensor::AppendLease::AppendLease(AppendLease &&other) noexcept
    : prefix_(std::move(other.prefix_)), available_(other.available_.exchange(false)) {}

PersistentTensor::AppendLease &
PersistentTensor::AppendLease::operator=(AppendLease &&other) noexcept {
  if (this != &other) {
    prefix_ = std::move(other.prefix_);
    available_ = other.available_.exchange(false);
  }
  return *this;
}

std::optional<PersistentTensor> PersistentTensor::AppendLease::TryAppend(const Tensor &prefix,
                                                                         const Tensor &tail,
                                                                         const Shape &shape) {
  if (!available_.exchange(false) || !prefix_.Matches(prefix))
    return std::nullopt;
  const size_t logical = AppendSize(prefix, tail, shape);
  if (logical > prefix_.capacity_bytes_)
    return std::nullopt;
  Tensor view = Tensor::Borrow(prefix.name, prefix.data_type, shape, prefix.bytes(), logical,
                               prefix.borrowed_owner());
  if (tail.size_bytes() != 0)
    std::memmove(view.mutable_bytes() + prefix.size_bytes(), tail.bytes(), tail.size_bytes());
  PersistentTensor result(std::move(view));
  result.capacity_bytes_ = prefix_.capacity_bytes_;
  return result;
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
