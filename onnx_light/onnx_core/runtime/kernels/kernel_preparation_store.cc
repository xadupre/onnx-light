// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kernel_preparation_store.h"

#include <algorithm>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

void ValidateAllocatedValue(const RuntimeValue &value, RawBufferAllocator *allocator,
                            KernelPreparationSlot slot) {
  switch (value.kind) {
  case RuntimeValue::Kind::kTensor:
    EXT_ENFORCE(value.tensor.has_allocation(), "Kernel preparation slot ", slot,
                " contains a tensor which is not allocator-backed.");
    EXT_ENFORCE(value.tensor.allocation_owner() == allocator, "Kernel preparation slot ", slot,
                " contains a tensor owned by another allocator.");
    return;
  case RuntimeValue::Kind::kStruct:
    for (const auto &[name, field] : value.fields) {
      (void)name;
      ValidateAllocatedValue(field, allocator, slot);
    }
    return;
  case RuntimeValue::Kind::kSequence:
    for (const RuntimeValue &element : value.elements) {
      ValidateAllocatedValue(element, allocator, slot);
    }
    return;
  case RuntimeValue::Kind::kEncoded:
    return;
  }
}

} // namespace

RawBuffer *KernelPreparationStore::Allocator::Allocate(size_t n_bytes) {
  auto buffer = std::make_unique<RawBuffer>(n_bytes);
  RawBuffer *result = buffer.get();
  buffers_.emplace(result, std::move(buffer));
  total_allocated_size_ += n_bytes;
  peak_allocated_size_ = std::max(peak_allocated_size_, total_allocated_size_);
  return result;
}

void KernelPreparationStore::Allocator::Free(RawBuffer *buffer) {
  auto found = buffers_.find(buffer);
  EXT_ENFORCE_INVALID(found != buffers_.end(),
                      "KernelPreparationStore allocator does not own the buffer.");
  total_allocated_size_ -= found->second->size();
  buffers_.erase(found);
}

KernelPreparationSlot KernelPreparationStore::Bind(std::string key) {
  EXT_ENFORCE(!key.empty(), "Kernel preparation keys must not be empty.");
  auto found = slots_by_key_.find(key);
  if (found != slots_by_key_.end()) {
    return found->second;
  }
  EXT_ENFORCE(slots_.size() < kInvalidKernelPreparationSlot,
              "Kernel preparation slot capacity was exceeded.");
  const KernelPreparationSlot slot = static_cast<KernelPreparationSlot>(slots_.size());
  slots_.emplace_back();
  slots_by_key_.emplace(std::move(key), slot);
  return slot;
}

bool KernelPreparationStore::IsReady(KernelPreparationSlot slot) const {
  return ValueAt(slot).has_value();
}

void KernelPreparationStore::Publish(KernelPreparationSlot slot, RuntimeValue value) {
  std::optional<RuntimeValue> &entry = ValueAt(slot);
  EXT_ENFORCE(!entry.has_value(), "Kernel preparation slot ", slot,
              " was published more than once.");
  ValidateAllocatedValue(value, &allocator_, slot);
  entry.emplace(std::move(value));
}

const RuntimeValue &KernelPreparationStore::Get(KernelPreparationSlot slot) const {
  const std::optional<RuntimeValue> &entry = ValueAt(slot);
  EXT_ENFORCE(entry.has_value(), "Kernel preparation slot ", slot, " is not ready.");
  return *entry;
}

std::optional<RuntimeValue> &KernelPreparationStore::ValueAt(KernelPreparationSlot slot) {
  EXT_ENFORCE(slot < slots_.size(), "Invalid kernel preparation slot ", slot, ".");
  return slots_[slot];
}

const std::optional<RuntimeValue> &
KernelPreparationStore::ValueAt(KernelPreparationSlot slot) const {
  EXT_ENFORCE(slot < slots_.size(), "Invalid kernel preparation slot ", slot, ".");
  return slots_[slot];
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
