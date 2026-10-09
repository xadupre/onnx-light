// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kernel_preparation_store.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

void ValidateAllocatedValue(const RuntimeValue &value,
                            const std::unordered_set<const void *> &allocators,
                            KernelPreparationSlot slot) {
  switch (value.kind) {
  case RuntimeValue::Kind::kTensor:
    EXT_ENFORCE(value.tensor.has_allocation(), "Kernel preparation slot ", slot,
                " contains a tensor which is not allocator-backed.");
    EXT_ENFORCE(allocators.contains(static_cast<const void *>(value.tensor.allocation_owner())),
                "Kernel preparation slot ", slot, " contains a tensor owned by another allocator.");
    return;
  case RuntimeValue::Kind::kStruct:
    for (const auto &[name, field] : value.fields) {
      (void)name;
      ValidateAllocatedValue(field, allocators, slot);
    }
    return;
  case RuntimeValue::Kind::kSequence:
    for (const RuntimeValue &element : value.elements) {
      ValidateAllocatedValue(element, allocators, slot);
    }
    return;
  case RuntimeValue::Kind::kEncoded:
    return;
  }
}

} // namespace

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

Tensor KernelPreparationStore::AllocateTensor(int32_t data_type, const Shape &shape,
                                              size_t n_bytes) {
  auto allocator = std::make_unique<ExecutionArena>(1);
  Tensor tensor = MakeOutputTensor(data_type, shape, n_bytes, allocator.get());
  allocator_index_.insert(allocator.get());
  allocators_.push_back(std::move(allocator));
  return tensor;
}

void KernelPreparationStore::Publish(KernelPreparationSlot slot, RuntimeValue value) {
  std::optional<RuntimeValue> &entry = ValueAt(slot);
  EXT_ENFORCE(!entry.has_value(), "Kernel preparation slot ", slot,
              " was published more than once.");
  ValidateAllocatedValue(value, allocator_index_, slot);
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

size_t KernelPreparationStore::prepared_bytes() const noexcept {
  size_t bytes = 0;
  for (const auto &allocator : allocators_) {
    bytes += allocator->TotalAllocatedSize();
  }
  return bytes;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
