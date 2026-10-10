// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "value_store.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

void ValidateAllocatedValue(const RuntimeValue &value,
                            const std::unordered_set<const void *> &allocators, uint32_t slot) {
  switch (value.kind) {
  case RuntimeValue::Kind::kTensor:
    EXT_ENFORCE(value.tensor.has_allocation(), "Value store slot ", slot,
                " contains a tensor which is not allocator-backed.");
    EXT_ENFORCE(allocators.contains(static_cast<const void *>(value.tensor.allocation_owner())),
                "Value store slot ", slot, " contains a tensor owned by another allocator.");
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

uint32_t ValueStore::Bind(std::string key) {
  EXT_ENFORCE(!key.empty(), "Value store keys must not be empty.");
  auto found = slots_by_key_.find(key);
  if (found != slots_by_key_.end()) {
    return found->second;
  }
  EXT_ENFORCE(slots_.size() < kInvalidValueStoreSlot, "Value store slot capacity was exceeded.");
  const uint32_t slot = static_cast<uint32_t>(slots_.size());
  slots_.emplace_back();
  slots_by_key_.emplace(std::move(key), slot);
  return slot;
}

bool ValueStore::IsReady(uint32_t slot) const { return ValueAt(slot).has_value(); }

Tensor ValueStore::AllocateTensor(int32_t data_type, const Shape &shape, size_t n_bytes) {
  auto allocator = std::make_unique<ExecutionArena>(1);
  Tensor tensor = MakeOutputTensor(data_type, shape, n_bytes, allocator.get());
  allocator_index_.insert(allocator.get());
  allocators_.push_back(std::move(allocator));
  return tensor;
}

void ValueStore::Publish(uint32_t slot, RuntimeValue value) {
  std::optional<RuntimeValue> &entry = ValueAt(slot);
  EXT_ENFORCE(!entry.has_value(), "Value store slot ", slot, " was published more than once.");
  ValidateAllocatedValue(value, allocator_index_, slot);
  entry.emplace(std::move(value));
}

const RuntimeValue &ValueStore::Get(uint32_t slot) const {
  const std::optional<RuntimeValue> &entry = ValueAt(slot);
  EXT_ENFORCE(entry.has_value(), "Value store slot ", slot, " is not ready.");
  return *entry;
}

std::optional<RuntimeValue> &ValueStore::ValueAt(uint32_t slot) {
  EXT_ENFORCE(slot < slots_.size(), "Invalid value store slot ", slot, ".");
  return slots_[slot];
}

const std::optional<RuntimeValue> &ValueStore::ValueAt(uint32_t slot) const {
  EXT_ENFORCE(slot < slots_.size(), "Invalid value store slot ", slot, ".");
  return slots_[slot];
}

size_t ValueStore::prepared_bytes() const noexcept {
  size_t bytes = 0;
  for (const auto &allocator : allocators_) {
    bytes += allocator->TotalAllocatedSize();
  }
  return bytes;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
