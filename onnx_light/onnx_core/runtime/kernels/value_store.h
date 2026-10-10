// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/runtime_value.h"
#include "onnx_light_helpers.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

inline constexpr uint32_t kInvalidValueStoreSlot = std::numeric_limits<uint32_t>::max();

/**
 * Stores immutable prepared runtime values shared by one root session and all
 * nested graph and model-local function sessions.
 *
 * The owning session hierarchy is serialized by contract. Registration,
 * publication, and indexed reads therefore require no mutex or atomic state.
 * A kernel retains the integer returned by :cpp:func:`Bind`, so execution is a
 * direct vector lookup. This synchronous store does not replace
 * :cpp:class:`PreparedObjectStore`, which owns asynchronous generation,
 * scheduling, residency, persistence, and eviction state.
 */
class ONNX_LIGHT_CORE_API ValueStore {
public:
  uint32_t Bind(std::string key);
  bool IsReady(uint32_t slot) const;
  Tensor AllocateTensor(int32_t data_type, const Shape &shape, size_t n_bytes);
  void Publish(uint32_t slot, RuntimeValue value);
  const RuntimeValue &Get(uint32_t slot) const;

  size_t prepared_bytes() const noexcept;
  size_t slot_count() const noexcept { return slots_.size(); }

private:
  std::optional<RuntimeValue> &ValueAt(uint32_t slot);
  const std::optional<RuntimeValue> &ValueAt(uint32_t slot) const;

  std::vector<std::unique_ptr<ExecutionArena>> allocators_;
  std::unordered_set<const void *> allocator_index_;
  std::unordered_map<std::string, uint32_t> slots_by_key_;
  std::vector<std::optional<RuntimeValue>> slots_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
