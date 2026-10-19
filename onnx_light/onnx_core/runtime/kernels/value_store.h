// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/runtime_value.h"
#include "onnx_light_helpers.h"

#include <cstddef>
#include <cstdint>
#include <deque>
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
 * Stores statically prepared, immutable kernel values, not invocation values.
 *
 * A root session fixes its owner at construction and explicitly shares it with
 * nested graph and model-local function sessions. Other sessions share it only
 * when their construction options supply the same owner. The store and its
 * allocations live until the last shared owner (including contexts) is released.
 * "Static" means publication once per slot followed by immutable replay, not
 * a global C++ object or mutation-free initialization.
 *
 * All sessions sharing the store are serialized by contract. Registration,
 * publication, and indexed reads therefore require no mutex or atomic state.
 * A kernel retains the integer returned by :cpp:func:`Bind`, so execution is a
 * direct indexed lookup. This synchronous store is separate from
 * :cpp:class:`PreparedExecutionState` and its :cpp:class:`PreparedObjectStore`;
 * the dynamic path manages asynchronous tasks, generations, residency,
 * persistence, and eviction.
 */
class ONNX_LIGHT_CORE_API ValueStore {
public:
  ValueStore() = default;
  ValueStore(const ValueStore &) = delete;
  ValueStore &operator=(const ValueStore &) = delete;
  ValueStore(ValueStore &&) = delete;
  ValueStore &operator=(ValueStore &&) = delete;

  uint32_t Bind(std::string key);
  bool IsReady(uint32_t slot) const;
  Tensor AllocateTensor(int32_t data_type, const Shape &shape, size_t n_bytes);
  void Publish(uint32_t slot, RuntimeValue value);
  const RuntimeValue &Get(uint32_t slot) const;

  /// Returns the sum of live buffer sizes in this static store's allocators.
  /// Includes allocated but unpublished values, not retained free capacity or
  /// dynamic PreparedObjectStore residency. No dynamic budget or eviction applies.
  size_t prepared_bytes() const noexcept;
  size_t slot_count() const noexcept { return slots_.size(); }

private:
  std::optional<RuntimeValue> &ValueAt(uint32_t slot);
  const std::optional<RuntimeValue> &ValueAt(uint32_t slot) const;

  std::vector<std::unique_ptr<ExecutionArena>> allocators_;
  std::unordered_set<const void *> allocator_index_;
  std::unordered_map<std::string, uint32_t> slots_by_key_;
  std::deque<std::optional<RuntimeValue>> slots_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
