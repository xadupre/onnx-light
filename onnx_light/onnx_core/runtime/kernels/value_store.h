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
#include <utility>
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
 * All sessions sharing the store are serialized by contract. Indexed reads
 * neither copy identity ownership nor require mutexes or atomic updates.
 * A kernel retains the store-bound slot returned by :cpp:func:`Bind`, so
 * execution checks its owner and performs a direct indexed lookup.
 * This synchronous store is separate from
 * :cpp:class:`PreparedExecutionState` and its :cpp:class:`PreparedObjectStore`;
 * the dynamic path manages asynchronous tasks, generations, residency,
 * persistence, and eviction.
 */
class ONNX_LIGHT_CORE_API ValueStore {
  struct Identity {};

public:
  /**
   * Identifies a stable index in one store, without retaining its allocations.
   *
   * A default-constructed or moved-from slot is unbound and rejected by store
   * operations. The identity token prevents a slot from matching another store,
   * even if the original store's address is reused after destruction.
   */
  class Slot {
  public:
    Slot() = default;

    /// Returns the numeric index for diagnostics, not an interchangeable handle.
    uint32_t index() const noexcept { return index_; }
    bool operator==(const Slot &) const = default;

  private:
    friend class ValueStore;
    Slot(std::shared_ptr<const Identity> owner, uint32_t index)
        : owner_(std::move(owner)), index_(index) {}

    std::shared_ptr<const Identity> owner_;
    uint32_t index_ = kInvalidValueStoreSlot;
  };

  ValueStore() = default;
  ValueStore(const ValueStore &) = delete;
  ValueStore &operator=(const ValueStore &) = delete;
  ValueStore(ValueStore &&) = delete;
  ValueStore &operator=(ValueStore &&) = delete;

  Slot Bind(std::string key);
  bool IsReady(const Slot &slot) const;
  Tensor AllocateTensor(int32_t data_type, const Shape &shape, size_t n_bytes);
  void Publish(const Slot &slot, RuntimeValue value);
  const RuntimeValue &Get(const Slot &slot) const;

  /// Returns the sum of live buffer sizes in this static store's allocators.
  /// Includes allocated but unpublished values, not retained free capacity or
  /// dynamic PreparedObjectStore residency. No dynamic budget or eviction applies.
  size_t prepared_bytes() const noexcept;
  size_t slot_count() const noexcept { return slots_.size(); }

private:
  std::optional<RuntimeValue> &ValueAt(const Slot &slot);
  const std::optional<RuntimeValue> &ValueAt(const Slot &slot) const;

  std::shared_ptr<const Identity> identity_ = std::make_shared<const Identity>();
  std::vector<std::unique_ptr<ExecutionArena>> allocators_;
  std::unordered_set<const void *> allocator_index_;
  std::unordered_map<std::string, uint32_t> slots_by_key_;
  std::deque<std::optional<RuntimeValue>> slots_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
