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
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

using KernelPreparationSlot = uint32_t;
inline constexpr KernelPreparationSlot kInvalidKernelPreparationSlot =
    std::numeric_limits<KernelPreparationSlot>::max();

/**
 * Stores immutable prepared runtime values shared by one root session and all
 * nested graph and model-local function sessions.
 *
 * The owning session hierarchy is serialized by contract. Registration,
 * publication, and indexed reads therefore require no mutex or atomic state.
 * A kernel retains the integer returned by :cpp:func:`Bind`, so execution is a
 * direct vector lookup.
 */
class ONNX_LIGHT_CORE_API KernelPreparationStore {
public:
  KernelPreparationSlot Bind(std::string key);
  bool IsReady(KernelPreparationSlot slot) const;
  RawBufferAllocator &allocator() noexcept { return allocator_; }
  void Publish(KernelPreparationSlot slot, RuntimeValue value);
  const RuntimeValue &Get(KernelPreparationSlot slot) const;

  size_t prepared_bytes() const noexcept { return allocator_.TotalAllocatedSize(); }
  size_t slot_count() const noexcept { return slots_.size(); }

private:
  class Allocator final : public RawBufferAllocator {
  public:
    RawBuffer *Allocate(size_t n_bytes) override;
    void Free(RawBuffer *buffer) override;
    size_t TotalAllocatedSize() const override { return total_allocated_size_; }
    size_t PeakAllocatedSize() const override { return peak_allocated_size_; }
    void ResetPeak() override { peak_allocated_size_ = total_allocated_size_; }

  private:
    std::unordered_map<RawBuffer *, std::unique_ptr<RawBuffer>> buffers_;
    size_t total_allocated_size_ = 0;
    size_t peak_allocated_size_ = 0;
  };

  std::optional<RuntimeValue> &ValueAt(KernelPreparationSlot slot);
  const std::optional<RuntimeValue> &ValueAt(KernelPreparationSlot slot) const;

  Allocator allocator_;
  std::unordered_map<std::string, KernelPreparationSlot> slots_by_key_;
  std::vector<std::optional<RuntimeValue>> slots_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
