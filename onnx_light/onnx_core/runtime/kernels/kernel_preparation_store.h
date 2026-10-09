// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_light_helpers.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

enum class KernelPreparationState {
  kEmpty,
  kReady,
};

/**
 * Owns one immutable kernel preparation for a session hierarchy.
 *
 * Slots are registered and published only while a session is initialized or
 * while its serialized first invocation reaches a preparation whose inputs
 * have just become available. The runtime never runs the same session
 * concurrently, so slots need neither a mutex nor atomic state.
 */
class ONNX_LIGHT_CORE_API KernelPreparationSlot {
public:
  const std::string &key() const noexcept { return key_; }
  const std::string &source() const noexcept { return source_; }
  KernelPreparationState state() const noexcept { return state_; }
  bool ready() const noexcept { return state_ == KernelPreparationState::kReady; }
  const RawBuffer &buffer() const {
    EXT_ENFORCE(ready(), "Kernel preparation '", key_, "' is not ready.");
    return buffer_;
  }

private:
  KernelPreparationSlot(std::string key, std::string source)
      : key_(std::move(key)), source_(std::move(source)) {}

  std::string key_;
  std::string source_;
  KernelPreparationState state_ = KernelPreparationState::kEmpty;
  RawBuffer buffer_;

  friend class KernelPreparationStore;
};

/**
 * Stores immutable prepared kernel objects shared by one root session and all
 * nested graph and model-local function sessions.
 *
 * The owning session hierarchy is serialized by contract. Registration,
 * publication, and reads therefore require no mutex. Kernels retain their
 * shared slot directly after binding, so execution performs no key lookup.
 */
class ONNX_LIGHT_CORE_API KernelPreparationStore {
public:
  std::shared_ptr<KernelPreparationSlot> Bind(std::string key, std::string source);
  void Publish(const std::shared_ptr<KernelPreparationSlot> &slot, RawBuffer buffer);

  size_t prepared_bytes() const noexcept { return prepared_bytes_; }
  size_t slot_count() const noexcept { return slots_.size(); }

private:
  std::unordered_map<std::string, std::shared_ptr<KernelPreparationSlot>> slots_;
  size_t prepared_bytes_ = 0;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
