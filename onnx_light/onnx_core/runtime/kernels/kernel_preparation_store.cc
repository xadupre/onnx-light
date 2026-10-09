// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kernel_preparation_store.h"

#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

std::shared_ptr<KernelPreparationSlot> KernelPreparationStore::Bind(std::string key,
                                                                    std::string source) {
  EXT_ENFORCE(!key.empty(), "Kernel preparation keys must not be empty.");
  auto found = slots_.find(key);
  if (found != slots_.end()) {
    return found->second;
  }
  auto slot = std::shared_ptr<KernelPreparationSlot>(
      new KernelPreparationSlot(std::move(key), std::move(source)));
  slots_.emplace(slot->key(), slot);
  return slot;
}

void KernelPreparationStore::Publish(const std::shared_ptr<KernelPreparationSlot> &slot,
                                     RawBuffer buffer) {
  EXT_ENFORCE(slot != nullptr, "Cannot publish a null kernel preparation slot.");
  EXT_ENFORCE(slot->state_ == KernelPreparationState::kEmpty, "Kernel preparation '", slot->key_,
              "' was published more than once.");
  prepared_bytes_ += buffer.size();
  slot->buffer_ = std::move(buffer);
  slot->state_ = KernelPreparationState::kReady;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
