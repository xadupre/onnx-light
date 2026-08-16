// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/runtime_parameters.h"

#include <thread>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

int32_t RuntimeParameters::EffectiveNumThreads() const noexcept {
  if (num_threads > 1) {
    return num_threads;
  }
  if (num_threads == 1) {
    return 1;
  }
  unsigned int cores = std::thread::hardware_concurrency();
  return cores == 0 ? 1 : static_cast<int32_t>(cores);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
