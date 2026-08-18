// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/runtime_parameters.h"

#include "onnx_core/platform/cpu_descriptor.h"

#include <thread>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

int32_t RuntimeParameters::EffectiveNumThreads() const noexcept {
  if (num_threads > 1) {
    return num_threads;
  }
  if (num_threads == 1) {
    return 1;
  }
  const platform::CpuDescriptor &descriptor = platform::GetCpuDescriptor();
  if (descriptor.physical_cores.has_value() && *descriptor.physical_cores != 0) {
    return static_cast<int32_t>(*descriptor.physical_cores);
  }
  if (descriptor.logical_cores.has_value() && *descriptor.logical_cores != 0) {
    return static_cast<int32_t>(*descriptor.logical_cores);
  }
  const unsigned int threads = std::thread::hardware_concurrency();
  return threads == 0 ? 1 : static_cast<int32_t>(threads);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
