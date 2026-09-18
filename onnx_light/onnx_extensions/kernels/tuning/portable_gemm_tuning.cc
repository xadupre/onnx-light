// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/tuning/portable_gemm_tuning.h"

#include "onnx_extensions/kernels/tuning/portable_parallel_tuning.h"

#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning {
namespace {

std::optional<std::string>
ValidateGemmTuning(const core::runtime::KernelTuningParameters &parameters) {
  const int64_t *value = parameters.TryGet<int64_t>(kGemmParallelMinimumTasks);
  if (value == nullptr || *value <= 0) {
    return std::string("Gemm ") + kGemmParallelMinimumTasks + " must be positive.";
  }
  const int64_t *configuration = parameters.TryGet<int64_t>(kGemmAlgorithmConfiguration);
  if (configuration == nullptr || *configuration < 0 ||
      *configuration >= kGemmAlgorithmConfigurationCount) {
    return std::string("Gemm ") + kGemmAlgorithmConfiguration + " must be an int64 in [0, " +
           std::to_string(kGemmAlgorithmConfigurationCount) + ").";
  }
  return std::nullopt;
}

core::runtime::KernelTuningParameters MakeGemmDefaults(int32_t element_type, uint32_t tuning_abi) {
  const GemmTuning defaults;
  return {MakePortableTuningKey("Gemm", element_type, tuning_abi),
          {{kGemmParallelMinimumTasks, defaults.parallel_minimum_tasks},
           {kGemmAlgorithmConfiguration, int64_t{0}}}};
}

} // namespace

void RegisterGemmTuningSchemas(std::span<const int32_t> supported_element_types,
                               uint32_t tuning_abi) {
  for (int32_t element_type : supported_element_types) {
    core::runtime::RegisterKernelTuningSchema(
        core::runtime::KernelTuningSchema::WithQueryValidation(
            MakeGemmDefaults(element_type, tuning_abi), ValidateGemmTuning));
  }
}

void ConfigureGemmTuning(const core::runtime::KernelTuningParameters &parameters,
                         GemmTuning &tuning, uint32_t tuning_abi) {
  if (parameters.key != MakePortableTuningKey("Gemm", parameters.key.element_type, tuning_abi)) {
    throw std::invalid_argument("Gemm tuning parameters have an incompatible key.");
  }
  if (std::optional<std::string> error = ValidateGemmTuning(parameters)) {
    throw std::invalid_argument(*error);
  }
  tuning = GemmTuning{};
  switch (*parameters.TryGet<int64_t>(kGemmAlgorithmConfiguration)) {
  case 1:
    tuning.tile_m = 32;
    break;
  case 2:
    tuning.tile_n = 64;
    break;
  case 3:
    tuning.tile_k = 64;
    break;
  case 4:
    tuning.pack_b_minimum_elements = 0;
    break;
  case 5:
    tuning.pack_b_minimum_elements = std::numeric_limits<int64_t>::max();
    break;
  case 6:
    tuning.tile_m = 32;
    tuning.tile_n = 64;
    tuning.tile_k = 128;
    break;
  default:
    break;
  }
  tuning.parallel_minimum_tasks = *parameters.TryGet<int64_t>(kGemmParallelMinimumTasks);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning
