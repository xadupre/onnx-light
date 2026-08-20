// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/tuning/portable_gemm_tuning.h"

#include "onnx_extensions/kernels/tuning/portable_parallel_tuning.h"

#include <optional>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning {
namespace {

std::optional<std::string>
ValidateGemmTuning(const core::runtime::KernelTuningParameters &parameters) {
  for (const char *name : {kGemmTileM, kGemmTileN, kGemmTileK, kGemmPackBMinimumElements,
                           kGemmSkinnyMLimit, kGemmParallelFmasPerWorkUnit,
                           kGemmParallelMinimumTasks, kGemmConversionParallelMinimumElements}) {
    const int64_t *value = parameters.TryGet<int64_t>(name);
    if (value == nullptr || *value <= 0) {
      return std::string("Gemm ") + name + " must be positive.";
    }
  }
  return std::nullopt;
}

core::runtime::KernelTuningParameters MakeGemmDefaults(int32_t element_type, uint32_t tuning_abi) {
  const GemmTuning defaults;
  return {
      MakePortableTuningKey("Gemm", element_type, tuning_abi),
      {{kGemmTileM, defaults.tile_m},
       {kGemmTileN, defaults.tile_n},
       {kGemmTileK, defaults.tile_k},
       {kGemmPackBMinimumElements, defaults.pack_b_minimum_elements},
       {kGemmSkinnyMLimit, defaults.skinny_m_limit},
       {kGemmParallelFmasPerWorkUnit, defaults.parallel_fmas_per_work_unit},
       {kGemmParallelMinimumTasks, defaults.parallel_minimum_tasks},
       {kGemmConversionParallelMinimumElements, defaults.conversion_parallel_minimum_elements}}};
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
  tuning.tile_m = *parameters.TryGet<int64_t>(kGemmTileM);
  tuning.tile_n = *parameters.TryGet<int64_t>(kGemmTileN);
  tuning.tile_k = *parameters.TryGet<int64_t>(kGemmTileK);
  tuning.pack_b_minimum_elements = *parameters.TryGet<int64_t>(kGemmPackBMinimumElements);
  tuning.skinny_m_limit = *parameters.TryGet<int64_t>(kGemmSkinnyMLimit);
  tuning.parallel_fmas_per_work_unit = *parameters.TryGet<int64_t>(kGemmParallelFmasPerWorkUnit);
  tuning.parallel_minimum_tasks = *parameters.TryGet<int64_t>(kGemmParallelMinimumTasks);
  tuning.conversion_parallel_minimum_elements =
      *parameters.TryGet<int64_t>(kGemmConversionParallelMinimumElements);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning
