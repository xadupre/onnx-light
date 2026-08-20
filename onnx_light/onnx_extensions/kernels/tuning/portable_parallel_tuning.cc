// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/tuning/portable_parallel_tuning.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning {
namespace {

constexpr std::string_view kTuningLibrary = "onnx_light";
constexpr std::string_view kTuningImplementation = "portable";

std::optional<std::string>
ValidateParallelTuning(std::string_view kernel,
                       const core::runtime::KernelTuningParameters &parameters) {
  const int64_t *minimum_elements = parameters.TryGet<int64_t>(kParallelMinimumElements);
  if (minimum_elements == nullptr || *minimum_elements <= 0) {
    return std::string(kernel) + " parallel.minimum_elements must be positive.";
  }
  return std::nullopt;
}

} // namespace

ParallelTunableKernel::ParallelTunableKernel(const core::runtime::KernelContext &ctx,
                                             std::string_view kernel,
                                             std::span<const int32_t> supported_element_types,
                                             int64_t portable_minimum_elements, uint32_t tuning_abi)
    : KernelBase(ctx), kernel_(kernel), supported_element_types_(supported_element_types),
      tuning_(portable_minimum_elements), tuning_abi_(tuning_abi) {}

core::runtime::KernelTuningKey ParallelTunableKernel::TuningKey(int32_t element_type) const {
  return IsSupportedElementType(element_type, supported_element_types_)
             ? MakePortableTuningKey(kernel_, element_type, tuning_abi_)
             : core::runtime::KernelTuningKey{};
}

void ParallelTunableKernel::Configure(const core::runtime::KernelTuningParameters &parameters) {
  ConfigureParallelTuning(kernel_, parameters, tuning_, tuning_abi_);
}

core::runtime::KernelTuningKey MakePortableTuningKey(std::string_view kernel, int32_t element_type,
                                                     uint32_t tuning_abi) {
  return {std::string(kTuningLibrary),        std::string(kernel),
          std::string(kTuningImplementation), element_type,
          core::symbolic::Device::kCPU,       tuning_abi};
}

bool IsSupportedElementType(int32_t element_type,
                            std::span<const int32_t> supported_element_types) {
  return std::find(supported_element_types.begin(), supported_element_types.end(), element_type) !=
         supported_element_types.end();
}

void RegisterParallelTuningSchemas(std::string_view kernel,
                                   std::span<const int32_t> supported_element_types,
                                   int64_t portable_minimum_elements, uint32_t tuning_abi) {
  const std::string kernel_name(kernel);
  for (int32_t element_type : supported_element_types) {
    core::runtime::RegisterKernelTuningSchema(
        core::runtime::KernelTuningSchema::WithQueryValidation(
            {MakePortableTuningKey(kernel, element_type, tuning_abi),
             {{std::string(kParallelMinimumElements), portable_minimum_elements}}},
            [kernel_name](const core::runtime::KernelTuningParameters &parameters) {
              return ValidateParallelTuning(kernel_name, parameters);
            }));
  }
}

void ConfigureParallelTuning(std::string_view kernel,
                             const core::runtime::KernelTuningParameters &parameters,
                             ParallelTuning &tuning, uint32_t tuning_abi) {
  if (parameters.key != MakePortableTuningKey(kernel, parameters.key.element_type, tuning_abi)) {
    throw std::invalid_argument(std::string(kernel) +
                                " tuning parameters have an incompatible key.");
  }
  if (std::optional<std::string> error = ValidateParallelTuning(kernel, parameters)) {
    throw std::invalid_argument(*error);
  }
  tuning.parallel_minimum_elements = *parameters.TryGet<int64_t>(kParallelMinimumElements);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning
