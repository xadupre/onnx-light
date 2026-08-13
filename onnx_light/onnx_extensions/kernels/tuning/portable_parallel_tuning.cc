// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/tuning/portable_parallel_tuning.h"

#include "onnx_core/runtime/parallel_for.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning {
namespace {

constexpr std::string_view kTuningLibrary = "onnx_light";
constexpr std::string_view kTuningImplementation = "portable";

void ValidateParallelTuning(std::string_view kernel,
                            const core::runtime::KernelTuningParameters &parameters) {
  if (parameters.Get<int64_t>(kParallelMinimumElements) <= 0) {
    throw std::invalid_argument(std::string(kernel) +
                                " parallel.minimum_elements must be positive.");
  }
}

} // namespace

int64_t CalibrateUnaryParallelMinimumElements(
    std::string_view kernel, const core::runtime::CpuExecutionDescriptor &execution,
    const core::runtime::CalibrationOptions &options, core::runtime::CalibrationReporter &reporter,
    int64_t portable_minimum, size_t element_size,
    const UnaryParallelCalibrationBenchmark &benchmark) {
  const uint32_t actual_threads = static_cast<uint32_t>(core::runtime::ParallelForThreadCount());
  if (execution.effective_threads != actual_threads) {
    throw std::invalid_argument(std::string(kernel) + " calibration requested " +
                                std::to_string(execution.effective_threads) +
                                " threads, but ParallelFor uses " + std::to_string(actual_threads) +
                                ".");
  }
  if (actual_threads == 1) {
    reporter.AddDiagnostic(std::string(kernel) +
                           " calibration kept the portable threshold because parallel execution "
                           "is unavailable.");
    return portable_minimum;
  }

  constexpr int64_t kFirstElements = int64_t{1} << 14;
  constexpr int64_t kMaximumElements = int64_t{1} << 23;
  constexpr int kRepetitions = 5;
  const uint64_t memory_budget =
      options.maximum_memory_bytes == 0 ? uint64_t{64} << 20 : options.maximum_memory_bytes;
  const int64_t budget_elements =
      static_cast<int64_t>(memory_budget / (3 * static_cast<uint64_t>(element_size)));
  const int64_t maximum_elements = std::min(kMaximumElements, budget_elements);
  if (maximum_elements < kFirstElements) {
    reporter.AddDiagnostic(std::string(kernel) +
                           " calibration memory budget is too small; kept the portable threshold.");
    return portable_minimum;
  }
  const uint64_t duration_ms = options.maximum_duration_ms == 0 ? 250 : options.maximum_duration_ms;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration_ms);

  int64_t first_winning_elements = 0;
  int consecutive_wins = 0;
  for (int64_t elements = kFirstElements; elements <= maximum_elements; elements *= 2) {
    const int64_t parallel_minimum = (elements + 1) / 2;
    const ParallelCalibrationMeasurement measurement =
        benchmark(elements, parallel_minimum, kRepetitions);
    if (measurement.parallel_nanoseconds * 100 <= measurement.serial_nanoseconds * 95) {
      if (consecutive_wins == 0) {
        first_winning_elements = elements;
      }
      ++consecutive_wins;
      if (consecutive_wins == 2) {
        const int64_t selected = (first_winning_elements + 1) / 2;
        reporter.AddDiagnostic(std::string(kernel) + " selected parallel.minimum_elements=" +
                               std::to_string(selected) + ".");
        return selected;
      }
    } else {
      consecutive_wins = 0;
      first_winning_elements = 0;
    }
    if (std::chrono::steady_clock::now() >= deadline || elements > maximum_elements / 2) {
      break;
    }
  }
  reporter.AddDiagnostic(std::string(kernel) +
                         " calibration found no stable parallel crossover; kept the portable "
                         "threshold.");
  return portable_minimum;
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
    core::runtime::RegisterKernelTuningSchema(core::runtime::KernelTuningSchema(
        {MakePortableTuningKey(kernel, element_type, tuning_abi),
         {{std::string(kParallelMinimumElements), portable_minimum_elements}}},
        [kernel_name](const core::runtime::KernelTuningParameters &parameters) {
          ValidateParallelTuning(kernel_name, parameters);
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
  ValidateParallelTuning(kernel, parameters);
  tuning.parallel_minimum_elements = parameters.Get<int64_t>(kParallelMinimumElements);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning
