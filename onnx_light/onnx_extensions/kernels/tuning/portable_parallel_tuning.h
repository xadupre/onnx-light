// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernel_tuning.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning {

inline constexpr std::string_view kParallelMinimumElements = "parallel.minimum_elements";

/** Stores the portable parallel configuration copied into one kernel instance. */
struct ParallelTuning {
  explicit ParallelTuning(int64_t minimum_elements) : parallel_minimum_elements(minimum_elements) {}

  int64_t parallel_minimum_elements;
};

/** Stores serial and parallel timings for one candidate element count. */
struct ParallelCalibrationMeasurement {
  int64_t serial_nanoseconds;
  int64_t parallel_nanoseconds;
};

/** Measures equivalent serial and parallel kernel invocations and validates their outputs. */
template <typename SerialRun, typename ParallelRun, typename OutputsEqual>
ParallelCalibrationMeasurement
MeasureParallelCalibrationRuns(std::string_view kernel, int repetitions, SerialRun run_serial,
                               ParallelRun run_parallel, OutputsEqual outputs_equal) {
  run_serial();
  run_parallel();
  if (!outputs_equal()) {
    throw std::runtime_error(std::string(kernel) +
                             " calibration parallel result differs from the serial result.");
  }

  const auto measure = [](const auto &run) {
    const auto begin = std::chrono::steady_clock::now();
    run();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                begin)
        .count();
  };
  std::vector<int64_t> serial_samples;
  std::vector<int64_t> parallel_samples;
  serial_samples.reserve(static_cast<size_t>(repetitions));
  parallel_samples.reserve(static_cast<size_t>(repetitions));
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    serial_samples.push_back(measure(run_serial));
    parallel_samples.push_back(measure(run_parallel));
  }
  const auto median = [](std::vector<int64_t> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  };
  return {median(std::move(serial_samples)), median(std::move(parallel_samples))};
}

using UnaryParallelCalibrationBenchmark = std::function<ParallelCalibrationMeasurement(
    int64_t elements, int64_t parallel_minimum, int repetitions)>;

/**
 * Finds the element-count crossover between serial and parallel unary kernel execution.
 *
 * The benchmark prepares and invokes the real kernel for one candidate count. This function owns
 * the common calibration policy: thread validation, resource limits, timing repetitions, and
 * stable-crossover selection.
 *
 * Returns:
 *   The selected positive ``parallel.minimum_elements`` value.
 */
int64_t CalibrateUnaryParallelMinimumElements(
    std::string_view kernel, const core::runtime::CpuExecutionDescriptor &execution,
    const core::runtime::CalibrationOptions &options, core::runtime::CalibrationReporter &reporter,
    int64_t portable_minimum, size_t element_size,
    const UnaryParallelCalibrationBenchmark &benchmark);

/**
 * Returns the tuning key for one portable onnx-light kernel implementation.
 *
 * Returns:
 *   The exact key for ``kernel`` and ``element_type``.
 */
core::runtime::KernelTuningKey MakePortableTuningKey(std::string_view kernel, int32_t element_type,
                                                     uint32_t tuning_abi = 1);

/**
 * Returns whether an element type is supported by a kernel.
 *
 * Returns:
 *   ``true`` when ``element_type`` appears in ``supported_element_types``.
 */
bool IsSupportedElementType(int32_t element_type, std::span<const int32_t> supported_element_types);

/** Registers one parallel tuning schema for every supported element type. */
void RegisterParallelTuningSchemas(std::string_view kernel,
                                   std::span<const int32_t> supported_element_types,
                                   int64_t portable_minimum_elements, uint32_t tuning_abi = 1);

/** Validates and copies resolved parallel parameters into a typed configuration. */
void ConfigureParallelTuning(std::string_view kernel,
                             const core::runtime::KernelTuningParameters &parameters,
                             ParallelTuning &tuning, uint32_t tuning_abi = 1);

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::tuning
