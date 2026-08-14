// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/parallel_for.h"
#include "onnx_core/runtime/random.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr uint32_t kTuningAbi = 1;
constexpr std::array<int32_t, 1> kSupportedElementTypes = {static_cast<int32_t>(DataType::BOOL)};

KernelTuningParameters CalibrateNot(const KernelTuningKey &key,
                                    const CpuExecutionDescriptor &execution,
                                    const CalibrationOptions &options,
                                    CalibrationReporter &reporter) {
  const int64_t portable_minimum = core::runtime::kParallelForGrainSize;
  const KernelContext context{DefaultOpset(1)};
  Not serial_kernel{context};
  Not parallel_kernel{context};
  serial_kernel.Configure(
      {key,
       {{std::string(tuning::kParallelMinimumElements), std::numeric_limits<int64_t>::max()}}});

  const uint32_t actual_threads = static_cast<uint32_t>(core::runtime::ParallelForThreadCount());
  if (execution.effective_threads != actual_threads) {
    throw std::invalid_argument(
        "Not calibration requested " + std::to_string(execution.effective_threads) +
        " threads, but ParallelFor uses " + std::to_string(actual_threads) + ".");
  }
  if (actual_threads == 1) {
    reporter.AddDiagnostic("Not calibration kept the portable threshold because parallel "
                           "execution is unavailable.");
    return {key, {{std::string(tuning::kParallelMinimumElements), portable_minimum}}};
  }

  constexpr int64_t kFirstElements = int64_t{1} << 14;
  constexpr int64_t kMaximumElements = int64_t{1} << 23;
  constexpr int kRepetitions = 5;
  const uint64_t memory_budget =
      options.maximum_memory_bytes == 0 ? uint64_t{64} << 20 : options.maximum_memory_bytes;
  const int64_t maximum_elements =
      std::min(kMaximumElements, static_cast<int64_t>(memory_budget / 3));
  if (maximum_elements < kFirstElements) {
    reporter.AddDiagnostic(
        "Not calibration memory budget is too small; kept the portable threshold.");
    return {key, {{std::string(tuning::kParallelMinimumElements), portable_minimum}}};
  }
  const uint64_t duration_ms = options.maximum_duration_ms == 0 ? 250 : options.maximum_duration_ms;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration_ms);
  const auto median = [](std::vector<int64_t> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  };
  const auto measure = [](const auto &run) {
    const auto begin = std::chrono::steady_clock::now();
    run();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                begin)
        .count();
  };

  int64_t minimum_elements = portable_minimum;
  int64_t first_winning_elements = 0;
  int consecutive_wins = 0;
  bool selected = false;
  for (int64_t elements = kFirstElements; elements <= maximum_elements; elements *= 2) {
    const Tensor input = RandnTensor(key.element_type, {elements}, /*seed=*/5);
    Tensor serial = MakeOutputTensor(DataType::BOOL, {elements}, input.size_bytes(), nullptr);
    Tensor parallel = MakeOutputTensor(DataType::BOOL, {elements}, input.size_bytes(), nullptr);
    parallel_kernel.Configure(
        {key, {{std::string(tuning::kParallelMinimumElements), (elements + 1) / 2}}});
    const auto run_serial = [&]() { serial_kernel(input, serial); };
    const auto run_parallel = [&]() { parallel_kernel(input, parallel); };
    run_serial();
    run_parallel();
    if (std::memcmp(serial.bytes(), parallel.bytes(), serial.size_bytes()) != 0) {
      throw std::runtime_error("Not calibration parallel result differs from the serial result.");
    }

    std::vector<int64_t> serial_samples;
    std::vector<int64_t> parallel_samples;
    serial_samples.reserve(kRepetitions);
    parallel_samples.reserve(kRepetitions);
    for (int repetition = 0; repetition < kRepetitions; ++repetition) {
      serial_samples.push_back(measure(run_serial));
      parallel_samples.push_back(measure(run_parallel));
    }
    const int64_t serial_nanoseconds = median(std::move(serial_samples));
    const int64_t parallel_nanoseconds = median(std::move(parallel_samples));
    if (parallel_nanoseconds * 100 <= serial_nanoseconds * 95) {
      if (consecutive_wins == 0) {
        first_winning_elements = elements;
      }
      ++consecutive_wins;
      if (consecutive_wins == 2) {
        minimum_elements = (first_winning_elements + 1) / 2;
        selected = true;
        reporter.AddDiagnostic(
            "Not selected parallel.minimum_elements=" + std::to_string(minimum_elements) + ".");
        break;
      }
    } else {
      consecutive_wins = 0;
      first_winning_elements = 0;
    }
    if (std::chrono::steady_clock::now() >= deadline || elements > maximum_elements / 2) {
      break;
    }
  }
  if (!selected) {
    reporter.AddDiagnostic(
        "Not calibration found no stable parallel crossover; kept the portable threshold.");
  }
  return {key, {{std::string(tuning::kParallelMinimumElements), minimum_elements}}};
}

} // namespace

Not::Not(const KernelContext &ctx) : KernelBase(ctx), tuning_(kParallelForGrainSize) {}

void Not::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Not", kSupportedElementTypes, kParallelForGrainSize,
                                        kTuningAbi);
  const KernelTuningKey key =
      tuning::MakePortableTuningKey("Not", static_cast<int32_t>(DataType::BOOL), kTuningAbi);
  core::runtime::RegisterKernelCalibrationFunction(key, CalibrateNot);
}

KernelTuningKey Not::TuningKey(int32_t element_type) const {
  return tuning::IsSupportedElementType(element_type, kSupportedElementTypes)
             ? tuning::MakePortableTuningKey("Not", element_type, kTuningAbi)
             : KernelTuningKey{};
}

void Not::Configure(const KernelTuningParameters &parameters) {
  tuning::ConfigureParallelTuning("Not", parameters, tuning_, kTuningAbi);
}

Tensor Not::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count());
  Tensor y = MakeOutputTensor(DataType::BOOL, x.shape, y_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(x, y);
  return y;
}

void Not::operator()(const Tensor &x, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::BOOL, "kernel::Not only supports BOOL tensors.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::BOOL,
                      "kernel::Not preallocated output must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Not preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n);
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes,
                      "kernel::Not preallocated output buffer has unexpected size in bytes.");
  const uint8_t *px = x.bytes();
  uint8_t *py = output.mutable_bytes();
  ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      py[static_cast<size_t>(i)] = static_cast<uint8_t>(px[i] == 0 ? 1 : 0);
    }
  });
}

void Not::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
