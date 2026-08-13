// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/parallel_for.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::Abs";
constexpr uint32_t kTuningAbi = 1;

constexpr std::array<int32_t, 8> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT),   static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::FLOAT16), static_cast<int32_t>(DataType::BFLOAT16),
    static_cast<int32_t>(DataType::INT8),    static_cast<int32_t>(DataType::INT16),
    static_cast<int32_t>(DataType::INT32),   static_cast<int32_t>(DataType::INT64),
};

template <typename T> T SignedAbs(T value) {
  if (value == std::numeric_limits<T>::min()) {
    return value;
  }
  return value < 0 ? static_cast<T>(-value) : value;
}

template <typename T, typename Generator>
void FillAbsCalibrationInput(Tensor &input, int64_t elements, Generator generate) {
  T *values = reinterpret_cast<T *>(input.mutable_bytes());
  for (int64_t i = 0; i < elements; ++i) {
    values[i] = generate(i);
  }
}

Tensor MakeAbsCalibrationInput(int32_t element_type, int64_t elements) {
  Tensor input = MakeOutputTensor(
      element_type, {elements}, static_cast<size_t>(elements) * ElementSize(element_type), nullptr);
  switch (static_cast<DataType>(element_type)) {
  case DataType::FLOAT:
    FillAbsCalibrationInput<float>(
        input, elements, [](int64_t i) { return static_cast<float>((i % 257) - 128) / 17.0f; });
    break;
  case DataType::DOUBLE:
    FillAbsCalibrationInput<double>(
        input, elements, [](int64_t i) { return static_cast<double>((i % 257) - 128) / 17.0; });
    break;
  case DataType::FLOAT16:
    FillAbsCalibrationInput<uint16_t>(input, elements, [](int64_t i) {
      return FloatToFloat16Bits(static_cast<float>((i % 257) - 128) / 17.0f);
    });
    break;
  case DataType::BFLOAT16:
    FillAbsCalibrationInput<uint16_t>(input, elements, [](int64_t i) {
      return FloatToBfloat16Bits(static_cast<float>((i % 257) - 128) / 17.0f);
    });
    break;
  case DataType::INT8:
    FillAbsCalibrationInput<int8_t>(input, elements,
                                    [](int64_t i) { return static_cast<int8_t>((i % 255) - 127); });
    break;
  case DataType::INT16:
    FillAbsCalibrationInput<int16_t>(
        input, elements, [](int64_t i) { return static_cast<int16_t>((i % 65535) - 32767); });
    break;
  case DataType::INT32:
    FillAbsCalibrationInput<int32_t>(
        input, elements, [](int64_t i) { return static_cast<int32_t>((i % 65535) - 32767); });
    break;
  case DataType::INT64:
    FillAbsCalibrationInput<int64_t>(input, elements,
                                     [](int64_t i) { return (i % 65535) - 32767; });
    break;
  default:
    throw std::invalid_argument("Abs calibration received an unsupported element type.");
  }
  return input;
}

int64_t CalibrateAbsParallelMinimumElements(const KernelTuningKey &key,
                                            const CpuExecutionDescriptor &execution,
                                            const CalibrationOptions &options,
                                            CalibrationReporter &reporter) {
  const int64_t portable_minimum = 32 * core::runtime::kParallelForGrainSize;
  const uint32_t actual_threads = static_cast<uint32_t>(core::runtime::ParallelForThreadCount());
  if (execution.effective_threads != actual_threads) {
    throw std::invalid_argument(
        "Abs calibration requested " + std::to_string(execution.effective_threads) +
        " threads, but ParallelFor uses " + std::to_string(actual_threads) + ".");
  }
  if (actual_threads == 1) {
    reporter.AddDiagnostic("Abs calibration kept the portable threshold because parallel "
                           "execution is unavailable.");
    return portable_minimum;
  }

  constexpr int64_t kFirstElements = int64_t{1} << 14;
  constexpr int64_t kMaximumElements = int64_t{1} << 23;
  constexpr int kRepetitions = 5;
  const uint64_t memory_budget =
      options.maximum_memory_bytes == 0 ? uint64_t{64} << 20 : options.maximum_memory_bytes;
  const int64_t budget_elements = static_cast<int64_t>(
      memory_budget / (3 * static_cast<uint64_t>(ElementSize(key.element_type))));
  const int64_t maximum_elements = std::min(kMaximumElements, budget_elements);
  if (maximum_elements < kFirstElements) {
    reporter.AddDiagnostic(
        "Abs calibration memory budget is too small; kept the portable threshold.");
    return portable_minimum;
  }
  const uint64_t duration_ms = options.maximum_duration_ms == 0 ? 250 : options.maximum_duration_ms;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration_ms);

  const KernelContext context{DefaultOpset(13)};
  Abs serial_kernel{context};
  Abs parallel_kernel{context};
  serial_kernel.Configure(
      {key, {{std::string(tuning::kParallelMinimumElements), kMaximumElements + 1}}});

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

  int64_t first_winning_elements = 0;
  int consecutive_wins = 0;
  for (int64_t elements = kFirstElements; elements <= maximum_elements; elements *= 2) {
    const Tensor input = MakeAbsCalibrationInput(key.element_type, elements);
    Tensor serial = MakeOutputTensor(key.element_type, {elements}, input.size_bytes(), nullptr);
    Tensor parallel = MakeOutputTensor(key.element_type, {elements}, input.size_bytes(), nullptr);
    const auto run_serial = [&]() { serial_kernel(input, serial); };
    const int64_t grain = (elements + 1) / 2;
    parallel_kernel.Configure({key, {{std::string(tuning::kParallelMinimumElements), grain}}});
    const auto run_parallel = [&]() { parallel_kernel(input, parallel); };

    run_serial();
    run_parallel();
    if (std::memcmp(serial.bytes(), parallel.bytes(), serial.size_bytes()) != 0) {
      throw std::runtime_error("Abs calibration parallel result differs from the serial result.");
    }

    std::vector<int64_t> serial_samples;
    std::vector<int64_t> parallel_samples;
    serial_samples.reserve(kRepetitions);
    parallel_samples.reserve(kRepetitions);
    for (int repetition = 0; repetition < kRepetitions; ++repetition) {
      serial_samples.push_back(measure(run_serial));
      parallel_samples.push_back(measure(run_parallel));
    }
    const int64_t serial_ns = median(std::move(serial_samples));
    const int64_t parallel_ns = median(std::move(parallel_samples));
    if (parallel_ns * 100 <= serial_ns * 95) {
      if (consecutive_wins == 0) {
        first_winning_elements = elements;
      }
      ++consecutive_wins;
      if (consecutive_wins == 2) {
        const int64_t selected = (first_winning_elements + 1) / 2;
        reporter.AddDiagnostic(
            "Abs selected parallel.minimum_elements=" + std::to_string(selected) +
            " for element type " + std::to_string(key.element_type) + ".");
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
  reporter.AddDiagnostic(
      "Abs calibration found no stable parallel crossover; kept the portable threshold.");
  return portable_minimum;
}

KernelTuningParameters CalibrateAbs(const KernelTuningKey &key,
                                    const CpuExecutionDescriptor &execution,
                                    const CalibrationOptions &options,
                                    CalibrationReporter &reporter) {
  const int64_t minimum_elements =
      CalibrateAbsParallelMinimumElements(key, execution, options, reporter);
  return {key, {{std::string(tuning::kParallelMinimumElements), minimum_elements}}};
}

} // namespace

Abs::Abs(const KernelContext &ctx)
    : KernelBase(ctx), tuning_(32 * core::runtime::kParallelForGrainSize) {}

void Abs::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Abs", kSupportedElementTypes,
                                        32 * core::runtime::kParallelForGrainSize, kTuningAbi);
  for (int32_t element_type : kSupportedElementTypes) {
    const KernelTuningKey key = tuning::MakePortableTuningKey("Abs", element_type, kTuningAbi);
    core::runtime::RegisterKernelCalibrationFunction(key, CalibrateAbs);
  }
}

KernelTuningKey Abs::TuningKey(int32_t element_type) const {
  return tuning::IsSupportedElementType(element_type, kSupportedElementTypes)
             ? tuning::MakePortableTuningKey("Abs", element_type, kTuningAbi)
             : KernelTuningKey{};
}

void Abs::Configure(const KernelTuningParameters &parameters) {
  tuning::ConfigureParallelTuning("Abs", parameters, tuning_, kTuningAbi);
}

Tensor Abs::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor y = MakeOutputTensor(x.data_type, x.shape, y_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(x, y);
  return y;
}

void Abs::operator()(const Tensor &x, Tensor &output) const {
  EXT_ENFORCE_INVALID(output.data_type == x.data_type, kName,
                      ": output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape, kName, ": output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == x.size_bytes(), kName,
                      ": output buffer size mismatch.");
  const int64_t n = x.element_count();
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT: {
    const float *px = x.AsFloat();
    float *py = output.AsFloat();
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = std::fabs(px[i]);
      }
    });
    return;
  }
  case DataType::DOUBLE: {
    const double *px = x.AsDouble();
    double *py = output.AsDouble();
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = std::fabs(px[i]);
      }
    });
    return;
  }
  case DataType::FLOAT16: {
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
    uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = FloatToFloat16Bits(std::fabs(Float16BitsToFloat(px[i])));
      }
    });
    return;
  }
  case DataType::BFLOAT16: {
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
    uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = FloatToBfloat16Bits(std::fabs(Bfloat16BitsToFloat(px[i])));
      }
    });
    return;
  }
  case DataType::INT8: {
    const int8_t *px = x.AsInt8();
    int8_t *py = output.AsInt8();
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = SignedAbs(px[i]);
      }
    });
    return;
  }
  case DataType::INT16: {
    const int16_t *px = x.AsInt16();
    int16_t *py = output.AsInt16();
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = SignedAbs(px[i]);
      }
    });
    return;
  }
  case DataType::INT32: {
    const int32_t *px = x.AsInt32();
    int32_t *py = output.AsInt32();
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = SignedAbs(px[i]);
      }
    });
    return;
  }
  case DataType::INT64: {
    const int64_t *px = x.AsInt64();
    int64_t *py = output.AsInt64();
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = SignedAbs(px[i]);
      }
    });
    return;
  }
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT8, INT16, INT32, and "
                      "INT64 tensors.");
  }
}

void Abs::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
