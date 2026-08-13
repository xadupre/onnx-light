// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/cast_helper.h"
#include "onnx_core/runtime/parallel_for.h"
#include "onnx_core/runtime/random.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

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

KernelTuningParameters CalibrateAbs(const KernelTuningKey &key,
                                    const CpuExecutionDescriptor &execution,
                                    const CalibrationOptions &options,
                                    CalibrationReporter &reporter) {
  const int64_t portable_minimum = 32 * core::runtime::kParallelForGrainSize;
  const KernelContext context{DefaultOpset(13)};
  Abs serial_kernel{context};
  Abs parallel_kernel{context};
  serial_kernel.Configure(
      {key,
       {{std::string(tuning::kParallelMinimumElements), std::numeric_limits<int64_t>::max()}}});

  const int64_t minimum_elements = tuning::CalibrateUnaryParallelMinimumElements(
      "Abs", execution, options, reporter, portable_minimum, ElementSize(key.element_type),
      [&](int64_t elements, int64_t parallel_minimum, int repetitions) {
        const Tensor input = RandnTensor(key.element_type, {elements}, /*seed=*/5);
        Tensor serial = MakeOutputTensor(key.element_type, {elements}, input.size_bytes(), nullptr);
        Tensor parallel =
            MakeOutputTensor(key.element_type, {elements}, input.size_bytes(), nullptr);
        parallel_kernel.Configure(
            {key, {{std::string(tuning::kParallelMinimumElements), parallel_minimum}}});
        return tuning::MeasureParallelCalibrationRuns(
            "Abs", repetitions, [&]() { serial_kernel(input, serial); },
            [&]() { parallel_kernel(input, parallel); },
            [&]() {
              return std::memcmp(serial.bytes(), parallel.bytes(), serial.size_bytes()) == 0;
            });
      });
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
