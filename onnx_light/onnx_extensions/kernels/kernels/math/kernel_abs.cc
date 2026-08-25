// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::Abs";
constexpr uint32_t kTuningAbi = 2;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;

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
  const KernelContext context{DefaultOpset(13)};
  Abs reference{context};
  Abs candidate{context};
  KernelCalibrationBenchmark benchmark;
  benchmark.portable_parameters = {
      key, {{std::string(tuning::kParallelMinimumElements), kPortableParallelMinimum}}};
  benchmark.parameter_name = std::string(tuning::kParallelMinimumElements);
  benchmark.cases = MakeElementwiseCalibrationCases(key.element_type, 1, int64_t{1} << 14,
                                                    int64_t{1} << 23, false);
  benchmark.reference.configure = [&](int64_t value) {
    reference.Configure({key, {{benchmark.parameter_name, value}}});
  };
  benchmark.reference.run = [&](std::span<const Tensor> inputs, Tensor &output) {
    reference(inputs[0], output);
  };
  benchmark.candidate.configure = [&](int64_t value) {
    candidate.Configure({key, {{benchmark.parameter_name, value}}});
  };
  benchmark.candidate.run = [&](std::span<const Tensor> inputs, Tensor &output) {
    candidate(inputs[0], output);
  };
  return CalibrateKernelBenchmark(key, execution, options, reporter, benchmark);
}

} // namespace

Abs::Abs(const KernelContext &ctx) : KernelBase(ctx), tuning_(kPortableParallelMinimum) {}

void Abs::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Abs", kSupportedElementTypes, kPortableParallelMinimum,
                                        kTuningAbi);
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
  Tensor y = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, y_n_bytes)
                : MakeOutputTensor(x.data_type, x.shape, y_n_bytes, nullptr);
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
    const uint32_t *px = reinterpret_cast<const uint32_t *>(x.bytes());
    uint32_t *py = reinterpret_cast<uint32_t *>(output.mutable_bytes());
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = px[i] & UINT32_C(0x7fffffff);
      }
    });
    return;
  }
  case DataType::DOUBLE: {
    const uint64_t *px = reinterpret_cast<const uint64_t *>(x.bytes());
    uint64_t *py = reinterpret_cast<uint64_t *>(output.mutable_bytes());
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = px[i] & UINT64_C(0x7fffffffffffffff);
      }
    });
    return;
  }
  case DataType::FLOAT16: {
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
    uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = static_cast<uint16_t>(px[i] & UINT16_C(0x7fff));
      }
    });
    return;
  }
  case DataType::BFLOAT16: {
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
    uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = static_cast<uint16_t>(px[i] & UINT16_C(0x7fff));
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
