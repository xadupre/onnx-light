// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/kernels/elementwise_helpers.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::Sign";
constexpr uint32_t kTuningAbi = 2;
constexpr int64_t kPortableParallelMinimum = core::runtime::kParallelForGrainSize;

constexpr std::array<int32_t, 12> kSupportedElementTypes = {
    static_cast<int32_t>(DataType::FLOAT),   static_cast<int32_t>(DataType::DOUBLE),
    static_cast<int32_t>(DataType::FLOAT16), static_cast<int32_t>(DataType::BFLOAT16),
    static_cast<int32_t>(DataType::UINT8),   static_cast<int32_t>(DataType::UINT16),
    static_cast<int32_t>(DataType::UINT32),  static_cast<int32_t>(DataType::UINT64),
    static_cast<int32_t>(DataType::INT8),    static_cast<int32_t>(DataType::INT16),
    static_cast<int32_t>(DataType::INT32),   static_cast<int32_t>(DataType::INT64),
};

// sign(x) for unsigned types: 0 when x == 0, 1 otherwise.
template <typename T> void SignUnsigned(const Tensor &x, Tensor &output, int64_t grain_size) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  ParallelFor(n, grain_size, [px, py](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      py[i] = px[i] > T{0} ? T{1} : T{0};
    }
  });
}

// sign(x) for signed types: -1, 0, or 1.
template <typename T> void SignSigned(const Tensor &x, Tensor &output, int64_t grain_size) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  ParallelFor(n, grain_size, [px, py](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      py[i] = px[i] > T{0} ? T{1} : (px[i] < T{0} ? T{-1} : T{0});
    }
  });
}

} // namespace

Sign::Sign(const KernelContext &ctx) : KernelBase(ctx), tuning_(kPortableParallelMinimum) {}

void Sign::RegisterTuningSchemas() {
  tuning::RegisterParallelTuningSchemas("Sign", kSupportedElementTypes, kPortableParallelMinimum,
                                        kTuningAbi);
}

KernelTuningKey Sign::TuningKey(int32_t element_type) const {
  return tuning::IsSupportedElementType(element_type, kSupportedElementTypes)
             ? tuning::MakePortableTuningKey("Sign", element_type, kTuningAbi)
             : KernelTuningKey{};
}

void Sign::Configure(const KernelTuningParameters &parameters) {
  tuning::ConfigureParallelTuning("Sign", parameters, tuning_, kTuningAbi);
}

Tensor Sign::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor y = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, y_n_bytes)
                : MakeOutputTensor(x.data_type, x.shape, y_n_bytes, nullptr);
  (*this)(x, y);
  return y;
}

void Sign::operator()(const Tensor &x, Tensor &output) const {
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
        py[i] = (px[i] > 0.0f) ? 1.0f : (px[i] < 0.0f ? -1.0f : 0.0f);
      }
    });
    return;
  }
  case DataType::DOUBLE: {
    const double *px = x.AsDouble();
    double *py = output.AsDouble();
    ParallelFor(n, tuning_.parallel_minimum_elements, [px, py](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        py[i] = (px[i] > 0.0) ? 1.0 : (px[i] < 0.0 ? -1.0 : 0.0);
      }
    });
    return;
  }
  case DataType::FLOAT16:
    detail::UnaryHalfElementwise(
        x, output, Float16BitsToFloat, FloatToFloat16Bits, tuning_.parallel_minimum_elements,
        [](float v) { return (v > 0.0f) ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); });
    return;
  case DataType::BFLOAT16:
    detail::UnaryHalfElementwise(
        x, output, Bfloat16BitsToFloat, FloatToBfloat16Bits, tuning_.parallel_minimum_elements,
        [](float v) { return (v > 0.0f) ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); });
    return;
  case DataType::UINT8:
    SignUnsigned<uint8_t>(x, output, tuning_.parallel_minimum_elements);
    return;
  case DataType::UINT16:
    SignUnsigned<uint16_t>(x, output, tuning_.parallel_minimum_elements);
    return;
  case DataType::UINT32:
    SignUnsigned<uint32_t>(x, output, tuning_.parallel_minimum_elements);
    return;
  case DataType::UINT64:
    SignUnsigned<uint64_t>(x, output, tuning_.parallel_minimum_elements);
    return;
  case DataType::INT8:
    SignSigned<int8_t>(x, output, tuning_.parallel_minimum_elements);
    return;
  case DataType::INT16:
    SignSigned<int16_t>(x, output, tuning_.parallel_minimum_elements);
    return;
  case DataType::INT32:
    SignSigned<int32_t>(x, output, tuning_.parallel_minimum_elements);
    return;
  case DataType::INT64:
    SignSigned<int64_t>(x, output, tuning_.parallel_minimum_elements);
    return;
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, UINT8, UINT16, UINT32, "
                      "UINT64, INT8, INT16, INT32, and INT64 tensors.");
  }
}

void Sign::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
