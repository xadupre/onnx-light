// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/_helpers/elementwise_helpers.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include "onnx_kernels/runtime_context.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kName = "kernel::Sign";

// sign(x) for unsigned types: 0 when x == 0, 1 otherwise.
template <typename T> void SignUnsigned(const Tensor &x, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  for (int64_t i = 0; i < n; ++i) {
    py[i] = px[i] > T{0} ? T{1} : T{0};
  }
}

// sign(x) for signed types: -1, 0, or 1.
template <typename T> void SignSigned(const Tensor &x, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  for (int64_t i = 0; i < n; ++i) {
    py[i] = px[i] > T{0} ? T{1} : (px[i] < T{0} ? T{-1} : T{0});
  }
}

} // namespace

Tensor Sign::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor y = MakeOutputTensor(x.data_type, x.shape, y_n_bytes, rt ? rt->allocator() : nullptr);
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
    for (int64_t i = 0; i < n; ++i) {
      py[i] = (px[i] > 0.0f) ? 1.0f : (px[i] < 0.0f ? -1.0f : 0.0f);
    }
    return;
  }
  case DataType::DOUBLE: {
    const double *px = x.AsDouble();
    double *py = output.AsDouble();
    for (int64_t i = 0; i < n; ++i) {
      py[i] = (px[i] > 0.0) ? 1.0 : (px[i] < 0.0 ? -1.0 : 0.0);
    }
    return;
  }
  case DataType::FLOAT16:
    kernel::detail::UnaryHalfElementwise(
        x, output, Float16BitsToFloat, FloatToFloat16Bits,
        [](float v) { return (v > 0.0f) ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); });
    return;
  case DataType::BFLOAT16:
    kernel::detail::UnaryHalfElementwise(
        x, output, Bfloat16BitsToFloat, FloatToBfloat16Bits,
        [](float v) { return (v > 0.0f) ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); });
    return;
  case DataType::UINT8:
    SignUnsigned<uint8_t>(x, output);
    return;
  case DataType::UINT16:
    SignUnsigned<uint16_t>(x, output);
    return;
  case DataType::UINT32:
    SignUnsigned<uint32_t>(x, output);
    return;
  case DataType::UINT64:
    SignUnsigned<uint64_t>(x, output);
    return;
  case DataType::INT8:
    SignSigned<int8_t>(x, output);
    return;
  case DataType::INT16:
    SignSigned<int16_t>(x, output);
    return;
  case DataType::INT32:
    SignSigned<int32_t>(x, output);
    return;
  case DataType::INT64:
    SignSigned<int64_t>(x, output);
    return;
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, UINT8, UINT16, UINT32, "
                      "UINT64, INT8, INT16, INT32, and INT64 tensors.");
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
