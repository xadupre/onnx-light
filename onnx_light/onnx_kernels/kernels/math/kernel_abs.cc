// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/_helpers/elementwise_helpers.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "onnx_kernels/runtime_context.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kName = "kernel::Abs";

} // namespace

Tensor Abs::operator()(const Tensor &x, RuntimeContext *rt) const {
  Tensor y("", x.data_type, x.shape,
           std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * x.element_size()));
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
    for (int64_t i = 0; i < n; ++i) {
      py[i] = std::fabs(px[i]);
    }
    return;
  }
  case DataType::DOUBLE: {
    const double *px = x.AsDouble();
    double *py = output.AsDouble();
    for (int64_t i = 0; i < n; ++i) {
      py[i] = std::fabs(px[i]);
    }
    return;
  }
  case DataType::FLOAT16:
    kernel::detail::UnaryHalfElementwise(x, output, Float16BitsToFloat, FloatToFloat16Bits,
                                         [](float v) { return std::fabs(v); });
    return;
  case DataType::BFLOAT16:
    kernel::detail::UnaryHalfElementwise(x, output, Bfloat16BitsToFloat, FloatToBfloat16Bits,
                                         [](float v) { return std::fabs(v); });
    return;
  case DataType::INT8: {
    const int8_t *px = x.AsInt8();
    int8_t *py = output.AsInt8();
    for (int64_t i = 0; i < n; ++i) {
      const int32_t v = static_cast<int32_t>(px[i]);
      py[i] = static_cast<int8_t>(v < 0 ? -v : v);
    }
    return;
  }
  case DataType::INT16: {
    const int16_t *px = x.AsInt16();
    int16_t *py = output.AsInt16();
    for (int64_t i = 0; i < n; ++i) {
      const int32_t v = static_cast<int32_t>(px[i]);
      py[i] = static_cast<int16_t>(v < 0 ? -v : v);
    }
    return;
  }
  case DataType::INT32: {
    const int32_t *px = x.AsInt32();
    int32_t *py = output.AsInt32();
    for (int64_t i = 0; i < n; ++i) {
      const int64_t v = static_cast<int64_t>(px[i]);
      py[i] = static_cast<int32_t>(v < 0 ? -v : v);
    }
    return;
  }
  case DataType::INT64: {
    const int64_t *px = x.AsInt64();
    int64_t *py = output.AsInt64();
    for (int64_t i = 0; i < n; ++i) {
      const uint64_t u = static_cast<uint64_t>(px[i]);
      py[i] = static_cast<int64_t>(px[i] < 0 ? (~u + 1) : u);
    }
    return;
  }
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, DOUBLE, FLOAT16, BFLOAT16, INT8, INT16, INT32, and "
                      "INT64 tensors.");
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
