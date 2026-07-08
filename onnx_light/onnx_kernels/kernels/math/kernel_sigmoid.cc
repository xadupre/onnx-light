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

constexpr const char *kName = "kernel::Sigmoid";

} // namespace

Tensor Sigmoid::operator()(const Tensor &x, RuntimeContext *rt) const {
  Tensor y("", x.data_type, x.shape,
           std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * x.element_size()));
  (*this)(x, y);
  return y;
}

void Sigmoid::operator()(const Tensor &x, Tensor &output) const {
  EXT_ENFORCE_INVALID(output.data_type == x.data_type, kName,
                      ": output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape, kName, ": output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == x.size_bytes(), kName, ": output buffer size mismatch.");
  const int64_t n = x.element_count();
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT: {
    const float *px = x.AsFloat();
    float *py = output.AsFloat();
    for (int64_t i = 0; i < n; ++i) {
      py[i] = 1.0f / (1.0f + std::exp(-px[i]));
    }
    return;
  }
  case DataType::DOUBLE: {
    const double *px = x.AsDouble();
    double *py = output.AsDouble();
    for (int64_t i = 0; i < n; ++i) {
      py[i] = 1.0 / (1.0 + std::exp(-px[i]));
    }
    return;
  }
  case DataType::FLOAT16:
    kernel::detail::UnaryHalfElementwise(x, output, Float16BitsToFloat, FloatToFloat16Bits,
                                         [](float v) { return 1.0f / (1.0f + std::exp(-v)); });
    return;
  case DataType::BFLOAT16:
    kernel::detail::UnaryHalfElementwise(x, output, Bfloat16BitsToFloat, FloatToBfloat16Bits,
                                         [](float v) { return 1.0f / (1.0f + std::exp(-v)); });
    return;
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, DOUBLE, FLOAT16, and BFLOAT16 tensors.");
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
