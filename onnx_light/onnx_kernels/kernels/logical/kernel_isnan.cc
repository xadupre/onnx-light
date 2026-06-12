// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/_helpers/cast_helper.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor IsNaN::operator()(const Tensor &x) const {
  Tensor y("", DataType::BOOL, x.shape,
           std::vector<uint8_t>(static_cast<size_t>(x.element_count())));
  (*this)(x, y);
  return y;
}

void IsNaN::operator()(const Tensor &x, Tensor &output) const {
  EXT_ENFORCE_INVALID(output.data_type == DataType::BOOL,
                      "kernel::IsNaN preallocated output must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::IsNaN preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n);
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      "kernel::IsNaN preallocated output buffer has unexpected size in bytes.");
  uint8_t *py = output.data.data();
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT: {
    const float *px = x.AsFloat();
    for (int64_t i = 0; i < n; ++i) {
      py[static_cast<size_t>(i)] = std::isnan(px[i]) ? 1u : 0u;
    }
    return;
  }
  case DataType::DOUBLE: {
    const double *px = x.AsDouble();
    for (int64_t i = 0; i < n; ++i) {
      py[static_cast<size_t>(i)] = std::isnan(px[i]) ? 1u : 0u;
    }
    return;
  }
  case DataType::FLOAT16: {
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.data.data());
    for (int64_t i = 0; i < n; ++i) {
      py[static_cast<size_t>(i)] = std::isnan(Float16BitsToFloat(px[i])) ? 1u : 0u;
    }
    return;
  }
  case DataType::BFLOAT16: {
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.data.data());
    for (int64_t i = 0; i < n; ++i) {
      py[static_cast<size_t>(i)] = std::isnan(Bfloat16BitsToFloat(px[i])) ? 1u : 0u;
    }
    return;
  }
  default:
    throw std::invalid_argument(
        "kernel::IsNaN only supports FLOAT, DOUBLE, FLOAT16 and BFLOAT16 tensors.");
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
