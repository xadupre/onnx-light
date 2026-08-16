// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor IsNaN::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count());
  Tensor y = MakeOutputTensor(DataType::BOOL, x.shape, y_n_bytes, rt ? rt->allocator() : nullptr);
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
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes,
                      "kernel::IsNaN preallocated output buffer has unexpected size in bytes.");
  uint8_t *py = output.mutable_bytes();
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
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
    for (int64_t i = 0; i < n; ++i) {
      py[static_cast<size_t>(i)] = std::isnan(Float16BitsToFloat(px[i])) ? 1u : 0u;
    }
    return;
  }
  case DataType::BFLOAT16: {
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
    for (int64_t i = 0; i < n; ++i) {
      py[static_cast<size_t>(i)] = std::isnan(Bfloat16BitsToFloat(px[i])) ? 1u : 0u;
    }
    return;
  }
  default:
    EXT_THROW_INVALID("unsupported data type ", x.data_type, ", ",
                      "kernel::IsNaN only supports FLOAT, DOUBLE, FLOAT16 and BFLOAT16 tensors.");
  }
}

void IsNaN::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
