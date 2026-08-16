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

Tensor IsInf::operator()(const Tensor &x, int64_t detect_positive, int64_t detect_negative,
                         RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count());
  Tensor y = MakeOutputTensor(DataType::BOOL, x.shape, y_n_bytes, rt ? rt->allocator() : nullptr);
  (*this)(x, detect_positive, detect_negative, y);
  return y;
}

void IsInf::operator()(const Tensor &x, int64_t detect_positive, int64_t detect_negative,
                       Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT || x.data_type == DataType::DOUBLE ||
                          x.data_type == DataType::FLOAT16,
                      "kernel::IsInf only supports FLOAT, DOUBLE and FLOAT16 tensors.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::BOOL,
                      "kernel::IsInf preallocated output must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::IsInf preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n);
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes,
                      "kernel::IsInf preallocated output buffer has unexpected size in bytes.");
  const bool report_pos = detect_positive != 0;
  const bool report_neg = detect_negative != 0;
  uint8_t *py = output.mutable_bytes();

  if (x.data_type == DataType::FLOAT) {
    const float *px = x.AsFloat();
    for (int64_t i = 0; i < n; ++i) {
      const float v = px[i];
      uint8_t r = 0;
      if (std::isinf(v)) {
        r = (v > 0.0f) ? (report_pos ? 1u : 0u) : (report_neg ? 1u : 0u);
      }
      py[static_cast<size_t>(i)] = r;
    }
  } else if (x.data_type == DataType::DOUBLE) {
    const double *px = x.AsDouble();
    for (int64_t i = 0; i < n; ++i) {
      const double v = px[i];
      uint8_t r = 0;
      if (std::isinf(v)) {
        r = (v > 0.0) ? (report_pos ? 1u : 0u) : (report_neg ? 1u : 0u);
      }
      py[static_cast<size_t>(i)] = r;
    }
  } else {
    // FLOAT16 — decode to float first.
    const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
    for (int64_t i = 0; i < n; ++i) {
      const float v = Float16BitsToFloat(px[i]);
      uint8_t r = 0;
      if (std::isinf(v)) {
        r = (v > 0.0f) ? (report_pos ? 1u : 0u) : (report_neg ? 1u : 0u);
      }
      py[static_cast<size_t>(i)] = r;
    }
  }
}

void IsInf::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const int64_t detect_positive = GetAttributeIntOrDefault(node, "detect_positive", 1);
  const int64_t detect_negative = GetAttributeIntOrDefault(node, "detect_negative", 1);
  onnx_kernels::kernel::IsInf k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, detect_positive, detect_negative, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
