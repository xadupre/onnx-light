// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

Tensor HardSigmoid::operator()(const Tensor &x, float alpha, float beta, RuntimeContext *rt) const {
  const size_t y_n_bytes = static_cast<size_t>(x.element_count()) * sizeof(float);
  Tensor y = rt ? rt->MakeOutputTensor(0, DataType::FLOAT, x.shape, y_n_bytes)
                : MakeOutputTensor(DataType::FLOAT, x.shape, y_n_bytes, nullptr);
  (*this)(x, alpha, beta, y);
  return y;
}

void HardSigmoid::operator()(const Tensor &x, float alpha, float beta, Tensor &output) const {
  EXT_ENFORCE_INVALID(x.data_type == DataType::FLOAT,
                      "kernel::HardSigmoid only supports FLOAT tensors.");
  EXT_ENFORCE_INVALID(output.data_type == DataType::FLOAT,
                      "kernel::HardSigmoid preallocated output must be a FLOAT tensor.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::HardSigmoid preallocated output shape must match input shape.");
  const int64_t n = x.element_count();
  const size_t expected_bytes = static_cast<size_t>(n) * sizeof(float);
  EXT_ENFORCE_INVALID(
      output.size_bytes() == expected_bytes,
      "kernel::HardSigmoid preallocated output buffer has unexpected size in bytes.");

  const float *px = x.AsFloat();
  float *py = output.AsFloat();
  for (int64_t i = 0; i < n; ++i) {
    const size_t idx = static_cast<size_t>(i);
    const float v = alpha * px[idx] + beta;
    py[idx] = std::max(0.0f, std::min(1.0f, v));
  }
}

void HardSigmoid::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const float alpha = GetAttributeFloatOrDefault(node, "alpha", 0.2f);
  const float beta = GetAttributeFloatOrDefault(node, "beta", 0.5f);
  onnx_kernels::kernel::HardSigmoid k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, alpha, beta, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
