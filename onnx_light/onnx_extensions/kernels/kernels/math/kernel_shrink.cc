// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::Shrink";

template <typename T> void ComputeInPlace(const Tensor &x, T bias, T lambd, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  const T neg_lambd = -lambd;
  for (int64_t i = 0; i < n; ++i) {
    const T v = px[i];
    if (v < neg_lambd) {
      py[i] = v + bias;
    } else if (v > lambd) {
      py[i] = v - bias;
    } else {
      py[i] = static_cast<T>(0);
    }
  }
}

void Dispatch(const Tensor &x, float bias, float lambd, Tensor &output) {
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    ComputeInPlace<float>(x, bias, lambd, output);
    return;
  case DataType::DOUBLE:
    ComputeInPlace<double>(x, static_cast<double>(bias), static_cast<double>(lambd), output);
    return;
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT and DOUBLE tensors.");
  }
}

void ValidateOutput(const Tensor &x, const Tensor &output) {
  EXT_ENFORCE_INVALID(output.data_type == x.data_type, kName,
                      ": output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape, kName, ": output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() == x.size_bytes(), kName,
                      ": output buffer size mismatch.");
}

} // namespace

Tensor Shrink::operator()(const Tensor &x, float bias, float lambd, RuntimeContext *rt) const {
  const size_t out_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor out = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(x.data_type, x.shape, out_n_bytes, nullptr);
  Dispatch(x, bias, lambd, out);
  return out;
}

void Shrink::operator()(const Tensor &x, float bias, float lambd, Tensor &output) const {
  ValidateOutput(x, output);
  Dispatch(x, bias, lambd, output);
}

void Shrink::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const float bias = GetAttributeFloatOrDefault(node, "bias", 0.0f);
  const float lambd = GetAttributeFloatOrDefault(node, "lambd", 0.5f);
  onnx_kernels::kernel::Shrink k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, bias, lambd, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
