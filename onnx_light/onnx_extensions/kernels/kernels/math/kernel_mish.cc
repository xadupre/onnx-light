// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kName = "kernel::Mish";

template <typename T> void ComputeInPlace(const Tensor &x, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  for (int64_t i = 0; i < n; ++i) {
    const T v = px[i];
    // Numerically stable softplus: log1p(exp(-|x|)) + max(x, 0).
    const T abs_x = std::fabs(v);
    const T sp = std::log1p(std::exp(-abs_x)) + std::fmax(v, static_cast<T>(0));
    py[i] = v * std::tanh(sp);
  }
}

void Dispatch(const Tensor &x, Tensor &output) {
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    ComputeInPlace<float>(x, output);
    return;
  case DataType::DOUBLE:
    ComputeInPlace<double>(x, output);
    return;
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT and DOUBLE tensors.");
  }
}

void ValidateOutput(const Tensor &x, const Tensor &output) {
  EXT_ENFORCE_INVALID(output.data_type == x.data_type, kName, ": unsupported data type ",
                      x.data_type, ", preallocated output must have the same dtype as input.");
  EXT_ENFORCE_INVALID(output.shape == x.shape, kName, ": unsupported data type ", x.data_type,
                      ", preallocated output shape must match input shape.");
  const size_t expected_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  EXT_ENFORCE_INVALID(output.size_bytes() == expected_bytes, kName, ": unsupported data type ",
                      x.data_type, ", preallocated output buffer has unexpected size in bytes.");
}

} // namespace

Tensor Mish::operator()(const Tensor &x, RuntimeContext *rt) const {
  const size_t out_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor out = MakeOutputTensor(x.data_type, x.shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  Dispatch(x, out);
  return out;
}

void Mish::operator()(const Tensor &x, Tensor &output) const {
  ValidateOutput(x, output);
  Dispatch(x, output);
}

void Mish::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, &rt), rt);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
