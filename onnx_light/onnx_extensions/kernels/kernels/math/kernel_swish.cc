// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/kernels/cast_helper.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::Swish";

template <typename T> void ComputeInPlace(const Tensor &x, T alpha, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  for (int64_t i = 0; i < n; ++i) {
    const T v = px[i];
    const T s = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-alpha * v));
    py[i] = v * s;
  }
}

using DecodeFunc = float (*)(uint16_t);
using EncodeFunc = uint16_t (*)(float);

void ComputeHalf(const Tensor &x, float alpha, Tensor &output, DecodeFunc decode,
                 EncodeFunc encode) {
  const int64_t n = x.element_count();
  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
  for (int64_t i = 0; i < n; ++i) {
    const float v = decode(px[i]);
    const float s = 1.0f / (1.0f + std::exp(-alpha * v));
    py[i] = encode(v * s);
  }
}

void Dispatch(const Tensor &x, float alpha, Tensor &output) {
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    ComputeInPlace<float>(x, alpha, output);
    return;
  case DataType::DOUBLE:
    ComputeInPlace<double>(x, static_cast<double>(alpha), output);
    return;
  case DataType::FLOAT16:
    ComputeHalf(x, alpha, output, Float16BitsToFloat, FloatToFloat16Bits);
    return;
  case DataType::BFLOAT16:
    ComputeHalf(x, alpha, output, Bfloat16BitsToFloat, FloatToBfloat16Bits);
    return;
  default:
    EXT_THROW_INVALID(kName, ": unsupported data type ", x.data_type,
                      ", only supports FLOAT, DOUBLE, FLOAT16, and BFLOAT16 tensors.");
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

Tensor Swish::operator()(const Tensor &x, float alpha, RuntimeContext *rt) const {
  const size_t out_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor out = rt ? rt->MakeOutputTensor(0, x.data_type, x.shape, out_n_bytes)
                  : MakeOutputTensor(x.data_type, x.shape, out_n_bytes, nullptr);
  Dispatch(x, alpha, out);
  return out;
}

void Swish::operator()(const Tensor &x, float alpha, Tensor &output) const {
  ValidateOutput(x, output);
  Dispatch(x, alpha, output);
}

void Swish::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const float alpha = GetAttributeFloatOrDefault(node, "alpha", 1.0f);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  SetOutput(node, 0, (*this)(x, alpha, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
