// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/cast_helper.h"

#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr const char *kName = "kernel::Gelu";

template <typename T> void ComputeExact(const Tensor &x, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  const T inv_sqrt2 = static_cast<T>(1.0L / 1.4142135623730951L);
  const T half = static_cast<T>(0.5);
  const T one = static_cast<T>(1);
  for (int64_t i = 0; i < n; ++i) {
    const T v = px[i];
    py[i] = half * v * (one + std::erf(v * inv_sqrt2));
  }
}

template <typename T> void ComputeTanh(const Tensor &x, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  const T sqrt_2_over_pi = static_cast<T>(0.7978845608028654L);
  const T c0 = static_cast<T>(0.044715);
  const T half = static_cast<T>(0.5);
  const T one = static_cast<T>(1);
  for (int64_t i = 0; i < n; ++i) {
    const T v = px[i];
    const T inner = sqrt_2_over_pi * (v + c0 * v * v * v);
    py[i] = half * v * (one + std::tanh(inner));
  }
}

using DecodeFunc = float (*)(uint16_t);
using EncodeFunc = uint16_t (*)(float);

void ComputeHalfExact(const Tensor &x, Tensor &output, DecodeFunc decode, EncodeFunc encode) {
  const int64_t n = x.element_count();
  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
  constexpr float inv_sqrt2 = 1.0f / 1.4142135623730951f;
  for (int64_t i = 0; i < n; ++i) {
    const float v = decode(px[i]);
    py[i] = encode(0.5f * v * (1.0f + std::erf(v * inv_sqrt2)));
  }
}

void ComputeHalfTanh(const Tensor &x, Tensor &output, DecodeFunc decode, EncodeFunc encode) {
  const int64_t n = x.element_count();
  const uint16_t *px = reinterpret_cast<const uint16_t *>(x.bytes());
  uint16_t *py = reinterpret_cast<uint16_t *>(output.mutable_bytes());
  constexpr float sqrt_2_over_pi = 0.7978845608028654f;
  constexpr float c0 = 0.044715f;
  for (int64_t i = 0; i < n; ++i) {
    const float v = decode(px[i]);
    const float inner = sqrt_2_over_pi * (v + c0 * v * v * v);
    py[i] = encode(0.5f * v * (1.0f + std::tanh(inner)));
  }
}

void Dispatch(const Tensor &x, const std::string &approximate, Tensor &output) {
  const bool tanh_approx = (approximate == "tanh");
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    if (tanh_approx)
      ComputeTanh<float>(x, output);
    else
      ComputeExact<float>(x, output);
    return;
  case DataType::DOUBLE:
    if (tanh_approx)
      ComputeTanh<double>(x, output);
    else
      ComputeExact<double>(x, output);
    return;
  case DataType::FLOAT16:
    if (tanh_approx)
      ComputeHalfTanh(x, output, Float16BitsToFloat, FloatToFloat16Bits);
    else
      ComputeHalfExact(x, output, Float16BitsToFloat, FloatToFloat16Bits);
    return;
  case DataType::BFLOAT16:
    if (tanh_approx)
      ComputeHalfTanh(x, output, Bfloat16BitsToFloat, FloatToBfloat16Bits);
    else
      ComputeHalfExact(x, output, Bfloat16BitsToFloat, FloatToBfloat16Bits);
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

void ValidateAttribute(const std::string &approximate) {
  EXT_ENFORCE_INVALID(approximate == "none" || approximate == "tanh", kName,
                      ": 'approximate' must be 'none' or 'tanh'.");
}

} // namespace

Tensor Gelu::operator()(const Tensor &x, const std::string &approximate, RuntimeContext *rt) const {
  ValidateAttribute(approximate);
  const size_t out_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor out = MakeOutputTensor(x.data_type, x.shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  Dispatch(x, approximate, out);
  return out;
}

void Gelu::operator()(const Tensor &x, const std::string &approximate, Tensor &output) const {
  ValidateAttribute(approximate);
  ValidateOutput(x, output);
  Dispatch(x, approximate, output);
}

void Gelu::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  const std::string approximate = GetAttributeStringOrDefault(node, "approximate", "none");
  onnx_kernels::kernel::Gelu k(rt.kernel_ctx());
  SetOutput(node, 0, k(x, approximate, &rt), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
