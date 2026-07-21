// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/onnx_kernels/kernels/math/include_math_kernels.h"

#include "onnx_core/runtime/cast_helper.h"

#include "onnx_core/runtime/runtime_context.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kName = "kernel::Elu";

template <typename T> void ComputeInPlace(const Tensor &x, T alpha, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  for (int64_t i = 0; i < n; ++i) {
    const T v = px[i];
    py[i] = v < static_cast<T>(0) ? alpha * (std::exp(v) - static_cast<T>(1)) : v;
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
    py[i] = encode(v < 0.0f ? alpha * (std::exp(v) - 1.0f) : v);
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

Tensor Elu::operator()(const Tensor &x, float alpha, RuntimeContext *rt) const {
  const size_t out_n_bytes = static_cast<size_t>(x.element_count()) * x.element_size();
  Tensor out = MakeOutputTensor(x.data_type, x.shape, out_n_bytes, rt ? rt->allocator() : nullptr);
  Dispatch(x, alpha, out);
  return out;
}

void Elu::operator()(const Tensor &x, float alpha, Tensor &output) const {
  ValidateOutput(x, output);
  Dispatch(x, alpha, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
