// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include "onnx_kernels/kernels/_helpers/cast_helper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "onnx_kernels/runtime_context.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kName = "kernel::Celu";

template <typename T> void ComputeInPlace(const Tensor &x, T alpha, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.mutable_bytes());
  for (int64_t i = 0; i < n; ++i) {
    const T v = px[i];
    // max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
    const T pos = std::max(static_cast<T>(0), v);
    const T neg = std::min(static_cast<T>(0), alpha * (std::exp(v / alpha) - static_cast<T>(1)));
    py[i] = pos + neg;
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
    const float pos = std::max(0.0f, v);
    const float neg = std::min(0.0f, alpha * (std::exp(v / alpha) - 1.0f));
    py[i] = encode(pos + neg);
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
  EXT_ENFORCE_INVALID(output.size_bytes() == x.size_bytes(), kName, ": output buffer size mismatch.");
}

} // namespace

Tensor Celu::operator()(const Tensor &x, float alpha, RuntimeContext *rt) const {
  Tensor out("", x.data_type, x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * x.element_size()));
  Dispatch(x, alpha, out);
  return out;
}

void Celu::operator()(const Tensor &x, float alpha, Tensor &output) const {
  ValidateOutput(x, output);
  Dispatch(x, alpha, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
