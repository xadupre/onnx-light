// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kName = "kernel::Gelu";

template <typename T> void ComputeExact(const Tensor &x, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.data.data());
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
  T *py = reinterpret_cast<T *>(output.data.data());
  // sqrt(2/pi)
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
  default:
    throw std::invalid_argument(std::string(kName) + " only supports FLOAT and DOUBLE tensors.");
  }
}

void ValidateOutput(const Tensor &x, const Tensor &output) {
  EXT_ENFORCE_INVALID(output.data_type == x.data_type,
                      std::string(kName) + ": output dtype must match input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      std::string(kName) + ": output shape must match input shape.");
  EXT_ENFORCE_INVALID(output.data.size() == x.data.size(),
                      std::string(kName) + ": output buffer size mismatch.");
}

void ValidateAttribute(const std::string &approximate) {
  EXT_ENFORCE_INVALID(approximate == "none" || approximate == "tanh",
                      std::string(kName) + ": 'approximate' must be 'none' or 'tanh'.");
}

} // namespace

Tensor Gelu::operator()(const Tensor &x, const std::string &approximate) const {
  ValidateAttribute(approximate);
  Tensor out("", x.data_type, x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * x.element_size()));
  Dispatch(x, approximate, out);
  return out;
}

void Gelu::operator()(const Tensor &x, const std::string &approximate, Tensor &output) const {
  ValidateAttribute(approximate);
  ValidateOutput(x, output);
  Dispatch(x, approximate, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
