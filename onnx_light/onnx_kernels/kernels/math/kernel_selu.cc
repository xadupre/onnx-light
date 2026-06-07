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

constexpr const char *kName = "kernel::Selu";

template <typename T> void ComputeInPlace(const Tensor &x, T alpha, T gamma, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.data.data());
  for (int64_t i = 0; i < n; ++i) {
    const T v = px[i];
    py[i] = v > static_cast<T>(0) ? gamma * v : gamma * (alpha * std::exp(v) - alpha);
  }
}

void Dispatch(const Tensor &x, float alpha, float gamma, Tensor &output) {
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    ComputeInPlace<float>(x, alpha, gamma, output);
    return;
  case DataType::DOUBLE:
    ComputeInPlace<double>(x, static_cast<double>(alpha), static_cast<double>(gamma), output);
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

} // namespace

Tensor Selu::operator()(const Tensor &x, float alpha, float gamma) const {
  Tensor out("", x.data_type, x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * x.element_size()));
  Dispatch(x, alpha, gamma, out);
  return out;
}

void Selu::operator()(const Tensor &x, float alpha, float gamma, Tensor &output) const {
  ValidateOutput(x, output);
  Dispatch(x, alpha, gamma, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
