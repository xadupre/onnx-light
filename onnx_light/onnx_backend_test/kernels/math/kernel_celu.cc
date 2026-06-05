// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/math/include_math_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

constexpr const char *kName = "kernel::Celu";

void ComputeInPlace(const Tensor &x, float alpha, Tensor &output) {
  const int64_t n = x.element_count();
  const float *px = reinterpret_cast<const float *>(x.data.data());
  float *py = reinterpret_cast<float *>(output.data.data());
  for (int64_t i = 0; i < n; ++i) {
    const float v = px[i];
    // max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
    const float pos = std::max(0.0f, v);
    const float neg = std::min(0.0f, alpha * (std::exp(v / alpha) - 1.0f));
    py[i] = pos + neg;
  }
}

void Dispatch(const Tensor &x, float alpha, Tensor &output) {
  switch (static_cast<DataType>(x.data_type)) {
  case DataType::FLOAT:
    ComputeInPlace(x, alpha, output);
    return;
  default:
    throw std::invalid_argument(std::string(kName) + " only supports FLOAT tensors.");
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

Tensor Celu::operator()(const Tensor &x, float alpha) const {
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
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
