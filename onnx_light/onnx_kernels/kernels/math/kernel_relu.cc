// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

constexpr const char *kName = "kernel::Relu";

template <typename T> void ComputeInPlace(const Tensor &x, Tensor &output) {
  const int64_t n = x.element_count();
  const T *px = reinterpret_cast<const T *>(x.bytes());
  T *py = reinterpret_cast<T *>(output.data.data());
  for (int64_t i = 0; i < n; ++i) {
    const T v = px[i];
    py[i] = v > static_cast<T>(0) ? v : static_cast<T>(0);
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
  case DataType::INT8:
    ComputeInPlace<int8_t>(x, output);
    return;
  case DataType::INT16:
    ComputeInPlace<int16_t>(x, output);
    return;
  case DataType::INT32:
    ComputeInPlace<int32_t>(x, output);
    return;
  case DataType::INT64:
    ComputeInPlace<int64_t>(x, output);
    return;
  default:
    throw std::invalid_argument(std::string(kName) +
                                " only supports FLOAT, DOUBLE and signed integer tensors.");
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

Tensor Relu::operator()(const Tensor &x) const {
  Tensor out("", x.data_type, x.shape,
             std::vector<uint8_t>(static_cast<size_t>(x.element_count()) * x.element_size()));
  Dispatch(x, out);
  return out;
}

void Relu::operator()(const Tensor &x, Tensor &output) const {
  ValidateOutput(x, output);
  Dispatch(x, output);
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
