// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

template <typename T> void ApplyThreshold(const Tensor &x, T threshold, T *out) {
  const T *px = x.As<T>();
  const int64_t n = x.element_count();
  for (int64_t i = 0; i < n; ++i) {
    out[i] = px[i] > threshold ? static_cast<T>(1) : static_cast<T>(0);
  }
}

template <typename T> void ValidateInput(const Tensor &x) {
  EXT_ENFORCE_INVALID(x.data_type == TensorElementType<T>::value,
                      "kernel::Binarizer input data_type does not match the requested element T.");
}

} // namespace

template <typename T> Tensor Binarizer::operator()(const Tensor &x, T threshold) const {
  ValidateInput<T>(x);
  const int64_t n = x.element_count();
  std::vector<uint8_t> bytes(static_cast<size_t>(n) * sizeof(T));
  Tensor out("", TensorElementType<T>::value, x.shape, std::move(bytes));
  ApplyThreshold<T>(x, threshold, reinterpret_cast<T *>(out.data.data()));
  return out;
}

template <typename T>
void Binarizer::operator()(const Tensor &x, T threshold, Tensor &output) const {
  ValidateInput<T>(x);
  EXT_ENFORCE_INVALID(output.data_type == TensorElementType<T>::value,
                      "kernel::Binarizer preallocated output dtype must match the input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Binarizer preallocated output shape must match the input shape.");
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(x.element_count()) * sizeof(T),
                      "kernel::Binarizer preallocated output buffer is incorrectly sized.");
  ApplyThreshold<T>(x, threshold, output.As<T>());
}

// Explicit instantiations for the supported element types.
#define ONNX_LIGHT_INSTANTIATE_BINARIZER(T)                                                        \
  template Tensor Binarizer::operator()(const Tensor &, T) const;                                  \
  template void Binarizer::operator()(const Tensor &, T, Tensor &) const

ONNX_LIGHT_INSTANTIATE_BINARIZER(float);
ONNX_LIGHT_INSTANTIATE_BINARIZER(double);
ONNX_LIGHT_INSTANTIATE_BINARIZER(int64_t);
ONNX_LIGHT_INSTANTIATE_BINARIZER(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_BINARIZER

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
