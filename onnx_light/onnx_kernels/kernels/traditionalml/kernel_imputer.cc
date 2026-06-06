// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

template <typename T> bool ImputerMatches(T value, T replaced) { return value == replaced; }

// Specialization for float: treat NaN as matching NaN.
template <> bool ImputerMatches<float>(float value, float replaced) {
  if (std::isnan(replaced)) {
    return std::isnan(value);
  }
  return value == replaced;
}

// Specialization for double: treat NaN as matching NaN.
template <> bool ImputerMatches<double>(double value, double replaced) {
  if (std::isnan(replaced)) {
    return std::isnan(value);
  }
  return value == replaced;
}

template <typename T> void ValidateInput(const Tensor &x) {
  EXT_ENFORCE_INVALID(x.data_type == TensorElementType<T>::value,
                      "kernel::Imputer input data_type does not match the requested element T.");
}

template <typename T>
void ValidateImputedValues(const std::vector<T> &imputed_values, int64_t last_dim) {
  EXT_ENFORCE_INVALID(!imputed_values.empty(),
                      "kernel::Imputer requires non-empty 'imputed_values'.");
  EXT_ENFORCE_INVALID(
      imputed_values.size() == 1u || static_cast<int64_t>(imputed_values.size()) == last_dim,
      "kernel::Imputer requires 'imputed_values' length to be 1 or to match the size of "
      "the last dimension of the input.");
}

int64_t LastDim(const std::vector<int64_t> &shape) { return shape.empty() ? 1 : shape.back(); }

template <typename T>
void ApplyImputer(const Tensor &x, const std::vector<T> &imputed_values, T replaced_value, T *out) {
  const T *px = x.As<T>();
  const int64_t n = x.element_count();
  const int64_t stride = static_cast<int64_t>(imputed_values.size());
  if (stride == 1) {
    const T imputed = imputed_values[0];
    for (int64_t i = 0; i < n; ++i) {
      out[i] = ImputerMatches(px[i], replaced_value) ? imputed : px[i];
    }
  } else {
    for (int64_t i = 0; i < n; ++i) {
      const int64_t k = i % stride;
      out[i] = ImputerMatches(px[i], replaced_value) ? imputed_values[k] : px[i];
    }
  }
}

} // namespace

template <typename T>
Tensor Imputer::operator()(const Tensor &x, const std::vector<T> &imputed_values,
                           T replaced_value) const {
  ValidateInput<T>(x);
  ValidateImputedValues<T>(imputed_values, LastDim(x.shape));
  const int64_t n = x.element_count();
  std::vector<uint8_t> bytes(static_cast<size_t>(n) * sizeof(T));
  Tensor out("", TensorElementType<T>::value, x.shape, std::move(bytes));
  ApplyImputer<T>(x, imputed_values, replaced_value, reinterpret_cast<T *>(out.data.data()));
  return out;
}

template <typename T>
void Imputer::operator()(const Tensor &x, const std::vector<T> &imputed_values, T replaced_value,
                         Tensor &output) const {
  ValidateInput<T>(x);
  ValidateImputedValues<T>(imputed_values, LastDim(x.shape));
  EXT_ENFORCE_INVALID(output.data_type == TensorElementType<T>::value,
                      "kernel::Imputer preallocated output dtype must match the input dtype.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::Imputer preallocated output shape must match the input shape.");
  EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(x.element_count()) * sizeof(T),
                      "kernel::Imputer preallocated output buffer is incorrectly sized.");
  ApplyImputer<T>(x, imputed_values, replaced_value, output.As<T>());
}

// Explicit instantiations for the supported element types.
#define ONNX_LIGHT_INSTANTIATE_IMPUTER(T)                                                          \
  template Tensor Imputer::operator()(const Tensor &, const std::vector<T> &, T) const;            \
  template void Imputer::operator()(const Tensor &, const std::vector<T> &, T, Tensor &) const

ONNX_LIGHT_INSTANTIATE_IMPUTER(float);
ONNX_LIGHT_INSTANTIATE_IMPUTER(double);
ONNX_LIGHT_INSTANTIATE_IMPUTER(int64_t);
ONNX_LIGHT_INSTANTIATE_IMPUTER(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_IMPUTER

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
