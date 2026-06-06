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

std::vector<int64_t> ComputeOutputShape(const Tensor &x, const Tensor &indices) {
  EXT_ENFORCE_INVALID(!x.shape.empty(),
                      "kernel::ArrayFeatureExtractor expects input X to have rank >= 1.");
  std::vector<int64_t> out_shape = x.shape;
  out_shape.back() = indices.element_count();
  return out_shape;
}

void ValidateIndices(const Tensor &x, const Tensor &indices) {
  EXT_ENFORCE_INVALID(indices.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::ArrayFeatureExtractor indices input must be int64.");
  const int64_t last_dim = x.shape.back();
  EXT_ENFORCE_INVALID(last_dim >= 0, "kernel::ArrayFeatureExtractor input X has invalid shape.");
  const int64_t *py = indices.AsInt64();
  for (int64_t i = 0; i < indices.element_count(); ++i) {
    const int64_t index = py[i];
    EXT_ENFORCE_INVALID(index >= 0 && index < last_dim,
                        "kernel::ArrayFeatureExtractor index is out of bounds for the last axis.");
  }
}

int64_t OuterSize(const Tensor &x) {
  const int64_t last_dim = x.shape.back();
  EXT_ENFORCE_INVALID(last_dim >= 0, "kernel::ArrayFeatureExtractor input X has invalid shape.");
  if (last_dim == 0) {
    return 0;
  }
  return x.element_count() / last_dim;
}

template <typename T> void ValidateInput(const Tensor &x) {
  EXT_ENFORCE_INVALID(
      x.data_type == TensorElementType<T>::value,
      "kernel::ArrayFeatureExtractor input data_type does not match the requested element T.");
}

template <typename T>
void GatherLastAxis(const Tensor &x, const Tensor &indices, const std::vector<int64_t> &out_shape,
                    T *out) {
  const T *px = x.As<T>();
  const int64_t *py = indices.AsInt64();
  const int64_t last_dim = x.shape.back();
  const int64_t n_indices = indices.element_count();
  const int64_t outer = OuterSize(x);
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t j = 0; j < n_indices; ++j) {
      const int64_t index = py[j];
      out[o * out_shape.back() + j] = px[o * last_dim + index];
    }
  }
}

} // namespace

template <typename T>
Tensor ArrayFeatureExtractor::operator()(const Tensor &x, const Tensor &indices) const {
  ValidateInput<T>(x);
  ValidateIndices(x, indices);
  const std::vector<int64_t> out_shape = ComputeOutputShape(x, indices);
  const int64_t outer = OuterSize(x);
  std::vector<uint8_t> bytes(static_cast<size_t>(indices.element_count() * outer) * sizeof(T));
  Tensor out("", TensorElementType<T>::value, out_shape, std::move(bytes));
  GatherLastAxis<T>(x, indices, out_shape, out.As<T>());
  return out;
}

template <typename T>
void ArrayFeatureExtractor::operator()(const Tensor &x, const Tensor &indices,
                                       Tensor &output) const {
  ValidateInput<T>(x);
  ValidateIndices(x, indices);
  const std::vector<int64_t> expected_shape = ComputeOutputShape(x, indices);
  const int64_t outer = OuterSize(x);
  EXT_ENFORCE_INVALID(
      output.data_type == TensorElementType<T>::value,
      "kernel::ArrayFeatureExtractor preallocated output dtype must match X dtype.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::ArrayFeatureExtractor preallocated output shape is incorrect.");
  EXT_ENFORCE_INVALID(
      output.data.size() == static_cast<size_t>(indices.element_count() * outer) * sizeof(T),
      "kernel::ArrayFeatureExtractor preallocated output buffer is incorrectly sized.");
  GatherLastAxis<T>(x, indices, expected_shape, output.As<T>());
}

#define ONNX_LIGHT_INSTANTIATE_ARRAY_FEATURE_EXTRACTOR(T)                                          \
  template Tensor ArrayFeatureExtractor::operator()<T>(const Tensor &, const Tensor &) const;      \
  template void ArrayFeatureExtractor::operator()<T>(const Tensor &, const Tensor &, Tensor &) const

ONNX_LIGHT_INSTANTIATE_ARRAY_FEATURE_EXTRACTOR(float);
ONNX_LIGHT_INSTANTIATE_ARRAY_FEATURE_EXTRACTOR(double);
ONNX_LIGHT_INSTANTIATE_ARRAY_FEATURE_EXTRACTOR(int64_t);
ONNX_LIGHT_INSTANTIATE_ARRAY_FEATURE_EXTRACTOR(int32_t);

#undef ONNX_LIGHT_INSTANTIATE_ARRAY_FEATURE_EXTRACTOR

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
