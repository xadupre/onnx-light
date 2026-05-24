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

template <typename KeyT> int32_t KeyDataType() noexcept { return TensorElementType<KeyT>::value; }

template <typename KeyT, typename ValueT>
void LookupAndFill(const Tensor &x, const std::vector<KeyT> &keys,
                   const std::vector<ValueT> &values, ValueT default_value, ValueT *out) {
  const KeyT *px = x.As<KeyT>();
  const int64_t n = x.element_count();
  const size_t k = keys.size();
  for (int64_t i = 0; i < n; ++i) {
    ValueT mapped = default_value;
    for (size_t j = 0; j < k; ++j) {
      if (px[i] == keys[j]) {
        mapped = values[j];
        break;
      }
    }
    out[i] = mapped;
  }
}

template <typename KeyT, typename ValueT>
void ValidateInputs(const Tensor &x, const std::vector<KeyT> &keys,
                    const std::vector<ValueT> &values) {
  if (x.data_type != KeyDataType<KeyT>()) {
    throw std::invalid_argument(
        "kernel::LabelEncoder input data_type does not match the requested KeyT.");
  }
  if (keys.size() != values.size()) {
    throw std::invalid_argument(
        "kernel::LabelEncoder requires keys and values to have the same length.");
  }
}

} // namespace

template <typename KeyT, typename ValueT>
Tensor LabelEncoder::operator()(const Tensor &x, const std::vector<KeyT> &keys,
                                const std::vector<ValueT> &values, ValueT default_value) const {
  ValidateInputs<KeyT, ValueT>(x, keys, values);
  const int64_t n = x.element_count();
  std::vector<uint8_t> bytes(static_cast<size_t>(n) * sizeof(ValueT));
  Tensor out("", TensorElementType<ValueT>::value, x.shape, std::move(bytes));
  LookupAndFill<KeyT, ValueT>(x, keys, values, default_value,
                              reinterpret_cast<ValueT *>(out.data.data()));
  return out;
}

template <typename KeyT, typename ValueT>
void LabelEncoder::operator()(const Tensor &x, const std::vector<KeyT> &keys,
                              const std::vector<ValueT> &values, ValueT default_value,
                              Tensor &output) const {
  ValidateInputs<KeyT, ValueT>(x, keys, values);
  if (output.data_type != TensorElementType<ValueT>::value) {
    throw std::invalid_argument(
        "kernel::LabelEncoder preallocated output dtype must match the requested ValueT.");
  }
  if (output.shape != x.shape) {
    throw std::invalid_argument(
        "kernel::LabelEncoder preallocated output shape must match the input shape.");
  }
  if (output.data.size() != static_cast<size_t>(x.element_count()) * sizeof(ValueT)) {
    throw std::invalid_argument(
        "kernel::LabelEncoder preallocated output buffer is incorrectly sized.");
  }
  LookupAndFill<KeyT, ValueT>(x, keys, values, default_value, output.As<ValueT>());
}

// Explicit instantiations for the supported (KeyT, ValueT) combinations.
#define ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER(KEY_T, VALUE_T)                                       \
  template Tensor LabelEncoder::operator()(const Tensor &, const std::vector<KEY_T> &,             \
                                           const std::vector<VALUE_T> &, VALUE_T) const;           \
  template void LabelEncoder::operator()(const Tensor &, const std::vector<KEY_T> &,               \
                                         const std::vector<VALUE_T> &, VALUE_T, Tensor &) const

ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER(int64_t, int64_t);
ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER(int64_t, float);
ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER(float, int64_t);
ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER(float, float);

#undef ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
