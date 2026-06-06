// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

template <typename K>
std::unordered_map<K, int64_t> BuildVocabIndex(const std::vector<K> &vocabulary) {
  std::unordered_map<K, int64_t> index;
  index.reserve(vocabulary.size());
  for (int64_t i = 0; i < static_cast<int64_t>(vocabulary.size()); ++i) {
    index.emplace(vocabulary[static_cast<size_t>(i)], i);
  }
  return index;
}

template <typename K, typename V, typename Writer>
void FillImpl(const std::vector<K> &input_keys, const std::vector<V> &input_values,
              const std::vector<K> &vocabulary, Writer write) {
  EXT_ENFORCE_INVALID(input_keys.size() == input_values.size(),
                      "kernel::DictVectorizer: keys/values must have the same length.");
  EXT_ENFORCE_INVALID(!vocabulary.empty(), "kernel::DictVectorizer: vocabulary must not be empty.");
  const std::unordered_map<K, int64_t> idx = BuildVocabIndex(vocabulary);
  for (size_t i = 0; i < input_keys.size(); ++i) {
    const auto it = idx.find(input_keys[i]);
    EXT_ENFORCE_INVALID(it != idx.end(),
                        "kernel::DictVectorizer: input key not present in the vocabulary.");
    write(static_cast<size_t>(it->second), input_values[i]);
  }
}

template <typename K, typename V>
void Fill(const std::vector<K> &input_keys, const std::vector<V> &input_values,
          const std::vector<K> &vocabulary, V *out) {
  FillImpl(input_keys, input_values, vocabulary,
           [out](size_t pos, const V &value) { out[pos] = value; });
}

} // namespace

template <typename K, typename V>
Tensor DictVectorizer::operator()(const std::vector<K> &input_keys,
                                  const std::vector<V> &input_values,
                                  const std::vector<K> &vocabulary) const {
  const int64_t c = static_cast<int64_t>(vocabulary.size());
  const std::vector<int64_t> shape{c};
  if constexpr (std::is_same_v<V, std::string>) {
    Tensor out = Tensor::FromStrings("", shape, std::vector<std::string>(vocabulary.size()));
    FillImpl(input_keys, input_values, vocabulary,
             [&out](size_t pos, const std::string &value) { out.string_data[pos] = value; });
    return out;
  } else {
    std::vector<uint8_t> bytes(static_cast<size_t>(c) * sizeof(V), 0u);
    Tensor out("", static_cast<int32_t>(TensorElementType<V>::value), shape, std::move(bytes));
    Fill<K, V>(input_keys, input_values, vocabulary, reinterpret_cast<V *>(out.data.data()));
    return out;
  }
}

template <typename K, typename V>
void DictVectorizer::operator()(const std::vector<K> &input_keys,
                                const std::vector<V> &input_values,
                                const std::vector<K> &vocabulary, Tensor &output) const {
  const int64_t c = static_cast<int64_t>(vocabulary.size());
  if constexpr (std::is_same_v<V, std::string>) {
    EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::STRING),
                        "kernel::DictVectorizer preallocated output dtype must be STRING.");
    EXT_ENFORCE_INVALID(output.shape == std::vector<int64_t>{c},
                        "kernel::DictVectorizer preallocated output shape must be [vocab_size].");
    output.string_data.assign(static_cast<size_t>(c), std::string());
    FillImpl(input_keys, input_values, vocabulary,
             [&output](size_t pos, const std::string &value) { output.string_data[pos] = value; });
  } else {
    EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(TensorElementType<V>::value),
                        "kernel::DictVectorizer preallocated output dtype must match value type.");
    EXT_ENFORCE_INVALID(output.shape == std::vector<int64_t>{c},
                        "kernel::DictVectorizer preallocated output shape must be [vocab_size].");
    EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(c) * sizeof(V),
                        "kernel::DictVectorizer preallocated output buffer is incorrectly sized.");
    std::fill(output.data.begin(), output.data.end(), uint8_t{0u});
    Fill<K, V>(input_keys, input_values, vocabulary, reinterpret_cast<V *>(output.data.data()));
  }
}

// Explicit instantiations for the supported (K, V) pairs.
#define ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(K, V)                                               \
  template Tensor DictVectorizer::operator()(const std::vector<K> &, const std::vector<V> &,       \
                                             const std::vector<K> &) const;                        \
  template void DictVectorizer::operator()(const std::vector<K> &, const std::vector<V> &,         \
                                           const std::vector<K> &, Tensor &) const

ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(std::string, int64_t);
ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(int64_t, std::string);
ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(int64_t, float);
ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(int64_t, double);
ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(std::string, float);
ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(std::string, double);

#undef ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
