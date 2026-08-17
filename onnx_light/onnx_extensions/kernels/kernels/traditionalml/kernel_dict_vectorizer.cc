// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

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
void FillImpl(std::span<const K> input_keys, std::span<const V> input_values,
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
void Fill(std::span<const K> input_keys, std::span<const V> input_values,
          const std::vector<K> &vocabulary, V *out) {
  FillImpl(input_keys, input_values, vocabulary,
           [out](size_t pos, const V &value) { out[pos] = value; });
}

} // namespace

template <typename K, typename V>
Tensor DictVectorizer::operator()(std::span<const K> input_keys, std::span<const V> input_values,
                                  const std::vector<K> &vocabulary, RuntimeContext *rt) const {
  const int64_t c = static_cast<int64_t>(vocabulary.size());
  const onnx_kernels::Shape shape{c};
  if constexpr (std::is_same_v<V, std::string>) {
    Tensor out = rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::STRING), shape, 0)
                    : Tensor::FromStrings("", shape, std::vector<std::string>(vocabulary.size()));
    out.string_data.assign(vocabulary.size(), std::string());
    FillImpl(input_keys, input_values, vocabulary,
             [&out](size_t pos, const std::string &value) { out.string_data[pos] = value; });
    return out;
  } else {
    const size_t n_bytes = static_cast<size_t>(c) * sizeof(V);
    Tensor out = rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(TensorElementType<V>::value),
                                           shape, n_bytes)
                    : MakeOutputTensor(static_cast<int32_t>(TensorElementType<V>::value), shape,
                                       n_bytes, ctx_.allocator);
    // ``Fill`` writes only the vocabulary positions present in the input and
    // leaves the rest untouched, so the output must start zeroed. Allocator
    // storage is no longer zero-initialised, so clear it explicitly.
    std::fill(out.mutable_bytes(), out.mutable_bytes() + out.size_bytes(), uint8_t{0u});
    Fill<K, V>(input_keys, input_values, vocabulary, reinterpret_cast<V *>(out.mutable_bytes()));
    return out;
  }
}

template <typename K, typename V>
void DictVectorizer::operator()(std::span<const K> input_keys, std::span<const V> input_values,
                                const std::vector<K> &vocabulary, Tensor &output) const {
  const int64_t c = static_cast<int64_t>(vocabulary.size());
  if constexpr (std::is_same_v<V, std::string>) {
    EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::STRING),
                        "kernel::DictVectorizer preallocated output dtype must be STRING.");
    EXT_ENFORCE_INVALID(output.shape == onnx_kernels::Shape{c},
                        "kernel::DictVectorizer preallocated output shape must be [vocab_size].");
    output.string_data.assign(static_cast<size_t>(c), std::string());
    FillImpl(input_keys, input_values, vocabulary,
             [&output](size_t pos, const std::string &value) { output.string_data[pos] = value; });
  } else {
    EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(TensorElementType<V>::value),
                        "kernel::DictVectorizer preallocated output dtype must match value type.");
    EXT_ENFORCE_INVALID(output.shape == onnx_kernels::Shape{c},
                        "kernel::DictVectorizer preallocated output shape must be [vocab_size].");
    EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<size_t>(c) * sizeof(V),
                        "kernel::DictVectorizer preallocated output buffer is incorrectly sized.");
    std::fill(output.mutable_bytes(), output.mutable_bytes() + output.size_bytes(), uint8_t{0u});
    Fill<K, V>(input_keys, input_values, vocabulary, reinterpret_cast<V *>(output.mutable_bytes()));
  }
}

// Explicit instantiations for the supported (K, V) pairs.
#define ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(K, V)                                               \
  template Tensor DictVectorizer::operator()(std::span<const K>, std::span<const V>,               \
                                             const std::vector<K> &, RuntimeContext *) const;      \
  template void DictVectorizer::operator()(std::span<const K>, std::span<const V>,                 \
                                           const std::vector<K> &, Tensor &) const

ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(std::string, int64_t);
ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(int64_t, std::string);
ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(int64_t, float);
ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(int64_t, double);
ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(std::string, float);
ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER(std::string, double);

#undef ONNX_LIGHT_INSTANTIATE_DICT_VECTORIZER

void DictVectorizer::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const std::string map_input = node.input(0);

  EXT_ENFORCE_INVALID(rt.HasMap(map_input), "RunNode: DictVectorizer map input '", map_input,
                      "' not found in the runtime context. Map-typed inputs "
                      "must be stored via PutMap before executing the graph.");
  const Map &dict_m = rt.GetMap(map_input);
  const Tensor &x_keys = dict_m.keys;
  const Tensor &x_values = dict_m.values;
  const AttributeProto *str_vocab = FindAttribute(node, "string_vocabulary");
  const AttributeProto *int_vocab = FindAttribute(node, "int64_vocabulary");
  const bool has_str = str_vocab != nullptr && str_vocab->strings_size() > 0;
  const bool has_int = int_vocab != nullptr && int_vocab->ints_size() > 0;
  EXT_ENFORCE_INVALID(has_str != has_int,
                      "RunNode: DictVectorizer requires exactly one of 'string_vocabulary' or "
                      "'int64_vocabulary' to be specified and non-empty.");
  onnx_kernels::kernel::DictVectorizer dict(rt.kernel_ctx());
  Tensor y;
  if (has_str) {
    EXT_ENFORCE_INVALID(!(x_keys.data_type != static_cast<int32_t>(DataType::STRING)),
                        "RunNode: DictVectorizer keys must be a STRING tensor when "
                        "'string_vocabulary' is set.");
    const std::vector<std::string> &keys = x_keys.AsStrings();
    std::vector<std::string> vocab;
    vocab.reserve(str_vocab->strings_size());
    for (int i = 0; i < static_cast<int>(str_vocab->strings_size()); ++i) {
      vocab.emplace_back(str_vocab->strings(i));
    }
    switch (x_values.data_type) {
    case static_cast<int32_t>(DataType::INT64): {
      y = dict.operator()<std::string, int64_t>(keys, TensorSpan<int64_t>(x_values), vocab, &rt);
      break;
    }
    case static_cast<int32_t>(DataType::FLOAT): {
      y = dict.operator()<std::string, float>(keys, TensorSpan<float>(x_values), vocab, &rt);
      break;
    }
    case static_cast<int32_t>(DataType::DOUBLE): {
      y = dict.operator()<std::string, double>(keys, TensorSpan<double>(x_values), vocab, &rt);
      break;
    }
    default:
      EXT_THROW_INVALID("RunNode: DictVectorizer values must be INT64, FLOAT, or DOUBLE when "
                        "'string_vocabulary' is set.");
    }
  } else {
    EXT_ENFORCE_INVALID(!(x_keys.data_type != static_cast<int32_t>(DataType::INT64)),
                        "RunNode: DictVectorizer keys must be an INT64 tensor when "
                        "'int64_vocabulary' is set.");
    const std::span<const int64_t> keys = TensorSpan<int64_t>(x_keys);
    std::vector<int64_t> vocab;
    vocab.reserve(int_vocab->ints_size());
    for (int i = 0; i < static_cast<int>(int_vocab->ints_size()); ++i) {
      vocab.push_back(int_vocab->ints(i));
    }
    switch (x_values.data_type) {
    case static_cast<int32_t>(DataType::FLOAT): {
      y = dict.operator()<int64_t, float>(keys, TensorSpan<float>(x_values), vocab, &rt);
      break;
    }
    case static_cast<int32_t>(DataType::DOUBLE): {
      y = dict.operator()<int64_t, double>(keys, TensorSpan<double>(x_values), vocab, &rt);
      break;
    }
    case static_cast<int32_t>(DataType::STRING): {
      const std::vector<std::string> &values = x_values.AsStrings();
      y = dict.operator()<int64_t, std::string>(keys, values, vocab, &rt);
      break;
    }
    default:
      EXT_THROW_INVALID("RunNode: DictVectorizer values must be FLOAT, DOUBLE, or STRING when "
                        "'int64_vocabulary' is set.");
    }
  }
  SetOutput(node, 0, std::move(y), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
