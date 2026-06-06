// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Convert one input map value (float or string) into the requested output
// element type.
template <typename V, typename OutT> OutT ConvertValue(const V &value);

template <> float ConvertValue<float, float>(const float &value) { return value; }
template <> int64_t ConvertValue<float, int64_t>(const float &value) {
  return static_cast<int64_t>(value);
}
template <> std::string ConvertValue<float, std::string>(const float &value) {
  return std::to_string(value);
}

template <> float ConvertValue<std::string, float>(const std::string &value) {
  return std::strtof(value.c_str(), nullptr);
}
template <> int64_t ConvertValue<std::string, int64_t>(const std::string &value) {
  return static_cast<int64_t>(std::strtoll(value.c_str(), nullptr, 10));
}
template <> std::string ConvertValue<std::string, std::string>(const std::string &value) {
  return value;
}

void ValidateAttributes(const std::vector<int64_t> &input_keys, std::size_t input_values_size,
                        const std::string &map_form, int64_t max_map) {
  EXT_ENFORCE_INVALID(input_keys.size() == input_values_size,
                      "kernel::CastMap: input keys/values must have the same length.");
  EXT_ENFORCE_INVALID(map_form == "DENSE" || map_form == "SPARSE",
                      "kernel::CastMap: 'map_form' must be 'DENSE' or 'SPARSE'.");
  if (map_form == "SPARSE") {
    EXT_ENFORCE_INVALID(max_map > 0,
                        "kernel::CastMap: 'max_map' must be strictly positive in 'SPARSE' mode.");
    for (int64_t key : input_keys) {
      EXT_ENFORCE_INVALID(key >= 0 && key < max_map,
                          "kernel::CastMap: input key out of range for the requested 'max_map'.");
    }
  }
}

int64_t OutputLength(const std::vector<int64_t> &input_keys, const std::string &map_form,
                     int64_t max_map) {
  return map_form == "SPARSE" ? max_map : static_cast<int64_t>(input_keys.size());
}

// Returns the input keys sorted in ascending order along with the original
// indices in ``input_values`` corresponding to each (sorted) key.
std::vector<std::pair<int64_t, std::size_t>>
SortKeysAscending(const std::vector<int64_t> &input_keys) {
  std::vector<std::pair<int64_t, std::size_t>> sorted;
  sorted.reserve(input_keys.size());
  for (std::size_t i = 0; i < input_keys.size(); ++i) {
    sorted.emplace_back(input_keys[i], i);
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const std::pair<int64_t, std::size_t> &a, const std::pair<int64_t, std::size_t> &b) {
              return a.first < b.first;
            });
  return sorted;
}

template <typename V, typename OutT>
void FillNumericOutput(const std::vector<int64_t> &input_keys, const std::vector<V> &input_values,
                       const std::string &map_form, OutT *out) {
  if (map_form == "SPARSE") {
    // Already zero-initialised by the caller; just scatter values by key.
    for (std::size_t i = 0; i < input_keys.size(); ++i) {
      out[static_cast<std::size_t>(input_keys[i])] = ConvertValue<V, OutT>(input_values[i]);
    }
  } else {
    const std::vector<std::pair<int64_t, std::size_t>> sorted = SortKeysAscending(input_keys);
    for (std::size_t i = 0; i < sorted.size(); ++i) {
      out[i] = ConvertValue<V, OutT>(input_values[sorted[i].second]);
    }
  }
}

template <typename V>
void FillStringOutput(const std::vector<int64_t> &input_keys, const std::vector<V> &input_values,
                      const std::string &map_form, std::vector<std::string> &out) {
  if (map_form == "SPARSE") {
    // ``out`` has already been resized and default-constructed by the caller.
    for (std::size_t i = 0; i < input_keys.size(); ++i) {
      out[static_cast<std::size_t>(input_keys[i])] = ConvertValue<V, std::string>(input_values[i]);
    }
  } else {
    const std::vector<std::pair<int64_t, std::size_t>> sorted = SortKeysAscending(input_keys);
    for (std::size_t i = 0; i < sorted.size(); ++i) {
      out[i] = ConvertValue<V, std::string>(input_values[sorted[i].second]);
    }
  }
}

} // namespace

template <typename V, typename OutT>
Tensor CastMap::operator()(const std::vector<int64_t> &input_keys,
                           const std::vector<V> &input_values, const std::string &cast_to,
                           const std::string &map_form, int64_t max_map) const {
  ValidateAttributes(input_keys, input_values.size(), map_form, max_map);
  (void)cast_to; // cast_to is encoded in OutT; only validated by the caller.

  const int64_t n = OutputLength(input_keys, map_form, max_map);
  const std::vector<int64_t> shape{n};

  if constexpr (std::is_same_v<OutT, std::string>) {
    Tensor out =
        Tensor::FromStrings("", shape, std::vector<std::string>(static_cast<std::size_t>(n)));
    FillStringOutput<V>(input_keys, input_values, map_form, out.string_data);
    return out;
  } else {
    std::vector<uint8_t> bytes(static_cast<std::size_t>(n) * sizeof(OutT), 0u);
    Tensor out("", static_cast<int32_t>(TensorElementType<OutT>::value), shape, std::move(bytes));
    FillNumericOutput<V, OutT>(input_keys, input_values, map_form,
                               reinterpret_cast<OutT *>(out.data.data()));
    return out;
  }
}

template <typename V, typename OutT>
void CastMap::operator()(const std::vector<int64_t> &input_keys, const std::vector<V> &input_values,
                         const std::string &cast_to, const std::string &map_form, int64_t max_map,
                         Tensor &output) const {
  ValidateAttributes(input_keys, input_values.size(), map_form, max_map);
  (void)cast_to;

  const int64_t n = OutputLength(input_keys, map_form, max_map);

  if constexpr (std::is_same_v<OutT, std::string>) {
    EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::STRING),
                        "kernel::CastMap preallocated output dtype must be STRING.");
    EXT_ENFORCE_INVALID(output.shape == std::vector<int64_t>{n},
                        "kernel::CastMap preallocated output shape must be [N].");
    output.string_data.assign(static_cast<std::size_t>(n), std::string());
    FillStringOutput<V>(input_keys, input_values, map_form, output.string_data);
  } else {
    EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(TensorElementType<OutT>::value),
                        "kernel::CastMap preallocated output dtype must match 'cast_to'.");
    EXT_ENFORCE_INVALID(output.shape == std::vector<int64_t>{n},
                        "kernel::CastMap preallocated output shape must be [N].");
    EXT_ENFORCE_INVALID(output.data.size() == static_cast<std::size_t>(n) * sizeof(OutT),
                        "kernel::CastMap preallocated output buffer is incorrectly sized.");
    std::fill(output.data.begin(), output.data.end(), uint8_t{0u});
    FillNumericOutput<V, OutT>(input_keys, input_values, map_form,
                               reinterpret_cast<OutT *>(output.data.data()));
  }
}

// Explicit instantiations for all supported (V, OutT) pairs.
#define ONNX_LIGHT_INSTANTIATE_CAST_MAP(V, OutT)                                                   \
  template Tensor CastMap::operator()<V, OutT>(const std::vector<int64_t> &,                       \
                                               const std::vector<V> &, const std::string &,        \
                                               const std::string &, int64_t) const;                \
  template void CastMap::operator()<V, OutT>(const std::vector<int64_t> &, const std::vector<V> &, \
                                             const std::string &, const std::string &, int64_t,    \
                                             Tensor &) const

ONNX_LIGHT_INSTANTIATE_CAST_MAP(float, float);
ONNX_LIGHT_INSTANTIATE_CAST_MAP(float, int64_t);
ONNX_LIGHT_INSTANTIATE_CAST_MAP(float, std::string);
ONNX_LIGHT_INSTANTIATE_CAST_MAP(std::string, float);
ONNX_LIGHT_INSTANTIATE_CAST_MAP(std::string, int64_t);
ONNX_LIGHT_INSTANTIATE_CAST_MAP(std::string, std::string);

#undef ONNX_LIGHT_INSTANTIATE_CAST_MAP

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
