// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

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

void ValidateAttributes(std::span<const int64_t> input_keys, std::size_t input_values_size,
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

int64_t OutputLength(std::span<const int64_t> input_keys, const std::string &map_form,
                     int64_t max_map) {
  return map_form == "SPARSE" ? max_map : static_cast<int64_t>(input_keys.size());
}

// Returns the input keys sorted in ascending order along with the original
// indices in ``input_values`` corresponding to each (sorted) key.
std::vector<std::pair<int64_t, std::size_t>>
SortKeysAscending(std::span<const int64_t> input_keys) {
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
void FillNumericOutput(std::span<const int64_t> input_keys, std::span<const V> input_values,
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
void FillStringOutput(std::span<const int64_t> input_keys, std::span<const V> input_values,
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
Tensor CastMap::operator()(std::span<const int64_t> input_keys, std::span<const V> input_values,
                           const std::string &cast_to, const std::string &map_form, int64_t max_map,
                           RuntimeContext * /*rt*/) const {
  ValidateAttributes(input_keys, input_values.size(), map_form, max_map);
  (void)cast_to; // cast_to is encoded in OutT; only validated by the caller.

  const int64_t n = OutputLength(input_keys, map_form, max_map);
  const onnx_kernels::Shape shape{n};

  if constexpr (std::is_same_v<OutT, std::string>) {
    Tensor out =
        Tensor::FromStrings("", shape, std::vector<std::string>(static_cast<std::size_t>(n)));
    FillStringOutput<V>(input_keys, input_values, map_form, out.string_data);
    return out;
  } else {
    Tensor out = MakeOutputTensor(static_cast<int32_t>(TensorElementType<OutT>::value), shape,
                                  static_cast<std::size_t>(n) * sizeof(OutT), ctx_.allocator);
    // ``FillNumericOutput`` scatters SPARSE entries by key and leaves the
    // remaining positions untouched, so the output must start zeroed. Allocator
    // storage is no longer zero-initialised, so clear it explicitly.
    std::fill(out.mutable_bytes(), out.mutable_bytes() + out.size_bytes(), uint8_t{0u});
    FillNumericOutput<V, OutT>(input_keys, input_values, map_form,
                               reinterpret_cast<OutT *>(out.mutable_bytes()));
    return out;
  }
}

template <typename V, typename OutT>
void CastMap::operator()(std::span<const int64_t> input_keys, std::span<const V> input_values,
                         const std::string &cast_to, const std::string &map_form, int64_t max_map,
                         Tensor &output) const {
  ValidateAttributes(input_keys, input_values.size(), map_form, max_map);
  (void)cast_to;

  const int64_t n = OutputLength(input_keys, map_form, max_map);

  if constexpr (std::is_same_v<OutT, std::string>) {
    EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::STRING),
                        "kernel::CastMap preallocated output dtype must be STRING.");
    EXT_ENFORCE_INVALID(output.shape == onnx_kernels::Shape{n},
                        "kernel::CastMap preallocated output shape must be [N].");
    output.string_data.assign(static_cast<std::size_t>(n), std::string());
    FillStringOutput<V>(input_keys, input_values, map_form, output.string_data);
  } else {
    EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(TensorElementType<OutT>::value),
                        "kernel::CastMap preallocated output dtype must match 'cast_to'.");
    EXT_ENFORCE_INVALID(output.shape == onnx_kernels::Shape{n},
                        "kernel::CastMap preallocated output shape must be [N].");
    EXT_ENFORCE_INVALID(output.size_bytes() == static_cast<std::size_t>(n) * sizeof(OutT),
                        "kernel::CastMap preallocated output buffer is incorrectly sized.");
    std::fill(output.mutable_bytes(), output.mutable_bytes() + output.size_bytes(), uint8_t{0u});
    FillNumericOutput<V, OutT>(input_keys, input_values, map_form,
                               reinterpret_cast<OutT *>(output.mutable_bytes()));
  }
}

// Explicit instantiations for all supported (V, OutT) pairs.
#define ONNX_LIGHT_INSTANTIATE_CAST_MAP(V, OutT)                                                   \
  template Tensor CastMap::operator()<V, OutT>(std::span<const int64_t>, std::span<const V>,       \
                                               const std::string &, const std::string &, int64_t,  \
                                               RuntimeContext *) const;                            \
  template void CastMap::operator()<V, OutT>(std::span<const int64_t>, std::span<const V>,         \
                                             const std::string &, const std::string &, int64_t,    \
                                             Tensor &) const

ONNX_LIGHT_INSTANTIATE_CAST_MAP(float, float);
ONNX_LIGHT_INSTANTIATE_CAST_MAP(float, int64_t);
ONNX_LIGHT_INSTANTIATE_CAST_MAP(float, std::string);
ONNX_LIGHT_INSTANTIATE_CAST_MAP(std::string, float);
ONNX_LIGHT_INSTANTIATE_CAST_MAP(std::string, int64_t);
ONNX_LIGHT_INSTANTIATE_CAST_MAP(std::string, std::string);

#undef ONNX_LIGHT_INSTANTIATE_CAST_MAP

void CastMap::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const std::string map_input = node.input(0);

  EXT_ENFORCE_INVALID(rt.HasMap(map_input), "RunNode: CastMap map input '", map_input,
                      "' not found in the runtime context. Map-typed inputs "
                      "must be stored via PutMap before executing the graph.");
  const Map &cast_m = rt.GetMap(map_input);
  const Tensor &x_keys = cast_m.keys;
  const Tensor &x_values = cast_m.values;
  EXT_ENFORCE_INVALID(!(x_keys.data_type != static_cast<int32_t>(DataType::INT64)),
                      "RunNode: CastMap keys must be an INT64 tensor.");
  const std::span<const int64_t> keys = TensorSpan<int64_t>(x_keys);
  const std::string cast_to = GetAttributeStringOrDefault(node, "cast_to", "TO_FLOAT");
  const std::string map_form = GetAttributeStringOrDefault(node, "map_form", "DENSE");
  const int64_t max_map = GetAttributeIntOrDefault(node, "max_map", 0);
  EXT_ENFORCE_INVALID(!(cast_to != "TO_FLOAT" && cast_to != "TO_INT64" && cast_to != "TO_STRING"),
                      "RunNode: CastMap attribute 'cast_to' must be 'TO_FLOAT', 'TO_INT64', or "
                      "'TO_STRING'.");
  onnx_kernels::kernel::CastMap cast_map(rt.kernel_ctx());
  Tensor y;
  switch (x_values.data_type) {
  case static_cast<int32_t>(DataType::FLOAT): {
    const std::span<const float> values = TensorSpan<float>(x_values);
    if (cast_to == "TO_FLOAT") {
      y = cast_map.operator()<float, float>(keys, values, cast_to, map_form, max_map);
    } else if (cast_to == "TO_INT64") {
      y = cast_map.operator()<float, int64_t>(keys, values, cast_to, map_form, max_map);
    } else {
      y = cast_map.operator()<float, std::string>(keys, values, cast_to, map_form, max_map);
    }
    break;
  }
  case static_cast<int32_t>(DataType::STRING): {
    const std::vector<std::string> &values = x_values.AsStrings();
    if (cast_to == "TO_FLOAT") {
      y = cast_map.operator()<std::string, float>(keys, values, cast_to, map_form, max_map);
    } else if (cast_to == "TO_INT64") {
      y = cast_map.operator()<std::string, int64_t>(keys, values, cast_to, map_form, max_map);
    } else {
      y = cast_map.operator()<std::string, std::string>(keys, values, cast_to, map_form, max_map);
    }
    break;
  }
  default:
    EXT_THROW_INVALID("RunNode: CastMap values must be a FLOAT or STRING tensor.");
  }
  SetOutput(node, 0, std::move(y), rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
