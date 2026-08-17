// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

template <typename KeyT> int32_t KeyDataType() noexcept { return TensorElementType<KeyT>::value; }

template <typename KeyT, typename ValueT>
void LookupAndFill(const Tensor &x, std::span<const KeyT> keys, std::span<const ValueT> values,
                   ValueT default_value, ValueT *out) {
  const int64_t n = x.element_count();
  const size_t k = keys.size();
  if constexpr (std::is_same_v<KeyT, std::string>) {
    const std::vector<std::string> &px = x.AsStrings();
    for (int64_t i = 0; i < n; ++i) {
      ValueT mapped = default_value;
      for (size_t j = 0; j < k; ++j) {
        if (px[static_cast<size_t>(i)] == keys[j]) {
          mapped = values[j];
          break;
        }
      }
      out[i] = mapped;
    }
  } else {
    const KeyT *px = x.As<KeyT>();
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
}

template <typename KeyT, typename ValueT>
void ValidateInputs(const Tensor &x, std::span<const KeyT> keys, std::span<const ValueT> values) {
  if constexpr (std::is_same_v<KeyT, std::string>) {
    EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::STRING),
                        "kernel::LabelEncoder input data_type does not match the requested KeyT.");
    EXT_ENFORCE_INVALID(
        static_cast<int64_t>(x.string_data.size()) == x.element_count(),
        "kernel::LabelEncoder STRING input string_data size does not match its shape.");
  } else {
    EXT_ENFORCE_INVALID(x.data_type == KeyDataType<KeyT>(),
                        "kernel::LabelEncoder input data_type does not match the requested KeyT.");
  }
  EXT_ENFORCE_INVALID(keys.size() == values.size(),
                      "kernel::LabelEncoder requires keys and values to have the same length.");
}

} // namespace

template <typename KeyT, typename ValueT>
Tensor LabelEncoder::operator()(const Tensor &x, std::span<const KeyT> keys,
                                std::span<const ValueT> values, ValueT default_value,
                                RuntimeContext *rt) const {
  ValidateInputs<KeyT, ValueT>(x, keys, values);
  const int64_t n = x.element_count();
  const size_t n_bytes = static_cast<size_t>(n) * sizeof(ValueT);
  Tensor out =
      rt ? rt->MakeOutputTensor(0, TensorElementType<ValueT>::value, x.shape, n_bytes)
         : MakeOutputTensor(TensorElementType<ValueT>::value, x.shape, n_bytes, ctx_.allocator);
  LookupAndFill<KeyT, ValueT>(x, keys, values, default_value,
                              reinterpret_cast<ValueT *>(out.mutable_bytes()));
  return out;
}

template <typename KeyT, typename ValueT>
void LabelEncoder::operator()(const Tensor &x, std::span<const KeyT> keys,
                              std::span<const ValueT> values, ValueT default_value,
                              Tensor &output) const {
  ValidateInputs<KeyT, ValueT>(x, keys, values);
  EXT_ENFORCE_INVALID(
      output.data_type == TensorElementType<ValueT>::value,
      "kernel::LabelEncoder preallocated output dtype must match the requested ValueT.");
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::LabelEncoder preallocated output shape must match the input shape.");
  EXT_ENFORCE_INVALID(output.size_bytes() ==
                          static_cast<size_t>(x.element_count()) * sizeof(ValueT),
                      "kernel::LabelEncoder preallocated output buffer is incorrectly sized.");
  LookupAndFill<KeyT, ValueT>(x, keys, values, default_value, output.As<ValueT>());
}

// Explicit instantiations for the supported (KeyT, ValueT) combinations.
#define ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER(KEY_T, VALUE_T)                                       \
  template Tensor LabelEncoder::operator()(const Tensor &, std::span<const KEY_T>,                 \
                                           std::span<const VALUE_T>, VALUE_T, RuntimeContext *)    \
      const;                                                                                       \
  template void LabelEncoder::operator()(const Tensor &, std::span<const KEY_T>,                   \
                                         std::span<const VALUE_T>, VALUE_T, Tensor &) const

ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER(int64_t, int64_t);
ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER(int64_t, float);
ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER(float, int64_t);
ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER(float, float);
ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER(std::string, int64_t);
ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER(std::string, int16_t);

#undef ONNX_LIGHT_INSTANTIATE_LABEL_ENCODER

void LabelEncoder::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  const Tensor &x = GetInput(node, 0, rt.tensors());

  // Identify the key source (exactly one of keys_int64s, keys_floats,
  // keys_strings, keys_tensor must be present per the ONNX spec).
  const AttributeProto *keys_int64s = FindAttribute(node, "keys_int64s");
  const AttributeProto *keys_floats = FindAttribute(node, "keys_floats");
  const AttributeProto *keys_strings = FindAttribute(node, "keys_strings");
  const AttributeProto *keys_tensor = FindAttribute(node, "keys_tensor");
  const int n_keys = (keys_int64s != nullptr) + (keys_floats != nullptr) +
                     (keys_strings != nullptr) + (keys_tensor != nullptr);
  EXT_ENFORCE_INVALID(n_keys == 1, "RunNode: LabelEncoder requires exactly one of 'keys_int64s', "
                                   "'keys_floats', 'keys_strings' or 'keys_tensor' to be set.");

  // Identify the value source.
  const AttributeProto *values_int64s = FindAttribute(node, "values_int64s");
  const AttributeProto *values_floats = FindAttribute(node, "values_floats");
  const AttributeProto *values_strings = FindAttribute(node, "values_strings");
  const AttributeProto *values_tensor = FindAttribute(node, "values_tensor");
  const int n_values = (values_int64s != nullptr) + (values_floats != nullptr) +
                       (values_strings != nullptr) + (values_tensor != nullptr);
  EXT_ENFORCE_INVALID(n_values == 1,
                      "RunNode: LabelEncoder requires exactly one of 'values_int64s', "
                      "'values_floats', 'values_strings' or 'values_tensor' to be set.");

  // Resolve KeyT. The tensor holder keeps the key data alive when the key
  // source is a tensor attribute; spans reference either the holder or an
  // attribute-sourced vector.
  enum class KeyKind { Int64, Float, String };
  KeyKind key_kind;
  Tensor keys_tensor_holder; // alive until after label_encoder call
  std::vector<int64_t> keys_i64_vec;
  std::vector<float> keys_f32_vec;
  std::vector<std::string> keys_str;
  std::span<const int64_t> keys_i64;
  std::span<const float> keys_f32;
  if (keys_int64s != nullptr) {
    key_kind = KeyKind::Int64;
    for (int64_t v : keys_int64s->ints()) {
      keys_i64_vec.push_back(v);
    }
    keys_i64 = keys_i64_vec;
  } else if (keys_floats != nullptr) {
    key_kind = KeyKind::Float;
    for (float v : keys_floats->floats()) {
      keys_f32_vec.push_back(v);
    }
    keys_f32 = keys_f32_vec;
  } else if (keys_strings != nullptr) {
    key_kind = KeyKind::String;
    for (const auto &v : keys_strings->strings()) {
      keys_str.push_back(v);
    }
  } else {
    keys_tensor_holder = TensorFromProto(keys_tensor->t());
    switch (keys_tensor_holder.data_type) {
    case static_cast<int32_t>(DataType::INT64):
      key_kind = KeyKind::Int64;
      keys_i64 = TensorSpan<int64_t>(keys_tensor_holder);
      break;
    case static_cast<int32_t>(DataType::FLOAT):
      key_kind = KeyKind::Float;
      keys_f32 = TensorSpan<float>(keys_tensor_holder);
      break;
    case static_cast<int32_t>(DataType::STRING):
      key_kind = KeyKind::String;
      keys_str = keys_tensor_holder.AsStrings();
      break;
    default:
      EXT_THROW_INVALID("RunNode: LabelEncoder 'keys_tensor' must have element type "
                        "INT64, FLOAT or STRING.");
    }
  }

  // Resolve ValueT and look up the (optional) default attribute.
  // Same pattern: tensor holder keeps value data alive for the span.
  enum class ValueKind { Int64, Float, Int16 };
  ValueKind value_kind;
  Tensor values_tensor_holder; // alive until after label_encoder call
  std::vector<int64_t> values_i64_vec;
  std::vector<float> values_f32_vec;
  std::vector<int16_t> values_i16_vec;
  std::span<const int64_t> values_i64;
  std::span<const float> values_f32;
  std::span<const int16_t> values_i16;
  if (values_int64s != nullptr) {
    value_kind = ValueKind::Int64;
    for (int64_t v : values_int64s->ints()) {
      values_i64_vec.push_back(v);
    }
    values_i64 = values_i64_vec;
  } else if (values_floats != nullptr) {
    value_kind = ValueKind::Float;
    for (float v : values_floats->floats()) {
      values_f32_vec.push_back(v);
    }
    values_f32 = values_f32_vec;
  } else if (values_strings != nullptr) {
    EXT_THROW_INVALID("RunNode: LabelEncoder with 'values_strings' is not supported "
                      "by this kernel registration.");
  } else {
    values_tensor_holder = TensorFromProto(values_tensor->t());
    switch (values_tensor_holder.data_type) {
    case static_cast<int32_t>(DataType::INT64):
      value_kind = ValueKind::Int64;
      values_i64 = TensorSpan<int64_t>(values_tensor_holder);
      break;
    case static_cast<int32_t>(DataType::FLOAT):
      value_kind = ValueKind::Float;
      values_f32 = TensorSpan<float>(values_tensor_holder);
      break;
    case static_cast<int32_t>(DataType::INT16):
      value_kind = ValueKind::Int16;
      values_i16 = TensorSpan<int16_t>(values_tensor_holder);
      break;
    default:
      EXT_THROW_INVALID("RunNode: LabelEncoder 'values_tensor' must have element "
                        "type INT64, FLOAT or INT16.");
    }
  }

  int64_t default_i64 = -1;
  float default_f32 = 0.0f;
  int16_t default_i16 = -1;
  const AttributeProto *default_int64 = FindAttribute(node, "default_int64");
  const AttributeProto *default_float = FindAttribute(node, "default_float");
  const AttributeProto *default_tensor_attr = FindAttribute(node, "default_tensor");
  if (default_int64 != nullptr) {
    default_i64 = default_int64->i();
  }
  if (default_float != nullptr) {
    default_f32 = default_float->f();
  }
  if (default_tensor_attr != nullptr) {
    const Tensor dt = TensorFromProto(default_tensor_attr->t());
    EXT_ENFORCE_INVALID(dt.element_count() == 1,
                        "RunNode: LabelEncoder 'default_tensor' must contain exactly one element.");
    switch (dt.data_type) {
    case static_cast<int32_t>(DataType::INT64):
      default_i64 = dt.AsInt64()[0];
      break;
    case static_cast<int32_t>(DataType::FLOAT):
      default_f32 = dt.AsFloat()[0];
      break;
    case static_cast<int32_t>(DataType::INT16):
      default_i16 = dt.AsInt16()[0];
      break;
    default:
      EXT_THROW_INVALID("RunNode: LabelEncoder 'default_tensor' must have element "
                        "type INT64, FLOAT or INT16.");
    }
  }

  onnx_kernels::kernel::LabelEncoder label_encoder(rt.kernel_ctx());
  Tensor out;
  if (key_kind == KeyKind::Int64 && value_kind == ValueKind::Int64) {
    out = label_encoder.operator()<int64_t, int64_t>(x, keys_i64, values_i64, default_i64, &rt);
  } else if (key_kind == KeyKind::Int64 && value_kind == ValueKind::Float) {
    out = label_encoder.operator()<int64_t, float>(x, keys_i64, values_f32, default_f32, &rt);
  } else if (key_kind == KeyKind::Float && value_kind == ValueKind::Int64) {
    out = label_encoder.operator()<float, int64_t>(x, keys_f32, values_i64, default_i64, &rt);
  } else if (key_kind == KeyKind::Float && value_kind == ValueKind::Float) {
    out = label_encoder.operator()<float, float>(x, keys_f32, values_f32, default_f32, &rt);
  } else if (key_kind == KeyKind::String && value_kind == ValueKind::Int64) {
    out = label_encoder.operator()<std::string, int64_t>(x, keys_str, values_i64, default_i64, &rt);
  } else if (key_kind == KeyKind::String && value_kind == ValueKind::Int16) {
    out = label_encoder.operator()<std::string, int16_t>(x, keys_str, values_i16, default_i16, &rt);
  } else {
    EXT_THROW_INVALID("RunNode: LabelEncoder key/value type combination is not supported.");
  }
  SetOutput(node, 0, std::move(out), rt.tensors());
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
