// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>
#include <string>
#include <vector>

#include "onnx/common/platform_helpers.h"
#include "onnx/common/proto_utils.h"

namespace ONNX_LIGHT_NAMESPACE {

// Creates a scalar TensorProto of the given value.
template <typename T> TensorProto ToTensor(const T &value);

// Creates a 1-D TensorProto from a vector.
template <typename T> TensorProto ToTensor(const std::vector<T> &values);

// Parses the data of a TensorProto into a typed vector.
template <typename T> std::vector<T> ParseData(const TensorProto *tensor_proto);

// -------------------------------------------------------------------------
// Inline template specialisations
// -------------------------------------------------------------------------

#define ONNX_LIGHT_DEFINE_TO_TENSOR_ONE(ctype, dtype_val, add_field)                               \
  template <> inline TensorProto ToTensor(const ctype &value) {                                    \
    TensorProto t;                                                                                  \
    t.data_type_ = TensorProto::dtype_val;                                                         \
    t.ref_##add_field##_data().push_back(static_cast<ctype>(value));                               \
    return t;                                                                                       \
  }

#define ONNX_LIGHT_DEFINE_TO_TENSOR_LIST(ctype, dtype_val, add_field)                              \
  template <> inline TensorProto ToTensor(const std::vector<ctype> &values) {                      \
    TensorProto t;                                                                                  \
    t.ref_##add_field##_data().clr_##add_field##_data();                                            \
    t.data_type_ = TensorProto::dtype_val;                                                         \
    for (const auto &val : values) {                                                                \
      t.ref_##add_field##_data().push_back(static_cast<ctype>(val));                               \
    }                                                                                               \
    return t;                                                                                       \
  }

// ToTensor specialisations – keep in sync with ONNX's tensor_proto_util.cc.
ONNX_LIGHT_DEFINE_TO_TENSOR_ONE(float, FLOAT, float)
ONNX_LIGHT_DEFINE_TO_TENSOR_ONE(bool, BOOL, int32)
ONNX_LIGHT_DEFINE_TO_TENSOR_ONE(int32_t, INT32, int32)
ONNX_LIGHT_DEFINE_TO_TENSOR_ONE(int64_t, INT64, int64)
ONNX_LIGHT_DEFINE_TO_TENSOR_ONE(uint64_t, UINT64, uint64)
ONNX_LIGHT_DEFINE_TO_TENSOR_ONE(double, DOUBLE, double)

#undef ONNX_LIGHT_DEFINE_TO_TENSOR_ONE
#undef ONNX_LIGHT_DEFINE_TO_TENSOR_LIST

// std::string scalar
template <> inline TensorProto ToTensor(const std::string &value) {
  TensorProto t;
  t.data_type_ = TensorProto::STRING;
  t.ref_string_data().add() = value;
  return t;
}

// Vector specialisations
template <> inline TensorProto ToTensor(const std::vector<float> &values) {
  TensorProto t;
  t.data_type_ = TensorProto::FLOAT;
  for (const auto &v : values)
    t.ref_float_data().push_back(v);
  return t;
}
template <> inline TensorProto ToTensor(const std::vector<bool> &values) {
  TensorProto t;
  t.data_type_ = TensorProto::BOOL;
  for (const bool v : values)
    t.ref_int32_data().push_back(static_cast<int32_t>(v));
  return t;
}
template <> inline TensorProto ToTensor(const std::vector<int32_t> &values) {
  TensorProto t;
  t.data_type_ = TensorProto::INT32;
  for (const auto &v : values)
    t.ref_int32_data().push_back(v);
  return t;
}
template <> inline TensorProto ToTensor(const std::vector<int64_t> &values) {
  TensorProto t;
  t.data_type_ = TensorProto::INT64;
  for (const auto &v : values)
    t.ref_int64_data().push_back(v);
  return t;
}
template <> inline TensorProto ToTensor(const std::vector<uint64_t> &values) {
  TensorProto t;
  t.data_type_ = TensorProto::UINT64;
  for (const auto &v : values)
    t.ref_uint64_data().push_back(v);
  return t;
}
template <> inline TensorProto ToTensor(const std::vector<double> &values) {
  TensorProto t;
  t.data_type_ = TensorProto::DOUBLE;
  for (const auto &v : values)
    t.ref_double_data().push_back(v);
  return t;
}
template <> inline TensorProto ToTensor(const std::vector<std::string> &values) {
  TensorProto t;
  t.data_type_ = TensorProto::STRING;
  for (const auto &v : values)
    t.ref_string_data().add() = v;
  return t;
}

// -------------------------------------------------------------------------
// ParseData helpers
// -------------------------------------------------------------------------

namespace detail {

template <typename T>
inline std::vector<T> parse_raw_bytes(const uint8_t *bytes, size_t byte_size, size_t num_elems) {
  std::vector<T> res(num_elems);
  // Make a mutable copy so we can byte-swap in place on big-endian platforms.
  std::vector<uint8_t> raw(bytes, bytes + byte_size);
  if (!is_processor_little_endian()) {
    constexpr size_t elem_sz = sizeof(T);
    for (size_t i = 0; i < num_elems; ++i) {
      auto *start = raw.data() + i * elem_sz;
      auto *end = start + elem_sz - 1;
      while (start < end) {
        std::swap(*start++, *end--);
      }
    }
  }
  memcpy(res.data(), raw.data(), num_elems * sizeof(T));
  return res;
}

} // namespace detail

#define ONNX_LIGHT_DEFINE_PARSE_DATA(ctype, data_field, dtype_enum)                                \
  template <> inline std::vector<ctype> ParseData(const TensorProto *tp) {                         \
    int64_t num_elems = 1;                                                                          \
    for (const auto d : tp->ref_dims())                                                             \
      num_elems *= static_cast<int64_t>(d);                                                         \
    if (tp->has_data_location() &&                                                                  \
        tp->data_location_ == TensorProto::DataLocation::EXTERNAL) {                               \
      fail_shape_inference("Cannot parse data from external tensors. Please load external data "    \
                           "into raw_data for tensor: ",                                            \
                           tp->ref_name().as_string());                                             \
    }                                                                                               \
    if (tp->has_raw_data()) {                                                                       \
      const auto &span = tp->raw_data_;                                                             \
      return detail::parse_raw_bytes<ctype>(                                                        \
          reinterpret_cast<const uint8_t *>(span.data()), span.size(),                              \
          static_cast<size_t>(num_elems));                                                          \
    }                                                                                               \
    const auto &data = tp->ref_##data_field##_data();                                               \
    std::vector<ctype> res;                                                                         \
    res.reserve(data.size());                                                                       \
    for (size_t i = 0; i < data.size(); ++i)                                                        \
      res.push_back(static_cast<ctype>(data[i]));                                                   \
    return res;                                                                                     \
  }

ONNX_LIGHT_DEFINE_PARSE_DATA(int32_t, int32, INT32)
ONNX_LIGHT_DEFINE_PARSE_DATA(int64_t, int64, INT64)
ONNX_LIGHT_DEFINE_PARSE_DATA(float, float, FLOAT)
ONNX_LIGHT_DEFINE_PARSE_DATA(double, double, DOUBLE)

#undef ONNX_LIGHT_DEFINE_PARSE_DATA

} // namespace ONNX_LIGHT_NAMESPACE
