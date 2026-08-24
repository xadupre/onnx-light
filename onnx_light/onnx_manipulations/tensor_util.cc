// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_util.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "onnx_lib/common/platform_helpers.h"

namespace ONNX_LIGHT_NAMESPACE {

int64_t RawDataElementCount(const TensorProto &tensor, size_t element_size, bool exact_fit) {
  const int64_t num_elements = safe_dim_product(tensor.dims(), [&](const char *msg) {
    ONNX_ASSERTM(false, msg, " for tensor: ", tensor.name())
  });
  const size_t size = tensor.ref_raw_data().size();
  const auto available = static_cast<uint64_t>(size / element_size);
  const auto required = static_cast<uint64_t>(num_elements);
  ONNX_ASSERTM(
      !(exact_fit ? (available != required || size % element_size != 0) : available < required),
      "Data size mismatch. Tensor: ", tensor.name(), " has ", size, " bytes of raw_data for ",
      num_elements, " elements of size ", element_size);
  return num_elements;
}

#define DEFINE_PARSE_DATA(type, typed_data_fetch)                                                  \
  template <> std::vector<type> ParseData(const TensorProto *tensor) {                             \
    std::vector<type> res;                                                                         \
    if (!tensor->is_raw_data()) {                                                                  \
      const auto &data = tensor->typed_data_fetch();                                               \
      res.insert(res.end(), data.begin(), data.end());                                             \
      return res;                                                                                  \
    }                                                                                              \
    /* A dimensionless local tensor is an unshaped payload, not necessarily a scalar. */           \
    if (tensor->dims_size() > 0) {                                                                 \
      return ParseRawData<type>(*tensor);                                                          \
    }                                                                                              \
    const utils::ByteSpan &raw_data = tensor->ref_raw_data();                                      \
    /* copy byte-wise: raw_data may be unaligned for type */                                       \
    /* require a whole number of elements */                                                       \
    ONNX_ASSERTM(raw_data.size() % sizeof(type) == 0, "raw_data size ", raw_data.size(),           \
                 " is not a multiple of element size ", sizeof(type));                             \
    res.resize(raw_data.size() / sizeof(type));                                                    \
    std::byte *bytes = reinterpret_cast<std::byte *>(res.data());                                  \
    std::copy_n(reinterpret_cast<const std::byte *>(raw_data.data()), raw_data.size(), bytes);     \
    /* swap byte order on big-endian hosts */                                                      \
    if (!is_processor_little_endian()) {                                                           \
      for (auto &element : res) {                                                                  \
        std::byte *start_byte = reinterpret_cast<std::byte *>(&element);                           \
        std::reverse(start_byte, start_byte + sizeof(type));                                       \
      }                                                                                            \
    }                                                                                              \
    return res;                                                                                    \
  }

DEFINE_PARSE_DATA(int32_t, ref_int32_data)
DEFINE_PARSE_DATA(int64_t, ref_int64_data)
DEFINE_PARSE_DATA(float, ref_float_data)
DEFINE_PARSE_DATA(double, ref_double_data)
DEFINE_PARSE_DATA(uint64_t, ref_uint64_data)

#undef DEFINE_PARSE_DATA

#define DEFINE_PARSE_DATA(type, typed_data_fetch)                                                  \
  template <> std::vector<type> ParseData(const Tensor *tensor) {                                  \
    std::vector<type> res;                                                                         \
    if (!tensor->is_raw_data()) {                                                                  \
      const auto &data = tensor->typed_data_fetch();                                               \
      res.insert(res.end(), data.begin(), data.end());                                             \
      return res;                                                                                  \
    }                                                                                              \
    const std::string &raw_data = tensor->raw();                                                   \
    /* copy byte-wise: raw_data may be unaligned for type */                                       \
    /* require a whole number of elements */                                                       \
    ONNX_ASSERTM(raw_data.size() % sizeof(type) == 0, "raw_data size ", raw_data.size(),           \
                 " is not a multiple of element size ", sizeof(type));                             \
    res.resize(raw_data.size() / sizeof(type));                                                    \
    std::byte *bytes = reinterpret_cast<std::byte *>(res.data());                                  \
    std::copy_n(reinterpret_cast<const std::byte *>(raw_data.data()), raw_data.size(), bytes);     \
    /* swap byte order on big-endian hosts */                                                      \
    if (!is_processor_little_endian()) {                                                           \
      for (auto &element : res) {                                                                  \
        std::byte *start_byte = reinterpret_cast<std::byte *>(&element);                           \
        std::reverse(start_byte, start_byte + sizeof(type));                                       \
      }                                                                                            \
    }                                                                                              \
    return res;                                                                                    \
  }

DEFINE_PARSE_DATA(int32_t, int32s)
DEFINE_PARSE_DATA(int64_t, int64s)
DEFINE_PARSE_DATA(float, floats)
DEFINE_PARSE_DATA(double, doubles)
DEFINE_PARSE_DATA(uint64_t, uint64s)

#undef DEFINE_PARSE_DATA

template <> TensorProto ToTensor<float>(const float &value) {
  TensorProto t;
  t.set_data_type(TensorProto::FLOAT);
  t.add_float_data(value);
  return t;
}

template <> TensorProto ToTensor<float>(const std::vector<float> &values) {
  TensorProto t;
  t.set_data_type(TensorProto::FLOAT);
  for (auto v : values)
    t.add_float_data(v);
  return t;
}

template <> TensorProto ToTensor<double>(const double &value) {
  TensorProto t;
  t.set_data_type(TensorProto::DOUBLE);
  t.add_double_data(value);
  return t;
}

template <> TensorProto ToTensor<double>(const std::vector<double> &values) {
  TensorProto t;
  t.set_data_type(TensorProto::DOUBLE);
  for (auto v : values)
    t.add_double_data(v);
  return t;
}

template <> TensorProto ToTensor<int64_t>(const int64_t &value) {
  TensorProto t;
  t.set_data_type(TensorProto::INT64);
  t.add_int64_data(value);
  return t;
}

template <> TensorProto ToTensor<int64_t>(const std::vector<int64_t> &values) {
  TensorProto t;
  t.set_data_type(TensorProto::INT64);
  for (auto v : values)
    t.add_int64_data(v);
  return t;
}

template <> TensorProto ToTensor<int32_t>(const int32_t &value) {
  TensorProto t;
  t.set_data_type(TensorProto::INT32);
  t.add_int32_data(value);
  return t;
}

template <> TensorProto ToTensor<int32_t>(const std::vector<int32_t> &values) {
  TensorProto t;
  t.set_data_type(TensorProto::INT32);
  for (auto v : values)
    t.add_int32_data(v);
  return t;
}

template <> TensorProto ToTensor<bool>(const bool &value) {
  TensorProto t;
  t.set_data_type(TensorProto::BOOL);
  t.add_int32_data(static_cast<int32_t>(value));
  return t;
}

template <> TensorProto ToTensor<uint8_t>(const uint8_t &value) {
  TensorProto t;
  t.set_data_type(TensorProto::UINT8);
  t.add_int32_data(static_cast<int32_t>(value));
  return t;
}

template <> TensorProto ToTensor<int8_t>(const int8_t &value) {
  TensorProto t;
  t.set_data_type(TensorProto::INT8);
  t.add_int32_data(static_cast<int32_t>(value));
  return t;
}

} // namespace ONNX_LIGHT_NAMESPACE
