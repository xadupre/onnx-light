// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_util.h"

#include <cstring>
#include <string>
#include <vector>

#include "onnx/common/platform_helpers.h"

namespace ONNX_LIGHT_NAMESPACE {

#define DEFINE_PARSE_DATA(type, typed_data_fetch)                                                  \
  template <> std::vector<type> ParseData(const TensorProto *tensor) {                             \
    std::vector<type> res;                                                                         \
    if (!tensor->is_raw_data()) {                                                                  \
      const auto &data = tensor->typed_data_fetch();                                               \
      res.insert(res.end(), data.begin(), data.end());                                             \
      return res;                                                                                  \
    }                                                                                              \
    /* make copy as we may have to reverse bytes */                                                \
    utils::ByteSpan raw_data = tensor->ref_raw_data();                                             \
    /* raw_data.data() returns a non-const pointer since raw_data is a local copy */               \
    unsigned char *bytes = raw_data.data();                                                        \
    /* onnx is always serialized as little endian - tweak byte order if needed */                  \
    if (!is_processor_little_endian()) {                                                           \
      size_t element_size = sizeof(type);                                                          \
      size_t num_elements = raw_data.size() / element_size;                                        \
      for (size_t i = 0; i < num_elements; ++i) {                                                  \
        unsigned char *start_byte = bytes + i * element_size;                                      \
        unsigned char *end_byte = start_byte + element_size - 1;                                   \
        /* keep swapping */                                                                        \
        for (size_t count = 0; count < element_size / 2; ++count) {                                \
          unsigned char temp = *start_byte;                                                        \
          *start_byte = *end_byte;                                                                 \
          *end_byte = temp;                                                                        \
          ++start_byte;                                                                            \
          --end_byte;                                                                              \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
    /* raw_data.c_str()/bytes is a byte array and may not be properly  */                          \
    /* aligned for the underlying type */                                                          \
    /* We need to copy the raw_data.c_str()/bytes as byte instead of  */                           \
    /* copying as the underlying type, otherwise we may hit memory   */                            \
    /* misalignment issues on certain platforms, such as arm32-v7a */                              \
    const size_t raw_data_size = raw_data.size();                                                  \
    res.resize(raw_data_size / sizeof(type));                                                      \
    memcpy(reinterpret_cast<char *>(res.data()), bytes, raw_data_size);                            \
    return res;                                                                                    \
  }

DEFINE_PARSE_DATA(int32_t, ref_int32_data)
DEFINE_PARSE_DATA(int64_t, ref_int64_data)
DEFINE_PARSE_DATA(float, ref_float_data)
DEFINE_PARSE_DATA(double, ref_double_data)
DEFINE_PARSE_DATA(uint64_t, ref_uint64_data)

#undef DEFINE_PARSE_DATA

#define DEFINE_PARSE_DATA(type, typed_data_fetch)                          \
  template <>                                                              \
  std::vector<type> ParseData(const Tensor* tensor) {                      \
    std::vector<type> res;                                                 \
    if (!tensor->is_raw_data()) {                                          \
      const auto& data = tensor->typed_data_fetch();                       \
      res.insert(res.end(), data.begin(), data.end());                     \
      return res;                                                          \
    }                                                                      \
    /* make copy as we may have to reverse bytes */                        \
    std::string raw_data = tensor->raw();                                  \
    /* okay to remove const qualifier as we have already made a copy */    \
    char* bytes = raw_data.data();                                         \
    /*onnx is little endian serialized always-tweak byte order if needed*/ \
    if (!is_processor_little_endian()) {                                   \
      const size_t element_size = sizeof(type);                            \
      const size_t num_elements = raw_data.size() / element_size;          \
      for (size_t i = 0; i < num_elements; ++i) {                          \
        char* start_byte = bytes + i * element_size;                       \
        char* end_byte = start_byte + element_size - 1;                    \
        /* keep swapping */                                                \
        for (size_t count = 0; count < element_size / 2; ++count) {        \
          char temp = *start_byte;                                         \
          *start_byte = *end_byte;                                         \
          *end_byte = temp;                                                \
          ++start_byte;                                                    \
          --end_byte;                                                      \
        }                                                                  \
      }                                                                    \
    }                                                                      \
    /* raw_data.c_str()/bytes is a byte array and may not be properly  */  \
    /* aligned for the underlying type */                                  \
    /* We need to copy the raw_data.c_str()/bytes as byte instead of  */   \
    /* copying as the underlying type, otherwise we may hit memory   */    \
    /* misalignment issues on certain platforms, such as arm32-v7a */      \
    const size_t raw_data_size = raw_data.size();                          \
    res.resize(raw_data_size / sizeof(type));                              \
    memcpy(reinterpret_cast<char*>(res.data()), bytes, raw_data_size);     \
    return res;                                                            \
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
