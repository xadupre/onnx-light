// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_proto_util.h"

#include <cstring>
#include <string>
#include <vector>

#include "onnx/common/common.h"
#include "onnx/common/platform_helpers.h"
#include "onnx/defs/data_type_utils.h"

namespace ONNX_NAMESPACE {

#define DEFINE_TO_TENSOR_ONE(type, enum_type, field, cast_expr)                                    \
  template <> TensorProto ToTensor<type>(const type &value) {                                      \
    TensorProto t;                                                                                 \
    t.set_data_type(enum_type);                                                                    \
    t.add_##field##_data(cast_expr);                                                               \
    return t;                                                                                      \
  }

#define DEFINE_TO_TENSOR_LIST(type, enum_type, field, cast_expr)                                   \
  template <> TensorProto ToTensor<type>(const std::vector<type> &values) {                        \
    TensorProto t;                                                                                 \
    t.clr_##field##_data();                                                                        \
    t.set_data_type(enum_type);                                                                    \
    for (const auto &val : values) {                                                               \
      t.add_##field##_data(cast_expr);                                                             \
    }                                                                                              \
    return t;                                                                                      \
  }

#define DEFINE_PARSE_DATA(type, typed_data_fetch, tensorproto_datatype)                            \
  template <> std::vector<type> ParseData(const TensorProto *tensor_proto) {                       \
    if (!tensor_proto->has_data_type() ||                                                          \
        tensor_proto->ref_data_type() == TensorProto::DataType::UNDEFINED) {                       \
      ONNX_THROW_EX(                                                                               \
          std::invalid_argument("The type of tensor is undefined so it cannot be parsed."));       \
    }                                                                                              \
    if (tensor_proto->ref_data_type() != tensorproto_datatype) {                                   \
      ONNX_THROW_EX(std::invalid_argument(ONNX_NAMESPACE::MakeString(                              \
          "ParseData type mismatch. Expected: ",                                                   \
          Utils::DataTypeUtils::ToDataTypeString(static_cast<int32_t>(tensorproto_datatype)),      \
          " Actual: ",                                                                             \
          Utils::DataTypeUtils::ToDataTypeString(                                                  \
              static_cast<int32_t>(tensor_proto->ref_data_type())))));                             \
    }                                                                                              \
    int64_t num_elements = 1;                                                                      \
    for (const auto &dim : tensor_proto->ref_dims()) {                                             \
      num_elements *= dim;                                                                         \
    }                                                                                              \
    std::vector<type> res;                                                                         \
    if (tensor_proto->has_data_location() &&                                                       \
        tensor_proto->ref_data_location() == TensorProto::DataLocation::EXTERNAL) {                \
      ONNX_THROW_EX(std::invalid_argument(                                                         \
          "Cannot parse data from external tensors. Please load external data into raw_data."));   \
    }                                                                                              \
    if (!tensor_proto->has_raw_data()) {                                                           \
      const auto &data = tensor_proto->ref_##typed_data_fetch();                                   \
      if (static_cast<int64_t>(data.size()) != num_elements) {                                     \
        ONNX_THROW_EX(std::invalid_argument(                                                       \
            ONNX_NAMESPACE::MakeString("Data size mismatch. Expected num elements ", num_elements, \
                                       " does not match actual num elements ", data.size())));     \
      }                                                                                            \
      res.insert(res.end(), data.begin(), data.end());                                             \
      return res;                                                                                  \
    }                                                                                              \
    if (tensor_proto->ref_data_type() == TensorProto::DataType::STRING) {                          \
      ONNX_THROW_EX(std::invalid_argument(                                                         \
          "raw_data type cannot be string; use string_data field for string tensors."));           \
    }                                                                                              \
    const auto &raw_span = tensor_proto->ref_raw_data();                                           \
    std::string raw_data(reinterpret_cast<const char *>(raw_span.data()), raw_span.size());        \
    constexpr size_t element_size = sizeof(type);                                                  \
    const auto required_bytes = static_cast<size_t>(num_elements) * element_size;                  \
    if (raw_data.size() < required_bytes) {                                                        \
      ONNX_THROW_EX(std::invalid_argument(ONNX_NAMESPACE::MakeString(                              \
          "Data size mismatch. Tensor does not have sufficient raw_data. Required bytes: ",        \
          required_bytes, ", actual bytes: ", raw_data.size())));                                  \
    }                                                                                              \
    raw_data.resize(required_bytes);                                                               \
    char *bytes = raw_data.data();                                                                 \
    if (!is_processor_little_endian()) {                                                           \
      const size_t count = raw_data.size() / element_size;                                         \
      for (size_t i = 0; i < count; ++i) {                                                         \
        char *start_byte = bytes + i * element_size;                                               \
        char *end_byte = start_byte + element_size - 1;                                            \
        for (size_t c = 0; c < element_size / 2; ++c) {                                            \
          char temp = *start_byte;                                                                 \
          *start_byte = *end_byte;                                                                 \
          *end_byte = temp;                                                                        \
          ++start_byte;                                                                            \
          --end_byte;                                                                              \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
    res.resize(static_cast<size_t>(num_elements));                                                 \
    memcpy(reinterpret_cast<char *>(res.data()), bytes, required_bytes);                           \
    return res;                                                                                    \
  }

DEFINE_TO_TENSOR_ONE(float, TensorProto::DataType::FLOAT, float, value)
DEFINE_TO_TENSOR_ONE(bool, TensorProto::DataType::BOOL, int32, static_cast<int32_t>(value))
DEFINE_TO_TENSOR_ONE(int32_t, TensorProto::DataType::INT32, int32, value)
DEFINE_TO_TENSOR_ONE(int64_t, TensorProto::DataType::INT64, int64, value)
DEFINE_TO_TENSOR_ONE(uint64_t, TensorProto::DataType::UINT64, uint64, value)
DEFINE_TO_TENSOR_ONE(double, TensorProto::DataType::DOUBLE, double, value)

DEFINE_TO_TENSOR_LIST(float, TensorProto::DataType::FLOAT, float, val)
DEFINE_TO_TENSOR_LIST(bool, TensorProto::DataType::BOOL, int32, static_cast<int32_t>(val))
DEFINE_TO_TENSOR_LIST(int32_t, TensorProto::DataType::INT32, int32, val)
DEFINE_TO_TENSOR_LIST(int64_t, TensorProto::DataType::INT64, int64, val)
DEFINE_TO_TENSOR_LIST(uint64_t, TensorProto::DataType::UINT64, uint64, val)
DEFINE_TO_TENSOR_LIST(double, TensorProto::DataType::DOUBLE, double, val)

DEFINE_PARSE_DATA(int32_t, int32_data, TensorProto::DataType::INT32)
DEFINE_PARSE_DATA(int64_t, int64_data, TensorProto::DataType::INT64)
DEFINE_PARSE_DATA(float, float_data, TensorProto::DataType::FLOAT)
DEFINE_PARSE_DATA(double, double_data, TensorProto::DataType::DOUBLE)

#undef DEFINE_PARSE_DATA
#undef DEFINE_TO_TENSOR_LIST
#undef DEFINE_TO_TENSOR_ONE

} // namespace ONNX_NAMESPACE
