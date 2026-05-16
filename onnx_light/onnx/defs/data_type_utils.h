// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file data_type_utils.h
 * @brief Declares helpers that convert between ONNX type descriptors.
 *
 * This header provides a canonical DataType identifier and conversion helpers
 * that map textual type strings and TypeProto values to one another.
 */

#ifndef ONNX_DEFS_DATA_TYPE_UTILS_H_
#define ONNX_DEFS_DATA_TYPE_UTILS_H_

#include <mutex>
#include <string>
#include <unordered_map>

#include "onnx/common/common.h"
#include "onnx/common/onnx_pb.h"

namespace ONNX_LIGHT_NAMESPACE {
/// Uses a canonical type-string pointer as a unique runtime type identifier.
using DataType = const std::string *;

namespace Utils {

/**
 * Converts ONNX type spellings into canonical DataType and TypeProto forms.
 *
 * The implementation stores a process-wide map from textual type strings to
 * TypeProto values. Returned DataType identifiers point to canonical strings in
 * that map.
 *
 * Grammar for data type strings:
 *
 * <type> ::= <data_type> |
 *            tensor(<data_type>) |
 *            seq(<type>) |
 *            map(<data_type>, <type>)
 *
 * <data_type> ::= float | int32 | string | bool | uint8
 *               | int8 | uint16 | int16 | int64 | float16 | double
 *
 * `type ::= data_type` denotes a scalar (zero-dimension) value.
 * Examples include `float` and `tensor(float)`.
 */
class DataTypeUtils final {
public:
  /**
   * This function converts a type string into a canonical DataType identifier.
   * @param type_str Type string following the grammar documented above.
   * @return Canonical DataType identifier for type_str.
   *
   * The function throws std::invalid_argument when type_str is invalid.
   * The function aborts when type_str is invalid and ONNX_NO_EXCEPTIONS is defined.
   */
  static DataType ToType(const std::string &type_str);

  /**
   * This function converts a TypeProto into a canonical DataType identifier.
   * @param type_proto TypeProto value to normalize.
   * @return Canonical DataType identifier for type_proto.
   *
   * The function throws std::invalid_argument when type_proto is invalid.
   * The function aborts when type_proto is invalid and ONNX_NO_EXCEPTIONS is defined.
   */
  static DataType ToType(const TypeProto &type_proto);

  /**
   * This function converts a canonical DataType identifier into its TypeProto form.
   * @param data_type Canonical DataType identifier.
   * @return Cached TypeProto corresponding to data_type.
   *
   * The function throws std::invalid_argument when data_type is invalid.
   * The function aborts when data_type is invalid and ONNX_NO_EXCEPTIONS is defined.
   */
  static const TypeProto &ToTypeProto(const DataType &data_type);

  /**
   * This function converts a TensorProto::DataType enum value into its type-string spelling.
   * @param tensor_data_type Numeric TensorProto::DataType enum value.
   * @return Textual primitive type name.
   */
  static std::string ToDataTypeString(int32_t tensor_data_type);

private:
  static void FromString(const std::string &type_str, TypeProto &type_proto);

  static int32_t FromDataTypeString(const std::string &type_str);

  static std::string ToString(const TypeProto &type_proto, const std::string &left = "",
                              const std::string &right = "");

  static bool IsValidDataTypeString(const std::string &type_str);

  static std::unordered_map<std::string, TypeProto> &GetTypeStrToProtoMap();

  /// Returns the lock guarding concurrent updates to the type-string map.
  static std::mutex &GetTypeStrLock();
};
} // namespace Utils
} // namespace ONNX_LIGHT_NAMESPACE

#endif // ONNX_DEFS_DATA_TYPE_UTILS_H_
