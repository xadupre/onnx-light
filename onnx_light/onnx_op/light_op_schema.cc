// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {

std::vector<std::string> TypesToStrings(const std::initializer_list<TensorType> types) {
  std::vector<std::string> allowed_types;
  allowed_types.reserve(types.size());
  for (const TensorType tensor_type : types) {
    allowed_types.emplace_back(ToTypeString(tensor_type));
  }
  return allowed_types;
}

std::vector<std::string> FloatTypeStrings() {
  return TypesToStrings({
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
  });
}

std::vector<std::string> NumericTypesForMathReductionStrings() {
  return TypesToStrings({
      TensorType::kUint32,
      TensorType::kUint64,
      TensorType::kInt32,
      TensorType::kInt64,
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
  });
}

std::vector<std::string> NumericTypesForMathReductionIr4Strings() {
  return TypesToStrings({
      TensorType::kUint32,
      TensorType::kUint64,
      TensorType::kInt32,
      TensorType::kInt64,
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
      TensorType::kBfloat16,
  });
}

std::vector<std::string> AllNumericTypesStrings() {
  return TypesToStrings({
      TensorType::kUint8,
      TensorType::kUint16,
      TensorType::kUint32,
      TensorType::kUint64,
      TensorType::kInt8,
      TensorType::kInt16,
      TensorType::kInt32,
      TensorType::kInt64,
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
  });
}

std::vector<std::string> AllNumericTypesIr4Strings() {
  return TypesToStrings({
      TensorType::kUint8,
      TensorType::kUint16,
      TensorType::kUint32,
      TensorType::kUint64,
      TensorType::kInt8,
      TensorType::kInt16,
      TensorType::kInt32,
      TensorType::kInt64,
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
      TensorType::kBfloat16,
  });
}

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
