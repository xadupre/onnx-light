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

std::vector<std::string> CastTypesV1V6Strings() {
  return {"tensor(float16)", "tensor(float)",  "tensor(double)", "tensor(int8)",
          "tensor(int16)",   "tensor(int32)",  "tensor(int64)",  "tensor(uint8)",
          "tensor(uint16)",  "tensor(uint32)", "tensor(uint64)", "tensor(bool)"};
}

std::vector<std::string> CastTypesV9Strings() {
  return {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(int8)",   "tensor(int16)",
          "tensor(int32)",   "tensor(int64)", "tensor(uint8)",  "tensor(uint16)", "tensor(uint32)",
          "tensor(uint64)",  "tensor(bool)",  "tensor(string)"};
}

std::vector<std::string> CastTypesV13Strings() {
  return {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(int8)",    "tensor(int16)",
          "tensor(int32)",   "tensor(int64)", "tensor(uint8)",  "tensor(uint16)",  "tensor(uint32)",
          "tensor(uint64)",  "tensor(bool)",  "tensor(string)", "tensor(bfloat16)"};
}

std::vector<std::string> CastTypesV19Strings() {
  return {"tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(int8)",           "tensor(int16)",      "tensor(int32)",
          "tensor(int64)",          "tensor(uint8)",      "tensor(uint16)",
          "tensor(uint32)",         "tensor(uint64)",     "tensor(bool)",
          "tensor(string)",         "tensor(bfloat16)",   "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)"};
}

std::vector<std::string> CastTypesV21Strings() {
  return {"tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(int8)",           "tensor(int16)",      "tensor(int32)",
          "tensor(int64)",          "tensor(uint8)",      "tensor(uint16)",
          "tensor(uint32)",         "tensor(uint64)",     "tensor(bool)",
          "tensor(string)",         "tensor(bfloat16)",   "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
          "tensor(uint4)",          "tensor(int4)"};
}

std::vector<std::string> CastTypesV23Strings() {
  return {"tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
          "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
          "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
          "tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
          "tensor(uint4)",          "tensor(int4)",       "tensor(float4e2m1)"};
}

std::vector<std::string> CastTypesV24Strings() {
  return {"tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
          "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
          "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
          "tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
          "tensor(uint4)",          "tensor(int4)",       "tensor(float4e2m1)",
          "tensor(float8e8m0)"};
}

std::vector<std::string> CastTypesV25Strings() {
  return {"tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
          "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
          "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
          "tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
          "tensor(uint4)",          "tensor(int4)",       "tensor(float4e2m1)",
          "tensor(float8e8m0)",     "tensor(uint2)",      "tensor(int2)"};
}

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
