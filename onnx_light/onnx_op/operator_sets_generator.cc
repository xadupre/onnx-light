// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_generator.h"
#include "onnx_op/operator_sets_generator_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace generator {

namespace {

using AllowedType = TypeConstraintParam::AllowedType;

std::vector<AllowedType> AllTensorTypes() {
  return {
      TensorType::kUint8,   TensorType::kUint16,    TensorType::kUint32,     TensorType::kUint64,
      TensorType::kInt8,    TensorType::kInt16,     TensorType::kInt32,      TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,     TensorType::kDouble,     TensorType::kString,
      TensorType::kBool,    TensorType::kComplex64, TensorType::kComplex128,
  };
}

std::vector<AllowedType> AllTensorTypesIr4() {
  return {
      TensorType::kUint8,    TensorType::kUint16,  TensorType::kUint32,    TensorType::kUint64,
      TensorType::kInt8,     TensorType::kInt16,   TensorType::kInt32,     TensorType::kInt64,
      TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,     TensorType::kDouble,
      TensorType::kString,   TensorType::kBool,    TensorType::kComplex64, TensorType::kComplex128,
  };
}

std::vector<AllowedType> AllTensorTypesIr9() {
  return {
      TensorType::kUint8,      TensorType::kUint16,         TensorType::kUint32,
      TensorType::kUint64,     TensorType::kInt8,           TensorType::kInt16,
      TensorType::kInt32,      TensorType::kInt64,          TensorType::kBfloat16,
      TensorType::kFloat16,    TensorType::kFloat,          TensorType::kDouble,
      TensorType::kString,     TensorType::kBool,           TensorType::kComplex64,
      TensorType::kComplex128, TensorType::kFloat8e4m3fn,   TensorType::kFloat8e4m3fnuz,
      TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
  };
}

std::vector<AllowedType> AllTensorTypesIr10() {
  std::vector<AllowedType> types = AllTensorTypesIr9();
  types.emplace_back(TensorType::kUint4);
  types.emplace_back(TensorType::kInt4);
  return types;
}

std::vector<AllowedType> AllTensorTypesIr11() {
  std::vector<AllowedType> types = AllTensorTypesIr10();
  types.emplace_back(TensorType::kFloat4e2m1);
  return types;
}

std::vector<AllowedType> AllTensorTypesIr12() {
  std::vector<AllowedType> types = AllTensorTypesIr11();
  types.emplace_back(TensorType::kFloat8e8m0);
  return types;
}

std::vector<AllowedType> AllTensorTypesIr13() {
  std::vector<AllowedType> types = AllTensorTypesIr12();
  types.emplace_back(TensorType::kUint2);
  types.emplace_back(TensorType::kInt2);
  return types;
}

std::vector<AllowedType> ConstantTypes(int since_version) {
  switch (since_version) {
  case 25:
    return AllTensorTypesIr13();
  case 24:
    return AllTensorTypesIr12();
  case 23:
    return AllTensorTypesIr11();
  case 21:
    return AllTensorTypesIr10();
  case 19:
    return AllTensorTypesIr9();
  case 13:
    return AllTensorTypesIr4();
  case 12:
  case 11:
  case 9:
    return AllTensorTypes();
  case 1:
    return FloatTypes();
  default:
    throw SchemaError("Unsupported Constant since_version: " + std::to_string(since_version));
  }
}

LightOpSchema MakeConstantSchema(int since_version) {
  return LightOpSchema(
      "Constant", kOnnxDomain, since_version, MakeConstantDoc(since_version), {},
      {
          {"output", "Output tensor containing the same value of the provided tensor.", "T"},
      },
      {
          {"T", ConstantTypes(since_version),
           since_version == 1 ? "Constrain input and output types to float tensors."
                              : "Constrain input and output types to all tensor types."},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpGeneratorSchemasWithHistory() {
  return std::vector<LightOpSchema>{
      MakeConstantSchema(25), MakeConstantSchema(24), MakeConstantSchema(23),
      MakeConstantSchema(21), MakeConstantSchema(19), MakeConstantSchema(13),
      MakeConstantSchema(12), MakeConstantSchema(11), MakeConstantSchema(9),
      MakeConstantSchema(1),
  };
}

} // namespace generator
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
