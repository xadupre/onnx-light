// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_controlflow.h"
#include "onnx_op/operator_sets_controlflow_doc.h"

#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace controlflow {

namespace {

using AllowedType = TypeConstraintParam::AllowedType;

std::vector<AllowedType> OptionalSequenceTypes() {
  return {
      TensorType::kUint8,    TensorType::kUint16,  TensorType::kUint32,    TensorType::kUint64,
      TensorType::kInt8,     TensorType::kInt16,   TensorType::kInt32,     TensorType::kInt64,
      TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,     TensorType::kDouble,
      TensorType::kString,   TensorType::kBool,    TensorType::kComplex64, TensorType::kComplex128,
  };
}

std::vector<AllowedType> TensorTypesIr4() { return OptionalSequenceTypes(); }

std::vector<AllowedType> TensorTypesIr9() {
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

std::vector<AllowedType> TensorTypesIr10() {
  return {
      TensorType::kUint8,      TensorType::kUint16,         TensorType::kUint32,
      TensorType::kUint64,     TensorType::kInt8,           TensorType::kInt16,
      TensorType::kInt32,      TensorType::kInt64,          TensorType::kBfloat16,
      TensorType::kFloat16,    TensorType::kFloat,          TensorType::kDouble,
      TensorType::kString,     TensorType::kBool,           TensorType::kComplex64,
      TensorType::kComplex128, TensorType::kFloat8e4m3fn,   TensorType::kFloat8e4m3fnuz,
      TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz, TensorType::kUint4,
      TensorType::kInt4,
  };
}

std::vector<AllowedType> TensorTypesIr11() {
  std::vector<AllowedType> types = TensorTypesIr10();
  types.emplace_back(TensorType::kFloat4e2m1);
  return types;
}

std::vector<AllowedType> TensorTypesIr12() {
  std::vector<AllowedType> types = TensorTypesIr11();
  types.emplace_back(TensorType::kFloat8e8m0);
  return types;
}

std::vector<AllowedType> TensorTypesIr13() {
  std::vector<AllowedType> types = TensorTypesIr12();
  types.emplace_back(TensorType::kUint2);
  types.emplace_back(TensorType::kInt2);
  return types;
}

std::vector<AllowedType> MakeSequenceTypes(const std::vector<AllowedType> &tensor_types) {
  std::vector<AllowedType> types;
  types.reserve(tensor_types.size());
  for (const AllowedType &tensor_type : tensor_types) {
    types.emplace_back("seq(" + ToTypeString(tensor_type) + ")");
  }
  return types;
}

std::vector<AllowedType> MakeOptionalTypes(const std::vector<AllowedType> &tensor_types) {
  const std::vector<AllowedType> sequence_types = OptionalSequenceTypes();
  std::vector<AllowedType> types;
  types.reserve(sequence_types.size() + tensor_types.size());
  for (const AllowedType &sequence_type : sequence_types) {
    types.emplace_back("optional(seq(" + ToTypeString(sequence_type) + "))");
  }
  for (const AllowedType &tensor_type : tensor_types) {
    types.emplace_back("optional(" + ToTypeString(tensor_type) + ")");
  }
  return types;
}

std::vector<AllowedType> MakeControlFlowTypes(const std::vector<AllowedType> &tensor_types) {
  std::vector<AllowedType> types = tensor_types;
  std::vector<AllowedType> sequence_types = MakeSequenceTypes(tensor_types);
  std::vector<AllowedType> optional_types = MakeOptionalTypes(tensor_types);
  types.insert(types.end(), std::make_move_iterator(sequence_types.begin()),
               std::make_move_iterator(sequence_types.end()));
  types.insert(types.end(), std::make_move_iterator(optional_types.begin()),
               std::make_move_iterator(optional_types.end()));
  return types;
}

std::vector<AllowedType> IfTypesForVersion(int since_version) {
  switch (since_version) {
  case 25:
    return MakeControlFlowTypes(TensorTypesIr13());
  case 24:
    return MakeControlFlowTypes(TensorTypesIr12());
  case 23:
    return MakeControlFlowTypes(TensorTypesIr11());
  case 21:
    return MakeControlFlowTypes(TensorTypesIr10());
  case 19:
    return MakeControlFlowTypes(TensorTypesIr9());
  case 16:
    return MakeControlFlowTypes(TensorTypesIr4());
  default:
    throw std::invalid_argument("Unsupported If schema version.");
  }
}

LightOpSchema MakeIfSchema(int since_version) {
  return LightOpSchema(
      "If", kOnnxDomain, since_version, MakeIfDoc(),
      {
          {"cond", "Condition for the if. The tensor must contain a single element.", "B"},
      },
      {
          {"outputs", MakeIfOutputDescription(), "V"},
      },
      {
          {"V", IfTypesForVersion(since_version),
           MakeIfValueTypeConstraintDescription(since_version)},
          {"B", {TensorType::kBool}, "Only bool"},
      });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpControlFlowSchemasWithHistory() {
  return {
      MakeIfSchema(25), MakeIfSchema(24), MakeIfSchema(23),
      MakeIfSchema(21), MakeIfSchema(19), MakeIfSchema(16),
  };
}

} // namespace controlflow
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
