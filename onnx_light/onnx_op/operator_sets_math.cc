// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"

#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {
namespace {

constexpr int kMathMinVersion = 1;
constexpr int kMathCurrentVersion = 14;
constexpr const char *kOnnxDomain = "ai.onnx";

using VersionToSchemaMap = std::map<int, OpSchema>;
using OpNameToSchemasMap = std::unordered_map<std::string, VersionToSchemaMap>;

OpNameToSchemasMap &Registry() {
  static OpNameToSchemasMap registry;
  return registry;
}

std::vector<std::string> NumericTypeStrings() {
  constexpr TensorType kAllowedNumericTypes[] = {
      TensorType::kUint8,     TensorType::kUint16,     TensorType::kUint32, TensorType::kUint64,
      TensorType::kInt8,      TensorType::kInt16,      TensorType::kInt32,  TensorType::kInt64,
      TensorType::kFloat16,   TensorType::kFloat,      TensorType::kDouble, TensorType::kBfloat16,
      TensorType::kComplex64, TensorType::kComplex128,
  };

  std::vector<std::string> allowed_types;
  allowed_types.reserve(sizeof(kAllowedNumericTypes) / sizeof(kAllowedNumericTypes[0]));
  for (const TensorType tensor_type : kAllowedNumericTypes) {
    allowed_types.emplace_back(ToTypeString(tensor_type));
  }
  return allowed_types;
}

OpSchema BuildElementwiseMathSchema(const char *op_name, const char *doc) {
  return OpSchema(
      op_name, kOnnxDomain, kMathCurrentVersion, doc,
      {
          {"A", "First operand.", "T"},
          {"B", "Second operand.", "T"},
      },
      {
          {"C", "Result, has same element type as two inputs", "T"},
      },
      {
          {"T", NumericTypeStrings(), "Constrain input and output types to all numeric tensors."},
      });
}

void RegisterOneSchema(OpSchema schema, bool fail_duplicate_schema) {
  auto &schema_versions = Registry()[schema.name()];
  const int version = schema.since_version();
  if (schema_versions.count(version) != 0) {
    if (fail_duplicate_schema) {
      std::ostringstream error;
      error << "Duplicate schema registration for op " << schema.name() << " domain "
            << schema.domain() << " version " << version;
      throw SchemaError(error.str());
    }
    return;
  }

  schema_versions.emplace(version, std::move(schema));
}

} // namespace

void RegisterOnnxOpMathOperatorSetSchema(bool fail_duplicate_schema) {
  RegisterOneSchema(
      BuildElementwiseMathSchema("Add", "Performs element-wise binary addition with broadcasting."),
      fail_duplicate_schema);
  RegisterOneSchema(BuildElementwiseMathSchema("Sub",
                                               "Performs element-wise binary subtraction with "
                                               "broadcasting."),
                    fail_duplicate_schema);
  RegisterOneSchema(BuildElementwiseMathSchema("Mul", "Performs element-wise binary multiplication "
                                                      "with broadcasting."),
                    fail_duplicate_schema);
  RegisterOneSchema(
      BuildElementwiseMathSchema("Div", "Performs element-wise binary division with broadcasting."),
      fail_duplicate_schema);
}

std::vector<OpSchema> GetAllOnnxOpMathSchemasWithHistory() {
  std::vector<OpSchema> schemas;
  for (const auto &by_name : Registry()) {
    for (const auto &by_version : by_name.second) {
      schemas.push_back(by_version.second);
    }
  }
  return schemas;
}

static_assert(kMathMinVersion <= kMathCurrentVersion, "Invalid math version range.");

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
