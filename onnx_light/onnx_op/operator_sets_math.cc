// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"

#include <iterator>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {
namespace {

constexpr int kMathMinVersion = 1;
constexpr int kMathCurrentVersion = 14;
constexpr const char *kOnnxDomain = "ai.onnx";
constexpr int kMathSupportedVersions[] = {14, 13, 7, 6, 1};

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

LightOpSchema BuildElementwiseMathSchema(const char *op_name, int since_version, const char *doc) {
  return LightOpSchema(
      op_name, kOnnxDomain, since_version, doc,
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

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpMathSchemasWithHistory() {
  struct OpDoc {
    const char *name;
    const char *doc;
  };

  constexpr OpDoc kOps[] = {
      {"Add", "Performs element-wise binary addition with broadcasting."},
      {"Div", "Performs element-wise binary division with broadcasting."},
      {"Mul", "Performs element-wise binary multiplication with broadcasting."},
      {"Sub", "Performs element-wise binary subtraction with broadcasting."},
  };

  std::vector<LightOpSchema> schemas;
  schemas.reserve(std::size(kMathSupportedVersions) * std::size(kOps));
  for (const OpDoc &op : kOps) {
    for (const int version : kMathSupportedVersions) {
      schemas.push_back(BuildElementwiseMathSchema(op.name, version, op.doc));
    }
  }
  return schemas;
}

static_assert(kMathMinVersion <= kMathCurrentVersion, "Invalid math version range.");

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
