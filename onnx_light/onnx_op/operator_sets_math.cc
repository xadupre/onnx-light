// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"

#include <limits>
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

using VersionToSchemaMap = std::map<int, MathOpSchema>;
using OpNameToSchemasMap = std::unordered_map<std::string, VersionToSchemaMap>;

OpNameToSchemasMap &Registry() {
  static OpNameToSchemasMap registry;
  return registry;
}

MathOpSchema BuildElementwiseMathSchema(const char *op_name, const char *doc) {
  return MathOpSchema(op_name, OnnxOpMathDomain(), kMathCurrentVersion, doc,
                      {
                          {"A", "First operand.", "T"},
                          {"B", "Second operand.", "T"},
                      },
                      {
                          {"C", "Result, has same element type as two inputs", "T"},
                      },
                      {
                          {"T",
                           {
                               "tensor(uint8)",
                               "tensor(uint16)",
                               "tensor(uint32)",
                               "tensor(uint64)",
                               "tensor(int8)",
                               "tensor(int16)",
                               "tensor(int32)",
                               "tensor(int64)",
                               "tensor(float16)",
                               "tensor(float)",
                               "tensor(double)",
                               "tensor(bfloat16)",
                               "tensor(complex64)",
                               "tensor(complex128)",
                           },
                           "Constrain input and output types to all numeric tensors."},
                      });
}

void RegisterOneSchema(MathOpSchema schema, int target_version, bool fail_duplicate_schema) {
  if (target_version != 0 && schema.since_version() > target_version) {
    return;
  }

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

const std::string &OnnxOpMathDomain() {
  static const std::string domain = "ai.onnx.math.noshape";
  return domain;
}

void RegisterOnnxOpMathOperatorSetSchema(int target_version, bool fail_duplicate_schema) {
  RegisterOneSchema(
      BuildElementwiseMathSchema("Add", "Performs element-wise binary addition with broadcasting."),
      target_version, fail_duplicate_schema);
  RegisterOneSchema(BuildElementwiseMathSchema("Sub",
                                               "Performs element-wise binary subtraction with "
                                               "broadcasting."),
                    target_version, fail_duplicate_schema);
  RegisterOneSchema(BuildElementwiseMathSchema("Mul", "Performs element-wise binary multiplication "
                                                      "with broadcasting."),
                    target_version, fail_duplicate_schema);
  RegisterOneSchema(
      BuildElementwiseMathSchema("Div", "Performs element-wise binary division with broadcasting."),
      target_version, fail_duplicate_schema);
}

const MathOpSchema *GetOnnxOpMathSchema(const std::string &op_type, int max_inclusive_version) {
  const auto &registry = Registry();
  const auto op_it = registry.find(op_type);
  if (op_it == registry.end() || op_it->second.empty()) {
    return nullptr;
  }

  const auto &versions = op_it->second;
  if (max_inclusive_version == std::numeric_limits<int>::max()) {
    return &versions.rbegin()->second;
  }

  auto pos = versions.lower_bound(max_inclusive_version);
  if (pos == versions.end() || pos->first > max_inclusive_version) {
    if (pos == versions.begin()) {
      return nullptr;
    }
    --pos;
  }
  return &pos->second;
}

std::vector<MathOpSchema> GetAllOnnxOpMathSchemasWithHistory() {
  std::vector<MathOpSchema> schemas;
  for (const auto &by_name : Registry()) {
    for (const auto &by_version : by_name.second) {
      schemas.push_back(by_version.second);
    }
  }
  return schemas;
}

void DeregisterOnnxOpMathOperatorSetSchema() { Registry().clear(); }

static_assert(kMathMinVersion <= kMathCurrentVersion, "Invalid math version range.");

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
