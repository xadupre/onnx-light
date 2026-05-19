// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/math/operator_sets.h"

#include <algorithm>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {
namespace {

OpSchema BuildElementwiseMathSchema(const char *op_name, const char *doc) {
  OpSchema schema;
  schema.SetName(op_name)
      .SetDomain(OnnxOpMathDomain())
      .SinceVersion(14)
      .SetDoc(doc)
      .Input(0, "A", "First operand.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Input(1, "B", "Second operand.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
      .Output(0, "C", "Result, has same element type as two inputs", "T", OpSchema::Single, true, 1,
              OpSchema::Differentiable)
      .TypeConstraint("T", OpSchema::all_numeric_types_ir4(),
                      "Constrain input and output types to all numeric tensors.");
  return schema;
}

void EnsureMathDomainVersionRange() {
  auto &domain_to_version = OpSchemaRegistry::DomainToVersionRange::Instance();
  const auto &version_map = domain_to_version.Map();
  const auto it = version_map.find(OnnxOpMathDomain());
  if (it == version_map.end()) {
    domain_to_version.AddDomainToVersion(OnnxOpMathDomain(), 1, 14, 14);
    return;
  }

  const int min_version = std::min(it->second.first, 1);
  const int max_version = std::max(it->second.second, 14);
  domain_to_version.UpdateDomainToVersion(OnnxOpMathDomain(), min_version, max_version, 14);
}

} // namespace

const std::string &OnnxOpMathDomain() {
  static const std::string domain = "ai.onnx.math.noshape";
  return domain;
}

void RegisterOnnxOpMathOperatorSetSchema(int target_version, bool fail_duplicate_schema) {
  EnsureMathDomainVersionRange();

  RegisterSchema(
      BuildElementwiseMathSchema("Add", "Performs element-wise binary addition with broadcasting."),
      target_version, fail_duplicate_schema);
  RegisterSchema(BuildElementwiseMathSchema("Sub", "Performs element-wise binary subtraction with "
                                                   "broadcasting."),
                 target_version, fail_duplicate_schema);
  RegisterSchema(BuildElementwiseMathSchema("Mul",
                                            "Performs element-wise binary multiplication with "
                                            "broadcasting."),
                 target_version, fail_duplicate_schema);
  RegisterSchema(BuildElementwiseMathSchema("Div", "Performs element-wise binary division with "
                                                   "broadcasting."),
                 target_version, fail_duplicate_schema);
}

void DeregisterOnnxOpMathOperatorSetSchema() {
  OpSchemaRegistry::Instance()->OpSchemaDeregisterAll(OnnxOpMathDomain());
}

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
