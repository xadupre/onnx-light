// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_logical.h"
#include "onnx_op/operator_sets_logical_utils.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace logical {
namespace {

constexpr const char *kOnnxDomain = "ai.onnx";

LightOpSchema BuildBinaryLogicalSchema(const char *op_type, const char *op_name,
                                       int since_version) {
  if (since_version == 1) {
    return LightOpSchema(op_type, kOnnxDomain, 1, detail::MakeBinaryLogicalOperatorDoc(op_name, 1),
                         {
                             {"A", "Left input tensor for the logical operator.", "T"},
                             {"B", "Right input tensor for the logical operator.", "T"},
                         },
                         {
                             {"C", "Result tensor.", "T1"},
                         },
                         {
                             {"T", {"tensor(bool)"}, "Constrain input to boolean tensor."},
                             {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
                         });
  }

  return LightOpSchema(op_type, kOnnxDomain, 7, detail::MakeBinaryLogicalOperatorDoc(op_name, 7),
                       {
                           {"A", "First input operand for the logical operator.", "T"},
                           {"B", "Second input operand for the logical operator.", "T"},
                       },
                       {
                           {"C", "Result tensor.", "T1"},
                       },
                       {
                           {"T", {"tensor(bool)"}, "Constrain input to boolean tensor."},
                           {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
                       });
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(7);
  schemas.push_back(BuildBinaryLogicalSchema("And", "and", 1));
  schemas.push_back(BuildBinaryLogicalSchema("And", "and", 7));
  schemas.push_back(BuildBinaryLogicalSchema("Or", "or", 1));
  schemas.push_back(BuildBinaryLogicalSchema("Or", "or", 7));
  schemas.push_back(BuildBinaryLogicalSchema("Xor", "xor", 1));
  schemas.push_back(BuildBinaryLogicalSchema("Xor", "xor", 7));
  schemas.push_back(
      LightOpSchema("Not", kOnnxDomain, 1, detail::MakeNotLogicalOperatorDoc(),
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", {"tensor(bool)"}, "Constrain input/output to boolean tensors."},
                    }));
  return schemas;
}

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
