// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_logical.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace logical {
namespace {

constexpr const char *kOnnxDomain = "ai.onnx";

std::string BuildBinaryLogicalOperatorDoc(const char *op_name, int since_version) {
  if (since_version == 1) {
    return R"DOC(
Returns the tensor resulted from performing the `)DOC" +
           std::string(op_name) +
           R"DOC(` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
)DOC";
  }

  return R"DOC(
Returns the tensor resulted from performing the `)DOC" +
         std::string(op_name) +
         R"DOC(` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
)DOC";
}

LightOpSchema BuildBinaryLogicalSchema(const char *op_type, const char *op_name,
                                       int since_version) {
  if (since_version == 1) {
    return LightOpSchema(op_type, kOnnxDomain, 1, BuildBinaryLogicalOperatorDoc(op_name, 1),
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

  return LightOpSchema(op_type, kOnnxDomain, 7, BuildBinaryLogicalOperatorDoc(op_name, 7),
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
      LightOpSchema("Not", kOnnxDomain, 1, R"DOC(
Returns the negation of the input tensor element-wise.
)DOC",
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
