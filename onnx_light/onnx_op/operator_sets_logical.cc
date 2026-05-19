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

std::string BuildAndOperatorDoc(int since_version) {
  if (since_version == 1) {
    return R"DOC(
Returns the tensor resulted from performing the `and` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
)DOC";
  }

  return R"DOC(
Returns the tensor resulted from performing the `and` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
)DOC";
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);
  schemas.push_back(
      LightOpSchema("And", kOnnxDomain, 1, BuildAndOperatorDoc(1),
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
                    }));
  schemas.push_back(
      LightOpSchema("And", kOnnxDomain, 7, BuildAndOperatorDoc(7),
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
                    }));
  return schemas;
}

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
