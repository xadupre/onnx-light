// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_logical.h"
#include "onnx_op/operator_sets_logical_doc.h"

#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace logical {

std::vector<LightOpSchema> BuildBinaryLogicalSchema(const char *op_type) {
  return std::vector<LightOpSchema>{
      LightOpSchema(op_type, kOnnxDomain, 1, MakeBinaryLogicalOperatorDoc(op_type, 1),
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
                    }),
      LightOpSchema(op_type, kOnnxDomain, 7, MakeBinaryLogicalOperatorDoc(op_type, 7),
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
                    })};
}

std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory() {
  std::vector<LightOpSchema> schemas;
  for (const char *op_type : {"And", "Or", "Xor"}) {
    std::vector<LightOpSchema> bin_ops = BuildBinaryLogicalSchema(op_type);
    schemas.insert(schemas.end(), std::make_move_iterator(bin_ops.begin()),
                   std::make_move_iterator(bin_ops.end()));
  }
  schemas.push_back(
      LightOpSchema("Not", kOnnxDomain, 1, MakeNotLogicalOperatorDoc(),
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
