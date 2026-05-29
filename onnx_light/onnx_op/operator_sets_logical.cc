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
                        {"T", {TensorType::kBool}, "Constrain input to boolean tensor."},
                        {"T1", {TensorType::kBool}, "Constrain output to boolean tensor."},
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
                        {"T", {TensorType::kBool}, "Constrain input to boolean tensor."},
                        {"T1", {TensorType::kBool}, "Constrain output to boolean tensor."},
                    })};
}

std::vector<LightOpSchema> BuildGreaterLessSchemas(const char *op_type) {
  return std::vector<LightOpSchema>{
      LightOpSchema(
          op_type, kOnnxDomain, 13, MakeBinaryLogicalOperatorDoc(op_type, 13),
          {
              {"A", "First input operand for the logical operator.", "T"},
              {"B", "Second input operand for the logical operator.", "T"},
          },
          {
              {"C", "Result tensor.", "T1"},
          },
          {
              {"T", AllNumericTypesIr4(), "Constrain input types to all numeric tensors."},
              {"T1", {TensorType::kBool}, "Constrain output to boolean tensor."},
          }),
      LightOpSchema(op_type, kOnnxDomain, 9, MakeBinaryLogicalOperatorDoc(op_type, 9),
                    {
                        {"A", "First input operand for the logical operator.", "T"},
                        {"B", "Second input operand for the logical operator.", "T"},
                    },
                    {
                        {"C", "Result tensor.", "T1"},
                    },
                    {
                        {"T", AllNumericTypes(), "Constrain input types to all numeric tensors."},
                        {"T1", {TensorType::kBool}, "Constrain output to boolean tensor."},
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
                        {"T", FloatTypes(), "Constrain input to float tensors."},
                        {"T1", {TensorType::kBool}, "Constrain output to boolean tensor."},
                    }),
      LightOpSchema(op_type, kOnnxDomain, 1, MakeBinaryLogicalOperatorDoc(op_type, 1),
                    {
                        {"A", "Left input tensor for the logical operator.", "T"},
                        {"B", "Right input tensor for the logical operator.", "T"},
                    },
                    {
                        {"C", "Result tensor.", "T1"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input to float tensors."},
                        {"T1", {TensorType::kBool}, "Constrain output to boolean tensor."},
                    })};
}

std::vector<LightOpSchema> BuildEqualSchemas() {
  return std::vector<LightOpSchema>{
      LightOpSchema(
          "Equal", kOnnxDomain, 19, MakeBinaryLogicalOperatorDoc("Equal", 19),
          {
              {"A", "First input operand for the logical operator.", "T"},
              {"B", "Second input operand for the logical operator.", "T"},
          },
          {
              {"C", "Result tensor.", "T1"},
          },
          {
              {"T", EqualTypesV19(), "Constrain input types to all (non-complex) tensors."},
              {"T1", {TensorType::kBool}, "Constrain output to boolean tensor."},
          }),
      LightOpSchema("Equal", kOnnxDomain, 13, MakeBinaryLogicalOperatorDoc("Equal", 13),
                    {
                        {"A", "First input operand for the logical operator.", "T"},
                        {"B", "Second input operand for the logical operator.", "T"},
                    },
                    {
                        {"C", "Result tensor.", "T1"},
                    },
                    {
                        {"T", EqualTypesV13(), "Constrain input types to all numeric tensors."},
                        {"T1", {TensorType::kBool}, "Constrain output to boolean tensor."},
                    }),
      LightOpSchema("Equal", kOnnxDomain, 11, MakeBinaryLogicalOperatorDoc("Equal", 11),
                    {
                        {"A", "First input operand for the logical operator.", "T"},
                        {"B", "Second input operand for the logical operator.", "T"},
                    },
                    {
                        {"C", "Result tensor.", "T1"},
                    },
                    {
                        {"T", EqualTypesV11(), "Constrain input types to all numeric tensors."},
                        {"T1", {TensorType::kBool}, "Constrain output to boolean tensor."},
                    }),
      LightOpSchema("Equal", kOnnxDomain, 7, MakeBinaryLogicalOperatorDoc("Equal", 7),
                    {
                        {"A", "First input operand for the logical operator.", "T"},
                        {"B", "Second input operand for the logical operator.", "T"},
                    },
                    {
                        {"C", "Result tensor.", "T1"},
                    },
                    {
                        {"T", EqualTypesV1V7(), "Constrain input to integral tensors."},
                        {"T1", {TensorType::kBool}, "Constrain output to boolean tensor."},
                    }),
      LightOpSchema("Equal", kOnnxDomain, 1, MakeBinaryLogicalOperatorDoc("Equal", 1),
                    {
                        {"A", "Left input tensor for the logical operator.", "T"},
                        {"B", "Right input tensor for the logical operator.", "T"},
                    },
                    {
                        {"C", "Result tensor.", "T1"},
                    },
                    {
                        {"T", EqualTypesV1V7(), "Constrain input to integral tensors."},
                        {"T1", {TensorType::kBool}, "Constrain output to boolean tensor."},
                    })};
}

std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory(bool init_doc,
                                                                 const std::string &op_type) {
  std::vector<LightOpSchema> schemas;
  for (const char *logical_op_type : {"And", "Or", "Xor"}) {
    std::vector<LightOpSchema> bin_ops = BuildBinaryLogicalSchema(logical_op_type);
    schemas.insert(schemas.end(), std::make_move_iterator(bin_ops.begin()),
                   std::make_move_iterator(bin_ops.end()));
  }
  for (const char *logical_op_type : {"Greater", "Less"}) {
    std::vector<LightOpSchema> comparison_ops = BuildGreaterLessSchemas(logical_op_type);
    schemas.insert(schemas.end(), std::make_move_iterator(comparison_ops.begin()),
                   std::make_move_iterator(comparison_ops.end()));
  }
  std::vector<LightOpSchema> equal_ops = BuildEqualSchemas();
  schemas.insert(schemas.end(), std::make_move_iterator(equal_ops.begin()),
                 std::make_move_iterator(equal_ops.end()));
  schemas.push_back(
      LightOpSchema("Not", kOnnxDomain, 1, MakeNotLogicalOperatorDoc(),
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", {TensorType::kBool}, "Constrain input/output to boolean tensors."},
                    }));
  return FilterSchemasByOpType(init_doc ? schemas : StripDocs(schemas), op_type);
}

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
