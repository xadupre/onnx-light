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

std::vector<TensorType> EqualTypesV1V7() {
  return {
      TensorType::kBool,
      TensorType::kInt32,
      TensorType::kInt64,
  };
}

std::vector<TensorType> EqualTypesV11() {
  return {
      TensorType::kBool,   TensorType::kUint8,   TensorType::kUint16, TensorType::kUint32,
      TensorType::kUint64, TensorType::kInt8,    TensorType::kInt16,  TensorType::kInt32,
      TensorType::kInt64,  TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
  };
}

std::vector<TensorType> EqualTypesV13() {
  std::vector<TensorType> types = EqualTypesV11();
  types.push_back(TensorType::kBfloat16);
  return types;
}

std::vector<TensorType> EqualTypesV19() {
  std::vector<TensorType> types = EqualTypesV13();
  types.push_back(TensorType::kString);
  return types;
}

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
          "Equal", kOnnxDomain, 19, MakeBinaryLogicalOperatorDoc("equal", 19),
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
      LightOpSchema("Equal", kOnnxDomain, 13, MakeBinaryLogicalOperatorDoc("equal", 13),
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
      LightOpSchema("Equal", kOnnxDomain, 11, MakeBinaryLogicalOperatorDoc("equal", 11),
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
      LightOpSchema("Equal", kOnnxDomain, 7, MakeBinaryLogicalOperatorDoc("equal", 7),
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
      LightOpSchema("Equal", kOnnxDomain, 1, MakeBinaryLogicalOperatorDoc("equal", 1),
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

std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory() {
  std::vector<LightOpSchema> schemas;
  for (const char *op_type : {"And", "Or", "Xor"}) {
    std::vector<LightOpSchema> bin_ops = BuildBinaryLogicalSchema(op_type);
    schemas.insert(schemas.end(), std::make_move_iterator(bin_ops.begin()),
                   std::make_move_iterator(bin_ops.end()));
  }
  for (const char *op_type : {"Greater", "Less"}) {
    std::vector<LightOpSchema> comparison_ops = BuildGreaterLessSchemas(op_type);
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
  return schemas;
}

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
