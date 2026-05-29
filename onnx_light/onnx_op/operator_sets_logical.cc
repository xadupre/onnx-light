// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_logical.h"
#include "onnx_op/operator_sets_logical_doc.h"

#include <map>
#include <string>
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

namespace {

const std::vector<TensorType> &BitwiseIntTypes() {
  // Bitwise binary operators (BitwiseAnd / BitwiseOr / BitwiseXor) and BitwiseNot
  // all share the same opset 18 integer type set.
  static const std::vector<TensorType> kBitwiseIntTypes = {
      TensorType::kUint8, TensorType::kUint16, TensorType::kUint32, TensorType::kUint64,
      TensorType::kInt8,  TensorType::kInt16,  TensorType::kInt32,  TensorType::kInt64,
  };
  return kBitwiseIntTypes;
}

std::vector<LightOpSchema> BuildBinaryBitwiseSchemas(const char *op_type) {
  std::vector<LightOpSchema> schemas;
  schemas.push_back(
      LightOpSchema(op_type, kOnnxDomain, 18, MakeBinaryBitwiseOperatorDoc(op_type),
                    {
                        {"A", "First input operand for the bitwise operator.", "T"},
                        {"B", "Second input operand for the bitwise operator.", "T"},
                    },
                    {
                        {"C", "Result tensor.", "T"},
                    },
                    {
                        {"T", BitwiseIntTypes(), "Constrain input to integer tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildBitwiseNotSchemas() {
  std::vector<LightOpSchema> schemas;
  schemas.push_back(
      LightOpSchema("BitwiseNot", kOnnxDomain, 18, MakeBitwiseNotOperatorDoc(),
                    {
                        {"X", "Input tensor", "T"},
                    },
                    {
                        {"Y", "Output tensor", "T"},
                    },
                    {
                        {"T", BitwiseIntTypes(), "Constrain input/output to integer tensors."},
                    }));
  return schemas;
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory(const std::string &op_type,
                                                                 bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"And", [] { return BuildBinaryLogicalSchema("And"); }},
      {"BitwiseAnd", [] { return BuildBinaryBitwiseSchemas("BitwiseAnd"); }},
      {"BitwiseNot", [] { return BuildBitwiseNotSchemas(); }},
      {"BitwiseOr", [] { return BuildBinaryBitwiseSchemas("BitwiseOr"); }},
      {"BitwiseXor", [] { return BuildBinaryBitwiseSchemas("BitwiseXor"); }},
      {"Equal", [] { return BuildEqualSchemas(); }},
      {"Greater", [] { return BuildGreaterLessSchemas("Greater"); }},
      {"Less", [] { return BuildGreaterLessSchemas("Less"); }},
      {"Not",
       [] {
         std::vector<LightOpSchema> schemas;
         schemas.push_back(LightOpSchema(
             "Not", kOnnxDomain, 1, MakeNotLogicalOperatorDoc(),
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
       }},
      {"Or", [] { return BuildBinaryLogicalSchema("Or"); }},
      {"Xor", [] { return BuildBinaryLogicalSchema("Xor"); }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
