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

std::vector<std::string> EqualTypesV19() {
  return {"tensor(bool)",     "tensor(uint8)",   "tensor(uint16)", "tensor(uint32)",
          "tensor(uint64)",   "tensor(int8)",    "tensor(int16)",  "tensor(int32)",
          "tensor(int64)",    "tensor(float16)", "tensor(float)",  "tensor(double)",
          "tensor(bfloat16)", "tensor(string)"};
}

std::vector<std::string> EqualTypesV13() {
  return {"tensor(bool)",  "tensor(uint8)",  "tensor(uint16)",  "tensor(uint32)", "tensor(uint64)",
          "tensor(int8)",  "tensor(int16)",  "tensor(int32)",   "tensor(int64)",  "tensor(float16)",
          "tensor(float)", "tensor(double)", "tensor(bfloat16)"};
}

std::vector<std::string> EqualTypesV11() {
  return {"tensor(bool)",   "tensor(uint8)",   "tensor(uint16)", "tensor(uint32)",
          "tensor(uint64)", "tensor(int8)",    "tensor(int16)",  "tensor(int32)",
          "tensor(int64)",  "tensor(float16)", "tensor(float)",  "tensor(double)"};
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

std::vector<LightOpSchema> BuildGreaterOrLessSchemas(const char *op_type) {
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
              {"T", AllNumericTypesIr4Strings(), "Constrain input types to all numeric tensors."},
              {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
          }),
      LightOpSchema(
          op_type, kOnnxDomain, 9, MakeBinaryLogicalOperatorDoc(op_type, 9),
          {
              {"A", "First input operand for the logical operator.", "T"},
              {"B", "Second input operand for the logical operator.", "T"},
          },
          {
              {"C", "Result tensor.", "T1"},
          },
          {
              {"T", AllNumericTypesStrings(), "Constrain input types to all numeric tensors."},
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
                        {"T", FloatTypeStrings(), "Constrain input to float tensors."},
                        {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
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
                        {"T", FloatTypeStrings(), "Constrain input to float tensors."},
                        {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
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
              {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
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
                        {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
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
                        {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
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
                        {"T",
                         {"tensor(bool)", "tensor(int32)", "tensor(int64)"},
                         "Constrain input to integral tensors."},
                        {"T1", {"tensor(bool)"}, "Constrain output to boolean tensor."},
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
                        {"T",
                         {"tensor(bool)", "tensor(int32)", "tensor(int64)"},
                         "Constrain input to integral tensors."},
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
  for (const char *op_type : {"Greater", "Less"}) {
    std::vector<LightOpSchema> logical_ops = BuildGreaterOrLessSchemas(op_type);
    schemas.insert(schemas.end(), std::make_move_iterator(logical_ops.begin()),
                   std::make_move_iterator(logical_ops.end()));
  }
  std::vector<LightOpSchema> equal_schemas = BuildEqualSchemas();
  schemas.insert(schemas.end(), std::make_move_iterator(equal_schemas.begin()),
                 std::make_move_iterator(equal_schemas.end()));
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
