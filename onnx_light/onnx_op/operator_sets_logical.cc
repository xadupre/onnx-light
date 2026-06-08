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

std::vector<LightOpSchema> BuildGreaterLessOrEqualSchemas(const char *op_type) {
  // ``GreaterOrEqual`` / ``LessOrEqual`` were introduced at opset 12 (using
  // the same numeric type set as ``Greater``/``Less`` v9) and extended at
  // opset 16 to ``all_numeric_types_ir4`` (matching ``Greater``/``Less`` v13).
  return std::vector<LightOpSchema>{
      LightOpSchema(
          op_type, kOnnxDomain, 16, MakeBinaryLogicalOperatorDoc(op_type, 16),
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
      LightOpSchema(op_type, kOnnxDomain, 12, MakeBinaryLogicalOperatorDoc(op_type, 12),
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

std::vector<LightOpSchema> BuildWhereSchemas() {
  const std::vector<TensorType> where_types_v9 = {
      TensorType::kUint8,   TensorType::kUint16,    TensorType::kUint32,    TensorType::kUint64,
      TensorType::kInt8,    TensorType::kInt16,     TensorType::kInt32,     TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,     TensorType::kDouble,    TensorType::kString,
      TensorType::kBool,    TensorType::kComplex64, TensorType::kComplex128};
  const std::vector<TensorType> where_types_v16 = {
      TensorType::kUint8,    TensorType::kUint16,  TensorType::kUint32,    TensorType::kUint64,
      TensorType::kInt8,     TensorType::kInt16,   TensorType::kInt32,     TensorType::kInt64,
      TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,     TensorType::kDouble,
      TensorType::kString,   TensorType::kBool,    TensorType::kComplex64, TensorType::kComplex128};
  return std::vector<LightOpSchema>{
      LightOpSchema(
          "Where", kOnnxDomain, 16, MakeWhereOperatorDoc(),
          {
              {"condition", "When True (nonzero), yield X, otherwise yield Y", "B"},
              {"X", "values selected at indices where condition is True", "T"},
              {"Y", "values selected at indices where condition is False", "T"},
          },
          {
              {"output", "Tensor of shape equal to the broadcasted shape of condition, X, and Y.",
               "T"},
          },
          {
              {"B", {TensorType::kBool}, "Constrain to boolean tensors."},
              {"T", where_types_v16,
               "Constrain input and output types to all tensor types (including bfloat)."},
          }),
      LightOpSchema(
          "Where", kOnnxDomain, 9, MakeWhereOperatorDoc(),
          {
              {"condition", "When True (nonzero), yield X, otherwise yield Y", "B"},
              {"X", "values selected at indices where condition is True", "T"},
              {"Y", "values selected at indices where condition is False", "T"},
          },
          {
              {"output", "Tensor of shape equal to the broadcasted shape of condition, X, and Y.",
               "T"},
          },
          {
              {"B", {TensorType::kBool}, "Constrain to boolean tensors."},
              {"T", where_types_v9, "Constrain input and output types to all tensor types."},
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

std::vector<LightOpSchema> BuildIsNaNSchemas() {
  static constexpr const char *kIsNaNDoc = R"DOC(Returns which elements of the input are NaN.)DOC";
  // ``IsNaN`` v20 widens the input dtype set to every IR-9 float type
  // (adds ``float8e*``). v13 added ``bfloat16``; v9 only accepted the
  // three legacy float types.
  static const std::vector<TensorType> kIsNaNTypesV20 = {
      TensorType::kBfloat16,   TensorType::kFloat16,        TensorType::kFloat,
      TensorType::kDouble,     TensorType::kFloat8e4m3fn,   TensorType::kFloat8e4m3fnuz,
      TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
  };
  static const std::vector<TensorType> kIsNaNTypesV13 = {
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
      TensorType::kBfloat16,
  };
  std::vector<LightOpSchema> schemas;
  schemas.reserve(3);
  schemas.push_back(
      LightOpSchema("IsNaN", kOnnxDomain, 20, kIsNaNDoc,
                    {
                        {"X", "input", "T1"},
                    },
                    {
                        {"Y", "output", "T2"},
                    },
                    {
                        {"T1", kIsNaNTypesV20, "Constrain input types to float tensors."},
                        {"T2", {TensorType::kBool}, "Constrain output types to boolean tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("IsNaN", kOnnxDomain, 13, kIsNaNDoc,
                    {
                        {"X", "input", "T1"},
                    },
                    {
                        {"Y", "output", "T2"},
                    },
                    {
                        {"T1", kIsNaNTypesV13, "Constrain input types to float tensors."},
                        {"T2", {TensorType::kBool}, "Constrain output types to boolean tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("IsNaN", kOnnxDomain, 9, kIsNaNDoc,
                    {
                        {"X", "input", "T1"},
                    },
                    {
                        {"Y", "output", "T2"},
                    },
                    {
                        {"T1", FloatTypes(), "Constrain input types to float tensors."},
                        {"T2", {TensorType::kBool}, "Constrain output types to boolean tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildIsInfSchemas() {
  static constexpr const char *kIsInfDoc =
      R"DOC(Map infinity to true and other values to false.)DOC";
  // ``IsInf`` v20 widens the input dtype set to every IR-9 float type;
  // v10 only accepted ``float`` and ``double``.
  static const std::vector<TensorType> kIsInfTypesV20 = {
      TensorType::kBfloat16,   TensorType::kFloat16,        TensorType::kFloat,
      TensorType::kDouble,     TensorType::kFloat8e4m3fn,   TensorType::kFloat8e4m3fnuz,
      TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
  };
  static const std::vector<TensorType> kIsInfTypesV10 = {
      TensorType::kFloat,
      TensorType::kDouble,
  };
  const std::vector<AttributeParam> kIsInfAttributes = {
      {"detect_positive",
       "(Optional) Whether map positive infinity to true. Default to 1 "
       "so that positive infinity induces true. Set this attribute to 0 "
       "if positive infinity should be mapped to false.",
       AttributeType::INT, /*required=*/false, static_cast<int64_t>(1)},
      {"detect_negative",
       "(Optional) Whether map negative infinity to true. Default to 1 "
       "so that negative infinity induces true. Set this attribute to 0 "
       "if negative infinity should be mapped to false.",
       AttributeType::INT, /*required=*/false, static_cast<int64_t>(1)},
  };
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);
  schemas.push_back(
      LightOpSchema("IsInf", kOnnxDomain, 20, kIsInfDoc,
                    {
                        {"X", "input", "T1"},
                    },
                    {
                        {"Y", "output", "T2"},
                    },
                    {
                        {"T1", kIsInfTypesV20, "Constrain input types to float tensors."},
                        {"T2", {TensorType::kBool}, "Constrain output types to boolean tensors."},
                    },
                    kIsInfAttributes));
  schemas.push_back(
      LightOpSchema("IsInf", kOnnxDomain, 10, kIsInfDoc,
                    {
                        {"X", "input", "T1"},
                    },
                    {
                        {"Y", "output", "T2"},
                    },
                    {
                        {"T1", kIsInfTypesV10, "Constrain input types to float tensors."},
                        {"T2", {TensorType::kBool}, "Constrain output types to boolean tensors."},
                    },
                    kIsInfAttributes));
  return schemas;
}

std::vector<LightOpSchema> BuildBitShiftSchemas() {
  // BitShift (opset 11): two integer inputs (unsigned dtypes) with
  // multidirectional broadcasting and a required ``direction`` string
  // attribute selecting LEFT or RIGHT shift.
  static const std::vector<TensorType> kBitShiftIntTypes = {
      TensorType::kUint8,
      TensorType::kUint16,
      TensorType::kUint32,
      TensorType::kUint64,
  };
  std::vector<AttributeParam> attributes;
  attributes.push_back({"direction",
                        "Direction of moving bits. It can be either \"RIGHT\" (for right shift) "
                        "or \"LEFT\" (for left shift).",
                        AttributeType::STRING, /*required=*/true});
  std::vector<LightOpSchema> schemas;
  schemas.push_back(LightOpSchema(
      "BitShift", kOnnxDomain, 11, MakeBitShiftOperatorDoc(),
      {
          {"X", "First operand, input to be shifted.", "T"},
          {"Y", "Second operand, amounts of shift.", "T"},
      },
      {
          {"Z", "Output tensor", "T"},
      },
      {
          {"T", kBitShiftIntTypes, "Constrain input and output types to integer tensors."},
      },
      std::move(attributes)));
  return schemas;
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpLogicalSchemasWithHistory(const std::string &op_type,
                                                                 bool init_doc) {
  static const std::map<std::string, SchemaBuilder> builders = {
      {"And", [] { return BuildBinaryLogicalSchema("And"); }},
      {"BitShift", [] { return BuildBitShiftSchemas(); }},
      {"BitwiseAnd", [] { return BuildBinaryBitwiseSchemas("BitwiseAnd"); }},
      {"BitwiseNot", [] { return BuildBitwiseNotSchemas(); }},
      {"BitwiseOr", [] { return BuildBinaryBitwiseSchemas("BitwiseOr"); }},
      {"BitwiseXor", [] { return BuildBinaryBitwiseSchemas("BitwiseXor"); }},
      {"Equal", [] { return BuildEqualSchemas(); }},
      {"Greater", [] { return BuildGreaterLessSchemas("Greater"); }},
      {"GreaterOrEqual", [] { return BuildGreaterLessOrEqualSchemas("GreaterOrEqual"); }},
      {"IsInf", [] { return BuildIsInfSchemas(); }},
      {"IsNaN", [] { return BuildIsNaNSchemas(); }},
      {"Less", [] { return BuildGreaterLessSchemas("Less"); }},
      {"LessOrEqual", [] { return BuildGreaterLessOrEqualSchemas("LessOrEqual"); }},
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
      {"Where", [] { return BuildWhereSchemas(); }},
      {"Xor", [] { return BuildBinaryLogicalSchema("Xor"); }},
  };
  return CollectSchemasFromBuilders(builders, op_type, init_doc);
}

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
