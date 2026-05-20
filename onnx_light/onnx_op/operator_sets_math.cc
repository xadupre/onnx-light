// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"
#include "onnx_op/operator_sets_math_doc.h"

#include <iterator>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {

std::vector<LightOpSchema> BuildElementwiseMathSchemaForVersion(const char *op_type) {
  return std::vector<LightOpSchema>{
      LightOpSchema(op_type, kOnnxDomain, 14, MakeElementwiseMathDoc(op_type, 14),
                    {
                        {"A", "First operand.", "T"},
                        {"B", "Second operand.", "T"},
                    },
                    {
                        {"C", "Result, has same element type as two inputs", "T"},
                    },
                    {
                        {"T", AllNumericTypesIr4Strings(),
                         "Constrain input and output types to all numeric tensors."},
                    }),

      LightOpSchema(op_type, kOnnxDomain, 13, MakeElementwiseMathDoc(op_type, 13),
                    {
                        {"A", "First operand.", "T"},
                        {"B", "Second operand.", "T"},
                    },
                    {
                        {"C", "Result, has same element type as two inputs", "T"},
                    },
                    {
                        {"T", NumericTypesForMathReductionIr4Strings(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }),
      LightOpSchema(op_type, kOnnxDomain, 7, MakeElementwiseMathDoc(op_type, 7),
                    {
                        {"A", "First operand.", "T"},
                        {"B", "Second operand.", "T"},
                    },
                    {
                        {"C", "Result, has same element type as two inputs", "T"},
                    },
                    {
                        {"T", NumericTypesForMathReductionStrings(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }),
      LightOpSchema(op_type, kOnnxDomain, 6, MakeElementwiseMathDoc(op_type, 6),
                    {
                        {"A", "First operand, should share the type with the second operand.", "T"},
                        {"B",
                         "Second operand. With broadcasting can be of smaller size than A. If "
                         "broadcasting is disabled it should be of the same size.",
                         "T"},
                    },
                    {
                        {"C", "Result, has same dimensions and type as A", "T"},
                    },
                    {
                        {"T", NumericTypesForMathReductionStrings(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }),
      LightOpSchema(
          op_type, kOnnxDomain, 1, MakeElementwiseMathDoc(op_type, 1),
          {
              {"A", "First operand, should share the type with the second operand.", "T"},
              {"B",
               "Second operand. With broadcasting can be of smaller size than A. If broadcasting "
               "is disabled it should be of the same size.",
               "T"},
          },
          {
              {"C", "Result, has same dimensions and type as A", "T"},
          },
          {
              {"T", FloatTypeStrings(), "Constrain input and output types to float tensors."},
          })};
}

std::vector<LightOpSchema> BuildModSchemas() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);
  schemas.push_back(
      LightOpSchema("Mod", kOnnxDomain, 13, "Performs an element-wise binary modulo operation.",
                    {
                        {"A", "Dividend tensor", "T"},
                        {"B", "Divisor tensor", "T"},
                    },
                    {
                        {"C", "Remainder tensor", "T"},
                    },
                    {
                        {"T", AllNumericTypesIr4Strings(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }));
  schemas.push_back(
      LightOpSchema("Mod", kOnnxDomain, 10, "Performs element-wise binary modulus.",
                    {
                        {"A", "Dividend tensor", "T"},
                        {"B", "Divisor tensor", "T"},
                    },
                    {
                        {"C", "Remainder tensor", "T"},
                    },
                    {
                        {"T", AllNumericTypesStrings(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }));
  return schemas;
}

std::vector<std::string> CastTypesV1V6() {
  return {"tensor(float16)", "tensor(float)",  "tensor(double)", "tensor(int8)",
          "tensor(int16)",   "tensor(int32)",  "tensor(int64)",  "tensor(uint8)",
          "tensor(uint16)",  "tensor(uint32)", "tensor(uint64)", "tensor(bool)"};
}

std::vector<std::string> CastTypesV9() {
  return {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(int8)",   "tensor(int16)",
          "tensor(int32)",   "tensor(int64)", "tensor(uint8)",  "tensor(uint16)", "tensor(uint32)",
          "tensor(uint64)",  "tensor(bool)",  "tensor(string)"};
}

std::vector<std::string> CastTypesV13() {
  return {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(int8)",    "tensor(int16)",
          "tensor(int32)",   "tensor(int64)", "tensor(uint8)",  "tensor(uint16)",  "tensor(uint32)",
          "tensor(uint64)",  "tensor(bool)",  "tensor(string)", "tensor(bfloat16)"};
}

std::vector<std::string> CastTypesV19() {
  return {"tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(int8)",           "tensor(int16)",      "tensor(int32)",
          "tensor(int64)",          "tensor(uint8)",      "tensor(uint16)",
          "tensor(uint32)",         "tensor(uint64)",     "tensor(bool)",
          "tensor(string)",         "tensor(bfloat16)",   "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)"};
}

std::vector<std::string> CastTypesV21() {
  return {"tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(int8)",           "tensor(int16)",      "tensor(int32)",
          "tensor(int64)",          "tensor(uint8)",      "tensor(uint16)",
          "tensor(uint32)",         "tensor(uint64)",     "tensor(bool)",
          "tensor(string)",         "tensor(bfloat16)",   "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
          "tensor(uint4)",          "tensor(int4)"};
}

std::vector<std::string> CastTypesV23() {
  return {"tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
          "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
          "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
          "tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
          "tensor(uint4)",          "tensor(int4)",       "tensor(float4e2m1)"};
}

std::vector<std::string> CastTypesV24() {
  return {"tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
          "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
          "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
          "tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
          "tensor(uint4)",          "tensor(int4)",       "tensor(float4e2m1)",
          "tensor(float8e8m0)"};
}

std::vector<std::string> CastTypesV25() {
  return {"tensor(uint8)",          "tensor(uint16)",     "tensor(uint32)",
          "tensor(uint64)",         "tensor(int8)",       "tensor(int16)",
          "tensor(int32)",          "tensor(int64)",      "tensor(bfloat16)",
          "tensor(float16)",        "tensor(float)",      "tensor(double)",
          "tensor(string)",         "tensor(bool)",       "tensor(float8e4m3fn)",
          "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)",
          "tensor(uint4)",          "tensor(int4)",       "tensor(float4e2m1)",
          "tensor(float8e8m0)",     "tensor(uint2)",      "tensor(int2)"};
}

std::vector<LightOpSchema> BuildCastSchemas() {
  const auto add_cast_schema = [](std::vector<LightOpSchema> &schemas, int version,
                                  const std::vector<std::string> &types,
                                  const char *input_constraint_description,
                                  const char *output_constraint_description) {
    schemas.push_back(
        LightOpSchema("Cast", kOnnxDomain, version,
                      "Casts the elements of an input tensor to a specified data type.",
                      {
                          {"input", "Input tensor to be cast.", "T1"},
                      },
                      {
                          {"output",
                           "Output tensor with the same shape as input with type specified by the "
                           "'to' argument",
                           "T2"},
                      },
                      {
                          {"T1", types, input_constraint_description},
                          {"T2", types, output_constraint_description},
                      }));
  };

  std::vector<LightOpSchema> schemas;
  schemas.reserve(9);
  const char *legacy_input_constraint =
      "Constrain input types. Casting from strings and complex are not supported.";
  const char *legacy_output_constraint =
      "Constrain output types. Casting to strings and complex are not supported.";
  const char *input_constraint = "Constrain input types. Casting from complex is not supported.";
  const char *output_constraint = "Constrain output types. Casting to complex is not supported.";
  add_cast_schema(schemas, 1, CastTypesV1V6(), legacy_input_constraint, legacy_output_constraint);
  add_cast_schema(schemas, 6, CastTypesV1V6(), legacy_input_constraint, legacy_output_constraint);
  add_cast_schema(schemas, 9, CastTypesV9(), input_constraint, output_constraint);
  add_cast_schema(schemas, 13, CastTypesV13(), input_constraint, output_constraint);
  add_cast_schema(schemas, 19, CastTypesV19(), input_constraint, output_constraint);
  add_cast_schema(schemas, 21, CastTypesV21(), input_constraint, output_constraint);
  add_cast_schema(schemas, 23, CastTypesV23(), input_constraint, output_constraint);
  add_cast_schema(schemas, 24, CastTypesV24(), input_constraint, output_constraint);
  add_cast_schema(schemas, 25, CastTypesV25(), input_constraint, output_constraint);
  return schemas;
}

std::vector<LightOpSchema> GetAllOnnxOpMathSchemasWithHistory() {
  std::vector<LightOpSchema> schemas;
  for (const auto &op_type : {"Add", "Div", "Mul", "Sub"}) {
    std::vector<LightOpSchema> bin_schemas = BuildElementwiseMathSchemaForVersion(op_type);
    schemas.insert(schemas.end(), std::make_move_iterator(bin_schemas.begin()),
                   std::make_move_iterator(bin_schemas.end()));
  }
  std::vector<LightOpSchema> mod_schemas = BuildModSchemas();
  schemas.insert(schemas.end(), std::make_move_iterator(mod_schemas.begin()),
                 std::make_move_iterator(mod_schemas.end()));
  std::vector<LightOpSchema> cast_schemas = BuildCastSchemas();
  schemas.insert(schemas.end(), std::make_move_iterator(cast_schemas.begin()),
                 std::make_move_iterator(cast_schemas.end()));
  return schemas;
}

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
