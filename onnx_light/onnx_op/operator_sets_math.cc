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
                        {"T", AllNumericTypesIr4(),
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
                        {"T", NumericTypesForMathReductionIr4(),
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
                        {"T", NumericTypesForMathReduction(),
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
                        {"T", NumericTypesForMathReduction(),
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
              {"T", FloatTypes(), "Constrain input and output types to float tensors."},
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
                        {"T", AllNumericTypesIr4(),
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
                        {"T", AllNumericTypes(),
                         "Constrain input and output types to high-precision numeric tensors."},
                    }));
  return schemas;
}

std::vector<LightOpSchema> BuildUnaryFloatMathSchemas(const char *op_type, int latest_version,
                                                      int previous_version) {
  const std::string doc = MakeUnaryMathDoc(op_type);
  const std::string output_description = MakeUnaryMathOutputDescription(op_type);
  return std::vector<LightOpSchema>{
      LightOpSchema(op_type, kOnnxDomain, latest_version, doc,
                    {
                        {"input", "Input tensor", "T"},
                    },
                    {
                        {"output", output_description, "T"},
                    },
                    {
                        {"T",
                         {TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,
                          TensorType::kDouble},
                         "Constrain input and output types to float tensors."},
                    }),
      LightOpSchema(op_type, kOnnxDomain, previous_version, doc,
                    {
                        {"input", "Input tensor", "T"},
                    },
                    {
                        {"output", output_description, "T"},
                    },
                    {
                        {"T", FloatTypes(), "Constrain input and output types to float tensors."},
                    })};
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
  std::vector<LightOpSchema> sin_schemas = BuildUnaryFloatMathSchemas("Sin", 22, 7);
  schemas.insert(schemas.end(), std::make_move_iterator(sin_schemas.begin()),
                 std::make_move_iterator(sin_schemas.end()));
  std::vector<LightOpSchema> cos_schemas = BuildUnaryFloatMathSchemas("Cos", 22, 7);
  schemas.insert(schemas.end(), std::make_move_iterator(cos_schemas.begin()),
                 std::make_move_iterator(cos_schemas.end()));
  std::vector<LightOpSchema> sinh_schemas = BuildUnaryFloatMathSchemas("Sinh", 22, 9);
  schemas.insert(schemas.end(), std::make_move_iterator(sinh_schemas.begin()),
                 std::make_move_iterator(sinh_schemas.end()));
  std::vector<LightOpSchema> cosh_schemas = BuildUnaryFloatMathSchemas("Cosh", 22, 9);
  schemas.insert(schemas.end(), std::make_move_iterator(cosh_schemas.begin()),
                 std::make_move_iterator(cosh_schemas.end()));
  return schemas;
}

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
