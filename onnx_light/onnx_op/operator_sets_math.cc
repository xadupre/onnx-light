// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"
#include "onnx_op/operator_sets_math_doc.h"

#include <cstring>
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

std::vector<LightOpSchema> BuildSinCosSchemas(const char *op_type) {
  const std::string output_desc = "The " +
                                  std::string(strcmp(op_type, "Sin") == 0 ? "sine" : "cosine") +
                                  " of the input tensor computed element-wise";
  using VersionTypes = std::pair<int, std::vector<std::string>>;
  const std::vector<VersionTypes> version_types = {
      {7, FloatTypeStrings()},
      {22, FloatTypeIr4Strings()},
  };
  std::vector<LightOpSchema> schemas;
  schemas.reserve(version_types.size());
  for (const auto &[version, types] : version_types) {
    schemas.push_back(
        LightOpSchema(op_type, kOnnxDomain, version, MakeSinCosDoc(op_type),
                      {{"input", "Input tensor", "T"}}, {{"output", output_desc, "T"}},
                      {{"T", types, "Constrain input and output types to float tensors."}}));
  }
  return schemas;
}

std::vector<LightOpSchema> BuildSinhCoshSchemas(const char *op_type) {
  const std::string output_desc = std::string("The hyperbolic ") +
                                  (strcmp(op_type, "Sinh") == 0 ? "sine" : "cosine") +
                                  " values of the input tensor computed element-wise";
  using VersionTypes = std::pair<int, std::vector<std::string>>;
  const std::vector<VersionTypes> version_types = {
      {9, FloatTypeStrings()},
      {22, FloatTypeIr4Strings()},
  };
  std::vector<LightOpSchema> schemas;
  schemas.reserve(version_types.size());
  for (const auto &[version, types] : version_types) {
    schemas.push_back(
        LightOpSchema(op_type, kOnnxDomain, version, MakeSinhCoshDoc(op_type),
                      {{"input", "Input tensor", "T"}}, {{"output", output_desc, "T"}},
                      {{"T", types, "Constrain input and output types to float tensors."}}));
  }
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
  for (const char *op_type : {"Sin", "Cos"}) {
    std::vector<LightOpSchema> trig_schemas = BuildSinCosSchemas(op_type);
    schemas.insert(schemas.end(), std::make_move_iterator(trig_schemas.begin()),
                   std::make_move_iterator(trig_schemas.end()));
  }
  for (const char *op_type : {"Sinh", "Cosh"}) {
    std::vector<LightOpSchema> hyp_schemas = BuildSinhCoshSchemas(op_type);
    schemas.insert(schemas.end(), std::make_move_iterator(hyp_schemas.begin()),
                   std::make_move_iterator(hyp_schemas.end()));
  }
  return schemas;
}

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
