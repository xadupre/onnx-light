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
namespace {

constexpr int kMathSupportedVersions[] = {14, 13, 7, 6, 1};

LightOpSchema BuildUnaryFloatMathSchemaForVersion(const char *op_name, int since_version,
                                                  const char *output_description) {
  if (since_version >= 22) {
    return LightOpSchema(
        op_name, kOnnxDomain, since_version, output_description,
        {
            {"input", "Input tensor", "T"},
        },
        {
            {"output", output_description, "T"},
        },
        {
            {"T",
             {"tensor(bfloat16)", "tensor(float16)", "tensor(float)", "tensor(double)"},
             "Constrain input and output types to float tensors."},
        });
  }
  return LightOpSchema(
      op_name, kOnnxDomain, since_version, output_description,
      {
          {"input", "Input tensor", "T"},
      },
      {
          {"output", output_description, "T"},
      },
      {
          {"T", FloatTypeStrings(), "Constrain input and output types to float tensors."},
      });
}

std::vector<LightOpSchema> BuildUnaryFloatMathSchemas(const char *op_name,
                                                      const std::vector<int> &versions,
                                                      const char *output_description) {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(versions.size());
  for (const int version : versions) {
    schemas.push_back(BuildUnaryFloatMathSchemaForVersion(op_name, version, output_description));
  }
  return schemas;
}

LightOpSchema BuildElementwiseMathSchemaForVersion(const char *op_name, int since_version) {
  if (since_version == 1) {
    return LightOpSchema(
        op_name, kOnnxDomain, since_version, MakeElementwiseMathDoc(op_name, since_version),
        {
            {"A", "First operand, should share the type with the second operand.", "T"},
            {"B",
             "Second operand. With broadcasting can be of smaller size than A. "
             "If broadcasting is disabled it should be of the same size.",
             "T"},
        },
        {
            {"C", "Result, has same dimensions and type as A", "T"},
        },
        {
            {"T", FloatTypeStrings(), "Constrain input and output types to float tensors."},
        });
  }

  if (since_version == 6) {
    return LightOpSchema(
        op_name, kOnnxDomain, since_version, MakeElementwiseMathDoc(op_name, since_version),
        {
            {"A", "First operand, should share the type with the second operand.", "T"},
            {"B",
             "Second operand. With broadcasting can be of smaller size than A. "
             "If broadcasting is disabled it should be of the same size.",
             "T"},
        },
        {
            {"C", "Result, has same dimensions and type as A", "T"},
        },
        {
            {"T", NumericTypesForMathReductionStrings(),
             "Constrain input and output types to high-precision numeric tensors."},
        });
  }

  if (since_version == 7) {
    return LightOpSchema(
        op_name, kOnnxDomain, since_version, MakeElementwiseMathDoc(op_name, since_version),
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
        });
  }

  if (since_version == 13) {
    return LightOpSchema(
        op_name, kOnnxDomain, since_version, MakeElementwiseMathDoc(op_name, since_version),
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
        });
  }

  return LightOpSchema(op_name, kOnnxDomain, since_version,
                       MakeElementwiseMathDoc(op_name, since_version),
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
                       });
}

std::vector<LightOpSchema> BuildElementwiseMathSchemas(const char *op_name) {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(std::size(kMathSupportedVersions));
  for (const int version : kMathSupportedVersions) {
    schemas.push_back(BuildElementwiseMathSchemaForVersion(op_name, version));
  }
  return schemas;
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

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpMathSchemasWithHistory() {
  const std::vector<int> kSinCosVersions = {22, 7};
  const std::vector<int> kSinhCoshVersions = {22, 9};

  std::vector<LightOpSchema> schemas;
  for (const auto &op_type : {"Add", "Div", "Mul", "Sub"}) {
    std::vector<LightOpSchema> bin_schemas = BuildElementwiseMathSchemas(op_type);
    schemas.insert(schemas.end(), std::make_move_iterator(bin_schemas.begin()),
                   std::make_move_iterator(bin_schemas.end()));
  }
  std::vector<LightOpSchema> sin_schemas = BuildUnaryFloatMathSchemas(
      "Sin", kSinCosVersions, "The sine of the input tensor computed element-wise");
  schemas.insert(schemas.end(), std::make_move_iterator(sin_schemas.begin()),
                 std::make_move_iterator(sin_schemas.end()));
  std::vector<LightOpSchema> cos_schemas = BuildUnaryFloatMathSchemas(
      "Cos", kSinCosVersions, "The cosine of the input tensor computed element-wise");
  schemas.insert(schemas.end(), std::make_move_iterator(cos_schemas.begin()),
                 std::make_move_iterator(cos_schemas.end()));
  std::vector<LightOpSchema> sinh_schemas = BuildUnaryFloatMathSchemas(
      "Sinh", kSinhCoshVersions,
      "The hyperbolic sine values of the input tensor computed element-wise");
  schemas.insert(schemas.end(), std::make_move_iterator(sinh_schemas.begin()),
                 std::make_move_iterator(sinh_schemas.end()));
  std::vector<LightOpSchema> cosh_schemas = BuildUnaryFloatMathSchemas(
      "Cosh", kSinhCoshVersions,
      "The hyperbolic cosine values of the input tensor computed element-wise");
  schemas.insert(schemas.end(), std::make_move_iterator(cosh_schemas.begin()),
                 std::make_move_iterator(cosh_schemas.end()));
  std::vector<LightOpSchema> mod_schemas = BuildModSchemas();
  schemas.insert(schemas.end(), std::make_move_iterator(mod_schemas.begin()),
                 std::make_move_iterator(mod_schemas.end()));
  return schemas;
}

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
