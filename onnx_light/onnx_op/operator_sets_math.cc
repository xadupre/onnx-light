// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"
#include "onnx_op/operator_sets_math_utils.h"

#include <iterator>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {
namespace {

constexpr const char *kOnnxDomain = "ai.onnx";
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
          {"T", detail::FloatTypeStrings(), "Constrain input and output types to float tensors."},
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

LightOpSchema BuildElementwiseMathSchemaForVersion(const char *op_name, int since_version,
                                                   const char *math_name) {
  if (since_version == 1) {
    return LightOpSchema(
        op_name, kOnnxDomain, since_version,
        detail::MakeElementwiseMathDoc(math_name, since_version),
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
            {"T", detail::FloatTypeStrings(), "Constrain input and output types to float tensors."},
        });
  }

  if (since_version == 6) {
    return LightOpSchema(
        op_name, kOnnxDomain, since_version,
        detail::MakeElementwiseMathDoc(math_name, since_version),
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
            {"T", detail::NumericTypesForMathReductionStrings(),
             "Constrain input and output types to high-precision numeric tensors."},
        });
  }

  if (since_version == 7) {
    return LightOpSchema(
        op_name, kOnnxDomain, since_version,
        detail::MakeElementwiseMathDoc(math_name, since_version),
        {
            {"A", "First operand.", "T"},
            {"B", "Second operand.", "T"},
        },
        {
            {"C", "Result, has same element type as two inputs", "T"},
        },
        {
            {"T", detail::NumericTypesForMathReductionStrings(),
             "Constrain input and output types to high-precision numeric tensors."},
        });
  }

  if (since_version == 13) {
    return LightOpSchema(
        op_name, kOnnxDomain, since_version,
        detail::MakeElementwiseMathDoc(math_name, since_version),
        {
            {"A", "First operand.", "T"},
            {"B", "Second operand.", "T"},
        },
        {
            {"C", "Result, has same element type as two inputs", "T"},
        },
        {
            {"T", detail::NumericTypesForMathReductionIr4Strings(),
             "Constrain input and output types to high-precision numeric tensors."},
        });
  }

  return LightOpSchema(op_name, kOnnxDomain, since_version,
                       detail::MakeElementwiseMathDoc(math_name, since_version),
                       {
                           {"A", "First operand.", "T"},
                           {"B", "Second operand.", "T"},
                       },
                       {
                           {"C", "Result, has same element type as two inputs", "T"},
                       },
                       {
                           {"T", detail::AllNumericTypesIr4Strings(),
                            "Constrain input and output types to all numeric tensors."},
                       });
}

std::vector<LightOpSchema> BuildElementwiseMathSchemas(const char *op_name, const char *math_name) {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(std::size(kMathSupportedVersions));
  for (const int version : kMathSupportedVersions) {
    schemas.push_back(BuildElementwiseMathSchemaForVersion(op_name, version, math_name));
  }
  return schemas;
}

} // namespace

std::vector<LightOpSchema> GetAllOnnxOpMathSchemasWithHistory() {
  struct OpDoc {
    const char *name;
    const char *math_name;
  };

  constexpr OpDoc kOps[] = {
      {"Add", "addition"},
      {"Div", "division"},
      {"Mul", "multiplication"},
      {"Sub", "subtraction"},
  };
  const std::vector<int> kSinCosVersions = {22, 7};
  const std::vector<int> kSinhCoshVersions = {22, 9};

  std::vector<LightOpSchema> schemas;
  schemas.reserve(std::size(kMathSupportedVersions) * std::size(kOps) + 2 * kSinCosVersions.size() +
                  2 * kSinhCoshVersions.size());
  for (const OpDoc &op : kOps) {
    std::vector<LightOpSchema> op_schemas = BuildElementwiseMathSchemas(op.name, op.math_name);
    schemas.insert(schemas.end(), std::make_move_iterator(op_schemas.begin()),
                   std::make_move_iterator(op_schemas.end()));
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
  return schemas;
}

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
