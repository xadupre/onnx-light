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

LightOpSchema BuildPowSchemaForVersion(int since_version) {
  if (since_version == 1) {
    return LightOpSchema(
        "Pow", kOnnxDomain, since_version, detail::MakeElementwiseMathDoc("power", since_version),
        {
            {"X", "Input tensor of any shape, base of the exponent.", "T"},
            {"Y", "Input tensor of any shape broadcastable to X shape, the exponent component.",
             "T"},
        },
        {
            {"Z", "Output tensor (same size as X)", "T"},
        },
        {
            {"T", detail::FloatTypeStrings(), "Constrain input and output types to float tensors."},
        });
  }

  return LightOpSchema(
      "Pow", kOnnxDomain, since_version, detail::MakeElementwiseMathDoc("power", since_version),
      {
          {"X", "First operand, base of the exponent.", "T"},
          {"Y", "Second operand, power of the exponent.", "T"},
      },
      {
          {"Z", "Output tensor.", "T"},
      },
      {
          {"T", detail::FloatTypeStrings(), "Constrain input and output types to float tensors."},
      });
}

std::vector<LightOpSchema> BuildPowSchemas() {
  const std::vector<int> kPowSupportedVersions = {1, 7};
  std::vector<LightOpSchema> schemas;
  schemas.reserve(kPowSupportedVersions.size());
  for (const int version : kPowSupportedVersions) {
    schemas.push_back(BuildPowSchemaForVersion(version));
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

  std::vector<LightOpSchema> schemas;
  schemas.reserve(std::size(kMathSupportedVersions) * std::size(kOps) + 2);
  for (const OpDoc &op : kOps) {
    std::vector<LightOpSchema> op_schemas = BuildElementwiseMathSchemas(op.name, op.math_name);
    schemas.insert(schemas.end(), std::make_move_iterator(op_schemas.begin()),
                   std::make_move_iterator(op_schemas.end()));
  }
  std::vector<LightOpSchema> pow_schemas = BuildPowSchemas();
  schemas.insert(schemas.end(), std::make_move_iterator(pow_schemas.begin()),
                 std::make_move_iterator(pow_schemas.end()));
  return schemas;
}

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
