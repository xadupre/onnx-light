// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math.h"

#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {
namespace {

constexpr const char *kOnnxDomain = "ai.onnx";
constexpr int kMathSupportedVersions[] = {14, 13, 7, 6, 1};

std::vector<std::string> TypesToStrings(const std::initializer_list<TensorType> types) {
  std::vector<std::string> allowed_types;
  allowed_types.reserve(types.size());
  for (const TensorType tensor_type : types) {
    allowed_types.emplace_back(ToTypeString(tensor_type));
  }
  return allowed_types;
}

std::vector<std::string> FloatTypeStrings() {
  return TypesToStrings({
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
  });
}

std::vector<std::string> NumericTypesForMathReductionStrings() {
  return TypesToStrings({
      TensorType::kUint32,
      TensorType::kUint64,
      TensorType::kInt32,
      TensorType::kInt64,
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
  });
}

std::vector<std::string> NumericTypesForMathReductionIr4Strings() {
  return TypesToStrings({
      TensorType::kUint32,
      TensorType::kUint64,
      TensorType::kInt32,
      TensorType::kInt64,
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
      TensorType::kBfloat16,
  });
}

std::vector<std::string> AllNumericTypesIr4Strings() {
  return TypesToStrings({
      TensorType::kUint8,
      TensorType::kUint16,
      TensorType::kUint32,
      TensorType::kUint64,
      TensorType::kInt8,
      TensorType::kInt16,
      TensorType::kInt32,
      TensorType::kInt64,
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
      TensorType::kBfloat16,
  });
}

std::string MakeElementwiseMathDoc(const char *math_name, int since_version) {
  const std::string_view math_name_view(math_name);
  if (since_version <= 6) {
    std::string doc = "Performs element-wise binary ";
    doc += math_name;
    doc += R"DOC( (with limited broadcast support).
If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.)DOC";
    if (since_version == 6 && math_name_view == "division") {
      doc += "\n\nFor integer inputs, the result is computed using truncating division "
             "(rounding toward zero).";
    }
    return doc;
  }

  std::string doc = "Performs element-wise binary ";
  doc += math_name;
  doc += R"DOC( (with Numpy-style broadcasting support).

This operator supports multidirectional (i.e., Numpy-style) broadcasting;
for more details please check the broadcasting behavior in ONNX.)DOC";
  if (math_name_view == "division") {
    doc += "\n\nFor integer inputs, the result is computed using truncating division "
           "(rounding toward zero).";
  }
  return doc;
}

LightOpSchema BuildElementwiseMathSchema(const char *op_name, int since_version,
                                         const char *math_name) {
  if (since_version == 1) {
    return LightOpSchema(
        op_name, kOnnxDomain, since_version, MakeElementwiseMathDoc(math_name, since_version),
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
        op_name, kOnnxDomain, since_version, MakeElementwiseMathDoc(math_name, since_version),
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
        op_name, kOnnxDomain, since_version, MakeElementwiseMathDoc(math_name, since_version),
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
        op_name, kOnnxDomain, since_version, MakeElementwiseMathDoc(math_name, since_version),
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
                       MakeElementwiseMathDoc(math_name, since_version),
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

std::string BuildAndOperatorDoc(int since_version) {
  if (since_version == 1) {
    return R"DOC(
Returns the tensor resulted from performing the `and` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
)DOC";
  }

  return R"DOC(
Returns the tensor resulted from performing the `and` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
)DOC";
}

std::vector<LightOpSchema> BuildAndSchemas() {
  std::vector<LightOpSchema> schemas;
  schemas.reserve(2);
  schemas.push_back(
      LightOpSchema("And", kOnnxDomain, 1, BuildAndOperatorDoc(1),
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
                    }));
  schemas.push_back(
      LightOpSchema("And", kOnnxDomain, 7, BuildAndOperatorDoc(7),
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
                    }));
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
    for (const int version : kMathSupportedVersions) {
      schemas.push_back(BuildElementwiseMathSchema(op.name, version, op.math_name));
    }
  }
  std::vector<LightOpSchema> and_schemas = BuildAndSchemas();
  schemas.insert(schemas.end(), std::make_move_iterator(and_schemas.begin()),
                 std::make_move_iterator(and_schemas.end()));
  return schemas;
}

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
