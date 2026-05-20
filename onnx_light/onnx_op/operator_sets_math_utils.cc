// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math_utils.h"

#include <initializer_list>
#include <string_view>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {
namespace detail {
namespace {

std::vector<std::string> TypesToStrings(const std::initializer_list<TensorType> types) {
  std::vector<std::string> allowed_types;
  allowed_types.reserve(types.size());
  for (const TensorType tensor_type : types) {
    allowed_types.emplace_back(ToTypeString(tensor_type));
  }
  return allowed_types;
}

} // namespace

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

std::vector<std::string> AllNumericTypesStrings() {
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

} // namespace detail
} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
