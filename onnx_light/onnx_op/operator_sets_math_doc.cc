// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_math_doc.h"

#include <initializer_list>
#include <map>
#include <string_view>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace math {

std::string MakeElementwiseMathDoc(const char *op_type, int since_version) {
  std::map<std::string, std::string> mapping{
      {"Add", "addition"},
      {"Div", "division"},
      {"Mul", "multiplication"},
      {"Sub", "subtraction"},
  };
  std::string math_name = mapping[op_type];

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

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
