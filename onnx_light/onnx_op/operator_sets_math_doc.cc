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
namespace {
const std::map<std::string, std::string> kUnaryMathOutputDescriptionMapping{
    {"Sin", "The sine of the input tensor computed element-wise"},
    {"Cos", "The cosine of the input tensor computed element-wise"},
    {"Exp", "The exponential of the input tensor computed element-wise"},
    {"Log", "The natural log of the input tensor computed element-wise"},
    {"Sinh", "The hyperbolic sine values of the input tensor computed element-wise"},
    {"Cosh", "The hyperbolic cosine values of the input tensor computed element-wise"},
    {"Asin", "The arcsine of the input tensor computed element-wise"},
    {"Acos", "The arccosine of the input tensor computed element-wise"},
    {"Asinh", "The hyperbolic arcsine values of the input tensor computed element-wise"},
    {"Acosh", "The hyperbolic arccosine values of the input tensor computed element-wise"},
    {"Atan", "The arctangent of the input tensor computed element-wise"},
    {"Atanh", "The hyperbolic arctangent values of the input tensor computed element-wise"},
    {"Tan", "The tangent of the input tensor computed element-wise"},
    {"Tanh", "The hyperbolic tangent values of the input tensor computed element-wise"},
};
}

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

std::string MakeUnaryMathDoc(const char *op_type) {
  return "Computes the " + std::string(op_type) + " value of the input tensor element-wise.";
}

std::string MakeBlackmanWindowDoc() {
  return "Generates a Blackman window as described in the paper "
         "https://ieeexplore.ieee.org/document/1455106.";
}

std::string MakeHannWindowDoc() {
  return "Generates a Hann window as described in the paper "
         "https://ieeexplore.ieee.org/document/1455106.";
}

std::string MakeHammingWindowDoc() {
  return "Generates a Hamming window as described in the paper "
         "https://ieeexplore.ieee.org/document/1455106.";
}

std::string MakePowDoc() { return "Performs element-wise exponentiation."; }

std::string MakeMatMulDoc() {
  return "\nMatrix product that behaves like "
         "[numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).\n";
}

std::string MakeGemmDoc(int since_version) {
  if (since_version <= 6) {
    return R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has
dimension (M X K), input tensor B has dimension (K X N), input tensor C and
output tensor Y have dimension (M X N).
If attribute broadcast is non-zero, input tensor C will be broadcasted to match
the dimension requirement. A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.
)DOC";
  }
  return R"DOC(General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
)DOC";
}

std::string MakeUnaryMathOutputDescription(const char *op_type) {
  const auto it = kUnaryMathOutputDescriptionMapping.find(op_type);
  if (it == kUnaryMathOutputDescriptionMapping.end()) {
    throw SchemaError("Unsupported unary math operator for output description: " +
                      std::string(op_type));
  }
  return it->second;
}

} // namespace math
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
