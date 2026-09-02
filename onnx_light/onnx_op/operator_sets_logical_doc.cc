// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_logical_doc.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op::logical {

std::string MakeBinaryLogicalOperatorDoc(const char *op_type, int since_version) {
  if (since_version == 1) {
    return R"DOC(
Returns the tensor resulted from performing the `)DOC" +
           std::string(op_type) +
           R"DOC(` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
)DOC";
  }

  return R"DOC(
Returns the tensor resulted from performing the `)DOC" +
         std::string(op_type) +
         R"DOC(` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
)DOC";
}

std::string MakeNotLogicalOperatorDoc() {
  return R"DOC(
Returns the negation of the input tensor element-wise.
)DOC";
}

std::string MakeWhereOperatorDoc() {
  return R"DOC(
Return elements, either from X or Y, depending on condition.
)DOC";
}

std::string MakeBinaryBitwiseOperatorDoc(const char *op_type) {
  return R"DOC(
Returns the tensor resulting from performing the bitwise `)DOC" +
         std::string(op_type) +
         R"DOC(` operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
)DOC";
}

std::string MakeBitwiseNotOperatorDoc() {
  return R"DOC(
Returns the bitwise not of the input tensor element-wise.
)DOC";
}

std::string MakeBitShiftOperatorDoc(int version) {
  const std::string broadcasting = R"DOC(
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
)DOC";
  if (version == 11) {
    return R"DOC(
Bitwise shift operator performs element-wise operation. For each input element, if the
attribute "direction" is "RIGHT", this operator moves its binary representation toward
the right side so that the input value is effectively decreased. If the attribute "direction"
is "LEFT", bits of binary representation moves toward the left side, which results the
increase of its actual value. The input X is the tensor to be shifted and another input
Y specifies the amounts of shifting. For example, if "direction" is "Right", X is [1, 4],
and S is [1, 1], the corresponding output Z would be [0, 2]. If "direction" is "LEFT" with
X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].

Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
not necessarily identical.
)DOC" + broadcasting;
  }
  return R"DOC(
Bitwise shift operator performs element-wise operation. For each input element, if the
attribute "direction" is "RIGHT", this operator moves its binary representation toward
the right side. If the attribute "direction" is "LEFT", bits of binary representation
move toward the left side. The input X is the tensor to be shifted and another
input Y specifies the amounts of shifting. For example, if "direction" is
"RIGHT", X is [1, 4], and Y is [1, 1], the corresponding output Z would be
[0, 2]. If "direction" is "LEFT" with X=[1, 2] and Y=[1, 2], the corresponding
output Z would be [2, 8].

For a signed T the right shift is an arithmetic shift (sign-extending). The
vacated high bits are filled with copies of the sign bit, so a negative X stays
negative. For a signed T a left shift can move bits into and past the sign bit,
and bits shifted past the sign bit are discarded.

If Y is negative, or is greater than or equal to the number of bits of T, then
the result is whatever the sign bit extension alone produces: -1 for a right
shift on a negative X, where the fill is a sign bit of 1, and 0 in every other
case.
)DOC" + broadcasting;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_op::logical
