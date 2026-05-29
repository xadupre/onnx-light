// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_logical_doc.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace logical {

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

} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
