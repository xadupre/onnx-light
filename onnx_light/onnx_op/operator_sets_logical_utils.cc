// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_logical_utils.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace logical {
namespace detail {

std::string MakeBinaryLogicalOperatorDoc(const char *op_name, int since_version) {
  if (since_version == 1) {
    return R"DOC(
Returns the tensor resulted from performing the `)DOC" +
           std::string(op_name) +
           R"DOC(` logical operation
elementwise on the input tensors `A` and `B`.

If broadcasting is enabled, the right-hand-side argument will be broadcasted
to match the shape of left-hand-side argument. See the doc of `Add` for a
detailed description of the broadcasting rules.
)DOC";
  }

  return R"DOC(
Returns the tensor resulted from performing the `)DOC" +
         std::string(op_name) +
         R"DOC(` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).
)DOC";
}

std::string MakeNotLogicalOperatorDoc() {
  return R"DOC(
Returns the negation of the input tensor element-wise.
)DOC";
}

} // namespace detail
} // namespace logical
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
