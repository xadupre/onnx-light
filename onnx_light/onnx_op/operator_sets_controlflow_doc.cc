// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_controlflow_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace controlflow {

std::string MakeIfDoc() { return "If conditional"; }

std::string MakeIfOutputDescription(int since_version) {
  if (since_version == 1) {
    return "Values that are live-out to the enclosing scope. The return values in the "
           "`then_branch` and `else_branch` must be of the same shape and same data type.";
  }
  return "Values that are live-out to the enclosing scope. The return values in the "
         "`then_branch` and `else_branch` must be of the same data type. "
         "The `then_branch` and `else_branch` may produce tensors with the same "
         "element type and different shapes. "
         "If corresponding outputs from the then-branch and the else-branch have "
         "static shapes S1 and S2, then the shape of the corresponding output "
         "variable of the if-node (if present) must be compatible with both S1 "
         "and S2 as it represents the union of both possible shapes."
         "For example, if in a model file, the first "
         "output of `then_branch` is typed float tensor with shape [2] and the "
         "first output of `else_branch` is another float tensor with shape [3], "
         "If's first output should have (a) no shape set, or (b) "
         "a shape of rank 1 with neither `dim_value` nor `dim_param` set, or (c) "
         "a shape of rank 1 with a unique `dim_param`. "
         "In contrast, the first output cannot have the shape [2] since [2] and "
         "[3] are not compatible.";
}

} // namespace controlflow
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
