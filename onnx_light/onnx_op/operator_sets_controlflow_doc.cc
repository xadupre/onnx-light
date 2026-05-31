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

namespace {

// Shared prose of all Loop doc revisions: a description of the looping
// semantics that does not change across opset versions.
constexpr const char *kLoopDocBody =
    R"DOC(Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).)DOC";

constexpr const char *kLoopDocVer13Suffix =
    "\n\nThe input/output of subgraph (produced by loop node) matching is based on order "
    "instead of name. The implementation will figure out the names based on this order.";

} // namespace

std::string MakeLoopDoc(int since_version) {
  std::string doc = kLoopDocBody;
  if (since_version >= 13) {
    doc += kLoopDocVer13Suffix;
  }
  return doc;
}

std::string MakeLoopOutputDescription(int since_version) {
  if (since_version >= 13) {
    return "Final N loop carried dependency values then K scan_outputs. "
           "Scan outputs must be Tensors.";
  }
  return "Final N loop carried dependency values then K scan_outputs";
}

std::string MakeLoopBodyAttributeDescription() {
  return "The graph run each iteration. It has 2+N inputs: (iteration_num, "
         "condition, loop carried dependencies...). It has 1+N+K outputs: "
         "(condition, loop carried dependencies..., scan_outputs...). Each "
         "scan_output is created by concatenating the value of the specified "
         "output value at the end of each iteration of the loop. It is an error"
         " if the dimensions or data type of these scan_outputs change across loop"
         " iterations.";
}

} // namespace controlflow
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
