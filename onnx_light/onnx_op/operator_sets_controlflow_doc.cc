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

std::string MakeIfThenBranchAttributeDescription() {
  return "Graph to run if condition is true. Has N outputs: values you wish to "
         "be live-out to the enclosing scope. The number of outputs must match"
         " the number of outputs in the else_branch.";
}

std::string MakeIfElseBranchAttributeDescription() {
  return "Graph to run if condition is false. Has N outputs: values you wish to"
         " be live-out to the enclosing scope. The number of outputs must match"
         " the number of outputs in the then_branch.";
}

namespace {

// Shared prose used by all Scan doc revisions: a description of the scan
// semantics that does not change across opset versions.
constexpr const char *kScanDocBody =
    R"DOC(Scan can be used to iterate over one or more scan_input tensors,
constructing zero or more scan_output tensors. It combines ideas from general recurrences,
functional programming constructs such as scan, fold, map, and zip, and is intended to enable
generalizations of RNN-like constructs for sequence-to-sequence processing.
Other tensors (referred to as state_variables here) can be used to carry a state
when iterating from one element to another (similar to hidden-state in RNNs, also referred
to as loop-carried dependences in the context of loops).
Many common usages involve a single scan_input tensor (where functionality
similar to scan, fold and map can be obtained). When more than one scan_input is used,
a behavior similar to zip is obtained.

The attribute body must be a graph, specifying the computation to be performed in
every iteration. It takes as input the current values of the state_variables and
the current iterated element of the scan_inputs. It must return the (updated) values
of the state_variables and zero or more scan_output_element tensors. The values of the
scan_output_element tensors are concatenated over all the iterations to produce the
scan_output values of the scan construct (similar to the concatenated intermediate
hidden-state values of RNN-like constructs). All the output tensors (state_variables as
well as scan_output_element tensors) are required to have the same shape in each iteration
of the loop (a restriction imposed to enable efficient memory allocation).)DOC";

constexpr const char *kScanDocOpset8Trailer =
    "\n\nNote that the iterated element passed to the body subgraph does not have a sequence "
    "axis. It will have a rank one less than the rank of the corresponding scan_input.";

constexpr const char *kScanDocOpset9Trailer =
    "\n\nThe operation supports batching, and the batch-axis is required to be 0. "
    "When multiple scan_input tensors are used, they must all have the same batch-size, "
    "and they must all have the same maximum-sequence-length (the dimensionality of the "
    "sequence axis or scan axis).";

} // namespace

std::string MakeScanDoc(int since_version) {
  std::string doc = kScanDocBody;
  if (since_version <= 8) {
    doc += kScanDocOpset8Trailer;
  } else {
    doc += kScanDocOpset9Trailer;
  }
  return doc;
}

std::string MakeScanBodyAttributeDescription() {
  return "The graph run each iteration. It has N+M inputs: "
         "(loop state variables..., scan_input_elts...). It has N+K outputs: "
         "(loop state variables..., scan_output_elts...). Each "
         "scan_output is created by concatenating the value of the specified "
         "scan_output_elt value at the end of each iteration of the loop. It is an error"
         " if the dimensions of these values change across loop iterations.";
}

std::string MakeScanNumScanInputsAttributeDescription() {
  return "An attribute specifying the number of scan_inputs M. ";
}

std::string MakeScanDirectionsAttributeDescription() {
  return "An optional list of M flags. The i-th element of the list specifies the direction "
         "to be scanned for the i-th scan_input tensor: 0 indicates forward direction and 1 "
         "indicates reverse direction. "
         "If omitted, all scan_input tensors will be scanned in the forward direction.";
}

std::string MakeScanInputDirectionsAttributeDescription() {
  return MakeScanDirectionsAttributeDescription();
}

std::string MakeScanOutputDirectionsAttributeDescription() {
  return "An optional list of K flags, one for each scan_output. The i-th element of the list "
         "specifies whether the i-th scan_output should be constructed by appending or "
         "prepending a new value in each iteration: 0 indicates appending and 1 "
         "indicates prepending. "
         "If omitted, all scan_output tensors will be produced by appending a value "
         "in each iteration.";
}

std::string MakeScanInputAxesAttributeDescription() {
  return "An optional list of M flags. The i-th element of the list specifies the axis "
         "to be scanned (the sequence axis) for the i-th scan_input. If omitted, 0 will "
         "be used as the scan axis for every scan_input. Negative value for an axis means "
         "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).";
}

std::string MakeScanOutputAxesAttributeDescription() {
  return "An optional list of K flags. The i-th element of the list specifies the axis "
         "for the i-th scan_output. The scan outputs are accumulated along the specified "
         "axis. If omitted, 0 will be used as the scan axis for every scan_output. "
         "Negative value for an axis means counting dimensions from the back. Accepted "
         "range is [-r, r-1].";
}

} // namespace controlflow
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
