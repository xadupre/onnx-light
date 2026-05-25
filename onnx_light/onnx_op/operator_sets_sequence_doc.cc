// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_sequence_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace sequence {

std::string MakeSequenceEmptyDoc() {
  return R"DOC(
Construct an empty tensor sequence, with given data type.
)DOC";
}

std::string MakeSequenceConstructDoc() {
  return R"DOC(
Construct a tensor sequence containing 'inputs' tensors.
All tensors in 'inputs' must have the same data type.
)DOC";
}

std::string MakeSequenceInsertDoc() {
  return R"DOC(
Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at 'position'.
'tensor' must have the same data type as 'input_sequence'.
Accepted range for 'position' is in `[-n, n]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it inserts 'tensor' to the back of 'input_sequence'.
)DOC";
}

std::string MakeSequenceAtDoc() {
  return R"DOC(
Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
)DOC";
}

std::string MakeSequenceEraseDoc() {
  return R"DOC(
Outputs a tensor sequence that removes the tensor at 'position' from 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it erases the last tensor from 'input_sequence'.
)DOC";
}

std::string MakeSequenceLengthDoc() {
  return R"DOC(
Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
)DOC";
}

std::string MakeSplitToSequenceDoc() {
  return R"DOC(
Split a tensor into a sequence of tensors, along the specified 'axis'.
Lengths of the parts can be specified using the optional argument 'split'.
If 'split' is not specified, 'input' is split into chunks of size 1 along the
specified 'axis'.
'split' must contain only positive numbers.
'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
If 'split' is a scalar, then 'input' will be split into chunks all of size 'split'
if possible. The last chunk alone may be smaller than 'split' if the 'input' size
along the given axis 'axis' is not divisible by 'split'.
If 'split' is a 1-dimensional tensor, the input tensor is split into 'size(split)' chunks,
with lengths of the parts on 'axis' specified in 'split'. In this scenario, the sum of entries
in 'split' must be equal to the dimension size of input tensor on 'axis'.
)DOC";
}

std::string MakeConcatFromSequenceDoc() {
  return R"DOC(
Concatenate a sequence of tensors into a single tensor.
All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
By default 'new_axis' is 0, the behavior is similar to numpy.concatenate.
When 'new_axis' is 1, the behavior is similar to numpy.stack.
)DOC";
}

std::string MakeSequenceMapDoc() {
  return R"DOC(
Applies a sub-graph to each sample in the input sequence(s).

Inputs can be either tensors or sequences, with the exception of the first input which must
be a sequence. The length of the first input sequence will determine the number of samples in the
outputs. Any other sequence inputs should have the same number of samples. The number of inputs
and outputs, should match the one of the subgraph.

For each i-th element in the output, a sample will be extracted from the input sequence(s) at
the i-th position and the sub-graph will be applied to it.
The outputs will contain the outputs of the sub-graph for each sample, in the same order as in
the input.

This operator assumes that processing each sample is independent and could executed in parallel
or in any order. Users cannot expect any specific ordering in which each subgraph is computed.)DOC";
}

} // namespace sequence
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
