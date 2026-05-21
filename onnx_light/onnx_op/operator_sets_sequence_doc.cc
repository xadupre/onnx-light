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

std::string MakeSequenceLengthDoc() {
  return R"DOC(
Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
)DOC";
}

} // namespace sequence
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
