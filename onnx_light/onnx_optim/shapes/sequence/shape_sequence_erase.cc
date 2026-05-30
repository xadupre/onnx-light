// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/sequence/shape_sequence.h"

#include <string>

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace sequence {

void ComputeShapeSequenceErase(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceErase", "ComputeShapeSequenceErase");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeSequenceErase: SequenceErase requires at least one input.");

  const std::string seq_name = node.input(0).as_string();
  const OptimSequence &seq = ctx.GetSequence(seq_name);
  const TensorType elem_dtype = seq.ElemDtype();

  // The output element type is always the same as the input sequence's.
  // The output length is (input length - 1) when the input length is a
  // concrete integer; otherwise it remains symbolic.
  // Because the erase position is a runtime value, per-element output
  // shapes are not inferred (we would need to know which element is
  // removed to enumerate the remaining shapes).
  OptimDim out_length;
  if (seq.Length().IsInt()) {
    const int64_t in_len = seq.Length().AsInt();
    out_length = OptimDim(in_len > 0 ? in_len - 1 : int64_t{0});
  } else {
    // Symbolic length: produce a fresh symbolic name for the output.
    out_length = OptimDim("SequenceErase_" + node.output(0).as_string() + "_len");
  }

  ctx.SetSequence(node.output(0), OptimSequence(elem_dtype, std::move(out_length)));
}

} // namespace sequence
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
