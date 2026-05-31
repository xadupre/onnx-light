// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/sequence/shape_sequence.h"

#include <limits>
#include <string>

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace sequence {

void ComputeShapeSequenceInsert(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceInsert", "ComputeShapeSequenceInsert");
  EXT_ENFORCE_INVALID(node.input_size() >= 2,
                      "ComputeShapeSequenceInsert: SequenceInsert requires at least two inputs.");

  const OptimSequence &seq = ctx.GetSequence(node.input(0).as_string());
  const OptimTensor &inserted = ctx.Get(node.input(1).as_string());

  const bool seq_has_dtype = seq.HasElemDtype();
  const TensorType out_dtype = seq_has_dtype ? seq.ElemDtype() : inserted.Dtype();
  if (seq_has_dtype) {
    EXT_ENFORCE_INVALID(seq.ElemDtype() == inserted.Dtype(),
                        "ComputeShapeSequenceInsert: sequence element dtype and inserted tensor "
                        "dtype must match.");
  }

  OptimDim out_length;
  if (seq.Length().IsInt()) {
    const int64_t in_len = seq.Length().AsInt();
    EXT_ENFORCE_INVALID(in_len < std::numeric_limits<int64_t>::max(),
                        "ComputeShapeSequenceInsert: input sequence length overflows int64.");
    out_length = OptimDim(in_len + 1);
  } else {
    out_length = OptimDim("SequenceInsert_" + node.output(0).as_string() + "_len");
  }

  ctx.SetSequence(node.output(0), OptimSequence(out_dtype, std::move(out_length)));
}

} // namespace sequence
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
