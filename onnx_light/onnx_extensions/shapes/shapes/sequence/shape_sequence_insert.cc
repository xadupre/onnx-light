// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/sequence/shape_sequence.h"

#include <limits>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_sequence.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::sequence {

void ComputeShapeSequenceInsert(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceInsert", "ComputeShapeSequenceInsert");
  EXT_ENFORCE_INVALID(node.input_size() >= 2,
                      "ComputeShapeSequenceInsert: SequenceInsert requires at least two inputs.");

  const SymSequence &seq = ctx.GetSequence(node.input(0));
  const SymTensor &inserted = ctx.Get(node.input(1));

  const bool seq_has_dtype = seq.HasElemDtype();
  const TensorType out_dtype = seq_has_dtype ? seq.ElemDtype() : inserted.Dtype();
  if (seq_has_dtype) {
    EXT_ENFORCE_INVALID(seq.ElemDtype() == inserted.Dtype(),
                        "ComputeShapeSequenceInsert: sequence element dtype and inserted tensor "
                        "dtype must match.");
  }

  SymDim out_length;
  if (seq.Length().IsInt()) {
    const int64_t in_len = seq.Length().AsInt();
    EXT_ENFORCE_INVALID(in_len < std::numeric_limits<int64_t>::max(),
                        "ComputeShapeSequenceInsert: input sequence length overflows int64.");
    out_length = SymDim(in_len + 1);
  } else {
    out_length = SymDim("SequenceInsert_" + node.output(0) + "_len");
  }

  ctx.SetSequence(node.output(0), SymSequence(out_dtype, std::move(out_length)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::sequence
