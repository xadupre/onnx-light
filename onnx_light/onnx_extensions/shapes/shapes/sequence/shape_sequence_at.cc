// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/sequence/shape_sequence.h"

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_sequence.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::sequence {

void ComputeShapeSequenceAt(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceAt", "ComputeShapeSequenceAt");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeSequenceAt: SequenceAt requires at least one input.");

  const SymSequence &seq = ctx.GetSequence(node.input(0));
  const TensorType elem_dtype = seq.ElemDtype();

  // The output element type matches the input sequence's element type. The
  // output shape can only be inferred when all per-element shapes are known
  // and identical, since the position is a runtime value.
  SymShape out_shape{};
  bool shape_known = false;
  if (seq.HasElemShapes() && !seq.ElemShapes().empty()) {
    const std::vector<SymShape> &shapes = seq.ElemShapes();
    bool all_equal = true;
    for (std::size_t i = 1; i < shapes.size(); ++i) {
      if (shapes[i] != shapes[0]) {
        all_equal = false;
        break;
      }
    }
    if (all_equal) {
      out_shape = shapes[0];
      shape_known = true;
    }
  }

  if (shape_known) {
    ctx.Set(node.output(0), SymTensor(nullptr, elem_dtype, out_shape));
  } else {
    ctx.Set(node.output(0), SymTensor(nullptr, elem_dtype, SymShape{}));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::sequence
