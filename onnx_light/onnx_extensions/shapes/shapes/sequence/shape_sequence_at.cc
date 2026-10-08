// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/sequence/shape_sequence.h"

#include <string>

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

  // The output element type matches the input sequence's element type. Merge
  // known element shapes dimension by dimension because the position is a
  // runtime value.
  SymShape out_shape{};
  bool shape_known = false;
  if (seq.HasElemShapes() && !seq.ElemShapes().empty()) {
    const std::vector<SymShape> &shapes = seq.ElemShapes();
    const std::size_t rank = shapes[0].Rank();
    bool common_rank = true;
    for (std::size_t i = 1; i < shapes.size(); ++i) {
      if (shapes[i].Rank() != rank) {
        common_rank = false;
        break;
      }
    }
    if (common_rank) {
      out_shape = shapes[0];
      for (std::size_t dim = 0; dim < rank; ++dim) {
        for (std::size_t i = 1; i < shapes.size(); ++i) {
          if (shapes[i][dim] != out_shape[dim]) {
            out_shape[dim] = SymDim("SequenceAt_" + node.output(0) + "_dim" + std::to_string(dim));
            break;
          }
        }
      }
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
