// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/sequence/shape_sequence.h"

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace sequence {

void ComputeShapeSequenceAt(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceAt", "ComputeShapeSequenceAt");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeSequenceAt: SequenceAt requires at least one input.");

  const OptimSequence &seq = ctx.GetSequence(node.input(0).as_string());
  const TensorType elem_dtype = seq.ElemDtype();

  // The output element type matches the input sequence's element type. The
  // output shape can only be inferred when all per-element shapes are known
  // and identical, since the position is a runtime value.
  OptimShape out_shape{};
  bool shape_known = false;
  if (seq.HasElemShapes() && !seq.ElemShapes().empty()) {
    const std::vector<OptimShape> &shapes = seq.ElemShapes();
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
    ctx.Set(node.output(0), OptimTensor(nullptr, elem_dtype, out_shape));
  } else {
    ctx.Set(node.output(0), OptimTensor(nullptr, elem_dtype, OptimShape{}));
  }
}

} // namespace sequence
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
