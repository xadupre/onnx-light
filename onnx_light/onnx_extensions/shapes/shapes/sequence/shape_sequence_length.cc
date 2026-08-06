// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/sequence/shape_sequence.h"

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::sequence {

void ComputeShapeSequenceLength(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceLength", "ComputeShapeSequenceLength");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeSequenceLength: SequenceLength requires one input.");
  (void)ctx.GetSequence(node.input(0));
  ctx.Set(node.output(0), SymTensor(nullptr, TensorType::kInt64, SymShape{}));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::sequence
