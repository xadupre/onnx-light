// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/sequence/shape_sequence.h"

#include <utility>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_sequence.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::sequence {

void ComputeShapeSequenceConstruct(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceConstruct", "ComputeShapeSequenceConstruct");

  const int n_inputs = node.input_size();

  if (n_inputs == 0) {
    // Zero-input form: empty sequence with no inferable element dtype.
    // The per-element shape vector is empty; length is a concrete zero.
    ctx.SetSequence(node.output(0), SymSequence(TensorType::kUndefined, std::vector<SymShape>{}));
    return;
  }

  const SymTensor &first = ctx.Get(node.input(0));
  const TensorType common_dtype = first.Dtype();

  std::vector<SymShape> elem_shapes;
  elem_shapes.reserve(static_cast<std::size_t>(n_inputs));
  elem_shapes.push_back(first.Shape());

  for (int i = 1; i < n_inputs; ++i) {
    const SymTensor &t = ctx.Get(node.input(i));
    EXT_ENFORCE_INVALID(t.Dtype() == common_dtype, "ComputeShapeSequenceConstruct: input '",
                        node.input(i), "' has a dtype that differs from input '", node.input(0),
                        "'.");
    elem_shapes.push_back(t.Shape());
  }

  ctx.SetSequence(node.output(0), SymSequence(common_dtype, std::move(elem_shapes)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::sequence
