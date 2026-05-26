// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/sequence/shape_sequence.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace sequence {

void ComputeShapeSequenceConstruct(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceConstruct", "ComputeShapeSequenceConstruct");

  const int n_inputs = node.input_size();

  if (n_inputs == 0) {
    // Zero-input form: empty sequence with no inferable element dtype.
    // The per-element shape vector is empty; length is a concrete zero.
    ctx.SetSequence(node.output(0),
                    OptimSequence(TensorType::kUndefined, std::vector<OptimShape>{}));
    return;
  }

  const OptimTensor &first = ctx.Get(node.input(0).as_string());
  const TensorType common_dtype = first.Dtype();

  std::vector<OptimShape> elem_shapes;
  elem_shapes.reserve(static_cast<std::size_t>(n_inputs));
  elem_shapes.push_back(first.Shape());

  for (int i = 1; i < n_inputs; ++i) {
    const OptimTensor &t = ctx.Get(node.input(i).as_string());
    if (t.Dtype() != common_dtype) {
      throw std::invalid_argument(
          "ComputeShapeSequenceConstruct: input '" + node.input(i).as_string() +
          "' has a dtype that differs from input '" + node.input(0).as_string() + "'.");
    }
    elem_shapes.push_back(t.Shape());
  }

  ctx.SetSequence(node.output(0), OptimSequence(common_dtype, std::move(elem_shapes)));
}

} // namespace sequence
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
