// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/sequence/shape_sequence.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace sequence {

namespace {

// Returns the broadcasted element shape: if both inputs share a known
// dim at position ``i`` it is kept as-is, otherwise the position is
// recorded as the symbolic placeholder ``"?"``. ``a`` and ``b`` must
// have the same rank; the caller is responsible for the rank check.
OptimShape MergeElemShapes(const OptimShape &a, const OptimShape &b) {
  OptimShape out;
  for (std::size_t i = 0; i < a.Rank(); ++i) {
    if (a[i] == b[i]) {
      out.PushBack(a[i]);
    } else {
      out.PushBack(OptimDim("?"));
    }
  }
  return out;
}

} // namespace

void ComputeShapeSequenceConstruct(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceConstruct", "ComputeShapeSequenceConstruct");

  const int n_inputs = node.input_size();
  OptimSequence out;

  if (n_inputs == 0) {
    // Zero-input form: empty sequence with no inferable element dtype
    // or shape. Length is a concrete zero.
    out.SetLength(OptimDim(static_cast<int64_t>(0)));
    ctx.SetSequence(node.output(0), std::move(out));
    return;
  }

  const OptimTensor &first = ctx.Get(node.input(0).as_string());
  TensorType common_dtype = first.Dtype();
  OptimShape common_shape = first.Shape();
  bool common_shape_known = true;

  for (int i = 1; i < n_inputs; ++i) {
    const OptimTensor &t = ctx.Get(node.input(i).as_string());
    if (t.Dtype() != common_dtype) {
      throw std::invalid_argument(
          "ComputeShapeSequenceConstruct: input '" + node.input(i).as_string() +
          "' has a dtype that differs from input '" + node.input(0).as_string() + "'.");
    }
    if (!common_shape_known) {
      continue;
    }
    if (t.Shape().Rank() != common_shape.Rank()) {
      // Rank mismatch: cannot describe a single element shape.
      common_shape_known = false;
      common_shape = OptimShape{};
      continue;
    }
    common_shape = MergeElemShapes(common_shape, t.Shape());
  }

  out.SetElemDtype(common_dtype);
  if (common_shape_known) {
    out.SetElemShape(std::move(common_shape));
  }
  out.SetLength(OptimDim(static_cast<int64_t>(n_inputs)));

  ctx.SetSequence(node.output(0), std::move(out));
}

} // namespace sequence
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
