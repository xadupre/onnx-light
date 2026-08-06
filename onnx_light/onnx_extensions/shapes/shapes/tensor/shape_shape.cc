// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <utility>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

namespace {

// Normalises ``raw`` against ``rank`` per the ONNX ``Shape`` spec: negative
// values count from the back and the result is clamped to ``[0, rank]``.
int64_t ClampAxis(int64_t raw, int64_t rank) {
  int64_t value = raw;
  if (value < 0) {
    value += rank;
  }
  if (value < 0) {
    value = 0;
  }
  if (value > rank) {
    value = rank;
  }
  return value;
}

} // namespace

void ComputeShapeShape(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Shape", "ComputeShapeShape");
  EXT_ENFORCE_INVALID(node.input_size() >= 1, "ComputeShapeShape: Shape requires one input.");

  const SymTensor &input = ctx.Get(node.input(0));
  const int64_t rank = static_cast<int64_t>(input.Shape().Rank());

  const int64_t start = ClampAxis(GetAttributeOr(node, "start", static_cast<int64_t>(0)), rank);
  const int64_t end = ClampAxis(GetAttributeOr(node, "end", rank), rank);

  const int64_t length = end > start ? end - start : 0;

  // The output is always a 1-D INT64 tensor whose single dimension is the
  // number of axes selected by ``[start, end)`` (a concrete value because
  // the rank of any registered ``SymTensor`` is known).
  SymShape out_shape;
  out_shape.PushBack(SymDim(length));

  SymTensor out_tensor(nullptr, TensorType::kInt64, std::move(out_shape));

  // The values produced by ``Shape`` are exactly the input's selected
  // dimensions; record them as ``ValueAsShape`` so that downstream
  // operators (``Concat``, ``Expand``, ``Reshape``, …) can propagate
  // the data even when the input shape carries symbolic dims.
  SymShape value_as_shape;
  for (int64_t i = start; i < end; ++i) {
    value_as_shape.PushBack(input.Shape()[static_cast<std::size_t>(i)]);
  }
  out_tensor.SetValueAsShape(std::move(value_as_shape));

  ctx.Set(node.output(0), std::move(out_tensor));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
