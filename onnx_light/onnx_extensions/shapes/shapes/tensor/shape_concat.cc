// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <string>
#include <utility>

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_core/symbolic/symbolic_helper.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes {

// Alias to the symbolic dimension-expression library, which lives in
// ``onnx_core`` so both ``onnx_op`` and ``onnx_shapes`` can share it.
namespace expressions = ::ONNX_LIGHT_NAMESPACE::core::expressions;
namespace shapes::tensor {

namespace {

SymDim FromDimType(const expressions::DimType &d) {
  if (std::holds_alternative<int64_t>(d)) {
    return SymDim(std::get<int64_t>(d));
  }
  return SymDim(std::get<std::string>(d));
}

// Merges ``in`` into ``out``. Two concrete integer dimensions must be
// equal. A concrete integer wins over a symbolic dimension. Two
// symbolic dimensions are merged into ``out`` only when they share
// the same expression; otherwise the previously-merged value in
// ``out`` is preserved.
void MergeDim(SymDim &out, const SymDim &in, int axis, int input_index) {
  if (out.IsInt() && in.IsInt()) {
    EXT_ENFORCE_INVALID(out.AsInt() == in.AsInt(), "ComputeShapeConcat: input ", input_index,
                        " dimension ", axis, " (", in.AsInt(),
                        ") differs from the previously-merged value (", out.AsInt(), ").");
    return;
  }
  if (in.IsInt()) {
    // Concrete value overrides a previously-symbolic dimension.
    out = in;
    return;
  }
  // ``in`` is symbolic; keep ``out`` as is.
}

} // namespace

void ComputeShapeConcat(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Concat", "ComputeShapeConcat");

  const int n_inputs = node.input_size();
  EXT_ENFORCE_INVALID(!(n_inputs < 1), "ComputeShapeConcat: Concat requires at least one input.");

  const SymTensor &first = ctx.Get(node.input(0));
  const TensorType common_dtype = first.Dtype();
  const SymShape &first_shape = first.Shape();
  const int rank = static_cast<int>(first_shape.Rank());

  // Resolve ``axis``. The default of ``1`` mirrors the opset 1
  // default; in later opsets ``axis`` is required so the default is
  // only used when the attribute was accidentally omitted.
  const int64_t axis_attr = GetAttributeOr<int64_t>(node, "axis", 1);
  const int64_t resolved_axis = axis_attr < 0 ? axis_attr + rank : axis_attr;
  EXT_ENFORCE_INVALID(!(resolved_axis < 0 || resolved_axis >= rank), "ComputeShapeConcat: axis ",
                      axis_attr, " is out of range for rank ", rank, ".");
  const std::size_t axis = static_cast<std::size_t>(resolved_axis);

  // Start the output shape from the first input.
  SymShape out_shape = first_shape;

  // Accumulate the concat-axis dimension using symbolic arithmetic so
  // that ``Concat([X[..,b], Y[..,c]], axis=1)`` produces ``..,b+c``
  // (or simplifications such as ``2*b`` when the same expression
  // repeats) rather than a fresh anonymous placeholder.
  expressions::DimType axis_dim = ToDimType(first_shape[axis]);

  for (int i = 1; i < n_inputs; ++i) {
    const SymTensor &t = ctx.Get(node.input(i));
    EXT_ENFORCE_INVALID(t.Dtype() == common_dtype, "ComputeShapeConcat: input '", node.input(i),
                        "' has a dtype that differs from input '", node.input(0), "'.");
    const SymShape &shape = t.Shape();
    EXT_ENFORCE_INVALID(!(static_cast<int>(shape.Rank()) != rank), "ComputeShapeConcat: input ", i,
                        " has rank ", shape.Rank(), " != ", rank, " (first input).");
    for (std::size_t d = 0; d < shape.Rank(); ++d) {
      if (d == axis) {
        axis_dim = expressions::dim_add(axis_dim, ToDimType(shape[d]));
      } else {
        MergeDim(out_shape[d], shape[d], static_cast<int>(d), i);
      }
    }
  }

  out_shape[axis] = FromDimType(axis_dim);

  SymTensor out_tensor(nullptr, common_dtype, std::move(out_shape));

  // Propagate ``ValueAsShape`` when concatenating along axis 0 of 1-D
  // tensors and every input already carries a ``ValueAsShape``
  // annotation. The resulting annotation is the concatenation of the
  // per-input annotations in input order.
  if (resolved_axis == 0 && rank == 1) {
    bool all_have_vas = first.HasValueAsShape();
    for (int i = 1; all_have_vas && i < n_inputs; ++i) {
      all_have_vas = ctx.Get(node.input(i)).HasValueAsShape();
    }
    if (all_have_vas) {
      SymShape value_as_shape;
      const auto append = [&](const SymShape &src) {
        for (std::size_t i = 0; i < src.Rank(); ++i) {
          value_as_shape.PushBack(src[i]);
        }
      };
      append(first.ValueAsShape());
      for (int i = 1; i < n_inputs; ++i) {
        append(ctx.Get(node.input(i)).ValueAsShape());
      }
      out_tensor.SetValueAsShape(std::move(value_as_shape));
    }
  }

  ctx.Set(node.output(0), std::move(out_tensor));
}

} // namespace shapes::tensor
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes
