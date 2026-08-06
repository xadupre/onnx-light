// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes {

// Alias to the symbolic dimension-expression library, which lives in
// ``onnx_core`` so both ``onnx_op`` and ``onnx_shapes`` can share it.
namespace expressions = ::ONNX_LIGHT_NAMESPACE::core::expressions;
namespace shapes::tensor {

namespace {

// Attempts to read a 1-D INT64 tensor as a vector of concrete integer values.
// Returns ``std::nullopt`` when the tensor's value is unknown or contains a
// symbolic dimension.
std::optional<std::vector<int64_t>> TryReadIntVector(const SymTensor &t) {
  if (!t.HasValueAsShape()) {
    return std::nullopt;
  }
  std::vector<int64_t> values;
  const SymShape &shape = t.ValueAsShape();
  values.reserve(shape.Rank());
  for (size_t i = 0; i < shape.Rank(); ++i) {
    if (!shape[i].IsInt()) {
      return std::nullopt;
    }
    values.push_back(shape[i].AsInt());
  }
  return values;
}

// Resolves the per-output split sizes when ``num_outputs`` is used and the
// axis dimension is a concrete integer. Mirrors :class:`kernel::Split`'s
// resolution: the last chunk absorbs the remainder.
std::vector<int64_t> SplitByNumOutputs(int64_t axis_dim, int64_t num_outputs) {
  const int64_t chunk = (axis_dim + num_outputs - 1) / num_outputs;
  std::vector<int64_t> sizes(static_cast<size_t>(num_outputs), chunk);
  int64_t remaining = axis_dim - chunk * (num_outputs - 1);
  sizes.back() = remaining < 0 ? 0 : remaining;
  return sizes;
}

// Wraps :cpp:func:`expressions::simplify_expression` and converts its variant
// result into an ``SymDim``: integer alternatives become concrete dims,
// strings become symbolic dims.
SymDim SimplifyToDim(const std::string &expr) {
  expressions::SimplifyResult r = expressions::simplify_expression(expr);
  if (std::holds_alternative<int64_t>(r)) {
    return SymDim(std::get<int64_t>(r));
  }
  return SymDim(std::get<std::string>(r));
}

// Builds per-output symbolic axis dims for ``Split(d, num_outputs=n)`` when
// ``d`` is purely symbolic. Mirrors the integer-arithmetic resolution in
// :func:`SplitByNumOutputs`: the first ``n - 1`` outputs each get
// ``ceil(d/n) = (d + n - 1) // n``, and the last output absorbs the remainder
// ``d - (n - 1) * ceil(d/n)``. For ``n == 2`` the remainder simplifies to
// ``d // 2`` (integer arithmetic for ``d >= 0``). Each generated expression
// is run through :cpp:func:`expressions::simplify_expression` so that
// constant subexpressions collapse and the canonical form is preserved.
std::vector<SymDim> SymbolicSplitByNumOutputs(const std::string &d, int64_t num_outputs) {
  std::vector<SymDim> result;
  result.reserve(static_cast<size_t>(num_outputs));
  if (num_outputs == 1) {
    result.push_back(SimplifyToDim(d));
    return result;
  }
  const std::string ns = std::to_string(num_outputs);
  const std::string nm1 = std::to_string(num_outputs - 1);
  const std::string chunk_expr = "(" + d + "+" + nm1 + ")//" + ns;
  const SymDim chunk_dim = SimplifyToDim(chunk_expr);
  for (int64_t i = 0; i < num_outputs - 1; ++i) {
    result.push_back(chunk_dim);
  }
  if (num_outputs == 2) {
    // ``d - (d + 1) // 2 == d // 2`` in integer arithmetic for ``d >= 0``.
    result.push_back(SimplifyToDim("(" + d + ")//2"));
  } else {
    result.push_back(SimplifyToDim("(" + d + ")-" + nm1 + "*(" + chunk_expr + ")"));
  }
  return result;
}

} // namespace

void ComputeShapeSplit(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Split", "ComputeShapeSplit");
  EXT_ENFORCE_INVALID(!(node.input_size() < 1),
                      "ComputeShapeSplit: Split requires at least one input.");

  const SymTensor &input = ctx.Get(node.input(0));
  const SymShape &in_shape = input.Shape();
  const int64_t rank = static_cast<int64_t>(in_shape.Rank());

  const int64_t axis_attr = GetAttributeOr<int64_t>(node, "axis", 0);
  const int64_t resolved_axis = axis_attr < 0 ? axis_attr + rank : axis_attr;
  EXT_ENFORCE_INVALID(!(resolved_axis < 0 || resolved_axis >= rank), "ComputeShapeSplit: axis ",
                      axis_attr, " is out of range for rank ", rank, ".");
  const std::size_t axis = static_cast<std::size_t>(resolved_axis);

  const int num_outputs_decl = node.output_size();

  // Resolve the split sizes when possible. ``sizes`` stays empty when they
  // cannot be determined, in which case the per-output axis dimension is set
  // to a fresh symbolic placeholder.
  std::vector<int64_t> sizes;
  // Symbolic per-output axis dims, used when the axis dim is purely
  // symbolic (no concrete value, no ``ValueAsShape``) but ``num_outputs`` is
  // still known. Each entry, when present, takes precedence over the fresh
  // placeholder fallback.
  std::vector<SymDim> symbolic_sizes;

  // When the input is 1-D and carries a ``ValueAsShape``, the axis dimension
  // is exactly ``ValueAsShape().Rank()`` even if the declared input shape is
  // symbolic — the annotation *is* the value of the tensor.
  const bool vas_gives_axis_dim = resolved_axis == 0 && rank == 1 && input.HasValueAsShape();
  const int64_t effective_axis_dim =
      in_shape[axis].IsInt()
          ? in_shape[axis].AsInt()
          : (vas_gives_axis_dim ? static_cast<int64_t>(input.ValueAsShape().Rank()) : int64_t{-1});

  // 1) Opset 13+ takes ``split`` as an optional input; opset 1/2/11 carry it
  //    as an INTS attribute.
  if (node.input_size() >= 2 && !node.input(1).empty()) {
    const SymTensor &split_t = ctx.Get(node.input(1));
    if (std::optional<std::vector<int64_t>> v = TryReadIntVector(split_t); v.has_value()) {
      sizes = std::move(*v);
    }
  } else {
    std::vector<int64_t> attr_split;
    if (GetAttributeInts(node, "split", attr_split)) {
      sizes = std::move(attr_split);
    } else if (effective_axis_dim >= 0) {
      // 2) Fall back to ``num_outputs`` (opset 18+) or to the declared number
      //    of outputs (older opsets require the input axis dim to be evenly
      //    divisible by the output count). The axis dim is taken from the
      //    declared shape when concrete, or from the input's
      //    ``ValueAsShape().Rank()`` (a 1-D, axis-0 special case) when the
      //    declared dim is symbolic but the value-as-shape is known.
      const int64_t num_outputs =
          GetAttributeOr<int64_t>(node, "num_outputs", static_cast<int64_t>(num_outputs_decl));
      if (num_outputs > 0) {
        sizes = SplitByNumOutputs(effective_axis_dim, num_outputs);
      }
    } else if (in_shape[axis].IsExpr()) {
      // 3) The axis dim is purely symbolic and ``ValueAsShape`` is unknown,
      //    so concrete sizes can't be resolved. ``num_outputs`` (or the
      //    declared output count) is still enough to build per-output
      //    symbolic axis dims using integer-arithmetic chunking.
      const int64_t num_outputs =
          GetAttributeOr<int64_t>(node, "num_outputs", static_cast<int64_t>(num_outputs_decl));
      if (num_outputs > 0) {
        symbolic_sizes = SymbolicSplitByNumOutputs(in_shape[axis].AsExpr(), num_outputs);
      }
    }
  }

  // Validate the resolved sizes when both they and the axis dim are known.
  if (!sizes.empty() && in_shape[axis].IsInt()) {
    int64_t total = 0;
    for (int64_t s : sizes) {
      EXT_ENFORCE_INVALID(!(s < 0), "ComputeShapeSplit: 'split' entries must be non-negative.");
      total += s;
    }
    EXT_ENFORCE_INVALID(total == in_shape[axis].AsInt(), "ComputeShapeSplit: sum of 'split' (",
                        total, ") does not match the input dimension on 'axis' (",
                        in_shape[axis].AsInt(), ").");
  }

  EXT_ENFORCE_INVALID(!(!sizes.empty() && static_cast<int>(sizes.size()) != num_outputs_decl),
                      "ComputeShapeSplit: number of resolved split sizes (", sizes.size(),
                      ") does not match the number of node outputs (", num_outputs_decl, ").");
  EXT_ENFORCE_INVALID(
      !(!symbolic_sizes.empty() && static_cast<int>(symbolic_sizes.size()) != num_outputs_decl),
      "ComputeShapeSplit: number of resolved symbolic split sizes (", symbolic_sizes.size(),
      ") does not match the number of node outputs (", num_outputs_decl, ").");

  // Propagate ``ValueAsShape`` when splitting along axis 0 of a 1-D tensor
  // that already carries a ``ValueAsShape`` annotation and the split sizes
  // are known. Each output's ``ValueAsShape`` is the corresponding contiguous
  // slice of the input's ``ValueAsShape``. This mirrors :cpp:func:`Concat`'s
  // inverse propagation and keeps downstream consumers (e.g. ``Expand``,
  // ``Reshape``) able to recover concrete/symbolic dimensions through
  // ``Split`` nodes used in shape arithmetic.
  const bool propagate_vas =
      resolved_axis == 0 && rank == 1 && input.HasValueAsShape() && !sizes.empty();
  const SymShape *in_vas = propagate_vas ? &input.ValueAsShape() : nullptr;
  // The ``ValueAsShape`` length must agree with the resolved axis dimension
  // when both are known; if not, skip propagation rather than producing a
  // misaligned slice.
  size_t vas_total = 0;
  if (propagate_vas) {
    for (int64_t s : sizes) {
      vas_total += static_cast<size_t>(s);
    }
    if (vas_total != in_vas->Rank()) {
      in_vas = nullptr;
    }
  }

  size_t vas_offset = 0;
  for (int i = 0; i < num_outputs_decl; ++i) {
    const std::string &name = node.output(i);
    const int64_t size_i = sizes.empty() ? int64_t{0} : sizes[static_cast<size_t>(i)];
    if (name.empty()) {
      vas_offset += static_cast<size_t>(size_i);
      continue;
    }
    SymShape out_shape = in_shape;
    if (!sizes.empty()) {
      out_shape[axis] = SymDim(size_i);
    } else if (!symbolic_sizes.empty()) {
      out_shape[axis] = symbolic_sizes[static_cast<size_t>(i)];
    } else {
      out_shape[axis] =
          SymDim("Split_axis" + std::to_string(resolved_axis) + "_out" + std::to_string(i));
    }
    SymTensor out_tensor(nullptr, input.Dtype(), std::move(out_shape));
    if (in_vas != nullptr) {
      SymShape out_vas;
      for (size_t j = 0; j < static_cast<size_t>(size_i); ++j) {
        out_vas.PushBack((*in_vas)[vas_offset + j]);
      }
      out_tensor.SetValueAsShape(std::move(out_vas));
    }
    vas_offset += static_cast<size_t>(size_i);
    ctx.Set(name, std::move(out_tensor));
  }
}

} // namespace shapes::tensor
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes
