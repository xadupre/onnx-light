// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
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

// Builds the symbolic expression representing the product of ``dims``.
// Integer entries are emitted as their numeric value; symbolic entries are
// parenthesised to preserve precedence. Returns ``"1"`` for an empty product.
std::string BuildProductExpr(const std::vector<SymDim> &dims) {
  std::string expr;
  for (const SymDim &dim : dims) {
    if (!expr.empty()) {
      expr += "*";
    }
    if (dim.IsInt()) {
      expr += std::to_string(dim.AsInt());
    } else {
      expr += "(" + dim.AsExpr() + ")";
    }
  }
  return expr.empty() ? std::string("1") : expr;
}

// Tries to infer the Reshape ``-1`` dimension by building the symbolic
// expression ``(product of input dims) /: (product of out_shape dims excluding
// the -1 position)`` and running it through
// :cpp:func:`expressions::simplify_expression`. The ``/:`` operator is used
// because Reshape preserves the total number of elements, so the division is
// always exact (integer, no remainder). This allows the simplifier to freely
// distribute multiplication through the division, producing cleaner symbolic
// results (e.g. ``"batch*c/:c"`` → ``"batch"``). This handles purely concrete
// inputs (returns an int), purely symbolic inputs (returns a clean symbolic
// expression such as ``"c/:2"``) and mixed cases. Returns ``std::nullopt``
// when any output dim is concrete zero — division by zero is not meaningful
// and the caller already rejects that case explicitly.
std::optional<SymDim> InferNegOneFromFactors(const SymShape &data_shape, const SymShape &out_shape,
                                             int neg_one_dim_index) {
  std::vector<SymDim> input_factors(data_shape.Dims());

  std::vector<SymDim> output_factors;
  output_factors.reserve(out_shape.Rank());
  for (int i = 0; i < static_cast<int>(out_shape.Rank()); ++i) {
    if (i == neg_one_dim_index) {
      continue;
    }
    const SymDim &dim = out_shape[i];
    if (dim.IsInt() && dim.AsInt() == 0) {
      return std::nullopt;
    }
    output_factors.push_back(dim);
  }

  const std::string expr =
      "(" + BuildProductExpr(input_factors) + ")/:(" + BuildProductExpr(output_factors) + ")";
  expressions::SimplifyResult result = expressions::simplify_expression(expr);
  if (std::holds_alternative<int64_t>(result)) {
    return SymDim(std::get<int64_t>(result));
  }
  return SymDim(std::get<std::string>(result));
}

// Wraps ``InferNegOneFromFactors`` and falls back to a stable symbolic
// placeholder when no deterministic inference is possible.
SymDim InferredOrFallbackDim(const SymShape &data_shape, const SymShape &out_shape,
                             int neg_one_dim_index) {
  auto inferred = InferNegOneFromFactors(data_shape, out_shape, neg_one_dim_index);
  return inferred.has_value() ? inferred.value()
                              : SymDim("Reshape_neg1_" + std::to_string(neg_one_dim_index));
}

} // namespace

void ComputeShapeReshape(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Reshape", "ComputeShapeReshape");

  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeReshape: Reshape requires two inputs (data, shape).");

  const SymTensor &data = ctx.Get(node.input(0));
  const SymTensor &shape_input = ctx.Get(node.input(1));

  const TensorType dtype = data.Dtype();
  const SymShape &data_shape = data.Shape();
  const int64_t allowzero = GetAttributeOr<int64_t>(node, "allowzero", 0);

  // When the target shape is not known via data-propagation, fall
  // back to the rank exposed by the 1-D ``shape`` input itself (its
  // single static dim). Every output dimension is left symbolic.
  if (!shape_input.HasValueAsShape()) {
    SymShape out_shape;
    if (shape_input.Shape().Rank() == 1 && shape_input.Shape()[0].IsInt()) {
      const int64_t rank = shape_input.Shape()[0].AsInt();
      for (int64_t i = 0; i < rank; ++i) {
        out_shape.PushBack(SymDim("Reshape_dim" + std::to_string(i)));
      }
    } else {
      out_shape.PushBack(SymDim("Reshape_dim0"));
    }
    ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
    return;
  }

  const SymShape &target = shape_input.ValueAsShape();
  const int target_rank = static_cast<int>(target.Rank());
  const int data_rank = static_cast<int>(data_shape.Rank());

  SymShape out_shape;
  // Index of the ``-1`` entry in ``target`` (if any), so it can be
  // back-filled from the input element count.
  int neg_one_index = -1;
  // Product of the concrete dimensions written into ``out_shape``,
  // used to back-fill the ``-1`` dimension. ``output_product_valid``
  // becomes false as soon as a symbolic dim is forwarded.
  int64_t output_product = 1;
  bool output_product_valid = true;
  // Tracks ``0`` entries that have been resolved by copying from the
  // input shape (so that the corresponding input dims can be excluded
  // from the input-product computation below).
  std::vector<bool> unresolved_zeros(target_rank, false);

  for (int i = 0; i < target_rank; ++i) {
    const SymDim &dim = target[i];
    if (!dim.IsInt()) {
      out_shape.PushBack(dim);
      output_product_valid = false;
      continue;
    }
    const int64_t v = dim.AsInt();
    if (v == -1) {
      EXT_ENFORCE_INVALID(neg_one_index == -1,
                          "ComputeShapeReshape: target shape may not have multiple -1 dimensions.");
      neg_one_index = i;
      // Placeholder; resolved below.
      out_shape.PushBack(SymDim(static_cast<int64_t>(-1)));
    } else if (v == 0) {
      if (allowzero == 0) {
        EXT_ENFORCE_INVALID(!(i >= data_rank),
                            "ComputeShapeReshape: invalid position of 0 in target "
                            "shape (index ",
                            i, " out of input rank ", data_rank, ").");
        const SymDim &input_dim = data_shape[i];
        out_shape.PushBack(input_dim);
        if (input_dim.IsInt()) {
          output_product *= input_dim.AsInt();
        } else {
          unresolved_zeros[i] = true;
          output_product_valid = false;
        }
      } else {
        out_shape.PushBack(SymDim(static_cast<int64_t>(0)));
        output_product *= 0;
      }
    } else if (v > 0) {
      out_shape.PushBack(SymDim(v));
      output_product *= v;
    } else {
      EXT_THROW_INVALID("ComputeShapeReshape: invalid dimension value ", v, " in target shape.");
    }
  }

  if (neg_one_index != -1 && output_product_valid) {
    EXT_ENFORCE_INVALID(
        output_product != 0,
        "ComputeShapeReshape: invalid target shape product of 0 in combination with -1.");
    int64_t input_product = 1;
    bool input_product_valid = true;
    for (int i = 0; i < data_rank; ++i) {
      if (data_shape[i].IsInt()) {
        input_product *= data_shape[i].AsInt();
      } else if (i >= static_cast<int>(unresolved_zeros.size()) || !unresolved_zeros[i]) {
        input_product_valid = false;
        break;
      }
    }
    if (input_product_valid) {
      EXT_ENFORCE_INVALID(
          input_product % output_product == 0,
          "ComputeShapeReshape: dimension could not be inferred: incompatible shapes (input "
          "element count ",
          input_product, " is not a multiple of ", output_product, ").");
      out_shape[neg_one_index] = SymDim(input_product / output_product);
    } else {
      out_shape[neg_one_index] = InferredOrFallbackDim(data_shape, out_shape, neg_one_index);
    }
  } else if (neg_one_index != -1) {
    out_shape[neg_one_index] = InferredOrFallbackDim(data_shape, out_shape, neg_one_index);
  }

  SymTensor out_tensor(nullptr, dtype, std::move(out_shape));

  // Propagate ``ValueAsShape`` when the input ``data`` already carries one
  // and the output is 1-D with the same number of elements. ``ValueAsShape``
  // represents the flattened sequence of (symbolic or concrete) values held
  // by the tensor, so a 1-D ``Reshape`` that preserves the element count
  // simply forwards the same sequence. This keeps downstream consumers
  // (``Expand``, ``ConstantOfShape``, another ``Reshape``, …) able to
  // recover concrete/symbolic dimensions through ``Reshape`` nodes used in
  // shape arithmetic.
  if (data.HasValueAsShape() && out_tensor.Shape().Rank() == 1 && out_tensor.Shape()[0].IsInt() &&
      static_cast<std::size_t>(out_tensor.Shape()[0].AsInt()) == data.ValueAsShape().Rank()) {
    out_tensor.SetValueAsShape(data.ValueAsShape());
  }

  ctx.Set(node.output(0), std::move(out_tensor));
}

} // namespace shapes::tensor
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes
