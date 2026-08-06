// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_core/symbolic/symbolic_helper.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes {

// Alias to the symbolic dimension-expression library, which lives in
// ``onnx_core`` so both ``onnx_op`` and ``onnx_shapes`` can share it.
namespace expressions = ::ONNX_LIGHT_NAMESPACE::core::expressions;

// Alias to the fixed-capacity integer shape type, reused here to carry the
// resolved list of axes.
using Shape = ::ONNX_LIGHT_NAMESPACE::core::runtime::Shape;
namespace shapes::tensor {

namespace {

// Converts a ``DimType`` back to an ``SymDim``.
SymDim FromDimType(const expressions::DimType &d) {
  if (std::holds_alternative<int64_t>(d)) {
    return SymDim(std::get<int64_t>(d));
  }
  return SymDim(std::get<std::string>(d));
}

// Reads the optional ``axes`` attribute introduced at opset 18 and normalises
// negative entries to positive indices in ``[0, rank)``. When ``axes`` is
// absent the function returns ``{0, 1, ..., rank-1}``.
Shape ResolveAxes(const NodeProto &node, std::size_t rank) {
  const AttributeProto *axes_attr = FindAttribute(node, "axes");
  Shape axes;
  if (axes_attr == nullptr || axes_attr->ref_ints().size() == 0) {
    axes.reserve(rank);
    for (std::size_t i = 0; i < rank; ++i) {
      axes.push_back(static_cast<int64_t>(i));
    }
    return axes;
  }
  axes.reserve(axes_attr->ref_ints().size());
  for (int64_t a : axes_attr->ref_ints()) {
    int64_t na = a < 0 ? a + static_cast<int64_t>(rank) : a;
    EXT_ENFORCE_INVALID(!(na < 0 || na >= static_cast<int64_t>(rank)),
                        "ComputeShapeResize: 'axes' value ", a,
                        " is out of range for input of rank ", rank, ".");
    axes.push_back(na);
  }
  return axes;
}

// Tries to express a positive floating-point scale as a pure integer divisor
// (scale < 1) or integer multiplier (scale >= 1). Sets ``divisor`` and
// ``multiplier`` (exactly one will be > 1, the other will be 1) on success.
// Returns false when the scale cannot be expressed as a small exact rational
// fraction with integer numerator and denominator.
bool ScaleToRational(double scale, int64_t &divisor, int64_t &multiplier) {
  divisor = 1;
  multiplier = 1;
  if (scale <= 0.0 || std::isnan(scale) || std::isinf(scale)) {
    return false;
  }
  if (scale >= 1.0) {
    const double rounded = std::round(scale);
    if (rounded > 0.0 && std::abs(scale - rounded) <= 1e-9 * rounded) {
      multiplier = static_cast<int64_t>(rounded);
      return true;
    }
    return false;
  }
  // scale < 1: check if 1/scale is close to a positive integer.
  const double inv = 1.0 / scale;
  const double rounded = std::round(inv);
  if (rounded > 0.0 && std::abs(inv - rounded) <= 1e-9 * rounded) {
    divisor = static_cast<int64_t>(rounded);
    return true;
  }
  return false;
}

} // namespace

void ComputeShapeResize(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Resize", "ComputeShapeResize");
  EXT_ENFORCE_INVALID(!(node.input_size() < 1),
                      "ComputeShapeResize: Resize requires at least one input.");

  const SymTensor &input = ctx.Get(node.input(0));
  const SymShape &input_shape = input.Shape();
  const std::size_t rank = input_shape.Rank();

  // The v10 schema is ``(X, scales)``; v11+ is ``(X, roi, scales, sizes)``
  // where any of ``roi``/``scales``/``sizes`` may be skipped by providing
  // the empty string as input name.
  const bool has_v11_layout = node.input_size() >= 3;

  // Index of the ``sizes`` input when present and non-empty.
  std::string sizes_name;
  if (has_v11_layout && node.input_size() >= 4) {
    sizes_name = node.input(3);
  }

  // The ``sizes`` input (INT64) is tracked via the data-propagation lattice
  // and can drive concrete output dims when statically known.
  std::vector<int64_t> sizes_data;
  bool sizes_known = false;
  if (!sizes_name.empty()) {
    const SymTensor &sizes_tensor = ctx.Get(sizes_name);
    if (sizes_tensor.HasValueAsShape()) {
      const SymShape &val = sizes_tensor.ValueAsShape();
      sizes_known = true;
      for (std::size_t i = 0; i < val.Rank(); ++i) {
        if (!val[i].IsInt()) {
          sizes_known = false;
          break;
        }
        sizes_data.push_back(val[i].AsInt());
      }
    }
  }

  // The ``scales`` input is a FLOAT tensor. Per-element values are not tracked
  // by the data-propagation lattice, but the min/max bounds ARE set for
  // constant initialisers. When min == max every element is identical, so we
  // know the single scale value and can compute symbolic output dimensions.
  // For v11+ the scales input is at index 2; for v10 it is at index 1.
  bool scales_known = false;
  int64_t scale_divisor = 1;
  int64_t scale_multiplier = 1;
  if (!sizes_known) {
    std::string scales_name_str;
    if (has_v11_layout && node.input_size() >= 3) {
      scales_name_str = node.input(2);
    } else if (!has_v11_layout && node.input_size() >= 2) {
      scales_name_str = node.input(1);
    }
    if (!scales_name_str.empty() && ctx.Has(scales_name_str)) {
      const SymTensor &scales_tensor = ctx.Get(scales_name_str);
      if (scales_tensor.HasMin() && scales_tensor.HasMax() &&
          scales_tensor.Min() == scales_tensor.Max()) {
        scales_known = ScaleToRational(scales_tensor.Min(), scale_divisor, scale_multiplier);
      }
    }
  }

  // ``axes`` (v18+) selects the subset of axes that ``scales``/``sizes`` refer
  // to. When absent it defaults to all axes.
  Shape axes = ResolveAxes(node, rank);

  SymShape out_shape;
  for (std::size_t i = 0; i < rank; ++i) {
    out_shape.PushBack(SymDim("Resize_dim" + std::to_string(i)));
  }
  // Non-axis dims (when ``axes`` is provided) keep the input dim.
  if (axes.size() != rank) {
    for (std::size_t i = 0; i < rank; ++i) {
      out_shape[i] = input_shape[i];
    }
  }

  if (sizes_known && sizes_data.size() == axes.size()) {
    for (std::size_t i = 0; i < axes.size(); ++i) {
      out_shape[static_cast<std::size_t>(axes[i])] = SymDim(sizes_data[i]);
    }
  } else if (scales_known) {
    for (std::size_t i = 0; i < axes.size(); ++i) {
      const std::size_t axis = static_cast<std::size_t>(axes[i]);
      expressions::DimType dim = ToDimType(input_shape[axis]);
      if (scale_divisor > 1) {
        dim = expressions::dim_div(dim, expressions::DimType{scale_divisor});
      } else if (scale_multiplier > 1) {
        dim = expressions::dim_mul(dim, expressions::DimType{scale_multiplier});
      }
      out_shape[axis] = FromDimType(dim);
    }
  }

  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace shapes::tensor
} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes
