// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

namespace {

// Reads the optional ``axes`` attribute introduced at opset 18 and normalises
// negative entries to positive indices in ``[0, rank)``. When ``axes`` is
// absent the function returns ``{0, 1, ..., rank-1}``.
std::vector<int64_t> ResolveAxes(const NodeProto &node, std::size_t rank) {
  const AttributeProto *axes_attr = FindAttribute(node, "axes");
  std::vector<int64_t> axes;
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
    if (na < 0 || na >= static_cast<int64_t>(rank)) {
      throw std::invalid_argument("ComputeShapeResize: 'axes' value " + std::to_string(a) +
                                  " is out of range for input of rank " + std::to_string(rank) +
                                  ".");
    }
    axes.push_back(na);
  }
  return axes;
}

} // namespace

void ComputeShapeResize(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Resize", "ComputeShapeResize");
  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapeResize: Resize requires at least one input.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const OptimShape &input_shape = input.Shape();
  const std::size_t rank = input_shape.Rank();

  // The v10 schema is ``(X, scales)``; v11+ is ``(X, roi, scales, sizes)``
  // where any of ``roi``/``scales``/``sizes`` may be skipped by providing
  // the empty string as input name.
  const bool has_v11_layout = node.input_size() >= 3;

  // Index of the ``sizes`` input when present and non-empty.
  std::string sizes_name;
  if (has_v11_layout && node.input_size() >= 4) {
    sizes_name = node.input(3).as_string();
  }

  // The ``scales`` input is a FLOAT tensor whose values are not tracked by the
  // data-propagation lattice (which only carries integer shape values), so it
  // never contributes concrete output dims here. The ``sizes`` input (INT64)
  // is tracked and can drive concrete output dims when statically known.
  std::vector<int64_t> sizes_data;
  bool sizes_known = false;
  if (!sizes_name.empty()) {
    const OptimTensor &sizes_tensor = ctx.Get(sizes_name);
    if (sizes_tensor.HasValueAsShape()) {
      const OptimShape &val = sizes_tensor.ValueAsShape();
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

  // ``axes`` (v18+) selects the subset of axes that ``scales``/``sizes`` refer
  // to. When absent it defaults to all axes.
  std::vector<int64_t> axes = ResolveAxes(node, rank);

  OptimShape out_shape;
  for (std::size_t i = 0; i < rank; ++i) {
    out_shape.PushBack(OptimDim("Resize_dim" + std::to_string(i)));
  }
  // Non-axis dims (when ``axes`` is provided) keep the input dim.
  if (axes.size() != rank) {
    for (std::size_t i = 0; i < rank; ++i) {
      out_shape[i] = input_shape[i];
    }
  }

  if (sizes_known && sizes_data.size() == axes.size()) {
    for (std::size_t i = 0; i < axes.size(); ++i) {
      out_shape[static_cast<std::size_t>(axes[i])] = OptimDim(sizes_data[i]);
    }
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
