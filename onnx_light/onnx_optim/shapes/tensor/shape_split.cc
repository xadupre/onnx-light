// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <optional>
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

// Attempts to read a 1-D INT64 tensor as a vector of concrete integer values.
// Returns ``std::nullopt`` when the tensor's value is unknown or contains a
// symbolic dimension.
std::optional<std::vector<int64_t>> TryReadIntVector(const OptimTensor &t) {
  if (!t.HasValueAsShape()) {
    return std::nullopt;
  }
  std::vector<int64_t> values;
  const OptimShape &shape = t.ValueAsShape();
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

} // namespace

void ComputeShapeSplit(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Split", "ComputeShapeSplit");
  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapeSplit: Split requires at least one input.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const OptimShape &in_shape = input.Shape();
  const int64_t rank = static_cast<int64_t>(in_shape.Rank());

  const int64_t axis_attr = GetAttributeOr<int64_t>(node, "axis", 0);
  const int64_t resolved_axis = axis_attr < 0 ? axis_attr + rank : axis_attr;
  if (resolved_axis < 0 || resolved_axis >= rank) {
    throw std::invalid_argument("ComputeShapeSplit: axis " + std::to_string(axis_attr) +
                                " is out of range for rank " + std::to_string(rank) + ".");
  }
  const std::size_t axis = static_cast<std::size_t>(resolved_axis);

  const int num_outputs_decl = node.output_size();

  // Resolve the split sizes when possible. ``sizes`` stays empty when they
  // cannot be determined, in which case the per-output axis dimension is set
  // to a fresh symbolic placeholder.
  std::vector<int64_t> sizes;

  // 1) Opset 13+ takes ``split`` as an optional input; opset 1/2/11 carry it
  //    as an INTS attribute.
  if (node.input_size() >= 2 && !node.input(1).empty()) {
    const OptimTensor &split_t = ctx.Get(node.input(1).as_string());
    if (std::optional<std::vector<int64_t>> v = TryReadIntVector(split_t); v.has_value()) {
      sizes = std::move(*v);
    }
  } else {
    std::vector<int64_t> attr_split;
    if (GetAttributeInts(node, "split", attr_split)) {
      sizes = std::move(attr_split);
    } else if (in_shape[axis].IsInt()) {
      // 2) Fall back to ``num_outputs`` (opset 18+) or to the declared number
      //    of outputs (older opsets require the input axis dim to be evenly
      //    divisible by the output count).
      const int64_t axis_dim = in_shape[axis].AsInt();
      const int64_t num_outputs =
          GetAttributeOr<int64_t>(node, "num_outputs", static_cast<int64_t>(num_outputs_decl));
      if (num_outputs > 0) {
        sizes = SplitByNumOutputs(axis_dim, num_outputs);
      }
    }
  }

  // Validate the resolved sizes when both they and the axis dim are known.
  if (!sizes.empty() && in_shape[axis].IsInt()) {
    int64_t total = 0;
    for (int64_t s : sizes) {
      if (s < 0) {
        throw std::invalid_argument("ComputeShapeSplit: 'split' entries must be non-negative.");
      }
      total += s;
    }
    if (total != in_shape[axis].AsInt()) {
      throw std::invalid_argument("ComputeShapeSplit: sum of 'split' (" + std::to_string(total) +
                                  ") does not match the input dimension on 'axis' (" +
                                  std::to_string(in_shape[axis].AsInt()) + ").");
    }
  }

  if (!sizes.empty() && static_cast<int>(sizes.size()) != num_outputs_decl) {
    throw std::invalid_argument(
        "ComputeShapeSplit: number of resolved split sizes (" + std::to_string(sizes.size()) +
        ") does not match the number of node outputs (" + std::to_string(num_outputs_decl) + ").");
  }

  for (int i = 0; i < num_outputs_decl; ++i) {
    const std::string &name = node.output(i).as_string();
    if (name.empty()) {
      continue;
    }
    OptimShape out_shape = in_shape;
    if (!sizes.empty()) {
      out_shape[axis] = OptimDim(sizes[static_cast<size_t>(i)]);
    } else {
      out_shape[axis] =
          OptimDim("Split_axis" + std::to_string(resolved_axis) + "_out" + std::to_string(i));
    }
    ctx.Set(name, OptimTensor(nullptr, input.Dtype(), std::move(out_shape)));
  }
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
