// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/sequence/shape_sequence.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "onnx_optim/optim_sequence.h"
#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace sequence {

namespace {

// Attempts to read a 1-D or scalar integer tensor as a vector of
// concrete integer values. Returns ``std::nullopt`` when the tensor's
// value is unknown or contains a symbolic dimension.
std::optional<std::vector<int64_t>> TryReadIntVector(const OptimTensor &t) {
  if (!t.HasValueAsShape()) {
    return std::nullopt;
  }
  const OptimShape &shape = t.ValueAsShape();
  std::vector<int64_t> values;
  values.reserve(shape.Rank());
  for (std::size_t i = 0; i < shape.Rank(); ++i) {
    if (!shape[i].IsInt()) {
      return std::nullopt;
    }
    values.push_back(shape[i].AsInt());
  }
  return values;
}

} // namespace

void ComputeShapeSplitToSequence(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SplitToSequence", "ComputeShapeSplitToSequence");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeSplitToSequence: SplitToSequence requires at least one input.");

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const OptimShape &in_shape = input.Shape();
  const TensorType elem_dtype = input.Dtype();

  const int64_t axis_attr = GetAttributeOr<int64_t>(node, "axis", 0);
  const int64_t keepdims = GetAttributeOr<int64_t>(node, "keepdims", 1);

  const bool has_split = node.input_size() >= 2 && !node.input(1).empty();

  // Without a known input rank we can only forward the element dtype.
  if (in_shape.Rank() == 0) {
    ctx.SetSequence(node.output(0),
                    OptimSequence(elem_dtype, OptimDim("SplitToSequence_" +
                                                       node.output(0).as_string() + "_len")));
    return;
  }

  const int64_t rank = static_cast<int64_t>(in_shape.Rank());
  const int64_t resolved_axis = axis_attr < 0 ? axis_attr + rank : axis_attr;
  EXT_ENFORCE_INVALID(resolved_axis >= 0 && resolved_axis < rank,
                      "ComputeShapeSplitToSequence: axis " + std::to_string(axis_attr) +
                          " is out of range for rank " + std::to_string(rank) + ".");
  const std::size_t axis = static_cast<std::size_t>(resolved_axis);

  const bool axis_dim_known = in_shape[axis].IsInt();
  const int64_t axis_dim = axis_dim_known ? in_shape[axis].AsInt() : int64_t{-1};

  // Resolve the per-output split sizes when possible.
  std::optional<std::vector<int64_t>> sizes;
  bool split_is_scalar = false;
  if (!has_split) {
    if (axis_dim_known) {
      sizes = std::vector<int64_t>(static_cast<std::size_t>(axis_dim), int64_t{1});
    }
  } else {
    const OptimTensor &split_t = ctx.Get(node.input(1).as_string());
    const OptimShape &split_shape = split_t.Shape();
    split_is_scalar = split_shape.Rank() == 0;
    if (std::optional<std::vector<int64_t>> v = TryReadIntVector(split_t); v.has_value()) {
      if (split_is_scalar) {
        if (axis_dim_known) {
          EXT_ENFORCE_INVALID((*v).size() == 1, "ComputeShapeSplitToSequence: scalar 'split' "
                                                "must contain exactly one value.");
          const int64_t chunk = (*v)[0];
          EXT_ENFORCE_INVALID(chunk > 0,
                              "ComputeShapeSplitToSequence: scalar 'split' must be positive.");
          std::vector<int64_t> resolved;
          int64_t remaining = axis_dim;
          while (remaining > 0) {
            const int64_t take = remaining >= chunk ? chunk : remaining;
            resolved.push_back(take);
            remaining -= take;
          }
          if (resolved.empty()) {
            resolved.push_back(0);
          }
          sizes = std::move(resolved);
        }
      } else {
        // 1-D tensor: use entries as-is.
        if (axis_dim_known) {
          int64_t total = 0;
          for (int64_t s : *v) {
            EXT_ENFORCE_INVALID(s >= 0, "ComputeShapeSplitToSequence: 'split' entries must be "
                                        "non-negative.");
            total += s;
          }
          EXT_ENFORCE_INVALID(total == axis_dim, "ComputeShapeSplitToSequence: sum of 'split' (" +
                                                     std::to_string(total) +
                                                     ") does not match the input dim on 'axis' (" +
                                                     std::to_string(axis_dim) + ").");
        }
        sizes = std::move(*v);
      }
    }
  }

  // ``squeeze`` only applies when ``split`` is omitted and keepdims == 0.
  const bool squeeze = !has_split && keepdims == 0;

  if (!sizes.has_value()) {
    // Unknown number of chunks → only forward dtype and symbolic length.
    ctx.SetSequence(node.output(0),
                    OptimSequence(elem_dtype, OptimDim("SplitToSequence_" +
                                                       node.output(0).as_string() + "_len")));
    return;
  }

  std::vector<OptimShape> elem_shapes;
  elem_shapes.reserve(sizes->size());
  for (int64_t s : *sizes) {
    OptimShape out_shape;
    for (std::size_t d = 0; d < in_shape.Rank(); ++d) {
      if (d == axis) {
        if (!squeeze) {
          out_shape.PushBack(OptimDim(s));
        }
      } else {
        out_shape.PushBack(in_shape[d]);
      }
    }
    elem_shapes.push_back(std::move(out_shape));
  }

  ctx.SetSequence(node.output(0), OptimSequence(elem_dtype, std::move(elem_shapes)));
}

} // namespace sequence
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
