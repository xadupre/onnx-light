// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/reduction/shape_reduction.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace reduction {

namespace {

int64_t ResolveAxis(int64_t axis, int64_t rank, const std::string &op_type) {
  const int64_t resolved = axis < 0 ? axis + rank : axis;
  EXT_ENFORCE_INVALID(resolved >= 0 && resolved < rank,
                      "ComputeShape" + op_type + ": axis " + std::to_string(axis) +
                          " is out of range for rank " + std::to_string(rank) + ".");
  return resolved;
}

// Builds the output shape of a reduction given the boolean mask of reduced
// dimensions. Reduced positions are either dropped (``keepdims=false``) or
// replaced by ``1`` (``keepdims=true``).
OptimShape BuildOutputShape(const OptimShape &in_shape, const std::vector<bool> &is_reduced,
                            bool keepdims) {
  OptimShape out;
  for (std::size_t d = 0; d < in_shape.Rank(); ++d) {
    if (is_reduced[d]) {
      if (keepdims) {
        out.PushBack(OptimDim(static_cast<int64_t>(1)));
      }
    } else {
      out.PushBack(in_shape[d]);
    }
  }
  return out;
}

} // namespace

void ComputeShapeReduceCommon(ShapesContext &ctx, const NodeProto &node, const char *data,
                              const char *axes, const char *op_type) {
  const std::string op(op_type);
  CheckNodeOpAndOutput(node, op_type, ("ComputeShape" + op).c_str());

  const OptimTensor &input = ctx.Get(data);
  const OptimShape &in_shape = input.Shape();
  const int64_t rank = static_cast<int64_t>(in_shape.Rank());

  const bool keepdims = GetAttributeOr<int64_t>(node, "keepdims", 1) != 0;
  const bool noop_with_empty_axes = GetAttributeOr<int64_t>(node, "noop_with_empty_axes", 0) != 0;

  // Determine the opset version to decide where ``axes`` comes from. Default
  // to v13 (input form) when no opset has been recorded.
  const int opset = ctx.HasOpsetVersion(kOnnxDomain) ? ctx.OpsetVersion(kOnnxDomain) : 13;
  const bool axes_is_input = opset >= 13;

  std::vector<bool> is_reduced(static_cast<std::size_t>(rank), false);
  bool axes_known = false;
  bool axes_count_known = false;
  int64_t axes_count = 0;

  if (axes_is_input) {
    const bool has_axes_input = axes != nullptr && axes[0] != '\0' && ctx.Has(std::string(axes));
    if (has_axes_input) {
      const OptimTensor &axes_tensor = ctx.Get(std::string(axes));
      if (axes_tensor.HasValueAsShape()) {
        const OptimShape &av = axes_tensor.ValueAsShape();
        axes_known = true;
        axes_count_known = true;
        axes_count = static_cast<int64_t>(av.Rank());
        if (axes_count == 0) {
          if (!noop_with_empty_axes) {
            std::fill(is_reduced.begin(), is_reduced.end(), true);
          }
        } else {
          for (std::size_t i = 0; i < av.Rank(); ++i) {
            if (!av[i].IsInt()) {
              // Symbolic axis value — cannot resolve to a concrete index.
              axes_known = false;
              break;
            }
            const int64_t a = ResolveAxis(av[i].AsInt(), rank, op);
            is_reduced[static_cast<std::size_t>(a)] = true;
          }
        }
      } else {
        // Concrete values unknown but the rank-1 shape of ``axes`` may still
        // tell us how many axes will be reduced.
        const OptimShape &as = axes_tensor.Shape();
        if (as.Rank() == 1 && as[0].IsInt()) {
          axes_count_known = true;
          axes_count = as[0].AsInt();
          if (axes_count == 0 && !noop_with_empty_axes) {
            axes_known = true;
            std::fill(is_reduced.begin(), is_reduced.end(), true);
          }
        }
      }
    } else {
      // ``axes`` input omitted (or empty string). Default behaviour depends
      // on ``noop_with_empty_axes``: reduce all unless the noop flag is set.
      axes_known = true;
      if (!noop_with_empty_axes) {
        std::fill(is_reduced.begin(), is_reduced.end(), true);
      }
    }
  } else {
    // opset < 13: ``axes`` is an attribute.
    std::vector<int64_t> attr_axes;
    const bool has_axes_attr = GetAttributeInts(node, "axes", attr_axes);
    axes_known = true;
    if (!has_axes_attr || attr_axes.empty()) {
      std::fill(is_reduced.begin(), is_reduced.end(), true);
    } else {
      for (int64_t a : attr_axes) {
        const int64_t resolved = ResolveAxis(a, rank, op);
        is_reduced[static_cast<std::size_t>(resolved)] = true;
      }
    }
    axes_count_known = true;
    axes_count = static_cast<int64_t>(attr_axes.size());
  }

  OptimShape out_shape;
  if (axes_known) {
    out_shape = BuildOutputShape(in_shape, is_reduced, keepdims);
  } else if (keepdims) {
    // Rank is preserved; every dim is either kept as-is or replaced by 1.
    // Because we do not know which dims are reduced, mark every dim as a
    // fresh symbolic expression derived from the input dim.
    for (std::size_t d = 0; d < in_shape.Rank(); ++d) {
      const OptimDim &din = in_shape[d];
      const std::string expr =
          op + "(" + (din.IsInt() ? std::to_string(din.AsInt()) : din.AsExpr()) + ")";
      out_shape.PushBack(OptimDim(expr));
    }
  } else if (axes_count_known) {
    // Rank decreases by ``axes_count``; we cannot know which dims survive,
    // so produce a fully-symbolic shape of the right rank.
    const int64_t out_rank = rank - axes_count;
    if (out_rank < 0) {
      throw std::invalid_argument("ComputeShape" + op + ": number of axes (" +
                                  std::to_string(axes_count) + ") exceeds input rank (" +
                                  std::to_string(rank) + ").");
    }
    for (int64_t d = 0; d < out_rank; ++d) {
      out_shape.PushBack(OptimDim(op + "_d" + std::to_string(d)));
    }
  } else {
    // Neither the axes nor their count is known: not enough information to
    // infer the output shape.
    throw std::invalid_argument("ComputeShape" + op +
                                ": cannot infer output shape because neither the "
                                "axes values nor the number of axes is known and 'keepdims' is 0.");
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

void ComputeShapeReduceSum(ShapesContext &ctx, const NodeProto &node, const char *data,
                           const char *axes) {
  ComputeShapeReduceCommon(ctx, node, data, axes, "ReduceSum");
}

void ComputeShapeReduceMax(ShapesContext &ctx, const NodeProto &node, const char *data,
                           const char *axes) {
  ComputeShapeReduceCommon(ctx, node, data, axes, "ReduceMax");
}

void ComputeShapeReduceMin(ShapesContext &ctx, const NodeProto &node, const char *data,
                           const char *axes) {
  ComputeShapeReduceCommon(ctx, node, data, axes, "ReduceMin");
}

void ComputeShapeReduceL1(ShapesContext &ctx, const NodeProto &node, const char *data,
                          const char *axes) {
  ComputeShapeReduceCommon(ctx, node, data, axes, "ReduceL1");
}

void ComputeShapeReduceL2(ShapesContext &ctx, const NodeProto &node, const char *data,
                          const char *axes) {
  ComputeShapeReduceCommon(ctx, node, data, axes, "ReduceL2");
}

void ComputeShapeReduceSumSquare(ShapesContext &ctx, const NodeProto &node, const char *data,
                                 const char *axes) {
  ComputeShapeReduceCommon(ctx, node, data, axes, "ReduceSumSquare");
}

} // namespace reduction
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
