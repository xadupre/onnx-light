// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/sequence/shape_sequence.h"

#include <cstddef>
#include <cstdint>
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

// Merges ``in`` into ``out``. Two concrete integer dimensions must be
// equal. A concrete integer wins over a symbolic dimension. Two
// symbolic dimensions keep the previously-merged ``out`` value.
void MergeDim(OptimDim &out, const OptimDim &in, std::size_t axis, std::size_t input_index) {
  if (out.IsInt() && in.IsInt()) {
    EXT_ENFORCE_INVALID(
        out.AsInt() == in.AsInt(),
        "ComputeShapeConcatFromSequence: element " + std::to_string(input_index) + " dimension " +
            std::to_string(axis) + " (" + std::to_string(in.AsInt()) +
            ") differs from the previously-merged value (" + std::to_string(out.AsInt()) + ").");
    return;
  }
  if (in.IsInt()) {
    out = in;
  }
}

} // namespace

void ComputeShapeConcatFromSequence(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "ConcatFromSequence", "ComputeShapeConcatFromSequence");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeConcatFromSequence: ConcatFromSequence requires one input.");

  const std::string seq_name = node.input(0).as_string();
  const OptimSequence &seq = ctx.GetSequence(seq_name);
  const TensorType elem_dtype = seq.ElemDtype();

  // Resolve attributes. ``axis`` is required; ``new_axis`` defaults to 0.
  EXT_ENFORCE_INVALID(FindAttribute(node, "axis") != nullptr,
                      "ComputeShapeConcatFromSequence: required attribute 'axis' is missing.");
  const int64_t axis_attr = GetAttributeOr<int64_t>(node, "axis", 0);
  const int64_t new_axis = GetAttributeOr<int64_t>(node, "new_axis", 0);
  EXT_ENFORCE_INVALID(new_axis == 0 || new_axis == 1,
                      "ComputeShapeConcatFromSequence: new_axis must be either 0 or 1, got " +
                          std::to_string(new_axis) + ".");

  // Without per-element shapes we can only forward the element dtype.
  if (!seq.HasElemShapes() || seq.ElemShapes().empty()) {
    ctx.Set(node.output(0), OptimTensor(nullptr, elem_dtype, OptimShape{}));
    return;
  }

  const std::vector<OptimShape> &elem_shapes = seq.ElemShapes();
  const std::size_t n = elem_shapes.size();
  const int rank = static_cast<int>(elem_shapes[0].Rank());

  for (std::size_t i = 1; i < n; ++i) {
    EXT_ENFORCE_INVALID(static_cast<int>(elem_shapes[i].Rank()) == rank,
                        "ComputeShapeConcatFromSequence: element " + std::to_string(i) +
                            " has rank " + std::to_string(elem_shapes[i].Rank()) +
                            " which differs from rank " + std::to_string(rank) + " of element 0.");
  }

  const int upper_bound = (new_axis == 1) ? rank : rank - 1;
  const int lower_bound = (new_axis == 1) ? -rank - 1 : -rank;
  EXT_ENFORCE_INVALID(axis_attr >= lower_bound && axis_attr <= upper_bound,
                      "ComputeShapeConcatFromSequence: axis " + std::to_string(axis_attr) +
                          " is out of range [" + std::to_string(lower_bound) + ", " +
                          std::to_string(upper_bound) + "] for rank " + std::to_string(rank) +
                          " (new_axis=" + std::to_string(new_axis) + ").");

  const int resolved_axis =
      static_cast<int>(axis_attr < 0 ? axis_attr + upper_bound + 1 : axis_attr);

  OptimShape out_shape;
  if (new_axis == 1) {
    // Stack along a new axis at position ``resolved_axis``. The new
    // dimension is the sequence length (n); every other dimension is
    // merged across elements.
    OptimShape merged = elem_shapes[0];
    for (std::size_t i = 1; i < n; ++i) {
      for (std::size_t d = 0; d < static_cast<std::size_t>(rank); ++d) {
        MergeDim(merged[d], elem_shapes[i][d], d, i);
      }
    }
    for (int i = 0; i <= upper_bound; ++i) {
      if (i == resolved_axis) {
        out_shape.PushBack(OptimDim(static_cast<int64_t>(n)));
      } else {
        // Index into ``merged`` skipping over the new axis.
        const std::size_t src = static_cast<std::size_t>(i > resolved_axis ? i - 1 : i);
        out_shape.PushBack(merged[src]);
      }
    }
  } else {
    // Concatenate along ``resolved_axis``. The axis dimension is the
    // sum of per-element dimensions along ``axis`` when all are
    // concrete; otherwise a fresh symbolic name. Every other dimension
    // is merged across elements.
    OptimShape merged = elem_shapes[0];
    bool axis_dim_known = merged[resolved_axis].IsInt();
    int64_t axis_dim_total = axis_dim_known ? merged[resolved_axis].AsInt() : 0;

    for (std::size_t i = 1; i < n; ++i) {
      const OptimShape &shape = elem_shapes[i];
      for (std::size_t d = 0; d < static_cast<std::size_t>(rank); ++d) {
        if (static_cast<int>(d) == resolved_axis) {
          if (axis_dim_known && shape[d].IsInt()) {
            axis_dim_total += shape[d].AsInt();
          } else {
            axis_dim_known = false;
          }
        } else {
          MergeDim(merged[d], shape[d], d, i);
        }
      }
    }

    if (axis_dim_known) {
      merged[resolved_axis] = OptimDim(axis_dim_total);
    } else {
      // Disambiguate the synthetic symbolic dim by output name so multiple
      // ConcatFromSequence nodes in the same graph do not collide.
      merged[resolved_axis] = OptimDim("ConcatFromSequence_" + node.output(0).as_string() +
                                       "_axis" + std::to_string(resolved_axis));
    }
    out_shape = std::move(merged);
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, elem_dtype, std::move(out_shape)));
}

} // namespace sequence
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
