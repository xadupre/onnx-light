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

// Extracts the per-axis pad values from a known ``pads`` initializer (length
// ``2 * num_axes``) into ``out_begin``/``out_end`` (indexed by data axis).
// ``axes`` is the resolved list of data axes that ``pads`` applies to.
// Returns false when any value is unknown.
bool FillPads(const std::vector<int64_t> &pads_values, const std::vector<int64_t> &axes,
              std::vector<int64_t> &out_begin, std::vector<int64_t> &out_end,
              std::vector<bool> &has_pad) {
  const std::size_t num_axes = axes.size();
  if (pads_values.size() != 2 * num_axes) {
    throw std::invalid_argument("ComputeShapePad: 'pads' length (" +
                                std::to_string(pads_values.size()) + ") must equal 2 * num_axes (" +
                                std::to_string(2 * num_axes) + ").");
  }
  for (std::size_t i = 0; i < num_axes; ++i) {
    const int64_t axis = axes[i];
    out_begin[static_cast<std::size_t>(axis)] = pads_values[i];
    out_end[static_cast<std::size_t>(axis)] = pads_values[i + num_axes];
    has_pad[static_cast<std::size_t>(axis)] = true;
  }
  return true;
}

// Resolves ``axes`` (possibly negative) using the input rank; throws on out
// of range values. ``axes_values`` must already be populated.
std::vector<int64_t> NormalizeAxes(const std::vector<int64_t> &axes_values, int64_t rank) {
  std::vector<int64_t> out;
  out.reserve(axes_values.size());
  for (int64_t a : axes_values) {
    int64_t normalized = a < 0 ? a + rank : a;
    if (normalized < 0 || normalized >= rank) {
      throw std::invalid_argument("ComputeShapePad: axis " + std::to_string(a) +
                                  " is out of range for input rank " + std::to_string(rank) + ".");
    }
    out.push_back(normalized);
  }
  return out;
}

} // namespace

void ComputeShapePad(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Pad", "ComputeShapePad");

  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapePad: Pad requires at least one input.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const TensorType dtype = input.Dtype();
  const OptimShape &in_shape = input.Shape();
  const std::size_t rank = in_shape.Rank();

  // Resolve pads (per data axis). pad_begin / pad_end default to 0; has_pad
  // tracks whether the value is known (true) or unknown (false). For an
  // unknown pad on a particular axis the output dim becomes symbolic.
  std::vector<int64_t> pad_begin(rank, 0);
  std::vector<int64_t> pad_end(rank, 0);
  std::vector<bool> has_pad(rank, false);
  // When all_pads_known is true and all input dims are known, we can fold
  // input_dim + pad_begin + pad_end into a concrete output dim.
  bool all_pads_known = false;

  if (node.input_size() == 1) {
    // v1/v2: pads come from the ``paddings`` (v1) or ``pads`` (v2) INTS attribute
    // and apply to every input axis.
    std::vector<int64_t> attr_pads;
    if (!GetAttributeInts(node, "pads", attr_pads)) {
      GetAttributeInts(node, "paddings", attr_pads);
    }
    if (!attr_pads.empty()) {
      std::vector<int64_t> axes(rank);
      for (std::size_t i = 0; i < rank; ++i) {
        axes[i] = static_cast<int64_t>(i);
      }
      FillPads(attr_pads, axes, pad_begin, pad_end, has_pad);
      all_pads_known = true;
    }
  } else {
    // v11+: pads are input(1); v18+ adds optional axes input(3).
    const OptimTensor &pads_input = ctx.Get(node.input(1).as_string());
    std::vector<int64_t> axes_values;
    bool axes_known = true;
    if (node.input_size() >= 4 && !node.input(3).as_string().empty()) {
      const OptimTensor &axes_input = ctx.Get(node.input(3).as_string());
      if (axes_input.HasValueAsShape()) {
        const OptimShape &axes_shape = axes_input.ValueAsShape();
        for (std::size_t i = 0; i < axes_shape.Rank(); ++i) {
          const OptimDim &d = axes_shape[i];
          if (!d.IsInt()) {
            axes_known = false;
            break;
          }
          axes_values.push_back(d.AsInt());
        }
      } else {
        axes_known = false;
      }
    } else {
      axes_values.resize(rank);
      for (std::size_t i = 0; i < rank; ++i) {
        axes_values[i] = static_cast<int64_t>(i);
      }
    }

    if (axes_known && pads_input.HasValueAsShape()) {
      const OptimShape &pads_shape = pads_input.ValueAsShape();
      std::vector<int64_t> pads_values;
      bool pads_all_int = true;
      for (std::size_t i = 0; i < pads_shape.Rank(); ++i) {
        const OptimDim &d = pads_shape[i];
        if (!d.IsInt()) {
          pads_all_int = false;
          break;
        }
        pads_values.push_back(d.AsInt());
      }
      if (pads_all_int) {
        const std::vector<int64_t> norm_axes =
            NormalizeAxes(axes_values, static_cast<int64_t>(rank));
        FillPads(pads_values, norm_axes, pad_begin, pad_end, has_pad);
        all_pads_known = true;
      }
    }
  }

  OptimShape out_shape;
  for (std::size_t i = 0; i < rank; ++i) {
    const OptimDim &in_dim = in_shape[i];
    if (all_pads_known) {
      if (in_dim.IsInt()) {
        out_shape.PushBack(OptimDim(in_dim.AsInt() + pad_begin[i] + pad_end[i]));
      } else if (has_pad[i] && pad_begin[i] == 0 && pad_end[i] == 0) {
        out_shape.PushBack(in_dim);
      } else {
        out_shape.PushBack(OptimDim("Pad_dim" + std::to_string(i)));
      }
    } else {
      out_shape.PushBack(OptimDim("Pad_dim" + std::to_string(i)));
    }
  }
  ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
