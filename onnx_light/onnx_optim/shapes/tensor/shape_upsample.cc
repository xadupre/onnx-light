// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cmath>
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

OptimDim ScaleDim(const OptimDim &dim, float scale, std::size_t axis) {
  if (dim.IsInt()) {
    const double scaled = static_cast<double>(dim.AsInt()) * static_cast<double>(scale);
    return OptimDim(static_cast<int64_t>(std::floor(scaled)));
  }
  return OptimDim("Upsample_dim" + std::to_string(axis));
}

} // namespace

void ComputeShapeUpsample(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Upsample", "ComputeShapeUpsample");
  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapeUpsample: Upsample requires one input.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const OptimShape &input_shape = input.Shape();
  const std::size_t rank = input_shape.Rank();

  // Recover the per-axis float scales when statically known. Three
  // schema variants exist:
  //
  //   * v1: ``width_scale`` and ``height_scale`` FLOAT attributes on a
  //     4-D NCHW input (scales for N/C are implicitly 1).
  //   * v7: a ``scales`` FLOATS attribute with one entry per input axis.
  //   * v9/v10: a ``scales`` input tensor (1-D FLOAT). Its values are not
  //     tracked by the data-propagation lattice (which only carries int
  //     shape values), so we leave every output dim symbolic in that case.
  std::vector<float> scales;
  bool scales_known = false;

  const AttributeProto *scales_attr = FindAttribute(node, "scales");
  if (scales_attr != nullptr) {
    for (float s : scales_attr->ref_floats()) {
      scales.push_back(s);
    }
    scales_known = true;
  } else {
    const AttributeProto *width_attr = FindAttribute(node, "width_scale");
    const AttributeProto *height_attr = FindAttribute(node, "height_scale");
    if (width_attr != nullptr && height_attr != nullptr) {
      // v1 layout: NCHW with implicit scale 1 on N and C.
      scales.assign(rank, 1.0f);
      if (rank == 4) {
        scales[2] = height_attr->ref_f();
        scales[3] = width_attr->ref_f();
        scales_known = true;
      }
    }
  }

  if (scales_known && scales.size() != rank) {
    throw std::invalid_argument(
        "ComputeShapeUpsample: 'scales' length (" + std::to_string(scales.size()) +
        ") must equal the rank of input 'X' (" + std::to_string(rank) + ").");
  }

  OptimShape out_shape;
  for (std::size_t i = 0; i < rank; ++i) {
    if (scales_known) {
      out_shape.PushBack(ScaleDim(input_shape[i], scales[i], i));
    } else {
      out_shape.PushBack(OptimDim("Upsample_dim" + std::to_string(i)));
    }
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
