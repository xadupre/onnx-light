// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

void ComputeShapeCol2Im(ShapesContext &ctx, const NodeProto &node, const char *input,
                        const char *image_shape, const char *block_shape) {
  CheckNodeOpAndOutput(node, "Col2Im", "ComputeShapeCol2Im");

  const OptimTensor &input_tensor = ctx.Get(input);
  const OptimTensor &image_shape_tensor = ctx.Get(image_shape);
  const OptimTensor &block_shape_tensor = ctx.Get(block_shape);
  const OptimShape &input_shape = input_tensor.Shape();

  if (input_shape.Rank() != 3) {
    throw std::invalid_argument("ComputeShapeCol2Im: input '" + std::string(input) +
                                "' must have rank 3 (N, C * product(block_shape), L).");
  }

  // Number of spatial dims: prefer the ``image_shape`` initializer (its
  // ``ValueAsShape`` annotation), otherwise the single static dim of its
  // 1-D shape, otherwise the ``block_shape`` counterpart.
  int64_t n_spatial = -1;
  if (image_shape_tensor.HasValueAsShape()) {
    n_spatial = static_cast<int64_t>(image_shape_tensor.ValueAsShape().Rank());
  } else if (image_shape_tensor.Shape().Rank() == 1 && image_shape_tensor.Shape()[0].IsInt()) {
    n_spatial = image_shape_tensor.Shape()[0].AsInt();
  } else if (block_shape_tensor.HasValueAsShape()) {
    n_spatial = static_cast<int64_t>(block_shape_tensor.ValueAsShape().Rank());
  } else if (block_shape_tensor.Shape().Rank() == 1 && block_shape_tensor.Shape()[0].IsInt()) {
    n_spatial = block_shape_tensor.Shape()[0].AsInt();
  }

  // ``C`` can only be computed when the product of the block shape is known.
  int64_t block_product = -1;
  if (block_shape_tensor.HasValueAsShape()) {
    const OptimShape &bs = block_shape_tensor.ValueAsShape();
    block_product = 1;
    for (size_t i = 0; i < bs.Rank(); ++i) {
      if (!bs[i].IsInt()) {
        block_product = -1;
        break;
      }
      block_product *= bs[i].AsInt();
    }
  }

  OptimShape out_shape;
  // Dim 0: N (batch).
  out_shape.PushBack(input_shape[0]);

  // Dim 1: C = input.shape[1] / product(block_shape).
  if (block_product > 0 && input_shape[1].IsInt()) {
    out_shape.PushBack(OptimDim(input_shape[1].AsInt() / block_product));
  } else {
    out_shape.PushBack(OptimDim(std::string("Col2Im.") + input + ":1"));
  }

  // Spatial dims: copy from ``image_shape`` initializer when known,
  // otherwise emit symbolic dims.
  const OptimShape *image_shape_values =
      image_shape_tensor.HasValueAsShape() ? &image_shape_tensor.ValueAsShape() : nullptr;
  if (n_spatial < 0) {
    // Number of spatial dims unknown — append a single symbolic placeholder.
    out_shape.PushBack(OptimDim(std::string("Col2Im.") + input + ":2"));
  } else {
    for (int64_t i = 0; i < n_spatial; ++i) {
      if (image_shape_values != nullptr && static_cast<size_t>(i) < image_shape_values->Rank() &&
          (*image_shape_values)[static_cast<size_t>(i)].IsInt()) {
        out_shape.PushBack((*image_shape_values)[static_cast<size_t>(i)]);
      } else {
        out_shape.PushBack(OptimDim(std::string("Col2Im.") + input + ":" + std::to_string(2 + i)));
      }
    }
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, input_tensor.Dtype(), std::move(out_shape)));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
