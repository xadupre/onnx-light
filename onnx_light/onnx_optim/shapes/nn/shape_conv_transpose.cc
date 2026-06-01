// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

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
namespace nn {

namespace {

OptimDim ComputeConvTransposeSpatialDim(const OptimDim &in_dim, int64_t kernel, int64_t stride,
                                        int64_t pad_begin, int64_t pad_end, int64_t dilation,
                                        int64_t output_padding, const std::string &x_name,
                                        size_t spatial_axis) {
  const std::string symbolic =
      std::string("ConvTranspose.") + x_name + ":" + std::to_string(spatial_axis);
  if (!in_dim.IsInt() || kernel <= 0 || stride <= 0) {
    return OptimDim(symbolic);
  }
  const int64_t iD = in_dim.AsInt();
  const int64_t eff_k = dilation * (kernel - 1) + 1;
  const int64_t out = stride * (iD - 1) + output_padding + eff_k - pad_begin - pad_end;
  if (out < 0) {
    return OptimDim(symbolic);
  }
  return OptimDim(out);
}

} // namespace

void ComputeShapeConvTranspose(ShapesContext &ctx, const NodeProto &node, const char *x,
                               const char *w) {
  CheckNodeOpAndOutput(node, "ConvTranspose", "ComputeShapeConvTranspose");

  const OptimTensor &x_tensor = ctx.Get(x);
  const OptimTensor &w_tensor = ctx.Get(w);
  const OptimShape &x_shape = x_tensor.Shape();
  const OptimShape &w_shape = w_tensor.Shape();

  if (x_shape.Rank() < 3) {
    throw std::invalid_argument("ComputeShapeConvTranspose: input '" + std::string(x) +
                                "' must have rank >= 3 (N, C, D1, ...).");
  }
  if (w_shape.Rank() != x_shape.Rank()) {
    throw std::invalid_argument("ComputeShapeConvTranspose: weight '" + std::string(w) +
                                "' rank must match input rank.");
  }

  const size_t n_spatial = x_shape.Rank() - 2;
  const int64_t group = GetAttributeOr<int64_t>(node, "group", 1);
  const std::string auto_pad = GetAttributeOr<std::string>(node, "auto_pad", "NOTSET");

  std::vector<int64_t> kernel_shape;
  GetAttributeInts(node, "kernel_shape", kernel_shape);
  if (kernel_shape.empty()) {
    kernel_shape.reserve(n_spatial);
    for (size_t i = 0; i < n_spatial; ++i) {
      const OptimDim &kd = w_shape[i + 2];
      kernel_shape.push_back(kd.IsInt() ? kd.AsInt() : -1);
    }
  } else if (kernel_shape.size() != n_spatial) {
    throw std::invalid_argument(
        "ComputeShapeConvTranspose: 'kernel_shape' size does not match input spatial rank.");
  }

  std::vector<int64_t> strides;
  GetAttributeInts(node, "strides", strides);
  if (strides.empty()) {
    strides.assign(n_spatial, 1);
  } else if (strides.size() != n_spatial) {
    throw std::invalid_argument(
        "ComputeShapeConvTranspose: 'strides' size does not match input spatial rank.");
  }

  std::vector<int64_t> dilations;
  GetAttributeInts(node, "dilations", dilations);
  if (dilations.empty()) {
    dilations.assign(n_spatial, 1);
  } else if (dilations.size() != n_spatial) {
    throw std::invalid_argument(
        "ComputeShapeConvTranspose: 'dilations' size does not match input spatial rank.");
  }

  std::vector<int64_t> output_padding;
  GetAttributeInts(node, "output_padding", output_padding);
  if (output_padding.empty()) {
    output_padding.assign(n_spatial, 0);
  } else if (output_padding.size() != n_spatial) {
    throw std::invalid_argument(
        "ComputeShapeConvTranspose: 'output_padding' size does not match input spatial rank.");
  }

  std::vector<int64_t> pads;
  GetAttributeInts(node, "pads", pads);
  if (pads.empty()) {
    pads.assign(n_spatial * 2, 0);
  } else if (pads.size() != n_spatial * 2) {
    throw std::invalid_argument("ComputeShapeConvTranspose: 'pads' size must be 2 * spatial rank.");
  }

  std::vector<int64_t> output_shape_attr;
  GetAttributeInts(node, "output_shape", output_shape_attr);
  const bool has_output_shape = !output_shape_attr.empty();
  if (has_output_shape && output_shape_attr.size() != n_spatial) {
    throw std::invalid_argument(
        "ComputeShapeConvTranspose: 'output_shape' size does not match input spatial rank.");
  }

  // Output channel dimension: W.shape[1] * group.
  OptimDim out_channels;
  if (w_shape[1].IsInt()) {
    out_channels = OptimDim(w_shape[1].AsInt() * group);
  } else {
    out_channels = OptimDim(std::string("ConvTranspose.") + w + ":1");
  }

  OptimShape out_shape;
  out_shape.PushBack(x_shape[0]);
  out_shape.PushBack(out_channels);

  for (size_t i = 0; i < n_spatial; ++i) {
    if (has_output_shape) {
      out_shape.PushBack(OptimDim(output_shape_attr[i]));
      continue;
    }
    if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
      // ``output_spatial = iD * stride`` per upstream default when output_shape
      // is not provided and auto_pad is SAME_*.
      if (x_shape[i + 2].IsInt()) {
        out_shape.PushBack(OptimDim(x_shape[i + 2].AsInt() * strides[i]));
      } else {
        out_shape.PushBack(OptimDim(std::string("ConvTranspose.") + x + ":" + std::to_string(i)));
      }
      continue;
    }
    if (kernel_shape[i] <= 0) {
      out_shape.PushBack(OptimDim(std::string("ConvTranspose.") + x + ":" + std::to_string(i)));
      continue;
    }
    out_shape.PushBack(ComputeConvTransposeSpatialDim(x_shape[i + 2], kernel_shape[i], strides[i],
                                                      pads[i], pads[i + n_spatial], dilations[i],
                                                      output_padding[i], x, i));
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, x_tensor.Dtype(), std::move(out_shape)));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
