// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

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

// Computes one spatial output dimension following the upstream
// ``convPoolShapeInference`` rule (floor mode): when the input dim and
// padding are known, returns
// ``floor((D + pad_begin + pad_end - dilation * (k - 1) - 1) / stride) + 1``.
// Otherwise propagates a symbolic dim labeled ``"DeformConv.<x>:<axis>"``.
OptimDim ComputeSpatialDim(const OptimDim &in_dim, int64_t kernel, int64_t stride,
                           int64_t pad_begin, int64_t pad_end, int64_t dilation,
                           const std::string &x_name, size_t spatial_axis) {
  if (in_dim.IsInt()) {
    const int64_t effective_kernel = dilation * (kernel - 1) + 1;
    const int64_t numer = in_dim.AsInt() + pad_begin + pad_end - effective_kernel;
    if (numer < 0 || stride <= 0) {
      // Match upstream behavior: invalid → leave symbolic.
      return OptimDim(std::string("DeformConv.") + x_name + ":" + std::to_string(spatial_axis));
    }
    return OptimDim(numer / stride + 1);
  }
  return OptimDim(std::string("DeformConv.") + x_name + ":" + std::to_string(spatial_axis));
}

} // namespace

void ComputeShapeDeformConv(ShapesContext &ctx, const NodeProto &node, const char *x,
                            const char *w) {
  CheckNodeOpAndOutput(node, "DeformConv", "ComputeShapeDeformConv");

  const OptimTensor &x_tensor = ctx.Get(x);
  const OptimTensor &w_tensor = ctx.Get(w);
  const OptimShape &x_shape = x_tensor.Shape();
  const OptimShape &w_shape = w_tensor.Shape();

  if (x_shape.Rank() < 3) {
    throw std::invalid_argument("ComputeShapeDeformConv: input '" + std::string(x) +
                                "' must have rank >= 3 (N, C, D1, ...).");
  }
  if (w_shape.Rank() != x_shape.Rank()) {
    throw std::invalid_argument("ComputeShapeDeformConv: weight '" + std::string(w) +
                                "' rank must match input rank.");
  }

  const size_t n_spatial = x_shape.Rank() - 2;

  // Resolve kernel_shape: prefer explicit attribute, fall back to W[2..].
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
        "ComputeShapeDeformConv: 'kernel_shape' attribute size does not match input spatial rank.");
  }

  std::vector<int64_t> strides;
  GetAttributeInts(node, "strides", strides);
  if (strides.empty()) {
    strides.assign(n_spatial, 1);
  } else if (strides.size() != n_spatial) {
    throw std::invalid_argument(
        "ComputeShapeDeformConv: 'strides' attribute size does not match input spatial rank.");
  }

  std::vector<int64_t> dilations;
  GetAttributeInts(node, "dilations", dilations);
  if (dilations.empty()) {
    dilations.assign(n_spatial, 1);
  } else if (dilations.size() != n_spatial) {
    throw std::invalid_argument(
        "ComputeShapeDeformConv: 'dilations' attribute size does not match input spatial rank.");
  }

  std::vector<int64_t> pads;
  GetAttributeInts(node, "pads", pads);
  if (pads.empty()) {
    pads.assign(n_spatial * 2, 0);
  } else if (pads.size() != n_spatial * 2) {
    throw std::invalid_argument(
        "ComputeShapeDeformConv: 'pads' attribute size must be 2 * spatial rank.");
  }

  OptimShape out_shape;
  out_shape.PushBack(x_shape[0]); // N
  out_shape.PushBack(w_shape[0]); // oC
  for (size_t i = 0; i < n_spatial; ++i) {
    if (kernel_shape[i] <= 0) {
      out_shape.PushBack(OptimDim(std::string("DeformConv.") + x + ":" + std::to_string(i)));
      continue;
    }
    out_shape.PushBack(ComputeSpatialDim(x_shape[i + 2], kernel_shape[i], strides[i], pads[i],
                                         pads[i + n_spatial], dilations[i], x, i));
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, x_tensor.Dtype(), std::move(out_shape)));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
