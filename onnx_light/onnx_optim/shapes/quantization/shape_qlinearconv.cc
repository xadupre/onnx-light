// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/quantization/shape_quantization.h"

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
namespace quantization {

namespace {

OptimDim ComputeQLinearConvSpatialDim(const OptimDim &in_dim, int64_t kernel, int64_t stride,
                                      int64_t pad_begin, int64_t pad_end, int64_t dilation,
                                      const std::string &auto_pad, const std::string &x_name,
                                      size_t spatial_axis) {
  const std::string symbolic =
      std::string("QLinearConv.") + x_name + ":" + std::to_string(spatial_axis);
  if (!in_dim.IsInt()) {
    return OptimDim(symbolic);
  }
  const int64_t iD = in_dim.AsInt();
  const int64_t eff_k = dilation * (kernel - 1) + 1;
  if (stride <= 0 || kernel <= 0) {
    return OptimDim(symbolic);
  }
  if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
    return OptimDim((iD + stride - 1) / stride);
  }
  if (auto_pad == "VALID") {
    const int64_t numer = iD - eff_k;
    if (numer < 0) {
      return OptimDim(symbolic);
    }
    return OptimDim(numer / stride + 1);
  }
  const int64_t numer = iD + pad_begin + pad_end - eff_k;
  if (numer < 0) {
    return OptimDim(symbolic);
  }
  return OptimDim(numer / stride + 1);
}

} // namespace

void ComputeShapeQLinearConv(ShapesContext &ctx, const NodeProto &node, const char *x,
                             const char *w, const char *y_zero_point) {
  CheckNodeOpAndOutput(node, "QLinearConv", "ComputeShapeQLinearConv");

  const OptimTensor &x_tensor = ctx.Get(x);
  const OptimTensor &w_tensor = ctx.Get(w);
  const OptimTensor &yzp_tensor = ctx.Get(y_zero_point);
  const OptimShape &x_shape = x_tensor.Shape();
  const OptimShape &w_shape = w_tensor.Shape();

  if (x_shape.Rank() < 3) {
    throw std::invalid_argument("ComputeShapeQLinearConv: input '" + std::string(x) +
                                "' must have rank >= 3 (N, C, D1, ...).");
  }
  if (w_shape.Rank() != x_shape.Rank()) {
    throw std::invalid_argument("ComputeShapeQLinearConv: weight '" + std::string(w) +
                                "' rank must match input rank.");
  }

  const size_t n_spatial = x_shape.Rank() - 2;
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
        "ComputeShapeQLinearConv: 'kernel_shape' size does not match input spatial rank.");
  }

  std::vector<int64_t> strides;
  GetAttributeInts(node, "strides", strides);
  if (strides.empty()) {
    strides.assign(n_spatial, 1);
  } else if (strides.size() != n_spatial) {
    throw std::invalid_argument(
        "ComputeShapeQLinearConv: 'strides' size does not match input spatial rank.");
  }

  std::vector<int64_t> dilations;
  GetAttributeInts(node, "dilations", dilations);
  if (dilations.empty()) {
    dilations.assign(n_spatial, 1);
  } else if (dilations.size() != n_spatial) {
    throw std::invalid_argument(
        "ComputeShapeQLinearConv: 'dilations' size does not match input spatial rank.");
  }

  std::vector<int64_t> pads;
  GetAttributeInts(node, "pads", pads);
  if (pads.empty()) {
    pads.assign(n_spatial * 2, 0);
  } else if (pads.size() != n_spatial * 2) {
    throw std::invalid_argument("ComputeShapeQLinearConv: 'pads' size must be 2 * spatial rank.");
  }

  OptimShape out_shape;
  out_shape.PushBack(x_shape[0]); // N
  out_shape.PushBack(w_shape[0]); // M
  for (size_t i = 0; i < n_spatial; ++i) {
    if (kernel_shape[i] <= 0) {
      out_shape.PushBack(OptimDim(std::string("QLinearConv.") + x + ":" + std::to_string(i)));
      continue;
    }
    out_shape.PushBack(ComputeQLinearConvSpatialDim(x_shape[i + 2], kernel_shape[i], strides[i],
                                                    pads[i], pads[i + n_spatial], dilations[i],
                                                    auto_pad, x, i));
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, yzp_tensor.Dtype(), std::move(out_shape)));
}

} // namespace quantization
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
