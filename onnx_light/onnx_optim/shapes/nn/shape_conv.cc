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

// Computes one spatial output dimension following the upstream
// ``convPoolShapeInference`` rule for Conv-like ops, honoring ``auto_pad``.
// Returns a symbolic ``"<op>.<x>:<axis>"`` when the input dim is symbolic
// or the formula cannot be evaluated.
OptimDim ComputeConvSpatialDim(const OptimDim &in_dim, int64_t kernel, int64_t stride,
                               int64_t pad_begin, int64_t pad_end, int64_t dilation,
                               const std::string &auto_pad, const std::string &op_name,
                               const std::string &x_name, size_t spatial_axis) {
  const std::string symbolic = op_name + "." + x_name + ":" + std::to_string(spatial_axis);
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

// Shared implementation for ``Conv`` and ``ConvInteger``. The output dtype
// is provided by the caller (``X.dtype`` for Conv, ``int32`` for ConvInteger).
void ComputeShapeConvLike(ShapesContext &ctx, const NodeProto &node, const char *x, const char *w,
                          const char *op_name, TensorType out_dtype) {
  const OptimTensor &x_tensor = ctx.Get(x);
  const OptimTensor &w_tensor = ctx.Get(w);
  const OptimShape &x_shape = x_tensor.Shape();
  const OptimShape &w_shape = w_tensor.Shape();

  if (x_shape.Rank() < 3) {
    throw std::invalid_argument(std::string("ComputeShape") + op_name + ": input '" +
                                std::string(x) + "' must have rank >= 3 (N, C, D1, ...).");
  }
  if (w_shape.Rank() != x_shape.Rank()) {
    throw std::invalid_argument(std::string("ComputeShape") + op_name + ": weight '" +
                                std::string(w) + "' rank must match input rank.");
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
    throw std::invalid_argument(std::string("ComputeShape") + op_name +
                                ": 'kernel_shape' size does not match input spatial rank.");
  }

  std::vector<int64_t> strides;
  GetAttributeInts(node, "strides", strides);
  if (strides.empty()) {
    strides.assign(n_spatial, 1);
  } else if (strides.size() != n_spatial) {
    throw std::invalid_argument(std::string("ComputeShape") + op_name +
                                ": 'strides' size does not match input spatial rank.");
  }

  std::vector<int64_t> dilations;
  GetAttributeInts(node, "dilations", dilations);
  if (dilations.empty()) {
    dilations.assign(n_spatial, 1);
  } else if (dilations.size() != n_spatial) {
    throw std::invalid_argument(std::string("ComputeShape") + op_name +
                                ": 'dilations' size does not match input spatial rank.");
  }

  std::vector<int64_t> pads;
  GetAttributeInts(node, "pads", pads);
  if (pads.empty()) {
    pads.assign(n_spatial * 2, 0);
  } else if (pads.size() != n_spatial * 2) {
    throw std::invalid_argument(std::string("ComputeShape") + op_name +
                                ": 'pads' size must be 2 * spatial rank.");
  }

  OptimShape out_shape;
  out_shape.PushBack(x_shape[0]); // N
  out_shape.PushBack(w_shape[0]); // M
  for (size_t i = 0; i < n_spatial; ++i) {
    if (kernel_shape[i] <= 0) {
      out_shape.PushBack(OptimDim(std::string(op_name) + "." + x + ":" + std::to_string(i)));
      continue;
    }
    out_shape.PushBack(ComputeConvSpatialDim(x_shape[i + 2], kernel_shape[i], strides[i], pads[i],
                                             pads[i + n_spatial], dilations[i], auto_pad, op_name,
                                             x, i));
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, out_dtype, std::move(out_shape)));
}

} // namespace

void ComputeShapeConv(ShapesContext &ctx, const NodeProto &node, const char *x, const char *w) {
  CheckNodeOpAndOutput(node, "Conv", "ComputeShapeConv");
  ComputeShapeConvLike(ctx, node, x, w, "Conv", ctx.Get(x).Dtype());
}

void ComputeShapeConvInteger(ShapesContext &ctx, const NodeProto &node, const char *x,
                             const char *w) {
  CheckNodeOpAndOutput(node, "ConvInteger", "ComputeShapeConvInteger");
  ComputeShapeConvLike(ctx, node, x, w, "ConvInteger", TensorType::kInt32);
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
