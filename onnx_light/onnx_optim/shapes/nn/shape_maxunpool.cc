// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

void ComputeShapeMaxUnpool(ShapesContext &ctx, const NodeProto &node, const char *x, const char *I,
                           const char *output_shape) {
  CheckNodeOpAndOutput(node, "MaxUnpool", "ComputeShapeMaxUnpool");
  (void)I;

  const OptimTensor &input = ctx.Get(x);
  const OptimShape &in_shape = input.Shape();
  EXT_ENFORCE_INVALID(in_shape.Rank() >= 2,
                      "ComputeShapeMaxUnpool: input must have rank >= 2 (N, C, D1, ...).");
  const size_t n_input_dims = in_shape.Rank() - 2;

  std::vector<int64_t> kernel_shape;
  EXT_ENFORCE_INVALID(GetAttributeInts(node, "kernel_shape", kernel_shape),
                      "ComputeShapeMaxUnpool: required attribute 'kernel_shape' is missing.");
  EXT_ENFORCE_INVALID(
      kernel_shape.size() == n_input_dims,
      "ComputeShapeMaxUnpool: attribute 'kernel_shape' size must match input rank - 2.");

  std::vector<int64_t> strides;
  if (GetAttributeInts(node, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      throw std::invalid_argument(
          "ComputeShapeMaxUnpool: attribute 'strides' size must match input rank - 2.");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> pads;
  if (GetAttributeInts(node, "pads", pads)) {
    if (pads.size() != 2 * n_input_dims) {
      throw std::invalid_argument(
          "ComputeShapeMaxUnpool: attribute 'pads' size must be 2 * (input rank - 2).");
    }
  } else {
    pads.assign(2 * n_input_dims, 0);
  }

  // When ``output_shape`` is provided as a known initializer, take its
  // values directly; this overrides any computed shape (per the ONNX spec).
  const OptimShape *explicit_out_shape = nullptr;
  if (output_shape != nullptr) {
    const OptimTensor &out_shape_tensor = ctx.Get(output_shape);
    if (out_shape_tensor.HasValueAsShape()) {
      explicit_out_shape = &out_shape_tensor.ValueAsShape();
    }
  }

  OptimShape out_shape;
  out_shape.PushBack(in_shape[0]);
  out_shape.PushBack(in_shape[1]);
  for (size_t i = 0; i < n_input_dims; ++i) {
    if (explicit_out_shape != nullptr && static_cast<size_t>(i + 2) < explicit_out_shape->Rank()) {
      out_shape.PushBack((*explicit_out_shape)[i + 2]);
      continue;
    }
    const OptimDim &d = in_shape[i + 2];
    if (d.IsInt()) {
      const int64_t out_d = strides[i] * (d.AsInt() - 1) + kernel_shape[i] - pads[i] -
                            pads[i + n_input_dims];
      out_shape.PushBack(OptimDim(out_d));
    } else {
      std::string expr = "MaxUnpool(" + d.AsExpr() + ",k=" + std::to_string(kernel_shape[i]) +
                         ",s=" + std::to_string(strides[i]) + ",p=" + std::to_string(pads[i]) +
                         "+" + std::to_string(pads[i + n_input_dims]) + ")";
      out_shape.PushBack(OptimDim(expr));
    }
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
