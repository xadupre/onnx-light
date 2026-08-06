// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_extensions/kernels/kernels/auto_pad.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

using onnx_kernels::kernel::AutoPad;
using onnx_kernels::kernel::AutoPadFromString;
using onnx_kernels::kernel::AutoPadToString;

namespace {

// Computes the size of the output along a single spatial axis according to
// the ONNX ``LpPool`` formula with explicit padding. Mirrors
// :cpp:func:`onnx_kernels::kernel::LpPool::OutputDim`.
int64_t OutputDim(int64_t in_dim, int64_t kernel, int64_t stride, int64_t pad_begin,
                  int64_t pad_end, bool ceil_mode, int64_t dilation = 1) {
  const int64_t eff_kernel = dilation * (kernel - 1) + 1;
  const double numerator =
      static_cast<double>(in_dim + pad_begin + pad_end - eff_kernel) / static_cast<double>(stride);
  const double v = ceil_mode ? std::ceil(numerator) : std::floor(numerator);
  int64_t out = static_cast<int64_t>(v) + 1;
  if (ceil_mode && out > 0) {
    const int64_t last_start = (out - 1) * stride - pad_begin;
    if (last_start >= in_dim) {
      --out;
    }
  }
  return out;
}

// Resolves a single spatial axis under ``auto_pad`` other than NOTSET.
int64_t AutoPadOutputDim(AutoPad auto_pad, int64_t in_dim, int64_t kernel, int64_t stride,
                         int64_t dilation) {
  const int64_t eff_kernel = dilation * (kernel - 1) + 1;
  if (auto_pad == AutoPad::kValid) {
    const double numerator = static_cast<double>(in_dim - eff_kernel) / static_cast<double>(stride);
    return static_cast<int64_t>(std::floor(numerator)) + 1;
  }
  int64_t out_dim =
      static_cast<int64_t>(std::ceil(static_cast<double>(in_dim) / static_cast<double>(stride)));
  return out_dim < 0 ? 0 : out_dim;
}

} // namespace

void ComputeShapeLpPool(ShapesContext &ctx, const NodeProto &node, const char *x) {
  CheckNodeOpAndOutput(node, "LpPool", "ComputeShapeLpPool");

  const SymTensor &input = ctx.Get(x);
  const SymShape &in_shape = input.Shape();
  EXT_ENFORCE_INVALID(in_shape.Rank() >= 2,
                      "ComputeShapeLpPool: input must have rank >= 2 (N, C, D1, ...).");
  const size_t n_input_dims = in_shape.Rank() - 2;

  std::vector<int64_t> kernel_shape;
  EXT_ENFORCE_INVALID(GetAttributeInts(node, "kernel_shape", kernel_shape),
                      "ComputeShapeLpPool: required attribute 'kernel_shape' is missing.");
  EXT_ENFORCE_INVALID(
      kernel_shape.size() == n_input_dims,
      "ComputeShapeLpPool: attribute 'kernel_shape' size must match input rank - 2.");

  std::vector<int64_t> strides;
  if (GetAttributeInts(node, "strides", strides)) {
    EXT_ENFORCE_INVALID(strides.size() == n_input_dims,
                        "ComputeShapeLpPool: attribute 'strides' size must match input rank - 2.");
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> dilations;
  if (GetAttributeInts(node, "dilations", dilations)) {
    EXT_ENFORCE_INVALID(
        dilations.size() == n_input_dims,
        "ComputeShapeLpPool: attribute 'dilations' size must match input rank - 2.");
  } else {
    dilations.assign(n_input_dims, 1);
  }

  std::vector<int64_t> pads;
  if (GetAttributeInts(node, "pads", pads)) {
    EXT_ENFORCE_INVALID(pads.size() == 2 * n_input_dims,
                        "ComputeShapeLpPool: attribute 'pads' size must be 2 * (input rank - 2).");
  } else {
    pads.assign(2 * n_input_dims, 0);
  }

  const AutoPad auto_pad =
      AutoPadFromString(GetAttributeOr<std::string>(node, "auto_pad", std::string("NOTSET")));
  EXT_ENFORCE_INVALID(auto_pad == AutoPad::kNotSet || auto_pad == AutoPad::kValid ||
                          auto_pad == AutoPad::kSameUpper || auto_pad == AutoPad::kSameLower,
                      "ComputeShapeLpPool: auto_pad='",
                      GetAttributeOr<std::string>(node, "auto_pad", std::string("NOTSET")),
                      "' is not supported; must be one of NOTSET, SAME_UPPER, SAME_LOWER "
                      "or VALID.");
  const bool use_auto_pad = auto_pad != AutoPad::kNotSet;

  const bool ceil_mode = GetAttributeOr<int64_t>(node, "ceil_mode", 0) != 0;

  SymShape out_shape;
  out_shape.PushBack(in_shape[0]);
  out_shape.PushBack(in_shape[1]);
  for (size_t i = 0; i < n_input_dims; ++i) {
    const SymDim &d = in_shape[i + 2];
    if (d.IsInt()) {
      int64_t out_d;
      if (use_auto_pad) {
        out_d = AutoPadOutputDim(auto_pad, d.AsInt(), kernel_shape[i], strides[i], dilations[i]);
      } else {
        out_d = OutputDim(d.AsInt(), kernel_shape[i], strides[i], pads[i], pads[i + n_input_dims],
                          ceil_mode, dilations[i]);
      }
      out_shape.PushBack(SymDim(out_d));
    } else {
      // Symbolic spatial dimension: propagate as a fresh symbolic expression.
      std::string expr = "LpPool(" + d.AsExpr() + ",k=" + std::to_string(kernel_shape[i]) +
                         ",s=" + std::to_string(strides[i]) + ",d=" + std::to_string(dilations[i]);
      if (use_auto_pad) {
        expr += std::string(",auto_pad=") + AutoPadToString(auto_pad);
      } else {
        expr += ",p=" + std::to_string(pads[i]) + "+" + std::to_string(pads[i + n_input_dims]) +
                ",ceil=" + (ceil_mode ? "1" : "0");
      }
      expr += ")";
      out_shape.PushBack(SymDim(expr));
    }
  }

  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
