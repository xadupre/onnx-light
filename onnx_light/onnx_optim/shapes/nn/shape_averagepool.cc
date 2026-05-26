// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

namespace {

// Reads a repeated INTS attribute by name, returning ``true`` if found.
// On success, the values are appended to ``out``.
bool GetIntsAttribute(const NodeProto &node, const char *name, std::vector<int64_t> &out) {
  for (int i = 0; i < node.attribute_size(); ++i) {
    const AttributeProto &attr = node.attribute(i);
    if (attr.ref_name() == name) {
      for (int64_t v : attr.ref_ints()) {
        out.push_back(v);
      }
      return true;
    }
  }
  return false;
}

// Reads a scalar INT attribute by name. Returns ``default_value`` when absent.
int64_t GetIntAttribute(const NodeProto &node, const char *name, int64_t default_value) {
  for (int i = 0; i < node.attribute_size(); ++i) {
    const AttributeProto &attr = node.attribute(i);
    if (attr.ref_name() == name) {
      return attr.ref_i();
    }
  }
  return default_value;
}

// Reads a STRING attribute by name. Returns ``default_value`` when absent.
std::string GetStringAttribute(const NodeProto &node, const char *name, const char *default_value) {
  for (int i = 0; i < node.attribute_size(); ++i) {
    const AttributeProto &attr = node.attribute(i);
    if (attr.ref_name() == name) {
      return attr.ref_s().as_string();
    }
  }
  return std::string(default_value);
}

// Computes the size of the output along a single spatial axis according to
// the ONNX ``AveragePool`` formula. Mirrors
// ``onnx_backend_test::kernel::AveragePool::OutputDim`` and the spec rule
// "Sliding windows that would start in the right padded region are ignored".
int64_t OutputDim(int64_t in_dim, int64_t kernel, int64_t stride, int64_t pad_begin,
                  int64_t pad_end, bool ceil_mode) {
  const double numerator =
      static_cast<double>(in_dim + pad_begin + pad_end - kernel) / static_cast<double>(stride);
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

} // namespace

void ComputeShapeAveragePool(ShapesContext &ctx, const NodeProto &node, const char *x) {
  if (node.op_type() != "AveragePool") {
    throw std::invalid_argument("ComputeShapeAveragePool expects op_type='AveragePool', got '" +
                                node.op_type().as_string() + "'.");
  }
  if (node.output_size() < 1) {
    throw std::invalid_argument("ComputeShapeAveragePool: node has no output.");
  }

  const OptimTensor &input = ctx.Get(x);
  const OptimShape &in_shape = input.Shape();
  if (in_shape.Rank() < 2) {
    throw std::invalid_argument(
        "ComputeShapeAveragePool: input must have rank >= 2 (N, C, D1, ...).");
  }
  const size_t n_input_dims = in_shape.Rank() - 2;

  std::vector<int64_t> kernel_shape;
  if (!GetIntsAttribute(node, "kernel_shape", kernel_shape)) {
    throw std::invalid_argument(
        "ComputeShapeAveragePool: required attribute 'kernel_shape' is missing.");
  }
  if (kernel_shape.size() != n_input_dims) {
    throw std::invalid_argument(
        "ComputeShapeAveragePool: attribute 'kernel_shape' size must match input rank - 2.");
  }

  std::vector<int64_t> strides;
  if (GetIntsAttribute(node, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      throw std::invalid_argument(
          "ComputeShapeAveragePool: attribute 'strides' size must match input rank - 2.");
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> pads;
  if (GetIntsAttribute(node, "pads", pads)) {
    if (pads.size() != 2 * n_input_dims) {
      throw std::invalid_argument(
          "ComputeShapeAveragePool: attribute 'pads' size must be 2 * (input rank - 2).");
    }
  } else {
    pads.assign(2 * n_input_dims, 0);
  }

  const std::string auto_pad = GetStringAttribute(node, "auto_pad", "NOTSET");
  if (auto_pad != "NOTSET" && auto_pad != "VALID") {
    throw std::invalid_argument(
        "ComputeShapeAveragePool: auto_pad='" + auto_pad +
        "' is not supported; only NOTSET (with explicit pads) and VALID are handled.");
  }

  const bool ceil_mode = GetIntAttribute(node, "ceil_mode", 0) != 0;

  OptimShape out_shape;
  out_shape.PushBack(in_shape[0]);
  out_shape.PushBack(in_shape[1]);
  for (size_t i = 0; i < n_input_dims; ++i) {
    const OptimDim &d = in_shape[i + 2];
    if (d.IsInt()) {
      const int64_t out_d = OutputDim(d.AsInt(), kernel_shape[i], strides[i], pads[i],
                                      pads[i + n_input_dims], ceil_mode);
      out_shape.PushBack(OptimDim(out_d));
    } else {
      // Symbolic spatial dimension: propagate as a fresh symbolic
      // expression derived from the input expression and the pooling
      // parameters. This is intentionally opaque - downstream passes
      // should treat it as a free symbol.
      const std::string expr =
          "AveragePool(" + d.AsExpr() + ",k=" + std::to_string(kernel_shape[i]) +
          ",s=" + std::to_string(strides[i]) + ",p=" + std::to_string(pads[i]) + "+" +
          std::to_string(pads[i + n_input_dims]) + ",ceil=" + (ceil_mode ? "1" : "0") + ")";
      out_shape.PushBack(OptimDim(expr));
    }
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
