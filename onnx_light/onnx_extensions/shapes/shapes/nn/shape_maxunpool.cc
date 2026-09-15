// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <cstdint>
#include <string>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

void ComputeShapeMaxUnpool(ShapesContext &ctx, const NodeProto &node, const char *x, const char *I,
                           const char *output_shape) {
  CheckNodeOpAndOutput(node, "MaxUnpool", "ComputeShapeMaxUnpool");
  (void)I;

  const SymTensor &input = ctx.Get(x);
  const SymShape &in_shape = input.Shape();
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
    EXT_ENFORCE_INVALID(
        strides.size() == n_input_dims,
        "ComputeShapeMaxUnpool: attribute 'strides' size must match input rank - 2.");
  } else {
    strides.assign(n_input_dims, 1);
  }

  std::vector<int64_t> pads;
  if (GetAttributeInts(node, "pads", pads)) {
    EXT_ENFORCE_INVALID(
        pads.size() == 2 * n_input_dims,
        "ComputeShapeMaxUnpool: attribute 'pads' size must be 2 * (input rank - 2).");
  } else {
    pads.assign(2 * n_input_dims, 0);
  }

  // When ``output_shape`` is provided as a known initializer, take its
  // values directly; this overrides any computed shape (per the ONNX spec).
  const SymShape *explicit_out_shape = nullptr;
  if (output_shape != nullptr) {
    const SymTensor &out_shape_tensor = ctx.Get(output_shape);
    EXT_ENFORCE_INVALID(out_shape_tensor.Dtype() == TensorType::kInt64,
                        "ComputeShapeMaxUnpool: output_shape must be INT64.");
    EXT_ENFORCE_INVALID(out_shape_tensor.Shape().Rank() == 1,
                        "ComputeShapeMaxUnpool: output_shape must be rank 1.");
    const SymDim &length = out_shape_tensor.Shape()[0];
    EXT_ENFORCE_INVALID(!length.IsInt() || length.AsInt() == static_cast<int64_t>(in_shape.Rank()),
                        "ComputeShapeMaxUnpool: output_shape length must match input rank.");
    if (out_shape_tensor.HasValueAsShape()) {
      explicit_out_shape = &out_shape_tensor.ValueAsShape();
      EXT_ENFORCE_INVALID(explicit_out_shape->Rank() == in_shape.Rank(),
                          "ComputeShapeMaxUnpool: output_shape length must match input rank.");
      for (size_t i = 0; i < explicit_out_shape->Rank(); ++i) {
        const SymDim &dim = (*explicit_out_shape)[i];
        EXT_ENFORCE_INVALID(!dim.IsInt() || dim.AsInt() >= 0,
                            "ComputeShapeMaxUnpool: output_shape dimensions must be non-negative.");
      }
    }
    SymShape out_shape;
    for (size_t i = 0; i < in_shape.Rank(); ++i) {
      out_shape.PushBack(explicit_out_shape
                             ? (*explicit_out_shape)[i]
                             : SymDim(std::string(output_shape) + "[" + std::to_string(i) + "]"));
    }
    ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), std::move(out_shape)));
    return;
  }

  SymShape out_shape;
  out_shape.PushBack(in_shape[0]);
  out_shape.PushBack(in_shape[1]);
  for (size_t i = 0; i < n_input_dims; ++i) {
    const SymDim &d = in_shape[i + 2];
    if (d.IsInt()) {
      const int64_t out_d =
          strides[i] * (d.AsInt() - 1) + kernel_shape[i] - pads[i] - pads[i + n_input_dims];
      out_shape.PushBack(SymDim(out_d));
    } else {
      std::string expr = "MaxUnpool(" + d.AsExpr() + ",k=" + std::to_string(kernel_shape[i]) +
                         ",s=" + std::to_string(strides[i]) + ",p=" + std::to_string(pads[i]) +
                         "+" + std::to_string(pads[i + n_input_dims]) + ")";
      out_shape.PushBack(SymDim(expr));
    }
  }

  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
