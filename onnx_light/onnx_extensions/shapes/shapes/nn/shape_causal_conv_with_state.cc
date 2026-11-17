// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <limits>
#include <stdexcept>
#include <string>

#include "onnx_core/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn {

void ComputeShapeCausalConvWithState(ShapesContext &ctx, const NodeProto &node, const char *input,
                                     const char *weight) {
  CheckNodeOpAndOutput(node, "CausalConvWithState", "ComputeShapeCausalConvWithState");
  EXT_ENFORCE_INVALID(!(node.output_size() < 2),
                      "ComputeShapeCausalConvWithState: node must declare two outputs.");
  int64_t dilation = 1;
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "dilation") {
      EXT_ENFORCE_INVALID(attr.type() == AttributeProto::INT,
                          "ComputeShapeCausalConvWithState: dilation must be an integer.");
      dilation = attr.i();
    }
  }
  EXT_ENFORCE_INVALID(dilation >= 1, "ComputeShapeCausalConvWithState: dilation must be >= 1.");

  const SymTensor &in = ctx.Get(input);
  const SymTensor &w = ctx.Get(weight);
  const SymShape &in_shape = in.Shape();
  const SymShape &w_shape = w.Shape();

  EXT_ENFORCE_INVALID(in_shape.Rank() == 3, "ComputeShapeCausalConvWithState: input '", input,
                      "' must have rank 3, got ", in_shape.Rank(), ".");
  EXT_ENFORCE_INVALID(w_shape.Rank() == 3, "ComputeShapeCausalConvWithState: weight '", weight,
                      "' must have rank 3, got ", w_shape.Rank(), ".");

  SymDim history;
  if (w_shape[2].IsInt()) {
    const int64_t k = w_shape[2].AsInt();
    EXT_ENFORCE_INVALID(k >= 1, "ComputeShapeCausalConvWithState: kernel size K must be >= 1.");
    EXT_ENFORCE_INVALID(k - 1 <= (std::numeric_limits<int64_t>::max() - 1) / dilation,
                        "ComputeShapeCausalConvWithState: effective kernel size overflows int64.");
    const int64_t h = (k - 1) * dilation;
    history = SymDim(h);
  } else {
    const std::string &kernel_expr = w_shape[2].AsExpr();
    history =
        SymDim(kernel_expr.empty() ? ""
                                   : "((" + kernel_expr + ")-1)" +
                                         (dilation == 1 ? "" : "*" + std::to_string(dilation)));
  }
  if (node.input_size() > 3 && !node.input(3).empty() && ctx.Has(node.input(3))) {
    const SymShape &past = ctx.Get(node.input(3)).Shape();
    EXT_ENFORCE_INVALID(past.Rank() == 3,
                        "ComputeShapeCausalConvWithState: past_state must have rank 3.");
    for (size_t axis = 0; axis < 2; ++axis) {
      EXT_ENFORCE_INVALID(!past[axis].IsInt() || !in_shape[axis].IsInt() ||
                              past[axis].AsInt() == in_shape[axis].AsInt(),
                          "ComputeShapeCausalConvWithState: past_state batch/channel mismatch.");
    }
    if (past[2].IsInt()) {
      const int64_t h = past[2].AsInt();
      EXT_ENFORCE_INVALID(h >= 0 && h < std::numeric_limits<int64_t>::max() && h % dilation == 0,
                          "ComputeShapeCausalConvWithState: invalid past_state history length.");
      EXT_ENFORCE_INVALID(!history.IsInt() || history.AsInt() == h,
                          "ComputeShapeCausalConvWithState: past_state history mismatch.");
      history = SymDim(h);
    }
  }
  if (history.IsInt() && in_shape[2].IsInt()) {
    EXT_ENFORCE_INVALID(in_shape[2].AsInt() >= 0 &&
                            in_shape[2].AsInt() <=
                                std::numeric_limits<int64_t>::max() - history.AsInt(),
                        "ComputeShapeCausalConvWithState: padded length overflows int64.");
  }
  ctx.Set(node.output(0), SymTensor(nullptr, in.Dtype(), in_shape));
  SymShape present_shape{in_shape[0], in_shape[1], history};
  ctx.Set(node.output(1), SymTensor(nullptr, in.Dtype(), present_shape));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::nn
