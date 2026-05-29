// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

namespace {

// Returns ``num_directions`` derived from the ``direction`` attribute. For
// unknown values the dim is left as a fresh symbolic expression so callers
// can still propagate the rank.
OptimDim NumDirectionsDim(const std::string &op_type, const std::string &direction) {
  if (direction == "forward" || direction == "reverse") {
    return OptimDim(static_cast<int64_t>(1));
  }
  if (direction == "bidirectional") {
    return OptimDim(static_cast<int64_t>(2));
  }
  return OptimDim(op_type + ".num_directions(" + direction + ")");
}

// Returns the ``hidden_size`` dim, preferring the explicit ``hidden_size``
// attribute when set to a positive value. Falls back to ``R.shape[2]`` (the
// recurrence weight's last dim is always ``hidden_size`` for RNN / GRU /
// LSTM) when available, otherwise yields a fresh symbolic expression.
OptimDim HiddenSizeDim(const std::string &op_type, const NodeProto &node, const ShapesContext &ctx,
                       const char *r) {
  const int64_t hidden_size = GetAttributeOr<int64_t>(node, "hidden_size", -1);
  if (hidden_size > 0) {
    return OptimDim(hidden_size);
  }
  if (r != nullptr && ctx.Has(r)) {
    const OptimShape &r_shape = ctx.Get(r).Shape();
    if (r_shape.Rank() == 3u) {
      return r_shape[2];
    }
  }
  return OptimDim(op_type + ".hidden_size()");
}

bool IsRecurrentOp(const std::string &op_type) {
  return op_type == "RNN" || op_type == "GRU" || op_type == "LSTM";
}

} // namespace

void ComputeShapeRNN(ShapesContext &ctx, const NodeProto &node, const char *x, const char *r) {
  const std::string op_type = node.op_type().as_string();
  EXT_ENFORCE_INVALID(IsRecurrentOp(op_type),
                      "ComputeShapeRNN: node.op_type() must be one of RNN, GRU or LSTM, got '" +
                          op_type + "'.");
  EXT_ENFORCE_INVALID(node.output_size() > 0,
                      "ComputeShapeRNN: node '" + op_type + "' must declare at least one output.");

  const OptimTensor &input = ctx.Get(x);
  const OptimShape &x_shape = input.Shape();
  if (x_shape.Rank() != 3u) {
    throw std::invalid_argument("ComputeShapeRNN: input '" + std::string(x) +
                                "' must have rank 3 ([seq_length, batch_size, input_size] or "
                                "[batch_size, seq_length, input_size]).");
  }

  const std::string direction =
      GetAttributeOr<std::string>(node, "direction", std::string("forward"));
  const int64_t layout = GetAttributeOr<int64_t>(node, "layout", 0);

  const OptimDim num_directions = NumDirectionsDim(op_type, direction);
  const OptimDim hidden_size = HiddenSizeDim(op_type, node, ctx, r);
  const OptimDim seq_length = layout == 0 ? x_shape[0] : x_shape[1];
  const OptimDim batch_size = layout == 0 ? x_shape[1] : x_shape[0];

  // Output 0: Y. Optional in the schema (an empty output name skips it).
  const std::string &y_name = node.output(0).as_string();
  if (!y_name.empty()) {
    OptimShape y_shape;
    if (layout == 0) {
      y_shape.PushBack(seq_length);
      y_shape.PushBack(num_directions);
      y_shape.PushBack(batch_size);
      y_shape.PushBack(hidden_size);
    } else {
      y_shape.PushBack(batch_size);
      y_shape.PushBack(seq_length);
      y_shape.PushBack(num_directions);
      y_shape.PushBack(hidden_size);
    }
    ctx.Set(y_name, OptimTensor(nullptr, input.Dtype(), std::move(y_shape)));
  }

  // Outputs 1 (Y_h) and 2 (Y_c, LSTM only): rank-3 hidden / cell states.
  for (int i = 1; i < node.output_size(); ++i) {
    const std::string &name = node.output(i).as_string();
    if (name.empty()) {
      continue;
    }
    OptimShape h_shape;
    if (layout == 0) {
      h_shape.PushBack(num_directions);
      h_shape.PushBack(batch_size);
      h_shape.PushBack(hidden_size);
    } else {
      h_shape.PushBack(batch_size);
      h_shape.PushBack(num_directions);
      h_shape.PushBack(hidden_size);
    }
    ctx.Set(name, OptimTensor(nullptr, input.Dtype(), std::move(h_shape)));
  }
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
