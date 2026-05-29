// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/nn/shape_nn.h"

#include <cstdint>
#include <string>

#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace nn {

namespace {

// Returns the channel dimension of ``X``:
//   * dim at index 1 when ``X`` has rank >= 2 (the spec convention since
//     opset 9 and the only case for opset 1/6 which require rank 4),
//   * the single dim when ``X`` is rank 1 (opset 9+ accepts a single dim
//     input of size N and treats C as 1),
//   * a fresh symbolic ``BatchNormalization.C`` expression when the rank
//     is 0 or unknown.
OptimDim ChannelDim(const OptimShape &x_shape) {
  if (x_shape.Rank() >= 2u) {
    return x_shape[1];
  }
  if (x_shape.Rank() == 1u) {
    return OptimDim(static_cast<int64_t>(1));
  }
  return OptimDim(std::string("BatchNormalization.C()"));
}

// Returns the opset version of the ``ai.onnx`` domain recorded in ``ctx``.
// Defaults to 15 (the most recent supported version) when not recorded so
// that callers can simply omit the ``opset_import`` when invoking shape
// inference on a stand-alone node.
int GetOnnxOpsetVersion(const ShapesContext &ctx) {
  if (!ctx.HasOpsetVersion("ai.onnx")) {
    return 15;
  }
  return ctx.OpsetVersion("ai.onnx");
}

} // namespace

void ComputeShapeBatchNormalization(ShapesContext &ctx, const NodeProto &node, const char *x,
                                    const char *input_mean) {
  CheckNodeOpAndOutput(node, "BatchNormalization", "ComputeShapeBatchNormalization");

  const OptimTensor &input = ctx.Get(x);
  const OptimShape &in_shape = input.Shape();

  // Output 0 (Y) always mirrors X.
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), in_shape));

  const int n_outputs = node.output_size();
  if (n_outputs <= 1) {
    return;
  }

  // Determine the dtype to use for the secondary outputs. For opset 14+ the
  // mean/var inputs carry a (potentially) different element type; for older
  // opsets every output shares the dtype of X.
  TensorType secondary_dtype = input.Dtype();
  const int opset = GetOnnxOpsetVersion(ctx);
  if (opset >= 14 && input_mean != nullptr) {
    secondary_dtype = ctx.Get(input_mean).Dtype();
  }

  const OptimShape channel_shape{ChannelDim(in_shape)};

  for (int i = 1; i < n_outputs; ++i) {
    const std::string &name = node.output(i).as_string();
    if (name.empty()) {
      continue;
    }
    ctx.Set(name, OptimTensor(nullptr, secondary_dtype, channel_shape));
  }
}

} // namespace nn
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
