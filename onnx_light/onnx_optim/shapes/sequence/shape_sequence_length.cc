// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/sequence/shape_sequence.h"

#include <string>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace sequence {

void ComputeShapeSequenceLength(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "SequenceLength", "ComputeShapeSequenceLength");
  EXT_ENFORCE_INVALID(node.input_size() >= 1,
                      "ComputeShapeSequenceLength: SequenceLength requires one input.");
  (void)ctx.GetSequence(node.input(0).as_string());
  ctx.Set(node.output(0), OptimTensor(nullptr, TensorType::kInt64, OptimShape{}));
}

} // namespace sequence
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
