// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

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
namespace tensor {

void ComputeShapeTranspose(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Transpose", "ComputeShapeTranspose");
  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapeTranspose: Transpose requires one input.");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const OptimShape &input_shape = input.Shape();
  const std::size_t rank = input_shape.Rank();

  std::vector<int64_t> perm;
  const bool has_perm = GetAttributeInts(node, "perm", perm);
  if (!has_perm) {
    perm.reserve(rank);
    for (std::size_t i = 0; i < rank; ++i) {
      perm.push_back(static_cast<int64_t>(rank - 1 - i));
    }
  } else if (perm.size() != rank) {
    throw std::invalid_argument("ComputeShapeTranspose: perm length (" +
                                std::to_string(perm.size()) + ") must match input rank (" +
                                std::to_string(rank) + ").");
  }

  std::vector<bool> seen(rank, false);
  for (int64_t p : perm) {
    if (p < 0 || static_cast<std::size_t>(p) >= rank) {
      throw std::invalid_argument("ComputeShapeTranspose: perm contains out-of-range axis " +
                                  std::to_string(p) + " for rank " + std::to_string(rank) + ".");
    }
    if (seen[static_cast<std::size_t>(p)]) {
      throw std::invalid_argument("ComputeShapeTranspose: perm contains duplicate axis " +
                                  std::to_string(p) + ".");
    }
    seen[static_cast<std::size_t>(p)] = true;
  }

  OptimShape out_shape;
  for (int64_t p : perm) {
    out_shape.PushBack(input_shape[static_cast<std::size_t>(p)]);
  }
  ctx.Set(node.output(0), OptimTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
