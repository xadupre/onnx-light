// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/training/shape_training.h"

#include <stdexcept>
#include <string>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace training {

void ComputeShapeAdagrad(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Adagrad", "ComputeShapeAdagrad");

  // Input layout: [R, T, X_1..X_N, G_1..G_N, H_1..H_N].
  // Output layout: [X_1_new..X_N_new, H_1_new..H_N_new].
  const int n_inputs = node.input_size();
  EXT_ENFORCE_INVALID(n_inputs >= 5,
                      std::string("ComputeShapeAdagrad: Adagrad expects at least 5 inputs "
                                  "(R, T and one (X, G, H) tuple), got ") +
                          std::to_string(n_inputs) + ".");
  const int num_adjustable = n_inputs - 2;
  EXT_ENFORCE_INVALID(num_adjustable % 3 == 0,
                      std::string("ComputeShapeAdagrad: the count of optimised tensors, "
                                  "gradients and accumulated squared gradients (input_size - 2) "
                                  "must be a multiple of 3, got ") +
                          std::to_string(num_adjustable) + ".");
  const int num_optimized = num_adjustable / 3;
  EXT_ENFORCE_INVALID(node.output_size() == 2 * num_optimized,
                      std::string("ComputeShapeAdagrad: expected ") +
                          std::to_string(2 * num_optimized) + " outputs for " +
                          std::to_string(num_optimized) + " optimised tensor(s), got " +
                          std::to_string(node.output_size()) + ".");

  for (int i = 0; i < num_optimized; ++i) {
    // X_i  -> X_i_new        (output i)
    const OptimTensor &x = ctx.Get(node.input(2 + i).as_string());
    ctx.Set(node.output(i), OptimTensor(nullptr, x.Dtype(), x.Shape()));

    // H_i  -> H_i_new        (output num_optimized + i)
    const OptimTensor &h = ctx.Get(node.input(2 + 2 * num_optimized + i).as_string());
    ctx.Set(node.output(num_optimized + i), OptimTensor(nullptr, h.Dtype(), h.Shape()));
  }
}

} // namespace training
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
