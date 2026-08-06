// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/training/shape_training.h"

#include <stdexcept>
#include <string>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::training {

void ComputeShapeMomentum(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Momentum", "ComputeShapeMomentum");

  // Input layout: [R, T, X_1..X_N, G_1..G_N, V_1..V_N].
  // Output layout: [X_1_new..X_N_new, V_1_new..V_N_new].
  const int n_inputs = node.input_size();
  EXT_ENFORCE_INVALID(n_inputs >= 5,
                      "ComputeShapeMomentum: Momentum expects at least 5 inputs "
                      "(R, T and one (X, G, V) tuple), got ",
                      std::to_string(n_inputs), ".");
  const int num_adjustable = n_inputs - 2;
  EXT_ENFORCE_INVALID(num_adjustable % 3 == 0,
                      "ComputeShapeMomentum: the count of optimised tensors, "
                      "gradients and momenta (input_size - 2) "
                      "must be a multiple of 3, got ",
                      std::to_string(num_adjustable), ".");
  const int num_optimized = num_adjustable / 3;
  EXT_ENFORCE_INVALID(node.output_size() == 2 * num_optimized, "ComputeShapeMomentum: expected ",
                      std::to_string(2 * num_optimized), " outputs for ",
                      std::to_string(num_optimized), " optimised tensor(s), got ",
                      std::to_string(node.output_size()), ".");

  for (int i = 0; i < num_optimized; ++i) {
    // X_i  -> X_i_new        (output i)
    const SymTensor &x = ctx.Get(node.input(2 + i));
    ctx.Set(node.output(i), SymTensor(nullptr, x.Dtype(), x.Shape()));

    // V_i  -> V_i_new        (output num_optimized + i)
    const SymTensor &v = ctx.Get(node.input(2 + 2 * num_optimized + i));
    ctx.Set(node.output(num_optimized + i), SymTensor(nullptr, v.Dtype(), v.Shape()));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::training
