// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/training/shape_training.h"

#include <string>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::training {

void ComputeShapeAdam(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Adam", "ComputeShapeAdam");

  // Input layout: [R, T, X_1..X_N, G_1..G_N, V_1..V_N, H_1..H_N].
  // Output layout: [X_1_new..X_N_new, V_1_new..V_N_new, H_1_new..H_N_new].
  const int n_inputs = node.input_size();
  EXT_ENFORCE_INVALID(n_inputs >= 6,
                      "ComputeShapeAdam: Adam expects at least 6 inputs "
                      "(R, T and one (X, G, V, H) tuple), got ",
                      std::to_string(n_inputs), ".");
  const int num_adjustable = n_inputs - 2;
  EXT_ENFORCE_INVALID(num_adjustable % 4 == 0,
                      "ComputeShapeAdam: the count of optimised tensors, "
                      "gradients, momenta and squared-gradients (input_size - 2) "
                      "must be a multiple of 4, got ",
                      std::to_string(num_adjustable), ".");
  const int num_optimized = num_adjustable / 4;
  EXT_ENFORCE_INVALID(node.output_size() == 3 * num_optimized, "ComputeShapeAdam: expected ",
                      std::to_string(3 * num_optimized), " outputs for ",
                      std::to_string(num_optimized), " optimised tensor(s), got ",
                      std::to_string(node.output_size()), ".");

  for (int i = 0; i < num_optimized; ++i) {
    // X_i  -> X_i_new        (output i)
    const SymTensor &x = ctx.Get(node.input(2 + i));
    ctx.Set(node.output(i), SymTensor(nullptr, x.Dtype(), x.Shape()));

    // V_i  -> V_i_new        (output num_optimized + i)
    const SymTensor &v = ctx.Get(node.input(2 + 2 * num_optimized + i));
    ctx.Set(node.output(num_optimized + i), SymTensor(nullptr, v.Dtype(), v.Shape()));

    // H_i  -> H_i_new        (output 2 * num_optimized + i)
    const SymTensor &h = ctx.Get(node.input(2 + 3 * num_optimized + i));
    ctx.Set(node.output(2 * num_optimized + i), SymTensor(nullptr, h.Dtype(), h.Shape()));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::training
