// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shape_check.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace tensor {

void ComputeShapeReshape(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Reshape", "ComputeShapeReshape");

  if (node.input_size() < 2) {
    throw std::invalid_argument("ComputeShapeReshape: Reshape requires two inputs (data, shape).");
  }

  const OptimTensor &data = ctx.Get(node.input(0).as_string());
  const OptimTensor &shape_input = ctx.Get(node.input(1).as_string());

  const TensorType dtype = data.Dtype();
  const OptimShape &data_shape = data.Shape();
  const int64_t allowzero = GetAttributeOr<int64_t>(node, "allowzero", 0);

  // When the target shape is not known via data-propagation, fall
  // back to the rank exposed by the 1-D ``shape`` input itself (its
  // single static dim). Every output dimension is left symbolic.
  if (!shape_input.HasValueAsShape()) {
    OptimShape out_shape;
    if (shape_input.Shape().Rank() == 1 && shape_input.Shape()[0].IsInt()) {
      const int64_t rank = shape_input.Shape()[0].AsInt();
      for (int64_t i = 0; i < rank; ++i) {
        out_shape.PushBack(OptimDim("Reshape_dim" + std::to_string(i)));
      }
    } else {
      out_shape.PushBack(OptimDim("Reshape_dim0"));
    }
    ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
    return;
  }

  const OptimShape &target = shape_input.ValueAsShape();
  const int target_rank = static_cast<int>(target.Rank());
  const int data_rank = static_cast<int>(data_shape.Rank());

  OptimShape out_shape;
  // Index of the ``-1`` entry in ``target`` (if any), so it can be
  // back-filled from the input element count.
  int neg_one_index = -1;
  // Product of the concrete dimensions written into ``out_shape``,
  // used to back-fill the ``-1`` dimension. ``output_product_valid``
  // becomes false as soon as a symbolic dim is forwarded.
  int64_t output_product = 1;
  bool output_product_valid = true;
  // Tracks ``0`` entries that have been resolved by copying from the
  // input shape (so that the corresponding input dims can be excluded
  // from the input-product computation below).
  std::vector<bool> unresolved_zeros(target_rank, false);

  for (int i = 0; i < target_rank; ++i) {
    const OptimDim &dim = target[i];
    if (!dim.IsInt()) {
      out_shape.PushBack(dim);
      output_product_valid = false;
      continue;
    }
    const int64_t v = dim.AsInt();
    if (v == -1) {
      if (neg_one_index != -1) {
        throw std::invalid_argument(
            "ComputeShapeReshape: target shape may not have multiple -1 dimensions.");
      }
      neg_one_index = i;
      // Placeholder; resolved below.
      out_shape.PushBack(OptimDim(static_cast<int64_t>(-1)));
    } else if (v == 0) {
      if (allowzero == 0) {
        if (i >= data_rank) {
          throw std::invalid_argument("ComputeShapeReshape: invalid position of 0 in target "
                                      "shape (index " +
                                      std::to_string(i) + " out of input rank " +
                                      std::to_string(data_rank) + ").");
        }
        const OptimDim &input_dim = data_shape[i];
        out_shape.PushBack(input_dim);
        if (input_dim.IsInt()) {
          output_product *= input_dim.AsInt();
        } else {
          unresolved_zeros[i] = true;
          output_product_valid = false;
        }
      } else {
        out_shape.PushBack(OptimDim(static_cast<int64_t>(0)));
        output_product *= 0;
      }
    } else if (v > 0) {
      out_shape.PushBack(OptimDim(v));
      output_product *= v;
    } else {
      throw std::invalid_argument("ComputeShapeReshape: invalid dimension value " +
                                  std::to_string(v) + " in target shape.");
    }
  }

  if (neg_one_index != -1 && output_product_valid) {
    if (output_product == 0) {
      throw std::invalid_argument(
          "ComputeShapeReshape: invalid target shape product of 0 in combination with -1.");
    }
    int64_t input_product = 1;
    bool input_product_valid = true;
    for (int i = 0; i < data_rank; ++i) {
      if (data_shape[i].IsInt()) {
        input_product *= data_shape[i].AsInt();
      } else if (i >= static_cast<int>(unresolved_zeros.size()) || !unresolved_zeros[i]) {
        input_product_valid = false;
        break;
      }
    }
    if (input_product_valid) {
      if (input_product % output_product != 0) {
        throw std::invalid_argument(
            "ComputeShapeReshape: dimension could not be inferred: incompatible shapes (input "
            "element count " +
            std::to_string(input_product) + " is not a multiple of " +
            std::to_string(output_product) + ").");
      }
      out_shape[neg_one_index] = OptimDim(input_product / output_product);
    } else {
      out_shape[neg_one_index] = OptimDim("Reshape_neg1_" + std::to_string(neg_one_index));
    }
  } else if (neg_one_index != -1) {
    out_shape[neg_one_index] = OptimDim("Reshape_neg1_" + std::to_string(neg_one_index));
  }

  ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace tensor
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
