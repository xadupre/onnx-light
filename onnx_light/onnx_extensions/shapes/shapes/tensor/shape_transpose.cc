// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/tensor/shape_tensor.h"

#include <cstdint>
#include <vector>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor {

void ComputeShapeTranspose(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "Transpose", "ComputeShapeTranspose");
  EXT_ENFORCE_INVALID(!(node.input_size() < 1),
                      "ComputeShapeTranspose: Transpose requires one input.");

  const SymTensor &input = ctx.Get(node.input(0));
  const SymShape &input_shape = input.Shape();
  const std::size_t rank = input_shape.Rank();

  std::vector<int64_t> perm;
  const bool has_perm = GetAttributeInts(node, "perm", perm);
  if (!has_perm) {
    perm.reserve(rank);
    for (std::size_t i = 0; i < rank; ++i) {
      perm.push_back(static_cast<int64_t>(rank - 1 - i));
    }
  } else if (perm.size() != rank) {
    EXT_THROW_INVALID("ComputeShapeTranspose: perm length (", perm.size(),
                      ") must match input rank (", rank, ").");
  }

  std::vector<bool> seen(rank, false);
  for (int64_t p : perm) {
    EXT_ENFORCE_INVALID(!(p < 0 || static_cast<std::size_t>(p) >= rank),
                        "ComputeShapeTranspose: perm contains out-of-range axis ", p, " for rank ",
                        rank, ".");
    EXT_ENFORCE_INVALID(!(seen[static_cast<std::size_t>(p)]),
                        "ComputeShapeTranspose: perm contains duplicate axis ", p, ".");
    seen[static_cast<std::size_t>(p)] = true;
  }

  SymShape out_shape;
  for (int64_t p : perm) {
    out_shape.PushBack(input_shape[static_cast<std::size_t>(p)]);
  }
  ctx.Set(node.output(0), SymTensor(nullptr, input.Dtype(), std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::tensor
