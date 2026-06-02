// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/math/shape_math.h"

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
namespace math {

namespace {

// Reads a 0-D INT32/INT64 OptimTensor value if it has a backing constant
// buffer; returns ``true`` when a value was extracted.
bool ReadScalarInt(const OptimTensor &t, int64_t &out) {
  if (t.Data() == nullptr) {
    return false;
  }
  if (t.Shape().Rank() != 0 &&
      !(t.Shape().Rank() == 1 && t.Shape()[0].IsInt() && t.Shape()[0].AsInt() == 1)) {
    return false;
  }
  switch (t.Dtype()) {
  case TensorType::kInt32:
    out = static_cast<int64_t>(*reinterpret_cast<const int32_t *>(t.Data()));
    return true;
  case TensorType::kInt64:
    out = *reinterpret_cast<const int64_t *>(t.Data());
    return true;
  default:
    return false;
  }
}

} // namespace

void ComputeShapeDFT(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "DFT", "ComputeShapeDFT");
  if (node.input_size() < 1) {
    throw std::invalid_argument("ComputeShapeDFT: DFT requires at least one input (input).");
  }

  const OptimTensor &input = ctx.Get(node.input(0).as_string());
  const TensorType dtype = input.Dtype();
  const OptimShape &in_shape = input.Shape();
  const int64_t rank = static_cast<int64_t>(in_shape.Rank());

  const AttributeProto *onesided_attr = FindAttribute(node, "onesided");
  const AttributeProto *inverse_attr = FindAttribute(node, "inverse");
  const bool onesided = (onesided_attr != nullptr) && onesided_attr->ref_i() != 0;
  const bool inverse = (inverse_attr != nullptr) && inverse_attr->ref_i() != 0;
  const int64_t out_last = (onesided && inverse) ? 1 : 2;

  // Determine the signal axis. v17: from the ``axis`` attribute (default 1).
  // v20: from the third input. When the axis comes from an input that is not
  // a known constant, mark the axis as unknown (we still know rank and last
  // dim).
  bool axis_known = false;
  int64_t axis = 0;
  const AttributeProto *axis_attr = FindAttribute(node, "axis");
  if (axis_attr != nullptr) {
    axis = axis_attr->ref_i();
    axis_known = true;
  } else if (node.input_size() >= 3 && !node.input(2).as_string().empty() &&
             ctx.Has(node.input(2).as_string())) {
    int64_t v = 0;
    if (ReadScalarInt(ctx.Get(node.input(2).as_string()), v)) {
      axis = v;
      axis_known = true;
    }
  } else {
    // Spec default at v20 is -2 (last signal axis) when ``axis`` is omitted.
    axis = -2;
    axis_known = true;
  }
  if (axis_known && rank > 0) {
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      axis_known = false;
    }
  }

  // Try to read dft_length as a constant when available.
  bool dft_length_known = false;
  int64_t dft_length = 0;
  if (node.input_size() >= 2 && !node.input(1).as_string().empty() &&
      ctx.Has(node.input(1).as_string())) {
    int64_t v = 0;
    if (ReadScalarInt(ctx.Get(node.input(1).as_string()), v)) {
      dft_length = v;
      dft_length_known = true;
    }
  }

  // Build the output shape. Output rank matches input rank.
  OptimShape out_shape;
  if (rank == 0) {
    ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
    return;
  }
  const std::string sym = "DFT_" + node.output(0).as_string() + "_axis";
  for (int64_t d = 0; d < rank; ++d) {
    if (d == rank - 1) {
      out_shape.PushBack(OptimDim(out_last));
    } else if (axis_known && d == axis) {
      if (dft_length_known) {
        if (onesided && !inverse) {
          out_shape.PushBack(OptimDim((dft_length / 2) + 1));
        } else {
          out_shape.PushBack(OptimDim(dft_length));
        }
      } else if (onesided || inverse) {
        // Cannot infer the output axis dim without dft_length.
        out_shape.PushBack(OptimDim(sym));
      } else {
        // Standard DFT with no dft_length: signal axis dim is preserved.
        out_shape.PushBack(in_shape[static_cast<std::size_t>(d)]);
      }
    } else if (!axis_known && d != rank - 1) {
      // Without knowing the signal axis, all non-trailing dims are symbolic.
      out_shape.PushBack(OptimDim(sym + "_" + std::to_string(d)));
    } else {
      out_shape.PushBack(in_shape[static_cast<std::size_t>(d)]);
    }
  }
  ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
