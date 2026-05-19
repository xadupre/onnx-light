// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "onnx_lib/defs/shape_inference.h"

namespace ONNX_LIGHT_NAMESPACE {

inline void appendDimToTensorShapeProto(TensorShapeProto &tsp, const TensorShapeProto *input_data,
                                        int index) {
  const auto rank = static_cast<int>(input_data->ref_dim().size());
  if (index >= rank || index < -rank) {
    fail_shape_inference("indices (", index, ") must be in [-rank, rank-1] (rank=", rank, ").");
  } else {
    *tsp.add_dim() = input_data->ref_dim()[(index < 0) ? rank + index : index];
  }
}

// Determines whether the given axis attribute is 0.
inline bool axisIsZero(DataPropagationContext &ctx, bool defaultZero = false) {
  const auto *axisAttr = ctx.getAttribute("axis");
  if (!axisAttr) {
    if (defaultZero) {
      return true;
    }
    fail_shape_inference("Required attribute axis is missing");
    return false;
  }
  int axis = static_cast<int>(axisAttr->ref_i());
  if (axis >= 0) {
    return axis == 0;
  }

  const TypeProto *type = ctx.getInputType(0);
  if ((type == nullptr) || (!type->has_tensor_type()) || (!type->ref_tensor_type().has_shape())) {
    return false;
  }

  const int rank = static_cast<int>(type->ref_tensor_type().ref_shape().ref_dim().size());
  if (axis < -rank || axis >= rank) {
    fail_shape_inference("axis=", axis, " must be in [-rank, rank-1] (rank=", rank, ") (1).");
    return false;
  }
  axis += rank;
  return axis == 0;
}

inline void PropagateShapeDataFromInputToOutput(DataPropagationContext &ctx, int idx) {
  const auto *const input_data = ctx.getInputData(idx);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.CopyFrom(*input_data);
    ctx.addOutputData(0, std::move(tsp));
  }
}

inline void GatherOp13DataPropagator(DataPropagationContext &ctx) {
  if (!axisIsZero(ctx, true)) {
    return;
  }
  const auto *const input_data = ctx.getInputData(0);
  if (input_data == nullptr) {
    return;
  }
  const auto *const input_indices = ctx.getInputData(1);
  if (input_indices == nullptr) {
    return;
  }
  TensorShapeProto tsp;
  for (size_t i = 0; i < input_indices->ref_dim().size(); ++i) {
    if (input_indices->ref_dim()[i].has_dim_value()) {
      appendDimToTensorShapeProto(tsp, input_data,
                                  static_cast<int>(input_indices->ref_dim()[i].ref_dim_value()));
    } else {
      return;
    }
  }
  if (!tsp.ref_dim().empty()) {
    ctx.addOutputData(0, std::move(tsp));
  }
}

} // namespace ONNX_LIGHT_NAMESPACE
