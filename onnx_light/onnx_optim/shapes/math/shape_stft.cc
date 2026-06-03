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

void ComputeShapeSTFT(ShapesContext &ctx, const NodeProto &node) {
  CheckNodeOpAndOutput(node, "STFT", "ComputeShapeSTFT");
  if (node.input_size() < 2) {
    throw std::invalid_argument(
        "ComputeShapeSTFT: STFT requires at least two inputs (signal, frame_step).");
  }

  const OptimTensor &signal = ctx.Get(node.input(0).as_string());
  const TensorType dtype = signal.Dtype();
  const OptimShape &in_shape = signal.Shape();

  const AttributeProto *onesided_attr = FindAttribute(node, "onesided");
  // STFT spec: default for onesided is 1 (real input).
  const bool onesided = (onesided_attr == nullptr) || onesided_attr->ref_i() != 0;

  // Try to read frame_step as a known constant.
  bool frame_step_known = false;
  int64_t frame_step_value = 0;
  if (!node.input(1).as_string().empty() && ctx.Has(node.input(1).as_string())) {
    int64_t v = 0;
    if (ReadScalarInt(ctx.Get(node.input(1).as_string()), v)) {
      frame_step_value = v;
      frame_step_known = true;
    }
  }

  // Determine frame_length from the optional ``window`` (input 2) and
  // ``frame_length`` (input 3) inputs. ``frame_length`` is preferred when
  // known as a constant.
  bool frame_length_known = false;
  int64_t frame_length_value = 0;
  if (node.input_size() >= 4 && !node.input(3).as_string().empty() &&
      ctx.Has(node.input(3).as_string())) {
    int64_t v = 0;
    if (ReadScalarInt(ctx.Get(node.input(3).as_string()), v)) {
      frame_length_value = v;
      frame_length_known = true;
    }
  }
  if (!frame_length_known && node.input_size() >= 3 && !node.input(2).as_string().empty() &&
      ctx.Has(node.input(2).as_string())) {
    const OptimTensor &window = ctx.Get(node.input(2).as_string());
    if (window.Shape().Rank() == 1 && window.Shape()[0].IsInt()) {
      frame_length_value = window.Shape()[0].AsInt();
      frame_length_known = true;
    }
  }

  // Compute output shape: [batch_size, n_frames, dft_unique_bins, 2].
  const std::string sym = "STFT_" + node.output(0).as_string();
  OptimShape out_shape;
  // batch_size from signal input.
  if (in_shape.Rank() >= 1) {
    out_shape.PushBack(in_shape[0]);
  } else {
    out_shape.PushBack(OptimDim(sym + "_batch"));
  }

  // n_frames: derive from signal_length, frame_length, frame_step when known.
  bool n_frames_known = false;
  int64_t n_frames_value = 0;
  if (frame_length_known && frame_step_known && in_shape.Rank() >= 2 && in_shape[1].IsInt()) {
    const int64_t signal_length = in_shape[1].AsInt();
    if (frame_step_value > 0 && frame_length_value > 0 && signal_length >= frame_length_value) {
      n_frames_value = (signal_length - frame_length_value) / frame_step_value + 1;
      n_frames_known = true;
    }
  }
  if (n_frames_known) {
    out_shape.PushBack(OptimDim(n_frames_value));
  } else {
    out_shape.PushBack(OptimDim(sym + "_frames"));
  }

  // dft_unique_bins.
  if (frame_length_known) {
    if (onesided) {
      out_shape.PushBack(OptimDim((frame_length_value / 2) + 1));
    } else {
      out_shape.PushBack(OptimDim(frame_length_value));
    }
  } else {
    out_shape.PushBack(OptimDim(sym + "_bins"));
  }

  out_shape.PushBack(OptimDim(2));
  ctx.Set(node.output(0), OptimTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace math
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
