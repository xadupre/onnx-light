// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/shapes/shapes/math/shape_math.h"

#include <cstdint>
#include <string>
#include <utility>

#include "onnx_core/shapes/shape_check.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math {

namespace {

// Reads a 0-D INT32/INT64 SymTensor value if it has a backing constant
// buffer; returns ``true`` when a value was extracted.
bool ReadScalarInt(const SymTensor &t, int64_t &out) {
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
  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "ComputeShapeSTFT: STFT requires at least two inputs (signal, frame_step).");

  const SymTensor &signal = ctx.Get(node.input(0));
  const TensorType dtype = signal.Dtype();
  const SymShape &in_shape = signal.Shape();
  EXT_ENFORCE_INVALID(in_shape.Rank() == 3, "ComputeShapeSTFT: signal must have rank 3.");
  if (in_shape[2].IsInt()) {
    EXT_ENFORCE_INVALID(in_shape[2].AsInt() == 1 || in_shape[2].AsInt() == 2,
                        "ComputeShapeSTFT: signal's last dimension must have size 1 or 2.");
  }

  const AttributeProto *onesided_attr = FindAttribute(node, "onesided");
  const int64_t onesided_value = onesided_attr == nullptr ? 1 : onesided_attr->ref_i();
  EXT_ENFORCE_INVALID(onesided_value == 0 || onesided_value == 1,
                      "ComputeShapeSTFT: onesided must be 0 or 1.");
  const bool onesided = onesided_value == 1;
  EXT_ENFORCE_INVALID(!onesided || !in_shape[2].IsInt() || in_shape[2].AsInt() == 1,
                      "ComputeShapeSTFT: one-sided STFT requires real input.");

  // Try to read frame_step as a known constant.
  bool frame_step_known = false;
  int64_t frame_step_value = 0;
  if (!node.input(1).empty() && ctx.Has(node.input(1))) {
    const SymTensor &frame_step = ctx.Get(node.input(1));
    EXT_ENFORCE_INVALID(
        frame_step.Shape().Rank() == 0 ||
            (frame_step.Shape().Rank() == 1 &&
             (!frame_step.Shape()[0].IsInt() || frame_step.Shape()[0].AsInt() == 1)),
        "ComputeShapeSTFT: frame_step must be a scalar or a single-element vector.");
    int64_t v = 0;
    if (ReadScalarInt(frame_step, v)) {
      EXT_ENFORCE_INVALID(v > 0, "ComputeShapeSTFT: frame_step must be greater than 0.");
      frame_step_value = v;
      frame_step_known = true;
    }
  }

  bool window_length_known = false;
  int64_t window_length = 0;
  if (node.input_size() >= 3 && !node.input(2).empty() && ctx.Has(node.input(2))) {
    const SymTensor &window = ctx.Get(node.input(2));
    EXT_ENFORCE_INVALID(window.Shape().Rank() == 1, "ComputeShapeSTFT: window must have rank 1.");
    if (window.Shape()[0].IsInt()) {
      window_length = window.Shape()[0].AsInt();
      EXT_ENFORCE_INVALID(window_length > 0,
                          "ComputeShapeSTFT: window must have a positive length.");
      window_length_known = true;
    }
  }

  bool frame_length_known = false;
  int64_t frame_length_value = 0;
  if (node.input_size() >= 4 && !node.input(3).empty() && ctx.Has(node.input(3))) {
    const SymTensor &frame_length = ctx.Get(node.input(3));
    EXT_ENFORCE_INVALID(frame_length.Shape().Rank() == 0,
                        "ComputeShapeSTFT: frame_length must be a scalar.");
    int64_t v = 0;
    if (ReadScalarInt(frame_length, v)) {
      EXT_ENFORCE_INVALID(v > 0, "ComputeShapeSTFT: frame_length must be greater than 0.");
      EXT_ENFORCE_INVALID(!window_length_known || window_length == v,
                          "ComputeShapeSTFT: window length must match frame_length.");
      frame_length_value = v;
      frame_length_known = true;
    }
  }
  if (!frame_length_known && window_length_known) {
    frame_length_value = window_length;
    frame_length_known = true;
  }

  const bool has_window =
      node.input_size() >= 3 && !node.input(2).empty() && ctx.Has(node.input(2));
  const bool has_frame_length =
      node.input_size() >= 4 && !node.input(3).empty() && ctx.Has(node.input(3));
  const bool frame_length_defaults_to_signal = !has_window && !has_frame_length;
  if (frame_length_defaults_to_signal && in_shape[1].IsInt()) {
    frame_length_value = in_shape[1].AsInt();
    frame_length_known = true;
  }

  // Compute output shape: [batch_size, n_frames, dft_unique_bins, 2].
  const std::string sym = "STFT_" + node.output(0);
  SymShape out_shape;
  out_shape.PushBack(in_shape[0]);

  // n_frames: derive from signal_length, frame_length, frame_step when known.
  bool n_frames_known = false;
  int64_t n_frames_value = 0;
  if (frame_length_defaults_to_signal) {
    n_frames_value = 1;
    n_frames_known = true;
  } else if (frame_length_known && frame_step_known && in_shape[1].IsInt()) {
    const int64_t signal_length = in_shape[1].AsInt();
    if (signal_length >= frame_length_value) {
      n_frames_value = (signal_length - frame_length_value) / frame_step_value + 1;
      n_frames_known = true;
    }
  }
  if (n_frames_known) {
    out_shape.PushBack(SymDim(n_frames_value));
  } else {
    out_shape.PushBack(SymDim(sym + "_frames"));
  }

  // dft_unique_bins.
  if (frame_length_known) {
    if (onesided) {
      out_shape.PushBack(SymDim((frame_length_value / 2) + 1));
    } else {
      out_shape.PushBack(SymDim(frame_length_value));
    }
  } else {
    out_shape.PushBack(SymDim(sym + "_bins"));
  }

  out_shape.PushBack(SymDim(2));
  ctx.Set(node.output(0), SymTensor(nullptr, dtype, std::move(out_shape)));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_shapes::shapes::math
