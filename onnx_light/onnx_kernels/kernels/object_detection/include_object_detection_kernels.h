// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

#include <cstdint>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

// ---------------------------------------------------------------------------
// Reference implementations of the ``object_detection`` backend test
// kernels.
//
// Each kernel is exposed as a small class whose constructor takes a
// :ref:`KernelContext` (carrying the opset against which the kernel must
// behave) and whose ``operator()`` performs the computation.
//
// Two flavors of ``operator()`` are provided:
//
//   * The returning overload (``Tensor operator()(...) const``) allocates a
//     fresh ``Tensor`` whose data buffer is owned by the returned value.
//   * The in-place overload (``void operator()(..., Tensor &output) const``)
//     writes results into a caller-supplied output tensor whose buffer has
//     already been allocated. The caller is responsible for setting
//     ``output.data_type``, ``output.shape`` and sizing ``output.data`` to
//     match the operator's expected output; the kernel validates these
//     attributes and throws ``std::invalid_argument`` on mismatch.
//
// ``RoiAlign`` mirrors the ONNX ``RoiAlign`` operator (since opset 10 in the
// ai.onnx domain). The kernel is restricted to the FLOAT element type and
// supports the attribute set common to opset 10 and opset 16+:
//
//   * ``mode`` (string): ``"avg"`` (default) or ``"max"``.
//   * ``output_height`` / ``output_width`` (int, default 1): spatial size of
//     each pooled feature map.
//   * ``spatial_scale`` (float, default 1.0): multiplier applied to the
//     ``rois`` coordinates to map them onto the input feature map.
//   * ``sampling_ratio`` (int, default 0): number of sampling points per bin
//     along each spatial axis; ``0`` selects an adaptive value of
//     ``ceil(roi_size / output_size)``.
//   * ``coordinate_transformation_mode`` (string, default ``"half_pixel"``
//     starting at opset 16): if ``"half_pixel"`` a ``-0.5`` offset is
//     applied to the scaled roi coordinates, otherwise no offset is applied.
//     Opset 10 has no such attribute and behaves like
//     ``"output_half_pixel"``.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias an
// input tensor's buffer. ``RoiAlign``'s output has a different shape and
// element layout from any of its inputs and so cannot share storage.
// ---------------------------------------------------------------------------

/// Reference RoiAlign kernel restricted to FLOAT inputs/outputs.
class RoiAlign : public KernelBase {
public:
  /// Attributes carried by the ONNX ``RoiAlign`` operator. Defaults match
  /// the opset-16 schema; ``coordinate_transformation_mode`` should be set
  /// to ``"output_half_pixel"`` to reproduce the legacy opset-10 behaviour.
  struct Attributes {
    std::string mode = "avg";
    int64_t output_height = 1;
    int64_t output_width = 1;
    int64_t sampling_ratio = 0;
    float spatial_scale = 1.0f;
    std::string coordinate_transformation_mode = "half_pixel";
  };

  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &x, const Tensor &rois, const Tensor &batch_indices,
                    const Attributes &attrs) const;
  void operator()(const Tensor &x, const Tensor &rois, const Tensor &batch_indices,
                  const Attributes &attrs, Tensor &output) const;

  /// Output element layout (num_rois, C, output_height, output_width)
  /// fundamentally differs from any input, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference NonMaxSuppression kernel (since opset 10 in the ai.onnx domain).
///
/// Implements the upstream NMS reference: for each (batch, class) pair the
/// kernel sorts boxes by descending score, optionally pre-filters by
/// ``score_threshold``, then greedily picks boxes whose IoU with all
/// previously selected boxes is at most ``iou_threshold``. The selection is
/// capped at ``max_output_boxes_per_class`` per (batch, class) pair and the
/// final output rows are emitted in the upstream order: outer loop over
/// batches, then classes, then per-pair selection order (which itself is the
/// descending-score insertion order).
///
/// Inputs are restricted to FLOAT for ``boxes``/``scores``/``iou_threshold``/
/// ``score_threshold`` and INT64 for ``max_output_boxes_per_class``; the
/// optional scalar inputs (``max_output_boxes_per_class``, ``iou_threshold``,
/// ``score_threshold``) may be omitted by passing a ``nullptr`` ``Tensor``
/// pointer to ``operator()``. Defaults follow the ONNX schema:
/// ``max_output_boxes_per_class = 0`` (no output), ``iou_threshold = 0`` and
/// ``score_threshold = -inf`` (no score filtering).
class NonMaxSuppression : public KernelBase {
public:
  /// Attributes carried by the ONNX ``NonMaxSuppression`` operator. The
  /// default matches the schema (``center_point_box = 0`` — the
  /// ``[y1, x1, y2, x2]`` corner format).
  struct Attributes {
    int64_t center_point_box = 0;
  };

  using KernelBase::KernelBase;

  /// Computes selected indices and returns a freshly allocated INT64 tensor
  /// of shape ``(num_selected, 3)``. The optional scalar inputs may be
  /// ``nullptr`` to indicate "not provided" (in which case their schema
  /// defaults apply).
  Tensor operator()(const Tensor &boxes, const Tensor &scores,
                    const Tensor *max_output_boxes_per_class, const Tensor *iou_threshold,
                    const Tensor *score_threshold, const Attributes &attrs) const;

  /// Output element layout differs fundamentally from any input
  /// (FLOAT/INT64 mix, distinct rank and shape), so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
