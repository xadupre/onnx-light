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
class RoiAlign {
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

  explicit RoiAlign(const KernelContext &ctx) : ctx_(ctx) {}

  Tensor operator()(const Tensor &x, const Tensor &rois, const Tensor &batch_indices,
                    const Attributes &attrs) const;
  void operator()(const Tensor &x, const Tensor &rois, const Tensor &batch_indices,
                  const Attributes &attrs, Tensor &output) const;

  /// Output element layout (num_rois, C, output_height, output_width)
  /// fundamentally differs from any input, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  const KernelContext &ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
