// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

void ValidateInputs(const Tensor &x, const Tensor &rois, const MaxRoiPool::Attributes &attrs) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::MaxRoiPool: X must be FLOAT.");
  EXT_ENFORCE_INVALID(rois.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::MaxRoiPool: rois must be FLOAT.");
  EXT_ENFORCE_INVALID(x.shape.size() == 4, "kernel::MaxRoiPool: X must be 4-D (N, C, H, W).");
  EXT_ENFORCE_INVALID(rois.shape.size() == 2 && rois.shape[1] == 5,
                      "kernel::MaxRoiPool: rois must be 2-D with shape (num_rois, 5).");
  EXT_ENFORCE_INVALID(attrs.pooled_shape.size() == 2 && attrs.pooled_shape[0] > 0 &&
                          attrs.pooled_shape[1] > 0,
                      "kernel::MaxRoiPool: attribute 'pooled_shape' must contain two positive "
                      "values (height, width).");
}

} // namespace

Tensor MaxRoiPool::operator()(const Tensor &x, const Tensor &rois, const Attributes &attrs) const {
  ValidateInputs(x, rois, attrs);
  const int64_t num_rois = rois.shape[0];
  const int64_t C = x.shape[1];
  const std::vector<int64_t> out_shape = {num_rois, C, attrs.pooled_shape[0],
                                          attrs.pooled_shape[1]};
  int64_t out_elements = 1;
  for (int64_t d : out_shape) {
    out_elements *= d;
  }
  Tensor out("", x.data_type, out_shape,
             std::vector<uint8_t>(static_cast<size_t>(out_elements) * sizeof(float)));
  (*this)(x, rois, attrs, out);
  return out;
}

void MaxRoiPool::operator()(const Tensor &x, const Tensor &rois, const Attributes &attrs,
                            Tensor &output) const {
  ValidateInputs(x, rois, attrs);

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t H = x.shape[2];
  const int64_t W = x.shape[3];
  const int64_t num_rois = rois.shape[0];
  const int64_t pooled_h = attrs.pooled_shape[0];
  const int64_t pooled_w = attrs.pooled_shape[1];

  const std::vector<int64_t> expected_shape = {num_rois, C, pooled_h, pooled_w};
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::MaxRoiPool preallocated output must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::MaxRoiPool preallocated output shape must be "
                      "(num_rois, C, pooled_shape[0], pooled_shape[1]).");
  const size_t expected_bytes =
      static_cast<size_t>(num_rois * C * pooled_h * pooled_w) * sizeof(float);
  EXT_ENFORCE_INVALID(
      output.data.size() == expected_bytes,
      "kernel::MaxRoiPool preallocated output buffer has unexpected size in bytes.");

  const float *px = x.AsFloat();
  const float *prois = rois.AsFloat();
  float *py = output.AsFloat();

  for (int64_t r = 0; r < num_rois; ++r) {
    const int64_t batch_idx = static_cast<int64_t>(std::round(prois[r * 5 + 0]));
    EXT_ENFORCE_INVALID(batch_idx >= 0 && batch_idx < N,
                        "kernel::MaxRoiPool: rois batch_id out of range [0, N).");

    // Scale the RoI corner coordinates and round to the nearest integer
    // feature-map cell — this matches the canonical Fast R-CNN reference
    // implementation also used by onnxruntime and Caffe2.
    const int64_t roi_start_w = static_cast<int64_t>(
        std::round(prois[r * 5 + 1] * static_cast<double>(attrs.spatial_scale)));
    const int64_t roi_start_h = static_cast<int64_t>(
        std::round(prois[r * 5 + 2] * static_cast<double>(attrs.spatial_scale)));
    const int64_t roi_end_w = static_cast<int64_t>(
        std::round(prois[r * 5 + 3] * static_cast<double>(attrs.spatial_scale)));
    const int64_t roi_end_h = static_cast<int64_t>(
        std::round(prois[r * 5 + 4] * static_cast<double>(attrs.spatial_scale)));

    // The Fast R-CNN definition clamps RoI sizes to at least 1 cell to
    // avoid degenerate (empty) RoIs.
    const int64_t roi_height = std::max<int64_t>(roi_end_h - roi_start_h + 1, 1);
    const int64_t roi_width = std::max<int64_t>(roi_end_w - roi_start_w + 1, 1);

    const double bin_size_h = static_cast<double>(roi_height) / static_cast<double>(pooled_h);
    const double bin_size_w = static_cast<double>(roi_width) / static_cast<double>(pooled_w);

    for (int64_t c = 0; c < C; ++c) {
      const float *plane = px + (batch_idx * C + c) * H * W;
      float *out_plane = py + (r * C + c) * pooled_h * pooled_w;
      for (int64_t ph = 0; ph < pooled_h; ++ph) {
        int64_t hstart =
            static_cast<int64_t>(std::floor(static_cast<double>(ph) * bin_size_h)) + roi_start_h;
        int64_t hend =
            static_cast<int64_t>(std::ceil(static_cast<double>(ph + 1) * bin_size_h)) + roi_start_h;
        hstart = std::min<int64_t>(std::max<int64_t>(hstart, 0), H);
        hend = std::min<int64_t>(std::max<int64_t>(hend, 0), H);
        for (int64_t pw = 0; pw < pooled_w; ++pw) {
          int64_t wstart =
              static_cast<int64_t>(std::floor(static_cast<double>(pw) * bin_size_w)) + roi_start_w;
          int64_t wend = static_cast<int64_t>(std::ceil(static_cast<double>(pw + 1) * bin_size_w)) +
                         roi_start_w;
          wstart = std::min<int64_t>(std::max<int64_t>(wstart, 0), W);
          wend = std::min<int64_t>(std::max<int64_t>(wend, 0), W);

          const bool is_empty = (hend <= hstart) || (wend <= wstart);
          float val = is_empty ? 0.0f : -std::numeric_limits<float>::infinity();
          for (int64_t h = hstart; h < hend; ++h) {
            for (int64_t w = wstart; w < wend; ++w) {
              val = std::max(val, plane[h * W + w]);
            }
          }
          out_plane[ph * pooled_w + pw] = val;
        }
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
