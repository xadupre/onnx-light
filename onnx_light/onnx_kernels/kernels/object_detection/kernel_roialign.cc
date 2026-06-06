// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/object_detection/include_object_detection_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

// Performs bilinear interpolation of the input feature map of size H x W at
// floating-point coordinate (y, x). Returns 0 when (y, x) falls outside the
// [-1, H] x [-1, W] sampling window used by the canonical ONNX RoiAlign
// reference (which treats out-of-range samples as background).
float BilinearInterpolate(const float *plane, int64_t H, int64_t W, float y, float x) {
  if (y < -1.0f || y > static_cast<float>(H) || x < -1.0f || x > static_cast<float>(W)) {
    return 0.0f;
  }
  y = std::max(y, 0.0f);
  x = std::max(x, 0.0f);
  int64_t y_low = static_cast<int64_t>(std::floor(y));
  int64_t x_low = static_cast<int64_t>(std::floor(x));
  int64_t y_high = y_low + 1;
  int64_t x_high = x_low + 1;
  if (y_low >= H - 1) {
    y_low = y_high = H - 1;
    y = static_cast<float>(y_low);
  }
  if (x_low >= W - 1) {
    x_low = x_high = W - 1;
    x = static_cast<float>(x_low);
  }
  const float ly = y - static_cast<float>(y_low);
  const float lx = x - static_cast<float>(x_low);
  const float hy = 1.0f - ly;
  const float hx = 1.0f - lx;
  const float v1 = plane[y_low * W + x_low];
  const float v2 = plane[y_low * W + x_high];
  const float v3 = plane[y_high * W + x_low];
  const float v4 = plane[y_high * W + x_high];
  const float w1 = hy * hx;
  const float w2 = hy * lx;
  const float w3 = ly * hx;
  const float w4 = ly * lx;
  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

void ValidateInputs(const Tensor &x, const Tensor &rois, const Tensor &batch_indices,
                    const RoiAlign::Attributes &attrs) {
  EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RoiAlign: X must be FLOAT.");
  EXT_ENFORCE_INVALID(rois.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RoiAlign: rois must be FLOAT.");
  EXT_ENFORCE_INVALID(batch_indices.data_type == static_cast<int32_t>(DataType::INT64),
                      "kernel::RoiAlign: batch_indices must be INT64.");
  EXT_ENFORCE_INVALID(x.shape.size() == 4, "kernel::RoiAlign: X must be 4-D (N, C, H, W).");
  EXT_ENFORCE_INVALID(rois.shape.size() == 2 && rois.shape[1] == 4,
                      "kernel::RoiAlign: rois must be 2-D with shape (num_rois, 4).");
  EXT_ENFORCE_INVALID(batch_indices.shape.size() == 1 && batch_indices.shape[0] == rois.shape[0],
                      "kernel::RoiAlign: batch_indices must be 1-D with shape (num_rois,).");
  EXT_ENFORCE_INVALID(attrs.output_height > 0 && attrs.output_width > 0,
                      "kernel::RoiAlign: output_height and output_width must be positive.");
  EXT_ENFORCE_INVALID(attrs.mode == "avg" || attrs.mode == "max",
                      "kernel::RoiAlign: mode must be 'avg' or 'max'.");
  EXT_ENFORCE_INVALID(attrs.coordinate_transformation_mode == "half_pixel" ||
                          attrs.coordinate_transformation_mode == "output_half_pixel",
                      "kernel::RoiAlign: coordinate_transformation_mode must be 'half_pixel' or "
                      "'output_half_pixel'.");
}

} // namespace

Tensor RoiAlign::operator()(const Tensor &x, const Tensor &rois, const Tensor &batch_indices,
                            const Attributes &attrs) const {
  ValidateInputs(x, rois, batch_indices, attrs);
  const int64_t num_rois = rois.shape[0];
  const int64_t C = x.shape[1];
  const std::vector<int64_t> out_shape = {num_rois, C, attrs.output_height, attrs.output_width};
  int64_t out_elements = 1;
  for (int64_t d : out_shape) {
    out_elements *= d;
  }
  Tensor out("", x.data_type, out_shape,
             std::vector<uint8_t>(static_cast<size_t>(out_elements) * sizeof(float)));
  (*this)(x, rois, batch_indices, attrs, out);
  return out;
}

void RoiAlign::operator()(const Tensor &x, const Tensor &rois, const Tensor &batch_indices,
                          const Attributes &attrs, Tensor &output) const {
  ValidateInputs(x, rois, batch_indices, attrs);

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];
  const int64_t H = x.shape[2];
  const int64_t W = x.shape[3];
  const int64_t num_rois = rois.shape[0];
  const int64_t out_h = attrs.output_height;
  const int64_t out_w = attrs.output_width;

  const std::vector<int64_t> expected_shape = {num_rois, C, out_h, out_w};
  EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::RoiAlign preallocated output must be FLOAT.");
  EXT_ENFORCE_INVALID(output.shape == expected_shape,
                      "kernel::RoiAlign preallocated output shape must be "
                      "(num_rois, C, output_height, output_width).");
  const size_t expected_bytes = static_cast<size_t>(num_rois * C * out_h * out_w) * sizeof(float);
  EXT_ENFORCE_INVALID(output.data.size() == expected_bytes,
                      "kernel::RoiAlign preallocated output buffer has unexpected size in bytes.");

  const float *px = x.AsFloat();
  const float *prois = rois.AsFloat();
  const int64_t *pbi = batch_indices.AsInt64();
  float *py = output.AsFloat();

  // Offset applied to the (already scaled) roi coordinates. Opset 16+
  // distinguishes "half_pixel" (offset -0.5) and "output_half_pixel"
  // (offset 0); opset 10 has no such attribute and always uses 0.
  const float roi_offset = attrs.coordinate_transformation_mode == "half_pixel" ? -0.5f : 0.0f;

  for (int64_t r = 0; r < num_rois; ++r) {
    const int64_t batch_idx = pbi[r];
    EXT_ENFORCE_INVALID(batch_idx >= 0 && batch_idx < N,
                        "kernel::RoiAlign: batch_indices entry out of range [0, N).");
    const float roi_x1 = prois[r * 4 + 0] * attrs.spatial_scale + roi_offset;
    const float roi_y1 = prois[r * 4 + 1] * attrs.spatial_scale + roi_offset;
    const float roi_x2 = prois[r * 4 + 2] * attrs.spatial_scale + roi_offset;
    const float roi_y2 = prois[r * 4 + 3] * attrs.spatial_scale + roi_offset;

    float roi_width = roi_x2 - roi_x1;
    float roi_height = roi_y2 - roi_y1;
    if (attrs.coordinate_transformation_mode == "output_half_pixel") {
      // Legacy behaviour: clamp the roi size to be at least 1 pixel.
      roi_width = std::max(roi_width, 1.0f);
      roi_height = std::max(roi_height, 1.0f);
    }

    const float bin_size_h = roi_height / static_cast<float>(out_h);
    const float bin_size_w = roi_width / static_cast<float>(out_w);

    const int64_t roi_bin_grid_h =
        attrs.sampling_ratio > 0
            ? attrs.sampling_ratio
            : static_cast<int64_t>(std::ceil(roi_height / static_cast<float>(out_h)));
    const int64_t roi_bin_grid_w =
        attrs.sampling_ratio > 0
            ? attrs.sampling_ratio
            : static_cast<int64_t>(std::ceil(roi_width / static_cast<float>(out_w)));
    const int64_t count = std::max<int64_t>(roi_bin_grid_h * roi_bin_grid_w, 1);

    for (int64_t c = 0; c < C; ++c) {
      const float *plane = px + (batch_idx * C + c) * H * W;
      float *out_plane = py + (r * C + c) * out_h * out_w;
      for (int64_t ph = 0; ph < out_h; ++ph) {
        for (int64_t pw = 0; pw < out_w; ++pw) {
          float val = attrs.mode == "max" ? -std::numeric_limits<float>::infinity() : 0.0f;
          bool any_sample = false;
          for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
            const float y =
                roi_y1 + static_cast<float>(ph) * bin_size_h +
                (static_cast<float>(iy) + 0.5f) * bin_size_h / static_cast<float>(roi_bin_grid_h);
            for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
              const float xv =
                  roi_x1 + static_cast<float>(pw) * bin_size_w +
                  (static_cast<float>(ix) + 0.5f) * bin_size_w / static_cast<float>(roi_bin_grid_w);
              const float sample = BilinearInterpolate(plane, H, W, y, xv);
              if (attrs.mode == "max") {
                val = std::max(val, sample);
              } else {
                val += sample;
              }
              any_sample = true;
            }
          }
          if (attrs.mode == "avg") {
            val /= static_cast<float>(count);
          } else if (!any_sample) {
            val = 0.0f;
          }
          out_plane[ph * out_w + pw] = val;
        }
      }
    }
  }
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
