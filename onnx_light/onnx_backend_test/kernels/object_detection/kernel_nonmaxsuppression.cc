// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/object_detection/include_object_detection_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

namespace {

// Returns (y1, x1, y2, x2) in canonical "corner" order regardless of
// ``center_point_box`` and regardless of whether the upstream y1<y2/x1<x2
// ordering holds (the schema explicitly allows flipped coordinates).
struct CornerBox {
  float y1, x1, y2, x2;
};

CornerBox ToCornerBox(const float *box, int64_t center_point_box) {
  if (center_point_box == 1) {
    // [x_center, y_center, width, height] (Pytorch convention).
    const float xc = box[0];
    const float yc = box[1];
    const float w = box[2];
    const float h = box[3];
    return {yc - 0.5f * h, xc - 0.5f * w, yc + 0.5f * h, xc + 0.5f * w};
  }
  // [y1, x1, y2, x2] corner format; coordinates may be flipped, so
  // normalize them so y1<=y2 and x1<=x2 to make the IoU computation
  // well-defined.
  float y1 = box[0];
  float x1 = box[1];
  float y2 = box[2];
  float x2 = box[3];
  if (y1 > y2) {
    std::swap(y1, y2);
  }
  if (x1 > x2) {
    std::swap(x1, x2);
  }
  return {y1, x1, y2, x2};
}

float ComputeIoU(const CornerBox &a, const CornerBox &b) {
  const float inter_y1 = std::max(a.y1, b.y1);
  const float inter_x1 = std::max(a.x1, b.x1);
  const float inter_y2 = std::min(a.y2, b.y2);
  const float inter_x2 = std::min(a.x2, b.x2);
  const float ih = inter_y2 - inter_y1;
  const float iw = inter_x2 - inter_x1;
  if (ih <= 0.0f || iw <= 0.0f) {
    return 0.0f;
  }
  const float inter = ih * iw;
  const float area_a = (a.y2 - a.y1) * (a.x2 - a.x1);
  const float area_b = (b.y2 - b.y1) * (b.x2 - b.x1);
  const float uni = area_a + area_b - inter;
  if (uni <= 0.0f) {
    return 0.0f;
  }
  return inter / uni;
}

void ValidateInputs(const Tensor &boxes, const Tensor &scores,
                    const Tensor *max_output_boxes_per_class, const Tensor *iou_threshold,
                    const Tensor *score_threshold) {
  EXT_ENFORCE_INVALID(boxes.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::NonMaxSuppression: boxes must be FLOAT.");
  EXT_ENFORCE_INVALID(scores.data_type == static_cast<int32_t>(DataType::FLOAT),
                      "kernel::NonMaxSuppression: scores must be FLOAT.");
  EXT_ENFORCE_INVALID(boxes.shape.size() == 3 && boxes.shape[2] == 4,
                      "kernel::NonMaxSuppression: boxes must be 3-D with shape "
                      "(num_batches, spatial_dimension, 4).");
  EXT_ENFORCE_INVALID(scores.shape.size() == 3,
                      "kernel::NonMaxSuppression: scores must be 3-D with shape "
                      "(num_batches, num_classes, spatial_dimension).");
  EXT_ENFORCE_INVALID(boxes.shape[0] == scores.shape[0],
                      "kernel::NonMaxSuppression: boxes and scores must agree on num_batches.");
  EXT_ENFORCE_INVALID(boxes.shape[1] == scores.shape[2],
                      "kernel::NonMaxSuppression: boxes and scores must agree on "
                      "spatial_dimension.");
  if (max_output_boxes_per_class != nullptr) {
    EXT_ENFORCE_INVALID(max_output_boxes_per_class->data_type ==
                            static_cast<int32_t>(DataType::INT64),
                        "kernel::NonMaxSuppression: max_output_boxes_per_class must be INT64.");
    EXT_ENFORCE_INVALID(max_output_boxes_per_class->element_count() == 1,
                        "kernel::NonMaxSuppression: max_output_boxes_per_class must be a scalar "
                        "(or a 1-element 1-D tensor).");
  }
  if (iou_threshold != nullptr) {
    EXT_ENFORCE_INVALID(iou_threshold->data_type == static_cast<int32_t>(DataType::FLOAT),
                        "kernel::NonMaxSuppression: iou_threshold must be FLOAT.");
    EXT_ENFORCE_INVALID(iou_threshold->element_count() == 1,
                        "kernel::NonMaxSuppression: iou_threshold must be a scalar.");
  }
  if (score_threshold != nullptr) {
    EXT_ENFORCE_INVALID(score_threshold->data_type == static_cast<int32_t>(DataType::FLOAT),
                        "kernel::NonMaxSuppression: score_threshold must be FLOAT.");
    EXT_ENFORCE_INVALID(score_threshold->element_count() == 1,
                        "kernel::NonMaxSuppression: score_threshold must be a scalar.");
  }
}

} // namespace

Tensor NonMaxSuppression::operator()(const Tensor &boxes, const Tensor &scores,
                                     const Tensor *max_output_boxes_per_class,
                                     const Tensor *iou_threshold, const Tensor *score_threshold,
                                     const Attributes &attrs) const {
  ValidateInputs(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);

  const int64_t num_batches = boxes.shape[0];
  const int64_t spatial = boxes.shape[1];
  const int64_t num_classes = scores.shape[1];

  const int64_t max_boxes =
      max_output_boxes_per_class != nullptr ? max_output_boxes_per_class->AsInt64()[0] : 0;
  const float iou_thr = iou_threshold != nullptr ? iou_threshold->AsFloat()[0] : 0.0f;
  const float score_thr = score_threshold != nullptr ? score_threshold->AsFloat()[0]
                                                     : -std::numeric_limits<float>::infinity();

  const float *pboxes = boxes.AsFloat();
  const float *pscores = scores.AsFloat();

  // Pre-compute the per-batch box coordinates in canonical corner form.
  std::vector<std::vector<CornerBox>> corner_boxes(static_cast<size_t>(num_batches));
  for (int64_t b = 0; b < num_batches; ++b) {
    corner_boxes[b].resize(static_cast<size_t>(spatial));
    for (int64_t s = 0; s < spatial; ++s) {
      corner_boxes[b][s] = ToCornerBox(pboxes + (b * spatial + s) * 4, attrs.center_point_box);
    }
  }

  std::vector<int64_t> selected; // flat list of [batch, class, box] triples.
  std::vector<int32_t> candidate_indices;
  candidate_indices.reserve(static_cast<size_t>(spatial));

  for (int64_t b = 0; b < num_batches; ++b) {
    for (int64_t c = 0; c < num_classes; ++c) {
      const float *cs = pscores + (b * num_classes + c) * spatial;

      // Pre-filter by score_threshold and sort indices by descending score.
      candidate_indices.clear();
      for (int64_t s = 0; s < spatial; ++s) {
        if (cs[s] > score_thr) {
          candidate_indices.push_back(static_cast<int32_t>(s));
        }
      }
      std::stable_sort(candidate_indices.begin(), candidate_indices.end(),
                       [cs](int32_t lhs, int32_t rhs) { return cs[lhs] > cs[rhs]; });

      std::vector<int32_t> kept; // indices already selected for (b, c)
      kept.reserve(candidate_indices.size());
      for (int32_t idx : candidate_indices) {
        if (static_cast<int64_t>(kept.size()) >= max_boxes) {
          break;
        }
        const CornerBox &cand = corner_boxes[b][idx];
        bool suppressed = false;
        for (int32_t prev : kept) {
          if (ComputeIoU(cand, corner_boxes[b][prev]) > iou_thr) {
            suppressed = true;
            break;
          }
        }
        if (!suppressed) {
          kept.push_back(idx);
          selected.push_back(b);
          selected.push_back(c);
          selected.push_back(idx);
        }
      }
    }
  }

  const int64_t num_selected = static_cast<int64_t>(selected.size()) / 3;
  std::vector<int64_t> data(selected.begin(), selected.end());
  return Tensor::FromInt64("", {num_selected, 3}, data);
}

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
