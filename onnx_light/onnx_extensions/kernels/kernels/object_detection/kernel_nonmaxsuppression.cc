// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/object_detection/include_object_detection_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/temporary_buffer.h"
#include "onnx_core/runtime/runtime_context.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

constexpr int64_t kSelectedIndexTupleWidth = 3;
constexpr const char *kNonMaxSuppressionName = "kernel::NonMaxSuppression";

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
                                     const Attributes &attrs, RuntimeContext *rt) const {
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

  // ``max_per_pair`` may be non-positive when ``max_boxes <= 0``; the
  // ``max_selected <= 0`` guard below turns that into an empty output.
  const int64_t max_per_pair = std::min<int64_t>(spatial, max_boxes);
  const int64_t max_selected = num_batches * num_classes * max_per_pair;

  // Nothing can be selected (e.g. ``max_output_boxes_per_class <= 0`` or an
  // empty spatial/class/batch dimension), so the output is an empty
  // ``(0, kSelectedIndexTupleWidth)`` tensor and no working memory is needed.
  if (max_selected <= 0) {
    return (rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::INT64),
                                      {0, kSelectedIndexTupleWidth}, 0)
               : MakeOutputTensor(static_cast<int32_t>(DataType::INT64),
                                  {0, kSelectedIndexTupleWidth}, 0, nullptr));
  }

  RawBufferAllocator *allocator = rt != nullptr ? rt->execution_allocator() : nullptr;

  // Pre-compute the per-batch box coordinates in canonical corner form. The
  // scratch buffers below are drawn from the runtime allocator when one is
  // available, falling back to inline ``std::vector`` storage otherwise, so no
  // working memory is allocated outside the allocator.
  const std::size_t box_count =
      static_cast<std::size_t>(num_batches) * static_cast<std::size_t>(spatial);
  detail::TemporaryTypedBuffer<CornerBox> corner_boxes_buf(box_count, allocator,
                                                           kNonMaxSuppressionName);
  CornerBox *corner_boxes = corner_boxes_buf.data();
  for (int64_t b = 0; b < num_batches; ++b) {
    for (int64_t s = 0; s < spatial; ++s) {
      corner_boxes[static_cast<std::size_t>(b * spatial + s)] =
          ToCornerBox(pboxes + (b * spatial + s) * 4, attrs.center_point_box);
    }
  }

  // Flat list of [batch, class, box] triples.
  detail::TemporaryTypedBuffer<int64_t> selected_buf(
      static_cast<std::size_t>(max_selected * kSelectedIndexTupleWidth), allocator,
      kNonMaxSuppressionName);
  int64_t *selected = selected_buf.data();
  int64_t selected_count = 0; // number of int64 values written to ``selected``.

  // Candidate box indices for a single (batch, class) pair, sorted by score.
  detail::TemporaryTypedBuffer<int32_t> candidate_buf(static_cast<std::size_t>(spatial), allocator,
                                                      kNonMaxSuppressionName);
  int32_t *candidate_indices = candidate_buf.data();

  // Indices already selected for the current (batch, class) pair.
  detail::TemporaryTypedBuffer<int32_t> kept_buf(static_cast<std::size_t>(max_per_pair), allocator,
                                                 kNonMaxSuppressionName);
  int32_t *kept = kept_buf.data();

  for (int64_t b = 0; b < num_batches; ++b) {
    for (int64_t c = 0; c < num_classes; ++c) {
      const float *cs = pscores + (b * num_classes + c) * spatial;

      // Pre-filter by score_threshold and sort indices by descending score.
      int64_t candidate_count = 0;
      for (int64_t s = 0; s < spatial; ++s) {
        if (cs[s] > score_thr) {
          candidate_indices[static_cast<std::size_t>(candidate_count++)] = static_cast<int32_t>(s);
        }
      }
      std::stable_sort(candidate_indices, candidate_indices + candidate_count,
                       [cs](int32_t lhs, int32_t rhs) { return cs[lhs] > cs[rhs]; });

      int64_t kept_count = 0; // indices already selected for (b, c)
      for (int64_t ci = 0; ci < candidate_count; ++ci) {
        if (kept_count >= max_boxes) {
          break;
        }
        const int32_t idx = candidate_indices[static_cast<std::size_t>(ci)];
        const CornerBox &cand = corner_boxes[static_cast<std::size_t>(b * spatial + idx)];
        bool suppressed = false;
        for (int64_t ki = 0; ki < kept_count; ++ki) {
          const int32_t prev = kept[static_cast<std::size_t>(ki)];
          if (ComputeIoU(cand, corner_boxes[static_cast<std::size_t>(b * spatial + prev)]) >
              iou_thr) {
            suppressed = true;
            break;
          }
        }
        if (!suppressed) {
          kept[static_cast<std::size_t>(kept_count++)] = idx;
          selected[static_cast<std::size_t>(selected_count++)] = b;
          selected[static_cast<std::size_t>(selected_count++)] = c;
          selected[static_cast<std::size_t>(selected_count++)] = idx;
        }
      }
    }
  }

  const int64_t num_selected = selected_count / kSelectedIndexTupleWidth;
  const size_t selected_n_bytes = static_cast<size_t>(selected_count) * sizeof(int64_t);
  Tensor output =
      (rt ? rt->MakeOutputTensor(0, static_cast<int32_t>(DataType::INT64),
                                 {num_selected, kSelectedIndexTupleWidth}, selected_n_bytes)
          : MakeOutputTensor(static_cast<int32_t>(DataType::INT64),
                             {num_selected, kSelectedIndexTupleWidth}, selected_n_bytes, nullptr));
  if (selected_count > 0) {
    std::memcpy(output.mutable_bytes(), selected, selected_n_bytes);
  }
  return output;
}

void NonMaxSuppression::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  EXT_ENFORCE_INVALID(!(node.input_size() < 2 || node.input_size() > 5), "RunNode: op '",
                      node.op_type(), "' expects between 2 and 5 inputs, got ", node.input_size(),
                      ".");
  RequireOutputCount(node, 1);
  const Tensor &boxes = GetInput(node, 0, rt.tensors());
  const Tensor &scores = GetInput(node, 1, rt.tensors());
  const Tensor *max_output_boxes_per_class = GetOptionalInput(node, 2, rt.tensors());
  const Tensor *iou_threshold = GetOptionalInput(node, 3, rt.tensors());
  const Tensor *score_threshold = GetOptionalInput(node, 4, rt.tensors());
  // For ONNX NonMaxSuppression (opset 10+), this runtime path uses the
  // single schema attribute center_point_box (default 0).
  onnx_kernels::kernel::NonMaxSuppression::Attributes attrs;
  attrs.center_point_box = GetAttributeIntOrDefault(node, "center_point_box", 0);
  onnx_kernels::kernel::NonMaxSuppression k(rt.kernel_ctx());
  SetOutput(node, 0,
            k(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, attrs),
            rt.tensors());
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel
