// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/object_detection/include_object_detection_cases.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

// Builds a single-node NonMaxSuppression NodeProto with the five canonical
// inputs and one output, matching the upstream
// ``onnx/backend/test/case/node/nonmaxsuppression.py`` graph layout.
NodeProto MakeNmsNode(int64_t center_point_box = 0) {
  NodeProto node;
  node.set_op_type("NonMaxSuppression");
  node.add_input("boxes");
  node.add_input("scores");
  node.add_input("max_output_boxes_per_class");
  node.add_input("iou_threshold");
  node.add_input("score_threshold");
  node.add_output("selected_indices");
  if (center_point_box != 0) {
    AddAttribute<int64_t>(node, "center_point_box", center_point_box);
  }
  return node;
}

// Registers one ``test_cc_nonmaxsuppression_*`` case. ``boxes_shape`` /
// ``boxes_values`` describe the boxes input and ``scores_shape`` /
// ``scores_values`` describe the scores input. ``expected`` is the expected
// (num_selected, 3) INT64 selected_indices output.
void RegisterCase(std::vector<TestCase> &registry, const std::string &case_name,
                  const std::vector<int64_t> &boxes_shape, const std::vector<float> &boxes_values,
                  const std::vector<int64_t> &scores_shape, const std::vector<float> &scores_values,
                  int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
                  const std::vector<int64_t> &expected, int64_t center_point_box = 0) {
  const OpsetId opset = DefaultOpset(11);
  NodeProto node = MakeNmsNode(center_point_box);

  Tensor boxes = Tensor::FromFloat("", boxes_shape, boxes_values);
  Tensor scores = Tensor::FromFloat("", scores_shape, scores_values);
  Tensor max_out = Tensor::FromInt64("", {1}, {max_output_boxes_per_class});
  Tensor iou_thr = Tensor::FromFloat("", {1}, {iou_threshold});
  Tensor score_thr = Tensor::FromFloat("", {1}, {score_threshold});

  const int64_t num_selected = static_cast<int64_t>(expected.size()) / 3;
  Tensor selected = Tensor::FromInt64("", {num_selected, 3}, expected);

  Expect(node, {boxes, scores, max_out, iou_thr, score_thr}, {selected}, case_name, {opset},
         "backend-test", registry);
}

} // namespace

// ---------------------------------------------------------------------------
// NonMaxSuppression — greedy IoU-based non-maximum suppression (since opset 10
// in the ai.onnx domain; the only opset-11 change is the documentation
// clarification that the IoU comparison is strictly greater than).
//
// The ten cases registered here mirror the upstream ONNX reference suite
// ``onnx/backend/test/case/node/nonmaxsuppression.py`` (data inputs and
// expected ``selected_indices`` outputs match ``test_nonmaxsuppression_*``
// exactly; each case is registered under ``test_cc_<upstream_name>``).
// ---------------------------------------------------------------------------
void RegisterNonMaxSuppressionCases(std::vector<TestCase> &registry) {
  // Shared boxes/scores fixtures used by the corner-format cases (1 batch, 6
  // boxes, 1 class — three "groups" of boxes, two of which overlap heavily).
  const std::vector<int64_t> kCornerBoxesShape = {1, 6, 4};
  const std::vector<float> kCornerBoxes = {
      0.0f, 0.0f,   1.0f, 1.0f,  //
      0.0f, 0.1f,   1.0f, 1.1f,  //
      0.0f, -0.1f,  1.0f, 0.9f,  //
      0.0f, 10.0f,  1.0f, 11.0f, //
      0.0f, 10.1f,  1.0f, 11.1f, //
      0.0f, 100.0f, 1.0f, 101.0f //
  };
  const std::vector<int64_t> kSingleClassScoresShape = {1, 1, 6};
  const std::vector<float> kSingleClassScores = {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f};

  // test_cc_nonmaxsuppression_suppress_by_IOU
  RegisterCase(registry, "test_cc_nonmaxsuppression_suppress_by_IOU", kCornerBoxesShape,
               kCornerBoxes, kSingleClassScoresShape, kSingleClassScores,
               /*max_out=*/3, /*iou=*/0.5f, /*score=*/0.0f, {0, 0, 3, 0, 0, 0, 0, 0, 5});

  // test_cc_nonmaxsuppression_suppress_by_IOU_and_scores
  RegisterCase(registry, "test_cc_nonmaxsuppression_suppress_by_IOU_and_scores", kCornerBoxesShape,
               kCornerBoxes, kSingleClassScoresShape, kSingleClassScores,
               /*max_out=*/3, /*iou=*/0.5f, /*score=*/0.4f, {0, 0, 3, 0, 0, 0});

  // test_cc_nonmaxsuppression_flipped_coordinates — same scores but several
  // boxes are flipped along x and/or y; selection must be unchanged.
  RegisterCase(registry, "test_cc_nonmaxsuppression_flipped_coordinates", kCornerBoxesShape,
               {
                   1.0f, 1.0f,   0.0f, 0.0f,  //
                   0.0f, 0.1f,   1.0f, 1.1f,  //
                   0.0f, 0.9f,   1.0f, -0.1f, //
                   0.0f, 10.0f,  1.0f, 11.0f, //
                   1.0f, 10.1f,  0.0f, 11.1f, //
                   1.0f, 101.0f, 0.0f, 100.0f //
               },
               kSingleClassScoresShape, kSingleClassScores,
               /*max_out=*/3, /*iou=*/0.5f, /*score=*/0.0f, {0, 0, 3, 0, 0, 0, 0, 0, 5});

  // test_cc_nonmaxsuppression_limit_output_size — cap selections at 2 per class.
  RegisterCase(registry, "test_cc_nonmaxsuppression_limit_output_size", kCornerBoxesShape,
               kCornerBoxes, kSingleClassScoresShape, kSingleClassScores,
               /*max_out=*/2, /*iou=*/0.5f, /*score=*/0.0f, {0, 0, 3, 0, 0, 0});

  // test_cc_nonmaxsuppression_single_box — a 1-box input should always pick
  // that single box.
  RegisterCase(registry, "test_cc_nonmaxsuppression_single_box", /*boxes_shape=*/{1, 1, 4},
               {0.0f, 0.0f, 1.0f, 1.0f}, /*scores_shape=*/{1, 1, 1}, {0.9f},
               /*max_out=*/3, /*iou=*/0.5f, /*score=*/0.0f, {0, 0, 0});

  // test_cc_nonmaxsuppression_identical_boxes — 10 identical boxes & scores
  // must collapse to a single selection.
  {
    std::vector<float> identical_boxes(10 * 4);
    for (int i = 0; i < 10; ++i) {
      identical_boxes[4 * i + 0] = 0.0f;
      identical_boxes[4 * i + 1] = 0.0f;
      identical_boxes[4 * i + 2] = 1.0f;
      identical_boxes[4 * i + 3] = 1.0f;
    }
    RegisterCase(registry, "test_cc_nonmaxsuppression_identical_boxes",
                 /*boxes_shape=*/{1, 10, 4}, identical_boxes,
                 /*scores_shape=*/{1, 1, 10}, std::vector<float>(10, 0.9f),
                 /*max_out=*/3, /*iou=*/0.5f, /*score=*/0.0f, {0, 0, 0});
  }

  // test_cc_nonmaxsuppression_center_point_box_format — Pytorch-style
  // [x_center, y_center, w, h] boxes; center_point_box=1.
  RegisterCase(registry, "test_cc_nonmaxsuppression_center_point_box_format", kCornerBoxesShape,
               {
                   0.5f, 0.5f,   1.0f, 1.0f, //
                   0.5f, 0.6f,   1.0f, 1.0f, //
                   0.5f, 0.4f,   1.0f, 1.0f, //
                   0.5f, 10.5f,  1.0f, 1.0f, //
                   0.5f, 10.6f,  1.0f, 1.0f, //
                   0.5f, 100.5f, 1.0f, 1.0f, //
               },
               kSingleClassScoresShape, kSingleClassScores,
               /*max_out=*/3, /*iou=*/0.5f, /*score=*/0.0f, {0, 0, 3, 0, 0, 0, 0, 0, 5},
               /*center_point_box=*/1);

  // test_cc_nonmaxsuppression_two_classes — same boxes, two classes with
  // identical scores; selection is independent per (batch, class) pair.
  RegisterCase(registry, "test_cc_nonmaxsuppression_two_classes", kCornerBoxesShape, kCornerBoxes,
               /*scores_shape=*/{1, 2, 6},
               {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f, //
                0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f},
               /*max_out=*/2, /*iou=*/0.5f, /*score=*/0.0f, {0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0});

  // test_cc_nonmaxsuppression_two_batches — same per-batch fixture stacked
  // along the batch axis.
  {
    std::vector<float> two_batches_boxes = kCornerBoxes;
    two_batches_boxes.insert(two_batches_boxes.end(), kCornerBoxes.begin(), kCornerBoxes.end());
    std::vector<float> two_batches_scores = kSingleClassScores;
    two_batches_scores.insert(two_batches_scores.end(), kSingleClassScores.begin(),
                              kSingleClassScores.end());
    RegisterCase(registry, "test_cc_nonmaxsuppression_two_batches",
                 /*boxes_shape=*/{2, 6, 4}, two_batches_boxes,
                 /*scores_shape=*/{2, 1, 6}, two_batches_scores,
                 /*max_out=*/2, /*iou=*/0.5f, /*score=*/0.0f, {0, 0, 3, 0, 0, 0, 1, 0, 3, 1, 0, 0});
  }

  // test_cc_nonmaxsuppression_iou_threshold_boundary — IoU == threshold must
  // NOT suppress (the comparison is strictly greater than).
  {
    const float exact_iou = static_cast<float>(0.25 / 1.75);
    RegisterCase(registry, "test_cc_nonmaxsuppression_iou_threshold_boundary",
                 /*boxes_shape=*/{1, 2, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.5f, 0.5f, 1.5f, 1.5f},
                 /*scores_shape=*/{1, 1, 2}, {0.9f, 0.8f},
                 /*max_out=*/3, /*iou=*/exact_iou, /*score=*/0.0f, {0, 0, 0, 0, 0, 1});
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
