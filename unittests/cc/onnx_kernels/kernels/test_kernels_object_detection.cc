// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/object_detection/include_object_detection_kernels.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::NonMaxSuppression;
using onnx_kernels::kernel::RoiAlign;

namespace Test {

namespace {

Tensor MakeFeatureMap() {
  std::vector<float> values(100);
  for (int i = 0; i < 100; ++i) {
    values[i] = static_cast<float>(i) / 100.0f;
  }
  return Tensor::FromFloat("", {1, 1, 10, 10}, values);
}

} // namespace

TEST(KernelClass, RoiAlignAvgProducesExpectedShapeAndRange) {
  const KernelContext ctx{DefaultOpset(16)};
  RoiAlign roialign{ctx};
  Tensor x = MakeFeatureMap();
  Tensor rois = Tensor::FromFloat("", {2, 4}, {0.0f, 0.0f, 9.0f, 9.0f, 2.0f, 2.0f, 7.0f, 7.0f});
  Tensor batch_indices = Tensor::FromInt64("", {2}, {0, 0});

  RoiAlign::Attributes attrs;
  attrs.mode = "avg";
  attrs.output_height = 5;
  attrs.output_width = 5;
  attrs.sampling_ratio = 2;
  attrs.spatial_scale = 1.0f;
  attrs.coordinate_transformation_mode = "half_pixel";
  Tensor y = roialign(x, rois, batch_indices, attrs);

  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  const std::vector<int64_t> expected_shape = {2, 1, 5, 5};
  EXPECT_EQ(y.shape, expected_shape);
  ASSERT_EQ(y.element_count(), 2 * 1 * 5 * 5);

  // The interior roi (2,2)-(7,7) samples values bounded by the feature map
  // values 0..0.99, so every output element of the second RoI must fall in
  // that range.
  const float *py = y.AsFloat();
  for (int64_t i = 25; i < 50; ++i) {
    EXPECT_GE(py[i], 0.0f);
    EXPECT_LE(py[i], 1.0f);
  }
  // The first RoI covers the full map; the center pooled value should be
  // within the full input value range and roughly near the middle of it.
  const float center = py[2 * 5 + 2];
  EXPECT_GE(center, 0.0f);
  EXPECT_LE(center, 1.0f);
  EXPECT_NEAR(center, 0.5f, 0.2f);
}

TEST(KernelClass, RoiAlignMaxPickedSampleNotSmallerThanAvg) {
  const KernelContext ctx{DefaultOpset(16)};
  RoiAlign roialign{ctx};
  Tensor x = MakeFeatureMap();
  Tensor rois = Tensor::FromFloat("", {1, 4}, {0.0f, 0.0f, 9.0f, 9.0f});
  Tensor batch_indices = Tensor::FromInt64("", {1}, {0});

  RoiAlign::Attributes attrs_avg;
  attrs_avg.mode = "avg";
  attrs_avg.output_height = 2;
  attrs_avg.output_width = 2;
  attrs_avg.sampling_ratio = 2;
  attrs_avg.spatial_scale = 1.0f;
  attrs_avg.coordinate_transformation_mode = "output_half_pixel";
  Tensor y_avg = roialign(x, rois, batch_indices, attrs_avg);

  RoiAlign::Attributes attrs_max = attrs_avg;
  attrs_max.mode = "max";
  Tensor y_max = roialign(x, rois, batch_indices, attrs_max);

  ASSERT_EQ(y_avg.element_count(), 4);
  ASSERT_EQ(y_max.element_count(), 4);
  const float *pa = y_avg.AsFloat();
  const float *pm = y_max.AsFloat();
  for (int i = 0; i < 4; ++i) {
    EXPECT_GE(pm[i], pa[i] - 1e-6f);
    EXPECT_GE(pm[i], 0.0f);
    EXPECT_LE(pm[i], 1.0f);
  }
}

TEST(KernelClass, RoiAlignRejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(16)};
  RoiAlign roialign{ctx};
  Tensor x = MakeFeatureMap();
  Tensor rois = Tensor::FromFloat("", {1, 4}, {0.0f, 0.0f, 9.0f, 9.0f});
  Tensor batch_indices = Tensor::FromInt64("", {1}, {0});
  RoiAlign::Attributes attrs;
  attrs.output_height = 2;
  attrs.output_width = 2;

  // X must be FLOAT.
  Tensor bad_x = Tensor::FromInt32("", {1, 1, 10, 10}, std::vector<int32_t>(100, 0));
  EXPECT_THROW(roialign(bad_x, rois, batch_indices, attrs), std::invalid_argument);

  // rois must be 2-D with shape (num_rois, 4).
  Tensor bad_rois = Tensor::FromFloat("", {1, 3}, {0.0f, 0.0f, 9.0f});
  EXPECT_THROW(roialign(x, bad_rois, batch_indices, attrs), std::invalid_argument);

  // batch_indices length must match num_rois.
  Tensor bad_bi = Tensor::FromInt64("", {2}, {0, 0});
  EXPECT_THROW(roialign(x, rois, bad_bi, attrs), std::invalid_argument);

  // Unknown mode.
  RoiAlign::Attributes bad_mode = attrs;
  bad_mode.mode = "median";
  EXPECT_THROW(roialign(x, rois, batch_indices, bad_mode), std::invalid_argument);

  // Out-of-range batch index.
  Tensor oob_bi = Tensor::FromInt64("", {1}, {5});
  EXPECT_THROW(roialign(x, rois, oob_bi, attrs), std::invalid_argument);

  // Mismatched preallocated output shape.
  Tensor bad_out("", static_cast<int32_t>(onnx_kernels::DataType::FLOAT), {1, 1, 3, 3},
                 std::vector<uint8_t>(9 * sizeof(float)));
  EXPECT_THROW(roialign(x, rois, batch_indices, attrs, bad_out), std::invalid_argument);
}

TEST(KernelClass, NonMaxSuppressionSuppressByIoU) {
  const KernelContext ctx{onnx_backend_test::DefaultOpset(11)};
  NonMaxSuppression nms{ctx};
  // Mirror the upstream test_nonmaxsuppression_suppress_by_IOU fixture.
  Tensor boxes = Tensor::FromFloat("", {1, 6, 4}, {0.f, 0.f,   1.f, 1.f,   0.f, 0.1f,  1.f, 1.1f,
                                                   0.f, -0.1f, 1.f, 0.9f,  0.f, 10.f,  1.f, 11.f,
                                                   0.f, 10.1f, 1.f, 11.1f, 0.f, 100.f, 1.f, 101.f});
  Tensor scores = Tensor::FromFloat("", {1, 1, 6}, {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f});
  Tensor max_out = Tensor::FromInt64("", {1}, {3});
  Tensor iou = Tensor::FromFloat("", {1}, {0.5f});
  Tensor score = Tensor::FromFloat("", {1}, {0.0f});
  NonMaxSuppression::Attributes attrs;
  Tensor y = nms(boxes, scores, &max_out, &iou, &score, attrs);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
  EXPECT_EQ(y.shape, (std::vector<int64_t>{3, 3}));
  const int64_t *py = y.AsInt64();
  const std::vector<int64_t> expected = {0, 0, 3, 0, 0, 0, 0, 0, 5};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(py[i], expected[i]);
  }
}

TEST(KernelClass, NonMaxSuppressionDefaultsAndOptionalInputs) {
  const KernelContext ctx{onnx_backend_test::DefaultOpset(11)};
  NonMaxSuppression nms{ctx};
  // Two non-overlapping boxes; with no max_output_boxes_per_class (default 0)
  // the output must be empty.
  Tensor boxes =
      Tensor::FromFloat("", {1, 2, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 10.0f, 1.0f, 11.0f});
  Tensor scores = Tensor::FromFloat("", {1, 1, 2}, {0.9f, 0.8f});
  NonMaxSuppression::Attributes attrs;

  Tensor y_empty = nms(boxes, scores, /*max_out=*/nullptr, /*iou=*/nullptr,
                       /*score=*/nullptr, attrs);
  EXPECT_EQ(y_empty.shape, (std::vector<int64_t>{0, 3}));

  // With max=10 both boxes must be selected (they don't overlap).
  Tensor max_out = Tensor::FromInt64("", {1}, {10});
  Tensor y_two = nms(boxes, scores, &max_out, /*iou=*/nullptr, /*score=*/nullptr, attrs);
  EXPECT_EQ(y_two.shape, (std::vector<int64_t>{2, 3}));
  const int64_t *py = y_two.AsInt64();
  // Higher-scored box first.
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 0);
  EXPECT_EQ(py[2], 0);
  EXPECT_EQ(py[3], 0);
  EXPECT_EQ(py[4], 0);
  EXPECT_EQ(py[5], 1);
}

TEST(KernelClass, NonMaxSuppressionRejectsBadInputs) {
  const KernelContext ctx{onnx_backend_test::DefaultOpset(11)};
  NonMaxSuppression nms{ctx};
  Tensor boxes = Tensor::FromFloat("", {1, 1, 4}, {0.0f, 0.0f, 1.0f, 1.0f});
  Tensor scores = Tensor::FromFloat("", {1, 1, 1}, {0.9f});
  NonMaxSuppression::Attributes attrs;

  // boxes must be 3-D with last dim == 4.
  Tensor bad_boxes = Tensor::FromFloat("", {1, 1, 3}, {0.0f, 0.0f, 1.0f});
  EXPECT_THROW(nms(bad_boxes, scores, nullptr, nullptr, nullptr, attrs), std::invalid_argument);

  // boxes and scores must agree on batch.
  Tensor mismatch_scores = Tensor::FromFloat("", {2, 1, 1}, {0.9f, 0.8f});
  EXPECT_THROW(nms(boxes, mismatch_scores, nullptr, nullptr, nullptr, attrs),
               std::invalid_argument);

  // boxes must be FLOAT.
  Tensor int_boxes = Tensor::FromInt32("", {1, 1, 4}, std::vector<int32_t>(4, 0));
  EXPECT_THROW(nms(int_boxes, scores, nullptr, nullptr, nullptr, attrs), std::invalid_argument);

  // max_output_boxes_per_class must be INT64.
  Tensor bad_max = Tensor::FromInt32("", {1}, {1});
  EXPECT_THROW(nms(boxes, scores, &bad_max, nullptr, nullptr, attrs), std::invalid_argument);
}

} // namespace Test
