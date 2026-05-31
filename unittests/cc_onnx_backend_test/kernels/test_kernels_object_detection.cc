// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/object_detection/include_object_detection_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::RoiAlign;

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

TEST(BackendKernelClass, RoiAlignAvgProducesExpectedShapeAndRange) {
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

  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
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

TEST(BackendKernelClass, RoiAlignMaxPickedSampleNotSmallerThanAvg) {
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

TEST(BackendKernelClass, RoiAlignRejectsBadInputs) {
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
  Tensor bad_out("", static_cast<int32_t>(onnx_backend_test::DataType::FLOAT), {1, 1, 3, 3},
                 std::vector<uint8_t>(9 * sizeof(float)));
  EXPECT_THROW(roialign(x, rois, batch_indices, attrs, bad_out), std::invalid_argument);
}

} // namespace Test
