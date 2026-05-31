// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::Cast;
using onnx_backend_test::kernel::CastLike;
using onnx_backend_test::kernel::Concat;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::Squeeze;
using onnx_backend_test::kernel::Unsqueeze;

namespace Test {

TEST(BackendKernelClass, ConcatClassConcatenatesAxis0) {
  const KernelContext ctx{DefaultOpset(13)};
  Concat concat_kernel{ctx};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  Tensor y = concat_kernel({x0, x1}, /*axis=*/0);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{4, 2}));
  ASSERT_EQ(y.element_count(), 8);
  const float *py = y.AsFloat();
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(py[i], static_cast<float>(i + 1));
  }
}

TEST(BackendKernelClass, ConcatClassConcatenatesNegativeAxis) {
  const KernelContext ctx{DefaultOpset(13)};
  Concat concat_kernel{ctx};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {2, 3}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
  Tensor y = concat_kernel({x0, x1}, /*axis=*/-1);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 5}));
  ASSERT_EQ(y.element_count(), 10);
  const float *py = y.AsFloat();
  const std::vector<float> expected{1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 3.0f, 4.0f, 8.0f, 9.0f, 10.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]);
  }
}

TEST(BackendKernelClass, ConcatClassRejectsMismatchedShape) {
  const KernelContext ctx{DefaultOpset(13)};
  Concat concat_kernel{ctx};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {3, 2}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
  EXPECT_THROW((void)concat_kernel({x0, x1}, /*axis=*/1), std::invalid_argument);
}

TEST(BackendKernelClass, ConcatClassRejectsScalar) {
  const KernelContext ctx{DefaultOpset(13)};
  Concat concat_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {}, {1.0f});
  EXPECT_THROW((void)concat_kernel({x}, /*axis=*/0), std::invalid_argument);
}

TEST(BackendKernelClass, ConcatInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  Concat concat_kernel{ctx};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {2, 3}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
  Tensor y("out", onnx_backend_test::DataType::FLOAT, {2, 5},
           std::vector<uint8_t>(10 * sizeof(float)));
  concat_kernel({x0, x1}, /*axis=*/-1, y);
  EXPECT_EQ(y.name, "out");
  ASSERT_EQ(y.element_count(), 10);
  const float *py = y.AsFloat();
  const std::vector<float> expected{1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 3.0f, 4.0f, 8.0f, 9.0f, 10.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]);
  }
}

TEST(BackendKernelClass, ConcatInPlaceRejectsMismatchedShape) {
  const KernelContext ctx{DefaultOpset(13)};
  Concat concat_kernel{ctx};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  Tensor bad_shape("", onnx_backend_test::DataType::FLOAT, {3, 2},
                   std::vector<uint8_t>(6 * sizeof(float)));
  EXPECT_THROW(concat_kernel({x0, x1}, /*axis=*/0, bad_shape), std::invalid_argument);
}

TEST(BackendKernelClass, SqueezeClassRemovesSpecifiedAxes) {
  const KernelContext ctx{DefaultOpset(13)};
  Squeeze squeeze_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 1, 3, 1}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
  Tensor y = squeeze_kernel(x, {1, 3});
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  const float *py = y.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(py[i], static_cast<float>(i));
  }
}

TEST(BackendKernelClass, SqueezeClassWithEmptyAxesRemovesAllSingletonDims) {
  const KernelContext ctx{DefaultOpset(13)};
  Squeeze squeeze_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {1, 2, 1, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
  Tensor y = squeeze_kernel(x, {});
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
}

TEST(BackendKernelClass, UnsqueezeClassInsertsSpecifiedAxes) {
  const KernelContext ctx{DefaultOpset(13)};
  Unsqueeze unsqueeze_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
  Tensor y = unsqueeze_kernel(x, {0, 2});
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 2, 1, 3}));
  const float *py = y.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(py[i], static_cast<float>(i));
  }
}

TEST(BackendKernelClass, UnsqueezeRejectsDuplicateAxes) {
  const KernelContext ctx{DefaultOpset(13)};
  Unsqueeze unsqueeze_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
  EXPECT_THROW((void)unsqueeze_kernel(x, {1, 1}), std::invalid_argument);
}

TEST(BackendKernelClass, CastClassFloatToDouble) {
  const KernelContext ctx{DefaultOpset(13)};
  Cast cast_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-1.5f, 0.0f, 2.25f});
  Tensor y = cast_kernel(x, static_cast<int32_t>(onnx_backend_test::DataType::DOUBLE));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::DOUBLE));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3}));
  const double *py = y.AsDouble();
  EXPECT_DOUBLE_EQ(py[0], -1.5);
  EXPECT_DOUBLE_EQ(py[1], 0.0);
  EXPECT_DOUBLE_EQ(py[2], 2.25);
}

TEST(BackendKernelClass, CastClassFloatToInt32TruncatesTowardZero) {
  const KernelContext ctx{DefaultOpset(13)};
  Cast cast_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {4}, {-1.5f, 0.0f, 2.75f, 4.0f});
  Tensor y = cast_kernel(x, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  const int32_t *py = y.AsInt32();
  EXPECT_EQ(py[0], -1);
  EXPECT_EQ(py[1], 0);
  EXPECT_EQ(py[2], 2);
  EXPECT_EQ(py[3], 4);
}

TEST(BackendKernelClass, CastClassInt64ToFloat) {
  const KernelContext ctx{DefaultOpset(13)};
  Cast cast_kernel{ctx};
  Tensor x = Tensor::FromInt64("", {4}, {-3, 0, 7, 42});
  Tensor y = cast_kernel(x, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], -3.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 7.0f);
  EXPECT_FLOAT_EQ(py[3], 42.0f);
}

TEST(BackendKernelClass, CastClassIdentityCopiesBytes) {
  const KernelContext ctx{DefaultOpset(13)};
  Cast cast_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor y = cast_kernel(x, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(y.shape, x.shape);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 2.0f);
  EXPECT_FLOAT_EQ(py[2], 3.0f);
}

TEST(BackendKernelClass, CastClassRejectsUnsupportedTo) {
  const KernelContext ctx{DefaultOpset(13)};
  Cast cast_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {1}, {1.0f});
  // FLOAT16 is not in the supported set for the kernel today.
  EXPECT_THROW((void)cast_kernel(x, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT16)),
               std::invalid_argument);
}

TEST(BackendKernelClass, CastInPlaceRejectsDtypeMismatch) {
  const KernelContext ctx{DefaultOpset(13)};
  Cast cast_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  // Output dtype declared as FLOAT but the caller asks for INT32.
  Tensor wrong_out("", onnx_backend_test::DataType::FLOAT, {2},
                   std::vector<uint8_t>(2 * sizeof(float)));
  EXPECT_THROW(cast_kernel(x, static_cast<int32_t>(onnx_backend_test::DataType::INT32), wrong_out),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// CastLike kernel tests.
//
// CastLike forwards to Cast using the dtype of its second input; the values
// of the second input are ignored. These tests check both the returning and
// in-place overloads, plus that the second tensor's values truly do not
// influence the result.
// ---------------------------------------------------------------------------

TEST(BackendKernelClass, CastLikeClassFloatToDouble) {
  const KernelContext ctx{DefaultOpset(15)};
  CastLike castlike_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-1.5f, 0.0f, 2.25f});
  // target_type carries the destination dtype only; its value is ignored.
  Tensor target = Tensor::FromDouble("", {1}, {0.0});
  Tensor y = castlike_kernel(x, target);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::DOUBLE));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3}));
  const double *py = y.AsDouble();
  EXPECT_DOUBLE_EQ(py[0], -1.5);
  EXPECT_DOUBLE_EQ(py[1], 0.0);
  EXPECT_DOUBLE_EQ(py[2], 2.25);
}

TEST(BackendKernelClass, CastLikeClassIgnoresTargetValues) {
  // Different target values must produce the same output as long as their
  // dtypes match.
  const KernelContext ctx{DefaultOpset(15)};
  CastLike castlike_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {4}, {-1.5f, 0.0f, 2.75f, 4.0f});
  Tensor t1 = Tensor::FromInt32("", {1}, {0});
  Tensor t2 = Tensor::FromInt32("", {2}, {12345, -67890});
  Tensor y1 = castlike_kernel(x, t1);
  Tensor y2 = castlike_kernel(x, t2);
  ASSERT_EQ(y1.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  ASSERT_EQ(y2.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  ASSERT_EQ(y1.shape, x.shape);
  ASSERT_EQ(y2.shape, x.shape);
  const int32_t *p1 = y1.AsInt32();
  const int32_t *p2 = y2.AsInt32();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(p1[i], p2[i]);
  }
  // FLOAT->INT32 truncates toward zero.
  EXPECT_EQ(p1[0], -1);
  EXPECT_EQ(p1[1], 0);
  EXPECT_EQ(p1[2], 2);
  EXPECT_EQ(p1[3], 4);
}

TEST(BackendKernelClass, CastLikeInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(15)};
  CastLike castlike_kernel{ctx};
  Tensor x = Tensor::FromInt64("", {3}, {-3, 0, 42});
  Tensor target = Tensor::FromFloat("", {1}, {0.0f});
  Tensor out("", onnx_backend_test::DataType::FLOAT, {3}, std::vector<uint8_t>(3 * sizeof(float)));
  castlike_kernel(x, target, out);
  const float *po = out.AsFloat();
  EXPECT_FLOAT_EQ(po[0], -3.0f);
  EXPECT_FLOAT_EQ(po[1], 0.0f);
  EXPECT_FLOAT_EQ(po[2], 42.0f);
}

TEST(BackendKernelClass, CastLikeInPlaceRejectsDtypeMismatch) {
  const KernelContext ctx{DefaultOpset(15)};
  CastLike castlike_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor target = Tensor::FromInt32("", {1}, {0});
  // Pre-allocated output dtype does not match target_type.data_type.
  Tensor wrong_out("", onnx_backend_test::DataType::FLOAT, {2},
                   std::vector<uint8_t>(2 * sizeof(float)));
  EXPECT_THROW(castlike_kernel(x, target, wrong_out), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// AffineGrid kernel tests.
// ---------------------------------------------------------------------------

using onnx_backend_test::kernel::AffineGrid;

namespace {

// Returns the (2, 2, 3) theta batch produced by ``create_theta_2d()`` in
// ``onnx/backend/test/case/node/affinegrid.py``. Pre-computed once via the
// upstream numpy helper.
Tensor MakeUpstreamTheta2D() {
  return Tensor::FromFloat("", {2, 2, 3},
                           {1.0889444f, -3.2880466f, 5.0f, 2.0223253f, 1.0960155f, -3.3f,
                            0.83578837f, -0.55442286f, 2.5f, 0.78762794f, 0.8397114f, 1.1f});
}

// Returns the (2, 3, 4) theta batch produced by ``create_theta_3d()`` in
// ``onnx/backend/test/case/node/affinegrid.py``.
Tensor MakeUpstreamTheta3D() {
  return Tensor::FromFloat(
      "", {2, 3, 4}, {2.6830733f,   -0.7943316f, 0.21829216f, 5.0f,       0.62225395f, 3.2880466f,
                      -0.53033006f, -3.3f,       0.24721935f, 1.7241772f, 0.07809311f, -1.1f,
                      -0.35552558f, 1.0044229f,  1.3995191f,  2.5f,       0.17578839f, 0.060288567f,
                      -0.9240381f,  1.1f,        -1.1f,       -0.45f,     0.45f,       2.2f});
}

} // namespace

TEST(BackendKernelClass, AffineGrid2DMatchesUpstreamReference) {
  const KernelContext ctx{DefaultOpset(20)};
  AffineGrid ag_kernel{ctx};
  Tensor theta = MakeUpstreamTheta2D();
  Tensor size = Tensor::FromInt64("", {4}, {2, 3, 5, 6});

  // align_corners == 0 (the default in the ONNX schema).
  AffineGrid::Attributes attrs;
  attrs.align_corners = 0;
  Tensor grid = ag_kernel(theta, size, attrs);
  ASSERT_EQ(grid.shape, (std::vector<int64_t>{2, 5, 6, 2}));
  const float *g = grid.AsFloat();
  // Sample values pre-computed via the upstream Python reference
  // (apply_affine_transform on construct_original_grid((5, 6), 0)).
  EXPECT_NEAR(g[0], 6.7229834f, 1e-4f);  // [0, 0, 0, 0]
  EXPECT_NEAR(g[1], -5.8620834f, 1e-4f); // [0, 0, 0, 1]
  EXPECT_NEAR(g[(0 * 5 * 6 + 2 * 6 + 3) * 2 + 0], 5.181491f, 1e-4f);
  EXPECT_NEAR(g[(0 * 5 * 6 + 2 * 6 + 3) * 2 + 1], -2.9629457f, 1e-4f);
  EXPECT_NEAR(g[(1 * 5 * 6 + 4 * 6 + 5) * 2 + 0], 2.752952f, 1e-4f);
  EXPECT_NEAR(g[(1 * 5 * 6 + 4 * 6 + 5) * 2 + 1], 2.4281259f, 1e-4f);

  // align_corners == 1.
  attrs.align_corners = 1;
  Tensor grid_ac = ag_kernel(theta, size, attrs);
  ASSERT_EQ(grid_ac.shape, (std::vector<int64_t>{2, 5, 6, 2}));
  const float *gac = grid_ac.AsFloat();
  EXPECT_NEAR(gac[0], 7.1991024f, 1e-4f);
  EXPECT_NEAR(gac[1], -6.4183407f, 1e-4f);
  EXPECT_NEAR(gac[(0 * 5 * 6 + 2 * 6 + 3) * 2 + 0], 5.2177887f, 1e-4f);
  EXPECT_NEAR(gac[(0 * 5 * 6 + 2 * 6 + 3) * 2 + 1], -2.895535f, 1e-4f);
  EXPECT_NEAR(gac[(1 * 5 * 6 + 4 * 6 + 5) * 2 + 0], 2.7813654f, 1e-4f);
  EXPECT_NEAR(gac[(1 * 5 * 6 + 4 * 6 + 5) * 2 + 1], 2.7273393f, 1e-4f);
}

TEST(BackendKernelClass, AffineGrid3DMatchesUpstreamReference) {
  const KernelContext ctx{DefaultOpset(20)};
  AffineGrid ag_kernel{ctx};
  Tensor theta = MakeUpstreamTheta3D();
  Tensor size = Tensor::FromInt64("", {5}, {2, 3, 4, 5, 6});

  AffineGrid::Attributes attrs;
  attrs.align_corners = 0;
  Tensor grid = ag_kernel(theta, size, attrs);
  ASSERT_EQ(grid.shape, (std::vector<int64_t>{2, 4, 5, 6, 3}));
  const float *g = grid.AsFloat();
  EXPECT_NEAR(g[0], 3.2358518f, 1e-4f);  // [0, 0, 0, 0, 0]
  EXPECT_NEAR(g[1], -6.0512347f, 1e-4f); // [0, 0, 0, 0, 1]
  EXPECT_NEAR(g[2], -2.7439277f, 1e-4f); // [0, 0, 0, 0, 2]
  // Last sample of batch 1.
  const int64_t last_idx = (1 * 4 * 5 * 6 + 3 * 5 * 6 + 4 * 6 + 5) * 3;
  EXPECT_NEAR(g[last_idx + 0], 4.056906f, 1e-4f);
  EXPECT_NEAR(g[last_idx + 1], 0.6016926f, 1e-4f);
  EXPECT_NEAR(g[last_idx + 2], 1.2608334f, 1e-4f);

  attrs.align_corners = 1;
  Tensor grid_ac = ag_kernel(theta, size, attrs);
  ASSERT_EQ(grid_ac.shape, (std::vector<int64_t>{2, 4, 5, 6, 3}));
  const float *gac = grid_ac.AsFloat();
  EXPECT_NEAR(gac[0], 2.8929663f, 1e-4f);
  EXPECT_NEAR(gac[1], -6.6799703f, 1e-4f);
  EXPECT_NEAR(gac[2], -3.1494896f, 1e-4f);
  EXPECT_NEAR(gac[last_idx + 0], 4.5484166f, 1e-4f);
  EXPECT_NEAR(gac[last_idx + 1], 0.41203886f, 1e-4f);
  EXPECT_NEAR(gac[last_idx + 2], 1.1f, 1e-4f);
}

TEST(BackendKernelClass, AffineGridRejectsBadShapes) {
  const KernelContext ctx{DefaultOpset(20)};
  AffineGrid ag_kernel{ctx};
  // theta of rank 2 instead of 3.
  Tensor theta_bad = Tensor::FromFloat("", {2, 3}, {0, 0, 0, 0, 0, 0});
  Tensor size = Tensor::FromInt64("", {4}, {1, 1, 2, 2});
  EXPECT_THROW((void)ag_kernel(theta_bad, size, AffineGrid::Attributes{}), std::invalid_argument);

  // size of length 3 (neither 4 nor 5).
  Tensor theta = Tensor::FromFloat("", {1, 2, 3}, {1, 0, 0, 0, 1, 0});
  Tensor size_bad = Tensor::FromInt64("", {3}, {1, 2, 2});
  EXPECT_THROW((void)ag_kernel(theta, size_bad, AffineGrid::Attributes{}), std::invalid_argument);

  // size[0] != theta[0] (batch mismatch).
  Tensor size_mismatch = Tensor::FromInt64("", {4}, {2, 1, 2, 2});
  EXPECT_THROW((void)ag_kernel(theta, size_mismatch, AffineGrid::Attributes{}),
               std::invalid_argument);

  // theta inner dims (2,3) vs 3-D size of length 5.
  Tensor size3d = Tensor::FromInt64("", {5}, {1, 1, 2, 2, 2});
  EXPECT_THROW((void)ag_kernel(theta, size3d, AffineGrid::Attributes{}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// GridSample kernel tests.
// ---------------------------------------------------------------------------

using onnx_backend_test::kernel::GridSample;
TEST(BackendKernelClass, GridSampleBilinearMatchesUpstream) {
  // Matches ``test_gridsample_bilinear`` from
  // ``onnx/backend/test/case/node/gridsample.py``.
  const KernelContext ctx{DefaultOpset(20)};
  GridSample gs{ctx};
  Tensor X = Tensor::FromFloat("", {1, 1, 3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  Tensor Grid = Tensor::FromFloat("", {1, 2, 4, 2},
                                  {-1.0f, -1.0f, -0.5f, -0.5f, -0.2f, -0.2f, 0.0f, 0.0f, 0.0f, 0.0f,
                                   -0.2f, -0.2f, 0.5f, 0.5f, 1.0f, 1.0f});
  GridSample::Attributes attrs;
  attrs.mode = "linear";
  attrs.padding_mode = "zeros";
  attrs.align_corners = 0;
  Tensor Y = gs(X, Grid, attrs);
  ASSERT_EQ(Y.shape, (std::vector<int64_t>{1, 1, 2, 4}));
  const float expected[8] = {0.0f, 0.5f, 1.7f, 2.5f, 2.5f, 1.7f, 4.5f, 1.25f};
  const float *y = Y.AsFloat();
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(y[i], expected[i], 1e-4f) << "idx " << i;
  }
}

TEST(BackendKernelClass, GridSampleNearestAndAlignCorners) {
  // Matches ``test_gridsample_nearest`` and ``_aligncorners_true``.
  const KernelContext ctx{DefaultOpset(20)};
  GridSample gs{ctx};
  Tensor X = Tensor::FromFloat("", {1, 1, 3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  Tensor Grid = Tensor::FromFloat("", {1, 2, 4, 2},
                                  {-1.0f, -1.0f, -0.5f, -0.5f, -0.2f, -0.2f, 0.0f, 0.0f, 0.0f, 0.0f,
                                   -0.2f, -0.2f, 0.5f, 0.5f, 1.0f, 1.0f});

  GridSample::Attributes nearest_attrs;
  nearest_attrs.mode = "nearest";
  Tensor Y_nearest = gs(X, Grid, nearest_attrs);
  const float exp_nearest[8] = {0.0f, 0.0f, 2.0f, 2.0f, 2.0f, 2.0f, 5.0f, 0.0f};
  const float *yn = Y_nearest.AsFloat();
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(yn[i], exp_nearest[i], 1e-5f) << "nearest idx " << i;
  }

  GridSample::Attributes ac_attrs;
  ac_attrs.mode = "linear";
  ac_attrs.align_corners = 1;
  Tensor Y_ac = gs(X, Grid, ac_attrs);
  const float exp_ac[8] = {0.0f, 1.25f, 2.0f, 2.5f, 2.5f, 2.0f, 3.75f, 5.0f};
  const float *yac = Y_ac.AsFloat();
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(yac[i], exp_ac[i], 1e-4f) << "aligncorners idx " << i;
  }
}

TEST(BackendKernelClass, GridSamplePaddingModes) {
  const KernelContext ctx{DefaultOpset(20)};
  GridSample gs{ctx};
  Tensor X = Tensor::FromFloat("", {1, 1, 3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  Tensor Grid = Tensor::FromFloat("", {1, 2, 4, 2},
                                  {-10.0f, -10.0f, -5.0f, -5.0f, -0.2f, -0.2f, 10.0f, 10.0f, 10.0f,
                                   10.0f, -0.2f, -0.2f, 5.0f, 5.0f, 10.0f, 10.0f});

  GridSample::Attributes attrs;
  attrs.mode = "linear";
  attrs.padding_mode = "zeros";
  Tensor Y_zeros = gs(X, Grid, attrs);
  const float exp_zeros[8] = {0.0f, 0.0f, 1.7f, 0.0f, 0.0f, 1.7f, 0.0f, 0.0f};
  const float *y0 = Y_zeros.AsFloat();
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(y0[i], exp_zeros[i], 1e-4f) << "zeros idx " << i;
  }

  attrs.padding_mode = "border";
  Tensor Y_border = gs(X, Grid, attrs);
  const float exp_border[8] = {0.0f, 0.0f, 1.7f, 5.0f, 5.0f, 1.7f, 5.0f, 5.0f};
  const float *yb = Y_border.AsFloat();
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(yb[i], exp_border[i], 1e-4f) << "border idx " << i;
  }

  attrs.padding_mode = "reflection";
  Tensor Y_reflect = gs(X, Grid, attrs);
  const float exp_reflect[8] = {2.5f, 0.0f, 1.7f, 2.5f, 2.5f, 1.7f, 5.0f, 2.5f};
  const float *yr = Y_reflect.AsFloat();
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(yr[i], exp_reflect[i], 1e-4f) << "reflect idx " << i;
  }
}

TEST(BackendKernelClass, GridSampleRejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(20)};
  GridSample gs{ctx};
  Tensor X = Tensor::FromFloat("", {1, 1, 3, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  // grid last-dim != rank - 2 (must be 2 for 2D, but here we pass 3).
  Tensor Grid_bad = Tensor::FromFloat("", {1, 2, 4, 3}, std::vector<float>(24, 0.0f));
  EXPECT_THROW((void)gs(X, Grid_bad, GridSample::Attributes{}), std::invalid_argument);
  // rank mismatch.
  Tensor Grid_rank = Tensor::FromFloat("", {1, 2, 2}, {0, 0, 0, 0});
  EXPECT_THROW((void)gs(X, Grid_rank, GridSample::Attributes{}), std::invalid_argument);
  // Unknown mode.
  Tensor Grid_ok = Tensor::FromFloat("", {1, 2, 4, 2}, std::vector<float>(16, 0.0f));
  GridSample::Attributes bad_attrs;
  bad_attrs.mode = "quintic";
  EXPECT_THROW((void)gs(X, Grid_ok, bad_attrs), std::invalid_argument);
}

using onnx_backend_test::kernel::NonZero;

TEST(BackendKernelClass, NonZeroFloat2DReturnsRowMajorIndices) {
  const KernelContext ctx{DefaultOpset(13)};
  NonZero nonzero_kernel{ctx};
  // X = [[1, 0], [1, 1]]; non-zero in row-major order at (0,0),(1,0),(1,1).
  // Output is (rank=2, nnz=3) = [[0,1,1],[0,0,1]].
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 0.0f, 1.0f, 1.0f});
  Tensor y = nonzero_kernel(x);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT64));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 1);
  EXPECT_EQ(py[2], 1);
  EXPECT_EQ(py[3], 0);
  EXPECT_EQ(py[4], 0);
  EXPECT_EQ(py[5], 1);
}

TEST(BackendKernelClass, NonZero1DReturnsSingleRow) {
  const KernelContext ctx{DefaultOpset(13)};
  NonZero nonzero_kernel{ctx};
  Tensor x = Tensor::FromInt64("", {5}, {0, 1, 0, -1, 2});
  Tensor y = nonzero_kernel(x);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 3}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 1);
  EXPECT_EQ(py[1], 3);
  EXPECT_EQ(py[2], 4);
}

TEST(BackendKernelClass, NonZeroBoolRespectsTruthiness) {
  const KernelContext ctx{DefaultOpset(13)};
  NonZero nonzero_kernel{ctx};
  Tensor x = Tensor::FromBool("", {2, 3}, {1, 0, 1, 0, 1, 0});
  Tensor y = nonzero_kernel(x);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  const int64_t *py = y.AsInt64();
  // Non-zero positions: (0,0),(0,2),(1,1) -> rows [0,0,1],[0,2,1].
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 0);
  EXPECT_EQ(py[2], 1);
  EXPECT_EQ(py[3], 0);
  EXPECT_EQ(py[4], 2);
  EXPECT_EQ(py[5], 1);
}

TEST(BackendKernelClass, NonZeroScalarProducesShapeZeroByNnz) {
  const KernelContext ctx{DefaultOpset(13)};
  NonZero nonzero_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {}, {1.0f});
  Tensor y = nonzero_kernel(x);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{0, 1}));
}

TEST(BackendKernelClass, NonZeroAllZeroesProducesEmptyOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  NonZero nonzero_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});
  Tensor y = nonzero_kernel(x);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 0}));
  EXPECT_EQ(y.element_count(), 0);
}

TEST(BackendKernelClass, NonZeroRejectsUnsupportedDtype) {
  const KernelContext ctx{DefaultOpset(13)};
  NonZero nonzero_kernel{ctx};
  Tensor x = Tensor::FromStrings("", {2}, {"foo", "bar"});
  EXPECT_THROW((void)nonzero_kernel(x), std::invalid_argument);
}

} // namespace Test
