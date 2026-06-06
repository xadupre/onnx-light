// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_kernels/test_case.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_kernels::DefaultOpset;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::Cast;
using onnx_kernels::kernel::CastLike;
using onnx_kernels::kernel::Concat;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::Reshape;
using onnx_kernels::kernel::Slice;
using onnx_kernels::kernel::Squeeze;
using onnx_kernels::kernel::Unsqueeze;

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
  Tensor y("out", onnx_kernels::DataType::FLOAT, {2, 5},
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
  Tensor bad_shape("", onnx_kernels::DataType::FLOAT, {3, 2},
                   std::vector<uint8_t>(6 * sizeof(float)));
  EXPECT_THROW(concat_kernel({x0, x1}, /*axis=*/0, bad_shape), std::invalid_argument);
}

TEST(BackendKernelClass, ReshapeClassReordersDimensions) {
  const KernelContext ctx{DefaultOpset(13)};
  Reshape reshape_kernel{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  Tensor shape = Tensor::FromInt64("", {2}, {3, 2});
  Tensor y = reshape_kernel(data, shape);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3, 2}));
  ASSERT_EQ(y.element_count(), data.element_count());
  const float *py = y.AsFloat();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(py[i], static_cast<float>(i + 1));
  }
}

TEST(BackendKernelClass, ReshapeClassAllowZeroHonoursLiteralZero) {
  const KernelContext ctx{DefaultOpset(14)};
  Reshape reshape_kernel{ctx};
  Tensor data = Tensor::FromFloat("", {0, 2}, {});
  Tensor shape = Tensor::FromInt64("", {2}, {0, 2});
  Tensor y = reshape_kernel(data, shape, /*allowzero=*/1);
  EXPECT_EQ(y.shape, (std::vector<int64_t>{0, 2}));
}

TEST(BackendKernelClass, SliceClassSlicesWithAxesAndSteps) {
  const KernelContext ctx{DefaultOpset(13)};
  Slice slice_kernel{ctx};
  Tensor data = Tensor::FromFloat("", {2, 4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  Tensor starts = Tensor::FromInt64("", {2}, {1, 0});
  Tensor ends = Tensor::FromInt64("", {2}, {2, 3});
  Tensor axes = Tensor::FromInt64("", {2}, {0, 1});
  Tensor steps = Tensor::FromInt64("", {2}, {1, 2});
  Tensor y = slice_kernel(data, starts, ends, &axes, &steps);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 5.f);
  EXPECT_FLOAT_EQ(py[1], 7.f);
}

TEST(BackendKernelClass, SliceClassUsesDefaultAxesAndSteps) {
  const KernelContext ctx{DefaultOpset(13)};
  Slice slice_kernel{ctx};
  Tensor data = Tensor::FromFloat("", {2, 4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  Tensor starts = Tensor::FromInt64("", {2}, {0, 1});
  Tensor ends = Tensor::FromInt64("", {2}, {-1, 1000});
  Tensor y = slice_kernel(data, starts, ends);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 3}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 2.f);
  EXPECT_FLOAT_EQ(py[1], 3.f);
  EXPECT_FLOAT_EQ(py[2], 4.f);
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
  Tensor y = cast_kernel(x, static_cast<int32_t>(onnx_kernels::DataType::DOUBLE));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::DOUBLE));
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
  Tensor y = cast_kernel(x, static_cast<int32_t>(onnx_kernels::DataType::INT32));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT32));
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
  Tensor y = cast_kernel(x, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
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
  Tensor y = cast_kernel(x, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
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
  EXPECT_THROW((void)cast_kernel(x, static_cast<int32_t>(onnx_kernels::DataType::FLOAT16)),
               std::invalid_argument);
}

TEST(BackendKernelClass, CastInPlaceRejectsDtypeMismatch) {
  const KernelContext ctx{DefaultOpset(13)};
  Cast cast_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  // Output dtype declared as FLOAT but the caller asks for INT32.
  Tensor wrong_out("", onnx_kernels::DataType::FLOAT, {2},
                   std::vector<uint8_t>(2 * sizeof(float)));
  EXPECT_THROW(cast_kernel(x, static_cast<int32_t>(onnx_kernels::DataType::INT32), wrong_out),
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
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::DOUBLE));
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
  ASSERT_EQ(y1.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT32));
  ASSERT_EQ(y2.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT32));
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
  Tensor out("", onnx_kernels::DataType::FLOAT, {3}, std::vector<uint8_t>(3 * sizeof(float)));
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
  Tensor wrong_out("", onnx_kernels::DataType::FLOAT, {2},
                   std::vector<uint8_t>(2 * sizeof(float)));
  EXPECT_THROW(castlike_kernel(x, target, wrong_out), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// AffineGrid kernel tests.
// ---------------------------------------------------------------------------

using onnx_kernels::kernel::AffineGrid;

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

using onnx_kernels::kernel::GridSample;
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

using onnx_kernels::kernel::NonZero;

TEST(BackendKernelClass, NonZeroFloat2DReturnsRowMajorIndices) {
  const KernelContext ctx{DefaultOpset(13)};
  NonZero nonzero_kernel{ctx};
  // X = [[1, 0], [1, 1]]; non-zero in row-major order at (0,0),(1,0),(1,1).
  // Output is (rank=2, nnz=3) = [[0,1,1],[0,0,1]].
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 0.0f, 1.0f, 1.0f});
  Tensor y = nonzero_kernel(x);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
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

using onnx_kernels::kernel::Shape;

TEST(BackendKernelClass, ShapeDefaultReturnsFullShape) {
  const KernelContext ctx{DefaultOpset(15)};
  Shape shape_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3, 4}, std::vector<float>(24, 0.0f));
  Tensor y = shape_kernel(x);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 2);
  EXPECT_EQ(py[1], 3);
  EXPECT_EQ(py[2], 4);
}

TEST(BackendKernelClass, ShapeStartEndAndNegativesAreClamped) {
  const KernelContext ctx{DefaultOpset(15)};
  Shape shape_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3, 4, 5}, std::vector<float>(120, 0.0f));

  Shape::Attributes a;
  a.start = 1;
  a.end = 3;
  Tensor y = shape_kernel(x, a);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(y.AsInt64()[0], 3);
  EXPECT_EQ(y.AsInt64()[1], 4);

  Shape::Attributes neg;
  neg.start = -2;
  Tensor y2 = shape_kernel(x, neg);
  ASSERT_EQ(y2.shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(y2.AsInt64()[0], 4);
  EXPECT_EQ(y2.AsInt64()[1], 5);

  // start > end yields empty 1-D output.
  Shape::Attributes inv;
  inv.start = 3;
  inv.end = 1;
  Tensor y3 = shape_kernel(x, inv);
  EXPECT_EQ(y3.shape, (std::vector<int64_t>{0}));
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

// ---------------------------------------------------------------------------
// DepthToSpace kernel tests
// ---------------------------------------------------------------------------

TEST(BackendKernelClass, DepthToSpaceDCRMatchesNumpyReference) {
  // Cross-checks the DCR computation against the NumPy equivalent given in
  // the ONNX spec:
  //   tmp = reshape(x, [b, blocksize, blocksize, c/(b*b), h, w])
  //   tmp = transpose(tmp, [0, 3, 4, 1, 5, 2])
  //   y   = reshape(tmp, [b, c/(b*b), h*blocksize, w*blocksize])
  const KernelContext ctx{DefaultOpset(13)};
  onnx_kernels::kernel::DepthToSpace d2s{ctx};
  std::vector<float> values(8);
  for (int i = 0; i < 8; ++i)
    values[i] = static_cast<float>(i);
  // Input shape (1, 8, 1, 1): C=8, blocksize=2 -> C_out=2, H_out=2, W_out=2.
  Tensor x = Tensor::FromFloat("", {1, 8, 1, 1}, values);
  onnx_kernels::kernel::DepthToSpace::Attributes a;
  a.blocksize = 2;
  a.mode = "DCR";
  Tensor y = d2s(x, a);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 2, 2, 2}));
  // Reference computed by hand following the DCR rule:
  //   c_in = bh * blocksize * C_out + bw * C_out + c_out
  const std::vector<float> expected{0.f, 2.f, 4.f, 6.f, 1.f, 3.f, 5.f, 7.f};
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]);
  }
}

TEST(BackendKernelClass, DepthToSpaceCRDMatchesNumpyReference) {
  const KernelContext ctx{DefaultOpset(13)};
  onnx_kernels::kernel::DepthToSpace d2s{ctx};
  std::vector<float> values(8);
  for (int i = 0; i < 8; ++i)
    values[i] = static_cast<float>(i);
  Tensor x = Tensor::FromFloat("", {1, 8, 1, 1}, values);
  onnx_kernels::kernel::DepthToSpace::Attributes a;
  a.blocksize = 2;
  a.mode = "CRD";
  Tensor y = d2s(x, a);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 2, 2, 2}));
  // Reference computed by hand following the CRD rule:
  //   c_in = c_out * blocksize^2 + bh * blocksize + bw
  const std::vector<float> expected{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]);
  }
}

TEST(BackendKernelClass, DepthToSpaceRejectsNonRank4Input) {
  const KernelContext ctx{DefaultOpset(13)};
  onnx_kernels::kernel::DepthToSpace d2s{ctx};
  Tensor x = Tensor::FromFloat("", {1, 4}, {0.f, 1.f, 2.f, 3.f});
  onnx_kernels::kernel::DepthToSpace::Attributes a;
  a.blocksize = 2;
  EXPECT_THROW((void)d2s(x, a), std::invalid_argument);
}

TEST(BackendKernelClass, DepthToSpaceRejectsChannelNotDivisible) {
  const KernelContext ctx{DefaultOpset(13)};
  onnx_kernels::kernel::DepthToSpace d2s{ctx};
  Tensor x = Tensor::FromFloat("", {1, 5, 2, 2},
                               {0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,
                                10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f});
  onnx_kernels::kernel::DepthToSpace::Attributes a;
  a.blocksize = 2;
  EXPECT_THROW((void)d2s(x, a), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// SpaceToDepth kernel tests
// ---------------------------------------------------------------------------

TEST(BackendKernelClass, SpaceToDepthMatchesNumpyReference) {
  // Cross-checks the SpaceToDepth computation against the NumPy equivalent
  // given in the ONNX spec:
  //   tmp = reshape(x, [N, C, H/bs, bs, W/bs, bs])
  //   tmp = transpose(tmp, [0, 3, 5, 1, 2, 4])
  //   y   = reshape(tmp, [N, C*bs*bs, H/bs, W/bs])
  const KernelContext ctx{DefaultOpset(13)};
  onnx_kernels::kernel::SpaceToDepth s2d{ctx};
  std::vector<float> values(8);
  for (int i = 0; i < 8; ++i)
    values[i] = static_cast<float>(i);
  // Input shape (1, 2, 2, 2): blocksize=2 -> output shape (1, 8, 1, 1).
  Tensor x = Tensor::FromFloat("", {1, 2, 2, 2}, values);
  onnx_kernels::kernel::SpaceToDepth::Attributes a;
  a.blocksize = 2;
  Tensor y = s2d(x, a);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 8, 1, 1}));
  // Reference computed by hand from the spec.
  // c_out: bh,bw,c -> (bh*bs*C + bw*C + c)
  //   c_out=0: bh=0,bw=0,c=0 -> x[0,0,0,0]=0
  //   c_out=1: bh=0,bw=0,c=1 -> x[0,1,0,0]=4
  //   c_out=2: bh=0,bw=1,c=0 -> x[0,0,0,1]=1
  //   c_out=3: bh=0,bw=1,c=1 -> x[0,1,0,1]=5
  //   c_out=4: bh=1,bw=0,c=0 -> x[0,0,1,0]=2
  //   c_out=5: bh=1,bw=0,c=1 -> x[0,1,1,0]=6
  //   c_out=6: bh=1,bw=1,c=0 -> x[0,0,1,1]=3
  //   c_out=7: bh=1,bw=1,c=1 -> x[0,1,1,1]=7
  const std::vector<float> expected{0.f, 4.f, 1.f, 5.f, 2.f, 6.f, 3.f, 7.f};
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]);
  }
}

TEST(BackendKernelClass, SpaceToDepthIsInverseOfDepthToSpaceDCR) {
  // Round-trip x -> SpaceToDepth -> DepthToSpace(DCR) should be identity.
  const KernelContext ctx{DefaultOpset(13)};
  onnx_kernels::kernel::SpaceToDepth s2d{ctx};
  onnx_kernels::kernel::DepthToSpace d2s{ctx};
  std::vector<float> values(2 * 3 * 4 * 6);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>(i);
  Tensor x = Tensor::FromFloat("", {2, 3, 4, 6}, values);
  onnx_kernels::kernel::SpaceToDepth::Attributes a;
  a.blocksize = 2;
  Tensor y = s2d(x, a);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 12, 2, 3}));
  onnx_kernels::kernel::DepthToSpace::Attributes ad;
  ad.blocksize = 2;
  ad.mode = "DCR";
  Tensor z = d2s(y, ad);
  ASSERT_EQ(z.shape, x.shape);
  const float *pz = z.AsFloat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_FLOAT_EQ(pz[i], values[i]);
  }
}

TEST(BackendKernelClass, SpaceToDepthRejectsNonRank4Input) {
  const KernelContext ctx{DefaultOpset(13)};
  onnx_kernels::kernel::SpaceToDepth s2d{ctx};
  Tensor x = Tensor::FromFloat("", {1, 4}, {0.f, 1.f, 2.f, 3.f});
  onnx_kernels::kernel::SpaceToDepth::Attributes a;
  a.blocksize = 2;
  EXPECT_THROW((void)s2d(x, a), std::invalid_argument);
}

TEST(BackendKernelClass, SpaceToDepthRejectsSpatialDimNotDivisible) {
  const KernelContext ctx{DefaultOpset(13)};
  onnx_kernels::kernel::SpaceToDepth s2d{ctx};
  // H=3 is not divisible by blocksize=2.
  Tensor x = Tensor::FromFloat("", {1, 1, 3, 2}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
  onnx_kernels::kernel::SpaceToDepth::Attributes a;
  a.blocksize = 2;
  EXPECT_THROW((void)s2d(x, a), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Upsample kernel tests
// ---------------------------------------------------------------------------

namespace {

Tensor MakeUpsampleScales(const std::vector<float> &scales) {
  return Tensor::FromFloat("", {static_cast<int64_t>(scales.size())}, scales);
}

} // namespace

TEST(BackendKernelClass, UpsampleNearestMatchesUpstreamNCHWReference) {
  // Mirrors the upstream ``test_upsample_nearest`` node test exactly.
  const KernelContext ctx{DefaultOpset(9)};
  onnx_kernels::kernel::Upsample upsample{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {1.f, 2.f, 3.f, 4.f});
  Tensor scales = MakeUpsampleScales({1.f, 1.f, 2.f, 3.f});
  onnx_kernels::kernel::Upsample::Attributes a;
  a.mode = "nearest";
  Tensor y = upsample(x, scales, a);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 4, 6}));
  const std::vector<float> expected{
      1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f,
      3.f, 3.f, 3.f, 4.f, 4.f, 4.f, 3.f, 3.f, 3.f, 4.f, 4.f, 4.f,
  };
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "at index " << i;
  }
}

TEST(BackendKernelClass, UpsampleNearest1DDoublesEachElement) {
  const KernelContext ctx{DefaultOpset(9)};
  onnx_kernels::kernel::Upsample upsample{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {10.f, 20.f, 30.f});
  Tensor scales = MakeUpsampleScales({2.f});
  onnx_kernels::kernel::Upsample::Attributes a;
  a.mode = "nearest";
  Tensor y = upsample(x, scales, a);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{6}));
  const std::vector<float> expected{10.f, 10.f, 20.f, 20.f, 30.f, 30.f};
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]);
  }
}

TEST(BackendKernelClass, UpsampleLinear4DBilinearReference) {
  // Bilinear upsampling of [[1, 2], [3, 4]] by 2x in both spatial axes
  // using the "asymmetric" coordinate transformation (in_coord =
  // out_coord / scale, clamped to [0, in_dim - 1]) matches the upstream
  // Upsample v7 semantics.
  const KernelContext ctx{DefaultOpset(9)};
  onnx_kernels::kernel::Upsample upsample{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {1.f, 2.f, 3.f, 4.f});
  Tensor scales = MakeUpsampleScales({1.f, 1.f, 2.f, 2.f});
  onnx_kernels::kernel::Upsample::Attributes a;
  a.mode = "linear";
  Tensor y = upsample(x, scales, a);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 4, 4}));
  // Reference computed by hand from the bilinear formula on a 2x2 grid
  // with in_coord = out_coord / 2 (clamped).
  const std::vector<float> expected{
      1.0f, 1.5f, 2.0f, 2.0f, 2.0f, 2.5f, 3.0f, 3.0f,
      3.0f, 3.5f, 4.0f, 4.0f, 3.0f, 3.5f, 4.0f, 4.0f,
  };
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "at index " << i;
  }
}

TEST(BackendKernelClass, UpsampleRejectsUnknownMode) {
  const KernelContext ctx{DefaultOpset(9)};
  onnx_kernels::kernel::Upsample upsample{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {1.f, 2.f, 3.f, 4.f});
  Tensor scales = MakeUpsampleScales({1.f, 1.f, 2.f, 2.f});
  onnx_kernels::kernel::Upsample::Attributes a;
  a.mode = "cubic";
  EXPECT_THROW((void)upsample(x, scales, a), std::invalid_argument);
}

TEST(BackendKernelClass, UpsampleRejectsScalesLengthMismatch) {
  const KernelContext ctx{DefaultOpset(9)};
  onnx_kernels::kernel::Upsample upsample{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {1.f, 2.f, 3.f, 4.f});
  // 3 scales for a rank-4 input.
  Tensor scales = MakeUpsampleScales({1.f, 2.f, 2.f});
  onnx_kernels::kernel::Upsample::Attributes a;
  EXPECT_THROW((void)upsample(x, scales, a), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Trilu kernel tests
// ---------------------------------------------------------------------------

TEST(BackendKernelClass, TriluUpperDefaultKeepsUpperTriangle) {
  const KernelContext ctx{DefaultOpset(14)};
  onnx_kernels::kernel::Trilu trilu{ctx};
  Tensor x = Tensor::FromFloat("", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  onnx_kernels::kernel::Trilu::Attributes attrs;
  Tensor y = trilu(x, /*k=*/nullptr, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3, 3}));
  const float *py = y.AsFloat();
  const std::vector<float> expected{1.0f, 2.0f, 3.0f, 0.0f, 5.0f, 6.0f, 0.0f, 0.0f, 9.0f};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, TriluLowerKeepsLowerTriangle) {
  const KernelContext ctx{DefaultOpset(14)};
  onnx_kernels::kernel::Trilu trilu{ctx};
  Tensor x = Tensor::FromFloat("", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  onnx_kernels::kernel::Trilu::Attributes attrs;
  attrs.upper = 0;
  Tensor y = trilu(x, /*k=*/nullptr, attrs);
  const float *py = y.AsFloat();
  const std::vector<float> expected{1.0f, 0.0f, 0.0f, 4.0f, 5.0f, 0.0f, 7.0f, 8.0f, 9.0f};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, TriluUpperWithPositiveK) {
  const KernelContext ctx{DefaultOpset(14)};
  onnx_kernels::kernel::Trilu trilu{ctx};
  // 3x4 matrix; upper, k=1 -> keep elements with j >= i + 1.
  Tensor x = Tensor::FromInt64("", {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor k = Tensor::FromInt64("", {}, {1});
  onnx_kernels::kernel::Trilu::Attributes attrs;
  Tensor y = trilu(x, &k, attrs);
  const std::vector<int64_t> expected{0, 2, 3, 4, 0, 0, 7, 8, 0, 0, 0, 12};
  const int64_t *py = y.AsInt64();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, TriluLowerWithNegativeK) {
  const KernelContext ctx{DefaultOpset(14)};
  onnx_kernels::kernel::Trilu trilu{ctx};
  // 3x3 matrix; lower, k=-1 -> keep elements with j <= i - 1 (strict lower).
  Tensor x = Tensor::FromFloat("", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  Tensor k = Tensor::FromInt64("", {}, {-1});
  onnx_kernels::kernel::Trilu::Attributes attrs;
  attrs.upper = 0;
  Tensor y = trilu(x, &k, attrs);
  const std::vector<float> expected{0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f, 7.0f, 8.0f, 0.0f};
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, TriluBatchedAppliesPerMatrix) {
  const KernelContext ctx{DefaultOpset(14)};
  onnx_kernels::kernel::Trilu trilu{ctx};
  // Two 2x2 matrices stacked along a leading batch dim.
  Tensor x = Tensor::FromFloat("", {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  onnx_kernels::kernel::Trilu::Attributes attrs;
  Tensor y = trilu(x, /*k=*/nullptr, attrs);
  const std::vector<float> expected{1.f, 2.f, 0.f, 4.f, 5.f, 6.f, 0.f, 8.f};
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, TriluRejectsRankLessThanTwo) {
  const KernelContext ctx{DefaultOpset(14)};
  onnx_kernels::kernel::Trilu trilu{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {1.f, 2.f, 3.f});
  onnx_kernels::kernel::Trilu::Attributes attrs;
  EXPECT_THROW((void)trilu(x, /*k=*/nullptr, attrs), std::invalid_argument);
}

TEST(BackendKernelClass, TriluRejectsNonInt64K) {
  const KernelContext ctx{DefaultOpset(14)};
  onnx_kernels::kernel::Trilu trilu{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  Tensor k = Tensor::FromInt32("", {}, {1});
  onnx_kernels::kernel::Trilu::Attributes attrs;
  EXPECT_THROW((void)trilu(x, &k, attrs), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// TensorScatter (opset 24).
// ---------------------------------------------------------------------------

TEST(BackendKernelClass, TensorScatterLinearWritesUpdateSlices) {
  const KernelContext ctx{DefaultOpset(24)};
  onnx_kernels::kernel::TensorScatter ts{ctx};
  // past_cache shape (2,1,4,5); update (2,1,1,5); write_indices=[1,2].
  Tensor past = Tensor::FromFloat("", {2, 1, 4, 5},
                                  {1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0,
                                   1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0});
  Tensor update = Tensor::FromFloat("", {2, 1, 1, 5}, {5, 5, 5, 5, 5, 1, 1, 1, 1, 1});
  Tensor write = Tensor::FromInt64("", {2}, {1, 2});
  onnx_kernels::kernel::TensorScatter::Attributes attrs;
  Tensor y = ts(past, update, &write, attrs);
  ASSERT_EQ(y.shape, past.shape);
  const float *py = y.AsFloat();
  // Batch 0: row 1 -> [5,5,5,5,5]. Batch 1: row 2 -> [1,1,1,1,1].
  const std::vector<float> expected{1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0,
                                    1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 1, 1, 1, 1, 1, 4, 3, 2, 1, 0};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, TensorScatterCircularWrapsWriteIndex) {
  const KernelContext ctx{DefaultOpset(24)};
  onnx_kernels::kernel::TensorScatter ts{ctx};
  // Single-batch (1,4) where update length 2 starts at index 3 and wraps to 0.
  Tensor past = Tensor::FromFloat("", {1, 4}, {0, 0, 0, 0});
  Tensor update = Tensor::FromFloat("", {1, 2}, {1, 2});
  Tensor write = Tensor::FromInt64("", {1}, {3});
  onnx_kernels::kernel::TensorScatter::Attributes attrs;
  attrs.axis = 1;
  attrs.mode = "circular";
  Tensor y = ts(past, update, &write, attrs);
  const float *py = y.AsFloat();
  // index 3 := 1 then (3+1) % 4 == 0 := 2.
  const std::vector<float> expected{2, 0, 0, 1};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, TensorScatterDefaultsWriteIndicesToZero) {
  const KernelContext ctx{DefaultOpset(24)};
  onnx_kernels::kernel::TensorScatter ts{ctx};
  Tensor past = Tensor::FromFloat("", {2, 3}, {0, 0, 0, 0, 0, 0});
  Tensor update = Tensor::FromFloat("", {2, 1}, {7, 9});
  onnx_kernels::kernel::TensorScatter::Attributes attrs;
  attrs.axis = 1;
  Tensor y = ts(past, update, /*write_indices=*/nullptr, attrs);
  const float *py = y.AsFloat();
  // write_indices defaults to all zeros, so update goes into column 0.
  const std::vector<float> expected{7, 0, 0, 9, 0, 0};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, TensorScatterRejectsOutOfRangeLinearIndex) {
  const KernelContext ctx{DefaultOpset(24)};
  onnx_kernels::kernel::TensorScatter ts{ctx};
  Tensor past = Tensor::FromFloat("", {1, 4}, {0, 0, 0, 0});
  Tensor update = Tensor::FromFloat("", {1, 2}, {1, 2});
  Tensor write = Tensor::FromInt64("", {1}, {3}); // 3+2 > 4 in linear mode.
  onnx_kernels::kernel::TensorScatter::Attributes attrs;
  attrs.axis = 1;
  EXPECT_THROW((void)ts(past, update, &write, attrs), std::invalid_argument);
}

TEST(BackendKernelClass, TensorScatterRejectsAxisOnBatchDim) {
  const KernelContext ctx{DefaultOpset(24)};
  onnx_kernels::kernel::TensorScatter ts{ctx};
  Tensor past = Tensor::FromFloat("", {2, 3}, {0, 0, 0, 0, 0, 0});
  Tensor update = Tensor::FromFloat("", {1, 3}, {1, 2, 3});
  onnx_kernels::kernel::TensorScatter::Attributes attrs;
  attrs.axis = 0; // batch axis is forbidden.
  EXPECT_THROW((void)ts(past, update, /*write_indices=*/nullptr, attrs), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// ReverseSequence kernel tests
// ---------------------------------------------------------------------------

TEST(BackendKernelClass, ReverseSequenceTimeAxisMatchesSpecExample1) {
  const KernelContext ctx{DefaultOpset(10)};
  onnx_kernels::kernel::ReverseSequence rs{ctx};
  // Spec example 1: 4x4 with time_axis=0, batch_axis=1.
  Tensor x = Tensor::FromFloat("", {4, 4},
                               {0.0f, 4.0f, 8.0f, 12.0f, 1.0f, 5.0f, 9.0f, 13.0f, 2.0f, 6.0f, 10.0f,
                                14.0f, 3.0f, 7.0f, 11.0f, 15.0f});
  Tensor seq = Tensor::FromInt64("", {4}, {4, 3, 2, 1});
  onnx_kernels::kernel::ReverseSequence::Attributes attrs;
  attrs.time_axis = 0;
  attrs.batch_axis = 1;
  Tensor y = rs(x, seq, attrs);
  const std::vector<float> expected{3.0f, 6.0f, 9.0f,  12.0f, 2.0f, 5.0f, 8.0f,  13.0f,
                                    1.0f, 4.0f, 10.0f, 14.0f, 0.0f, 7.0f, 11.0f, 15.0f};
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, ReverseSequenceBatchAxisMatchesSpecExample2) {
  const KernelContext ctx{DefaultOpset(10)};
  onnx_kernels::kernel::ReverseSequence rs{ctx};
  // Spec example 2: 4x4 with time_axis=1, batch_axis=0.
  Tensor x = Tensor::FromFloat("", {4, 4},
                               {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                11.0f, 12.0f, 13.0f, 14.0f, 15.0f});
  Tensor seq = Tensor::FromInt64("", {4}, {1, 2, 3, 4});
  onnx_kernels::kernel::ReverseSequence::Attributes attrs;
  attrs.time_axis = 1;
  attrs.batch_axis = 0;
  Tensor y = rs(x, seq, attrs);
  const std::vector<float> expected{0.0f,  1.0f, 2.0f, 3.0f,  5.0f,  4.0f,  6.0f,  7.0f,
                                    10.0f, 9.0f, 8.0f, 11.0f, 15.0f, 14.0f, 13.0f, 12.0f};
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, ReverseSequenceRank3CopiesInnerDimension) {
  const KernelContext ctx{DefaultOpset(10)};
  onnx_kernels::kernel::ReverseSequence rs{ctx};
  // shape [3 (time), 2 (batch), 2 (inner)] with sequence_lens=[3,1].
  Tensor x = Tensor::FromFloat("", {3, 2, 2},
                               {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
  Tensor seq = Tensor::FromInt64("", {2}, {3, 1});
  onnx_kernels::kernel::ReverseSequence::Attributes attrs;
  attrs.time_axis = 0;
  attrs.batch_axis = 1;
  Tensor y = rs(x, seq, attrs);
  // batch 0: full reverse along time -> (t=0:{8,9}, t=1:{4,5}, t=2:{0,1})
  // batch 1: only first 1 element reversed (identity) -> (t=0:{2,3}, t=1:{6,7}, t=2:{10,11})
  const std::vector<float> expected{8.f, 9.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 0.f, 1.f, 10.f, 11.f};
  const float *py = y.AsFloat();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]) << "i=" << i;
  }
}

TEST(BackendKernelClass, ReverseSequenceRejectsRankLessThanTwo) {
  const KernelContext ctx{DefaultOpset(10)};
  onnx_kernels::kernel::ReverseSequence rs{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {1.f, 2.f, 3.f});
  Tensor seq = Tensor::FromInt64("", {3}, {1, 1, 1});
  onnx_kernels::kernel::ReverseSequence::Attributes attrs;
  EXPECT_THROW((void)rs(x, seq, attrs), std::invalid_argument);
}

TEST(BackendKernelClass, ReverseSequenceRejectsNonInt64SequenceLens) {
  const KernelContext ctx{DefaultOpset(10)};
  onnx_kernels::kernel::ReverseSequence rs{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  Tensor seq = Tensor::FromInt32("", {2}, {2, 1});
  onnx_kernels::kernel::ReverseSequence::Attributes attrs;
  EXPECT_THROW((void)rs(x, seq, attrs), std::invalid_argument);
}

TEST(BackendKernelClass, ReverseSequenceRejectsAxesEqual) {
  const KernelContext ctx{DefaultOpset(10)};
  onnx_kernels::kernel::ReverseSequence rs{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.f, 2.f, 3.f, 4.f});
  Tensor seq = Tensor::FromInt64("", {2}, {2, 1});
  onnx_kernels::kernel::ReverseSequence::Attributes attrs;
  attrs.time_axis = 0;
  attrs.batch_axis = 0;
  EXPECT_THROW((void)rs(x, seq, attrs), std::invalid_argument);
}

TEST(BackendKernelClass, SplitClassEqualPartsByNumOutputs) {
  const KernelContext ctx{DefaultOpset(18)};
  onnx_kernels::kernel::Split split{ctx};
  Tensor x = Tensor::FromFloat("", {6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  std::vector<Tensor> outs = split(x, /*axis=*/0, /*split=*/{}, /*num_outputs=*/3);
  ASSERT_EQ(outs.size(), 3u);
  EXPECT_EQ(outs[0].shape, (std::vector<int64_t>{2}));
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[0], 1.f);
  EXPECT_FLOAT_EQ(outs[1].AsFloat()[1], 4.f);
  EXPECT_FLOAT_EQ(outs[2].AsFloat()[1], 6.f);
}

TEST(BackendKernelClass, SplitClassVariablePartsBySplitSizes) {
  const KernelContext ctx{DefaultOpset(18)};
  onnx_kernels::kernel::Split split{ctx};
  Tensor x = Tensor::FromFloat("", {2, 6},
                               {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
  std::vector<Tensor> outs = split(x, /*axis=*/1, /*split=*/{2, 4}, /*num_outputs=*/0);
  ASSERT_EQ(outs.size(), 2u);
  EXPECT_EQ(outs[0].shape, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(outs[1].shape, (std::vector<int64_t>{2, 4}));
  // First output row 1: {1,2}; row 2: {7,8}.
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[0], 1.f);
  EXPECT_FLOAT_EQ(outs[0].AsFloat()[3], 8.f);
  // Second output row 1: {3,4,5,6}; row 2: {9,10,11,12}.
  EXPECT_FLOAT_EQ(outs[1].AsFloat()[0], 3.f);
  EXPECT_FLOAT_EQ(outs[1].AsFloat()[7], 12.f);
}

TEST(BackendKernelClass, SplitClassUnevenLastChunkSmaller) {
  const KernelContext ctx{DefaultOpset(18)};
  onnx_kernels::kernel::Split split{ctx};
  Tensor x = Tensor::FromFloat("", {7}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
  std::vector<Tensor> outs = split(x, /*axis=*/0, /*split=*/{}, /*num_outputs=*/4);
  ASSERT_EQ(outs.size(), 4u);
  EXPECT_EQ(outs[0].shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(outs[1].shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(outs[2].shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(outs[3].shape, (std::vector<int64_t>{1}));
  EXPECT_FLOAT_EQ(outs[3].AsFloat()[0], 7.f);
}

TEST(BackendKernelClass, SplitClassNegativeAxis) {
  const KernelContext ctx{DefaultOpset(18)};
  onnx_kernels::kernel::Split split{ctx};
  Tensor x = Tensor::FromFloat("", {2, 4}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  std::vector<Tensor> outs = split(x, /*axis=*/-1, /*split=*/{}, /*num_outputs=*/2);
  ASSERT_EQ(outs.size(), 2u);
  EXPECT_EQ(outs[0].shape, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(outs[1].shape, (std::vector<int64_t>{2, 2}));
  EXPECT_FLOAT_EQ(outs[1].AsFloat()[3], 8.f);
}

TEST(BackendKernelClass, SplitClassRejectsBadAxis) {
  const KernelContext ctx{DefaultOpset(18)};
  onnx_kernels::kernel::Split split{ctx};
  Tensor x = Tensor::FromFloat("", {4}, {1.f, 2.f, 3.f, 4.f});
  EXPECT_THROW((void)split(x, /*axis=*/2, /*split=*/{}, /*num_outputs=*/2), std::invalid_argument);
}

TEST(BackendKernelClass, SplitClassRejectsSplitMismatch) {
  const KernelContext ctx{DefaultOpset(18)};
  onnx_kernels::kernel::Split split{ctx};
  Tensor x = Tensor::FromFloat("", {4}, {1.f, 2.f, 3.f, 4.f});
  EXPECT_THROW((void)split(x, /*axis=*/0, /*split=*/{1, 1}, /*num_outputs=*/0),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// OneHot kernel tests
// ---------------------------------------------------------------------------

TEST(BackendKernelClass, OneHotDefaultAxisMatchesReference) {
  const KernelContext ctx{DefaultOpset(11)};
  onnx_kernels::kernel::OneHot one_hot{ctx};
  Tensor indices = Tensor::FromInt64("", {3}, {0, 7, 8});
  Tensor depth = Tensor::FromFloat("", {}, {12.0f});
  Tensor values = Tensor::FromInt32("", {2}, {2, 5});
  onnx_kernels::kernel::OneHot::Attributes attrs; // axis = -1
  Tensor y = one_hot(indices, depth, values, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3, 12}));
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT32));
  const int32_t *py = y.AsInt32();
  // Expected rows: position 0, 7, 8 are 5 (on_value); rest 2 (off_value).
  for (int64_t i = 0; i < 3; ++i) {
    const std::vector<int64_t> on_positions{0, 7, 8};
    for (int64_t k = 0; k < 12; ++k) {
      const int32_t expected = (k == on_positions[i]) ? 5 : 2;
      EXPECT_EQ(py[i * 12 + k], expected) << "i=" << i << " k=" << k;
    }
  }
}

TEST(BackendKernelClass, OneHotWithAxisInsertsDimensionAtPosition) {
  const KernelContext ctx{DefaultOpset(11)};
  onnx_kernels::kernel::OneHot one_hot{ctx};
  // indices shape (2, 2), axis = 1 -> output shape (2, 10, 2).
  Tensor indices = Tensor::FromFloat("", {2, 2}, {1.0f, 9.0f, 2.0f, 4.0f});
  Tensor depth = Tensor::FromFloat("", {}, {10.0f});
  Tensor values = Tensor::FromFloat("", {2}, {1.0f, 3.0f});
  onnx_kernels::kernel::OneHot::Attributes attrs;
  attrs.axis = 1;
  Tensor y = one_hot(indices, depth, values, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 10, 2}));
  const float *py = y.AsFloat();
  // For each (outer, inner) the entry is 3 at the slot corresponding to
  // the (outer, inner) value of ``indices``, otherwise 1.
  const std::vector<std::vector<int64_t>> idx_values{{1, 9}, {2, 4}};
  for (int64_t outer = 0; outer < 2; ++outer) {
    for (int64_t k = 0; k < 10; ++k) {
      for (int64_t inner = 0; inner < 2; ++inner) {
        const float expected = (k == idx_values[outer][inner]) ? 3.0f : 1.0f;
        EXPECT_FLOAT_EQ(py[(outer * 10 + k) * 2 + inner], expected)
            << "outer=" << outer << " k=" << k << " inner=" << inner;
      }
    }
  }
}

TEST(BackendKernelClass, OneHotNegativeIndicesWrapAroundDepth) {
  const KernelContext ctx{DefaultOpset(11)};
  onnx_kernels::kernel::OneHot one_hot{ctx};
  // axis = 1, depth = 10, indices = [0, -7, -8] -> selected slots 0, 3, 2.
  Tensor indices = Tensor::FromInt64("", {3}, {0, -7, -8});
  Tensor depth = Tensor::FromFloat("", {}, {10.0f});
  Tensor values = Tensor::FromFloat("", {2}, {1.0f, 3.0f});
  onnx_kernels::kernel::OneHot::Attributes attrs;
  attrs.axis = 1;
  Tensor y = one_hot(indices, depth, values, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3, 10}));
  const float *py = y.AsFloat();
  const std::vector<int64_t> selected{0, 3, 2};
  for (int64_t i = 0; i < 3; ++i) {
    for (int64_t k = 0; k < 10; ++k) {
      const float expected = (k == selected[i]) ? 3.0f : 1.0f;
      EXPECT_FLOAT_EQ(py[i * 10 + k], expected) << "i=" << i << " k=" << k;
    }
  }
}

TEST(BackendKernelClass, OneHotOutOfRangeIndicesLeaveOffValue) {
  const KernelContext ctx{DefaultOpset(11)};
  onnx_kernels::kernel::OneHot one_hot{ctx};
  Tensor indices = Tensor::FromInt64("", {2}, {5, 100});
  Tensor depth = Tensor::FromInt64("", {}, {3});
  Tensor values = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
  onnx_kernels::kernel::OneHot::Attributes attrs;
  Tensor y = one_hot(indices, depth, values, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 3}));
  const float *py = y.AsFloat();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(py[i], 0.0f);
  }
}

TEST(BackendKernelClass, OneHotRejectsValuesNotRankOneOfTwo) {
  const KernelContext ctx{DefaultOpset(11)};
  onnx_kernels::kernel::OneHot one_hot{ctx};
  Tensor indices = Tensor::FromInt64("", {2}, {0, 1});
  Tensor depth = Tensor::FromInt64("", {}, {3});
  Tensor bad_values = Tensor::FromFloat("", {3}, {0.0f, 1.0f, 2.0f});
  onnx_kernels::kernel::OneHot::Attributes attrs;
  EXPECT_THROW((void)one_hot(indices, depth, bad_values, attrs), std::invalid_argument);
}

TEST(BackendKernelClass, OneHotRejectsZeroDepth) {
  const KernelContext ctx{DefaultOpset(11)};
  onnx_kernels::kernel::OneHot one_hot{ctx};
  Tensor indices = Tensor::FromInt64("", {2}, {0, 1});
  Tensor depth = Tensor::FromInt64("", {}, {0});
  Tensor values = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
  onnx_kernels::kernel::OneHot::Attributes attrs;
  EXPECT_THROW((void)one_hot(indices, depth, values, attrs), std::invalid_argument);
}

TEST(BackendKernelClass, OneHotRejectsAxisOutOfRange) {
  const KernelContext ctx{DefaultOpset(11)};
  onnx_kernels::kernel::OneHot one_hot{ctx};
  Tensor indices = Tensor::FromInt64("", {2}, {0, 1});
  Tensor depth = Tensor::FromInt64("", {}, {3});
  Tensor values = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
  onnx_kernels::kernel::OneHot::Attributes attrs;
  attrs.axis = 5;
  EXPECT_THROW((void)one_hot(indices, depth, values, attrs), std::invalid_argument);
}

} // namespace Test
