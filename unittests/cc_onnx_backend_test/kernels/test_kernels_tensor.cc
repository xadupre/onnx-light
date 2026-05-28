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
using onnx_backend_test::kernel::Concat;
using onnx_backend_test::kernel::KernelContext;

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
  Tensor y("out", TensorProto::DataType::FLOAT, {2, 5}, std::vector<uint8_t>(10 * sizeof(float)));
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
  Tensor bad_shape("", TensorProto::DataType::FLOAT, {3, 2},
                   std::vector<uint8_t>(6 * sizeof(float)));
  EXPECT_THROW(concat_kernel({x0, x1}, /*axis=*/0, bad_shape), std::invalid_argument);
}

TEST(BackendKernelClass, CastClassFloatToDouble) {
  const KernelContext ctx{DefaultOpset(13)};
  Cast cast_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-1.5f, 0.0f, 2.25f});
  Tensor y = cast_kernel(x, static_cast<int32_t>(TensorProto::DataType::DOUBLE));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::DOUBLE));
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
  Tensor y = cast_kernel(x, static_cast<int32_t>(TensorProto::DataType::INT32));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::INT32));
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
  Tensor y = cast_kernel(x, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
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
  Tensor y = cast_kernel(x, static_cast<int32_t>(TensorProto::DataType::FLOAT));
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
  EXPECT_THROW((void)cast_kernel(x, static_cast<int32_t>(TensorProto::DataType::FLOAT16)),
               std::invalid_argument);
}

TEST(BackendKernelClass, CastInPlaceRejectsDtypeMismatch) {
  const KernelContext ctx{DefaultOpset(13)};
  Cast cast_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  // Output dtype declared as FLOAT but the caller asks for INT32.
  Tensor wrong_out("", TensorProto::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
  EXPECT_THROW(cast_kernel(x, static_cast<int32_t>(TensorProto::DataType::INT32), wrong_out),
               std::invalid_argument);
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

} // namespace Test
