// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::Attention;
using onnx_kernels::kernel::AveragePool;
using onnx_kernels::kernel::BatchNormalization;
using onnx_kernels::kernel::Dropout;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::MaxPool;
using onnx_kernels::kernel::MaxUnpool;
using onnx_kernels::kernel::MeanVarianceNormalization;

namespace Test {

TEST(KernelClass, AveragePool2DDefault) {
  const KernelContext ctx{DefaultOpset(19)};
  AveragePool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
  Tensor y = pool(x, /*kernel_shape=*/{2, 2});
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  const std::vector<int64_t> expected_shape = {1, 1, 3, 3};
  EXPECT_EQ(y.shape, expected_shape);
  const float *py = y.AsFloat();
  // Averages of 2x2 windows: (1+2+5+6)/4=3.5, (2+3+6+7)/4=4.5, ...
  EXPECT_FLOAT_EQ(py[0], 3.5f);
  EXPECT_FLOAT_EQ(py[1], 4.5f);
  EXPECT_FLOAT_EQ(py[2], 5.5f);
  EXPECT_FLOAT_EQ(py[3], 7.5f);
  EXPECT_FLOAT_EQ(py[4], 8.5f);
  EXPECT_FLOAT_EQ(py[5], 9.5f);
  EXPECT_FLOAT_EQ(py[6], 11.5f);
  EXPECT_FLOAT_EQ(py[7], 12.5f);
  EXPECT_FLOAT_EQ(py[8], 13.5f);
}

TEST(KernelClass, AveragePool2DStrides) {
  const KernelContext ctx{DefaultOpset(19)};
  AveragePool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 5, 5},
                               {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
  Tensor y = pool(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2});
  const std::vector<int64_t> expected_shape = {1, 1, 2, 2};
  EXPECT_EQ(y.shape, expected_shape);
  const float *py = y.AsFloat();
  // Window means: avg of 3x3 in top-left corner is mean(1..3,6..8,11..13)=7.
  EXPECT_FLOAT_EQ(py[0], 7.0f);
  EXPECT_FLOAT_EQ(py[1], 9.0f);
  EXPECT_FLOAT_EQ(py[2], 17.0f);
  EXPECT_FLOAT_EQ(py[3], 19.0f);
}

TEST(KernelClass, AveragePool2DPadsCountIncludePad) {
  const KernelContext ctx{DefaultOpset(19)};
  AveragePool pool{ctx};
  Tensor x =
      Tensor::FromFloat("", {1, 1, 3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  // 3x3 kernel, strides 1, pad 1 on every side, count_include_pad=true =>
  // every output divides by 9 (kernel area).
  Tensor y = pool(x, /*kernel_shape=*/{3, 3}, /*strides=*/{1, 1},
                  /*pads=*/{1, 1, 1, 1}, /*ceil_mode=*/false, /*count_include_pad=*/true);
  const std::vector<int64_t> expected_shape = {1, 1, 3, 3};
  EXPECT_EQ(y.shape, expected_shape);
  const float *py = y.AsFloat();
  // Top-left output = (0+0+0+0+1+2+0+4+5)/9 = 12/9.
  EXPECT_FLOAT_EQ(py[0], 12.0f / 9.0f);
  // Center output = (1+2+3+4+5+6+7+8+9)/9 = 5.0.
  EXPECT_FLOAT_EQ(py[4], 5.0f);
  // Bottom-right output = (5+6+0+8+9+0+0+0+0)/9 = 28/9.
  EXPECT_FLOAT_EQ(py[8], 28.0f / 9.0f);
}

TEST(KernelClass, AveragePool2DPadsCountExcludePad) {
  const KernelContext ctx{DefaultOpset(19)};
  AveragePool pool{ctx};
  Tensor x =
      Tensor::FromFloat("", {1, 1, 3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  Tensor y = pool(x, /*kernel_shape=*/{3, 3}, /*strides=*/{1, 1},
                  /*pads=*/{1, 1, 1, 1}, /*ceil_mode=*/false, /*count_include_pad=*/false);
  const float *py = y.AsFloat();
  // Top-left output: only 4 in-bounds elements (1,2,4,5) divided by 4 = 3.
  EXPECT_FLOAT_EQ(py[0], 3.0f);
  // Center stays 5.
  EXPECT_FLOAT_EQ(py[4], 5.0f);
  // Bottom-right: in-bounds (5,6,8,9) divided by 4 = 7.
  EXPECT_FLOAT_EQ(py[8], 7.0f);
}

TEST(KernelClass, AveragePoolCeilMode) {
  const KernelContext ctx{DefaultOpset(19)};
  AveragePool pool{ctx};
  // 1x1x4x4 input, 3x3 kernel, stride 2; floor gives 1x1, ceil gives 2x2.
  Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
  Tensor y_floor = pool(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2});
  EXPECT_EQ(y_floor.shape, (std::vector<int64_t>{1, 1, 1, 1}));
  Tensor y_ceil = pool(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2}, /*pads=*/{},
                       /*ceil_mode=*/true, /*count_include_pad=*/false);
  EXPECT_EQ(y_ceil.shape, (std::vector<int64_t>{1, 1, 2, 2}));
}

TEST(KernelClass, AveragePool1D) {
  const KernelContext ctx{DefaultOpset(19)};
  AveragePool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  Tensor y = pool(x, /*kernel_shape=*/{3});
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 1, 3}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 2.0f);
  EXPECT_FLOAT_EQ(py[1], 3.0f);
  EXPECT_FLOAT_EQ(py[2], 4.0f);
}

TEST(KernelClass, AveragePoolRejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(19)};
  AveragePool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
  // Empty kernel_shape is rejected.
  EXPECT_THROW(pool(x, /*kernel_shape=*/{}), std::invalid_argument);
  // Rank mismatch between x and kernel_shape (x is 4-D so kernel_shape must be size 2).
  EXPECT_THROW(pool(x, /*kernel_shape=*/{2}), std::invalid_argument);
  // Non-FLOAT input.
  Tensor bad_x = Tensor::FromInt32("", {1, 1, 4, 4}, std::vector<int32_t>(16, 1));
  EXPECT_THROW(pool(bad_x, /*kernel_shape=*/{2, 2}), std::invalid_argument);
  // Wrong-length strides.
  EXPECT_THROW(pool(x, /*kernel_shape=*/{2, 2}, /*strides=*/{1}), std::invalid_argument);
  // Wrong-length pads.
  EXPECT_THROW(pool(x, /*kernel_shape=*/{2, 2}, /*strides=*/{1, 1}, /*pads=*/{1, 1, 1}),
               std::invalid_argument);
}

TEST(KernelClass, AveragePool2DDilations) {
  // mirrors test_averagepool_2d_dilations: 4x4 input 1..16, kernel 2x2,
  // dilations (2,2), stride 1, ceil_mode -> 2x2 output [[6, 7], [10, 11]].
  const KernelContext ctx{DefaultOpset(19)};
  AveragePool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
  Tensor y = pool(x, /*kernel_shape=*/{2, 2}, /*strides=*/{1, 1}, /*pads=*/{},
                  /*ceil_mode=*/true, /*count_include_pad=*/false, /*dilations=*/{2, 2});
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 1, 2, 2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 6.0f);
  EXPECT_FLOAT_EQ(py[1], 7.0f);
  EXPECT_FLOAT_EQ(py[2], 10.0f);
  EXPECT_FLOAT_EQ(py[3], 11.0f);
}

TEST(KernelClass, AveragePool2DAutoPadSameUpperPrecomputed) {
  // mirrors test_averagepool_2d_precomputed_same_upper: 5x5 input 1..25,
  // kernel 3x3, stride 2, auto_pad=SAME_UPPER -> 3x3 output
  // [[4, 5.5, 7], [11.5, 13, 14.5], [19, 20.5, 22]].
  const KernelContext ctx{DefaultOpset(19)};
  AveragePool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 5, 5},
                               {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
  Tensor y = pool(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2}, /*pads=*/{},
                  /*ceil_mode=*/false, /*count_include_pad=*/false, /*dilations=*/{},
                  /*auto_pad=*/"SAME_UPPER");
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 1, 3, 3}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 4.0f);
  EXPECT_FLOAT_EQ(py[1], 5.5f);
  EXPECT_FLOAT_EQ(py[2], 7.0f);
  EXPECT_FLOAT_EQ(py[3], 11.5f);
  EXPECT_FLOAT_EQ(py[4], 13.0f);
  EXPECT_FLOAT_EQ(py[5], 14.5f);
  EXPECT_FLOAT_EQ(py[6], 19.0f);
  EXPECT_FLOAT_EQ(py[7], 20.5f);
  EXPECT_FLOAT_EQ(py[8], 22.0f);
}

TEST(KernelClass, AveragePoolAutoPadAndPadsAreMutuallyExclusive) {
  const KernelContext ctx{DefaultOpset(19)};
  AveragePool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
  // Non-empty ``pads`` together with auto_pad != NOTSET is rejected.
  EXPECT_THROW(pool(x, /*kernel_shape=*/{2, 2}, /*strides=*/{1, 1}, /*pads=*/{1, 1, 1, 1},
                    /*ceil_mode=*/false, /*count_include_pad=*/false, /*dilations=*/{},
                    /*auto_pad=*/"SAME_UPPER"),
               std::invalid_argument);
  // Unknown auto_pad value is rejected.
  EXPECT_THROW(pool(x, /*kernel_shape=*/{2, 2}, /*strides=*/{}, /*pads=*/{},
                    /*ceil_mode=*/false, /*count_include_pad=*/false, /*dilations=*/{},
                    /*auto_pad=*/"NONSENSE"),
               std::invalid_argument);
  // Wrong-length dilations is rejected.
  EXPECT_THROW(pool(x, /*kernel_shape=*/{2, 2}, /*strides=*/{1, 1}, /*pads=*/{},
                    /*ceil_mode=*/false, /*count_include_pad=*/false, /*dilations=*/{2}),
               std::invalid_argument);
}

TEST(KernelClass, BatchNormalizationInferenceMatchesFormula) {
  const KernelContext ctx{DefaultOpset(15)};
  BatchNormalization bn{ctx};
  // 1x2x1x3 input (the same shape as test_cc_batchnorm_example). With
  // mean = 0 / 3, var = 1 / 1.5, scale = 1 / 1.5, B = 0 / 1 and the default
  // epsilon = 1e-5, the inference formula reduces to:
  //   channel 0: y = x
  //   channel 1: y = (x - 3) * 1.5 / sqrt(1.5) + 1
  Tensor x = Tensor::FromFloat("", {1, 2, 1, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
  Tensor scale = Tensor::FromFloat("", {2}, {1.0f, 1.5f});
  Tensor bias = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
  Tensor mean = Tensor::FromFloat("", {2}, {0.0f, 3.0f});
  Tensor var = Tensor::FromFloat("", {2}, {1.0f, 1.5f});
  Tensor y = bn(x, scale, bias, mean, var);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  const std::vector<int64_t> expected_shape = {1, 2, 1, 3};
  EXPECT_EQ(y.shape, expected_shape);
  const float *py = y.AsFloat();
  // Channel 0: scale = 1, mean = 0, var = 1 (≈ identity at epsilon -> 0).
  EXPECT_NEAR(py[0], -1.0f, 1e-3f);
  EXPECT_NEAR(py[1], 0.0f, 1e-3f);
  EXPECT_NEAR(py[2], 1.0f, 1e-3f);
  // Channel 1: y = (x - 3) * 1.5 / sqrt(1.5) + 1 ≈ (x - 3) * 1.2247 + 1.
  const float k = 1.5f / std::sqrt(1.5f);
  EXPECT_NEAR(py[3], (2.0f - 3.0f) * k + 1.0f, 1e-3f);
  EXPECT_NEAR(py[4], (3.0f - 3.0f) * k + 1.0f, 1e-3f);
  EXPECT_NEAR(py[5], (4.0f - 3.0f) * k + 1.0f, 1e-3f);
}

TEST(KernelClass, BatchNormalizationRejectsWrongChannelSize) {
  const KernelContext ctx{DefaultOpset(15)};
  BatchNormalization bn{ctx};
  Tensor x = Tensor::FromFloat("", {1, 2, 1, 1}, {0.0f, 0.0f});
  // scale has size 3 but C is 2 → must be rejected.
  Tensor scale = Tensor::FromFloat("", {3}, {1.0f, 1.0f, 1.0f});
  Tensor bias = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  Tensor mean = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  Tensor var = Tensor::FromFloat("", {2}, {1.0f, 1.0f});
  EXPECT_THROW(bn(x, scale, bias, mean, var), std::invalid_argument);
}

TEST(KernelClass, BatchNormalizationRank1InputTreatsChannelAsOne) {
  const KernelContext ctx{DefaultOpset(15)};
  BatchNormalization bn{ctx};
  Tensor x = Tensor::FromFloat("", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor scale = Tensor::FromFloat("", {1}, {2.0f});
  Tensor bias = Tensor::FromFloat("", {1}, {-1.0f});
  Tensor mean = Tensor::FromFloat("", {1}, {2.5f});
  Tensor var = Tensor::FromFloat("", {1}, {1.0f});
  Tensor y = bn(x, scale, bias, mean, var);
  ASSERT_EQ(y.shape.size(), 1u);
  EXPECT_EQ(y.shape[0], 4);
  const float *py = y.AsFloat();
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(py[i], (static_cast<float>(i + 1) - 2.5f) * 2.0f + (-1.0f), 1e-3f);
  }
}

TEST(KernelClass, MeanVarianceNormalizationDefaultAxes) {
  const KernelContext ctx{DefaultOpset(13)};
  MeanVarianceNormalization mvn{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2, 1, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = mvn(x);
  ASSERT_EQ(y.shape, x.shape);
  const float *py = y.AsFloat();
  // Default axes [0,2,3] normalize channel-wise across N.
  EXPECT_NEAR(py[0], -1.0f, 1e-5f);
  EXPECT_NEAR(py[1], -1.0f, 1e-5f);
  EXPECT_NEAR(py[2], 1.0f, 1e-5f);
  EXPECT_NEAR(py[3], 1.0f, 1e-5f);
}

TEST(KernelClass, MeanVarianceNormalizationCustomAxes) {
  const KernelContext ctx{DefaultOpset(13)};
  MeanVarianceNormalization mvn{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 3.0f, 5.0f, 7.0f});
  Tensor y = mvn(x, {1});
  ASSERT_EQ(y.shape, x.shape);
  const float *py = y.AsFloat();
  EXPECT_NEAR(py[0], -1.0f, 1e-5f);
  EXPECT_NEAR(py[1], 1.0f, 1e-5f);
  EXPECT_NEAR(py[2], -1.0f, 1e-5f);
  EXPECT_NEAR(py[3], 1.0f, 1e-5f);
}

TEST(KernelClass, MeanVarianceNormalizationRejectsAxisOutOfRange) {
  const KernelContext ctx{DefaultOpset(13)};
  MeanVarianceNormalization mvn{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_THROW(mvn(x, {2}), std::invalid_argument);
}

TEST(KernelClass, DropoutInferenceModeCopiesInputAndOnesMask) {
  const KernelContext ctx{DefaultOpset(22)};
  Dropout dropout{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
  Tensor mask("", static_cast<int32_t>(onnx_kernels::DataType::BOOL), x.shape,
              std::vector<uint8_t>(6, 0));
  Tensor y = dropout(x, /*ratio=*/0.5f, /*training_mode=*/false, mask);
  ASSERT_EQ(y.shape, x.shape);
  ASSERT_EQ(y.data_type, x.data_type);
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(y.AsFloat()[i], x.AsFloat()[i]);
    EXPECT_EQ(mask.AsBool()[i], static_cast<uint8_t>(1));
  }
}

TEST(KernelClass, DropoutTrainingModeIsDeterministicForSeed) {
  const KernelContext ctx{DefaultOpset(22)};
  Dropout dropout{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
  auto y0 = dropout(x, /*ratio=*/0.4f, /*training_mode=*/true, /*seed=*/123);
  auto y1 = dropout(x, /*ratio=*/0.4f, /*training_mode=*/true, /*seed=*/123);
  ASSERT_EQ(y0.first.data, y1.first.data);
  ASSERT_EQ(y0.second.data, y1.second.data);
}

TEST(KernelClass, DropoutRejectsInvalidRatio) {
  const KernelContext ctx{DefaultOpset(22)};
  Dropout dropout{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor mask("", static_cast<int32_t>(onnx_kernels::DataType::BOOL), x.shape,
              std::vector<uint8_t>(2, 0));
  EXPECT_THROW(dropout(x, /*ratio=*/1.0f, /*training_mode=*/true, mask), std::invalid_argument);
}

// ---- Attention -----------------------------------------------------------

namespace {

KernelContext AttentionKernelContext() { return KernelContext(DefaultOpset(23)); }

} // namespace

TEST(KernelClass, AttentionMatchesHandComputedSingleHead) {
  // batch=1, heads=1, q_len=1, k_len=2, head_size=2, v_head_size=2,
  // default scale = 1/sqrt(head_size) = 1/sqrt(2).
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  const KernelContext ctx = AttentionKernelContext();
  const Attention attention{ctx};
  const Tensor Y = attention(Q, K, V);
  ASSERT_EQ(Y.shape, (std::vector<int64_t>{1, 1, 1, 2}));

  const double s = 1.0 / std::sqrt(2.0);
  const double e0 = std::exp(s);
  const double e1 = 1.0;
  const double p0 = e0 / (e0 + e1);
  const double p1 = e1 / (e0 + e1);
  EXPECT_NEAR(Y.AsFloat()[0], static_cast<float>(p0 * 1.0 + p1 * 3.0), 1e-6);
  EXPECT_NEAR(Y.AsFloat()[1], static_cast<float>(p0 * 2.0 + p1 * 4.0), 1e-6);
}

TEST(KernelClass, AttentionExplicitScaleMatchesDefault) {
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  const KernelContext ctx = AttentionKernelContext();
  const Attention attention{ctx};
  const Tensor Y_default = attention(Q, K, V);
  const Tensor Y_explicit = attention(Q, K, V, 1.0f / std::sqrt(2.0f));
  ASSERT_EQ(Y_default.shape, Y_explicit.shape);
  for (int64_t i = 0; i < Y_default.element_count(); ++i) {
    EXPECT_NEAR(Y_default.AsFloat()[i], Y_explicit.AsFloat()[i], 1e-6);
  }
}

TEST(KernelClass, AttentionSupportsGQAHeadSharing) {
  // q_num_heads=2, kv_num_heads=1 (group_size=2). Single key/value position
  // → softmax probability is 1, so every Q head must reproduce V[0] exactly.
  const Tensor Q = Tensor::FromFloat("", {1, 2, 1, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 1, 2}, {0.5f, -0.5f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 1, 2}, {7.0f, -3.0f});

  const KernelContext ctx = AttentionKernelContext();
  const Attention attention{ctx};
  const Tensor Y = attention(Q, K, V);
  ASSERT_EQ(Y.shape, (std::vector<int64_t>{1, 2, 1, 2}));
  for (int64_t h = 0; h < 2; ++h) {
    EXPECT_FLOAT_EQ(Y.AsFloat()[h * 2 + 0], 7.0f);
    EXPECT_FLOAT_EQ(Y.AsFloat()[h * 2 + 1], -3.0f);
  }
}

TEST(KernelClass, AttentionCausalMasksFuturePositions) {
  // q_seq=2, kv_seq=2. ``is_causal`` allows row 0 to attend only to col 0,
  // and row 1 to attend to cols 0 and 1. With Q=I row 0 must therefore
  // reproduce V[0] exactly regardless of K[1]/V[1].
  const Tensor Q = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {10.0f, -10.0f, 100.0f, -100.0f});

  const KernelContext ctx = AttentionKernelContext();
  const Attention attention{ctx};
  Attention::Attributes attrs;
  attrs.is_causal = true;
  const Tensor Y = attention(Q, K, V, attrs).Y;
  ASSERT_EQ(Y.shape, (std::vector<int64_t>{1, 1, 2, 2}));
  // Row 0 = V[0] exactly (only attends to position 0).
  EXPECT_FLOAT_EQ(Y.AsFloat()[0], 10.0f);
  EXPECT_FLOAT_EQ(Y.AsFloat()[1], -10.0f);
  // Row 1 attends to both; output stays within [V[0], V[1]] bounds.
  EXPECT_GT(Y.AsFloat()[2], 10.0f);
  EXPECT_LT(Y.AsFloat()[2], 100.0f);
  EXPECT_LT(Y.AsFloat()[3], -10.0f);
  EXPECT_GT(Y.AsFloat()[3], -100.0f);
}

TEST(KernelClass, AttentionBoolMaskExcludesFalsePositions) {
  // mask=[true, false] forces softmax to put all weight on position 0,
  // so the output must equal V[0] exactly.
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {5.0f, 7.0f, 11.0f, 13.0f});
  const Tensor mask = Tensor::FromBool("", {1, 1, 1, 2}, {1, 0});

  const KernelContext ctx = AttentionKernelContext();
  const Attention attention{ctx};
  Attention::Attributes attrs;
  const Tensor Y = attention(Q, K, V, attrs, &mask).Y;
  ASSERT_EQ(Y.shape, (std::vector<int64_t>{1, 1, 1, 2}));
  EXPECT_FLOAT_EQ(Y.AsFloat()[0], 5.0f);
  EXPECT_FLOAT_EQ(Y.AsFloat()[1], 7.0f);
}

TEST(KernelClass, AttentionFloatMaskAddsBias) {
  // A large negative bias on position 1 effectively excludes it, mirroring
  // the BOOL-false case.
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {5.0f, 7.0f, 11.0f, 13.0f});
  const Tensor mask = Tensor::FromFloat("", {1, 1, 1, 2}, {0.0f, -1e9f});

  const KernelContext ctx = AttentionKernelContext();
  const Attention attention{ctx};
  const Tensor Y = attention(Q, K, V, /*scale=*/1.0f, mask);
  ASSERT_EQ(Y.shape, (std::vector<int64_t>{1, 1, 1, 2}));
  EXPECT_NEAR(Y.AsFloat()[0], 5.0f, 1e-5f);
  EXPECT_NEAR(Y.AsFloat()[1], 7.0f, 1e-5f);
}

TEST(KernelClass, AttentionSoftcapSaturatesExtremeScores) {
  // With huge raw scores ``s`` (here ``scale = 1e3``) the softcap
  // saturates them to ±softcap. The two key positions become
  // [+softcap, -softcap] which is far enough apart for softmax to put
  // ~all weight on position 0, hence Y ≈ V[0].
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 1}, {1.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 1}, {1.0f, -1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 1}, {2.0f, -3.0f});

  const KernelContext ctx = AttentionKernelContext();
  const Attention attention{ctx};
  Attention::Attributes attrs;
  attrs.has_scale = true;
  attrs.scale = 1e3f;
  attrs.softcap = 1.0f;
  const Tensor Y = attention(Q, K, V, attrs).Y;
  ASSERT_EQ(Y.shape, (std::vector<int64_t>{1, 1, 1, 1}));
  // exp(1)/(exp(1)+exp(-1)) ≈ 0.8808; weighted output ≈ 0.8808*2 + 0.1192*-3.
  const double w0 = std::exp(1.0) / (std::exp(1.0) + std::exp(-1.0));
  const double w1 = 1.0 - w0;
  EXPECT_NEAR(Y.AsFloat()[0], static_cast<float>(w0 * 2.0 + w1 * -3.0), 1e-5f);
}

TEST(KernelClass, AttentionPastKVConcatenatesIntoPresent) {
  // present_key/present_value are the concatenation along axis 2 of past_*
  // and the current K/V.
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 1, 2}, {3.0f, 4.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 1, 2}, {30.0f, 40.0f});
  const Tensor past_key = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor past_value = Tensor::FromFloat("", {1, 1, 2, 2}, {10.0f, 11.0f, 12.0f, 13.0f});

  const KernelContext ctx = AttentionKernelContext();
  const Attention attention{ctx};
  Attention::Attributes attrs;
  auto r = attention(Q, K, V, attrs, /*attn_mask=*/nullptr, &past_key, &past_value);
  ASSERT_EQ(r.present_key.shape, (std::vector<int64_t>{1, 1, 3, 2}));
  ASSERT_EQ(r.present_value.shape, (std::vector<int64_t>{1, 1, 3, 2}));
  // present_key prefix matches past_key.
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(r.present_key.AsFloat()[i], past_key.AsFloat()[i]);
    EXPECT_FLOAT_EQ(r.present_value.AsFloat()[i], past_value.AsFloat()[i]);
  }
  // present_key suffix == current K.
  EXPECT_FLOAT_EQ(r.present_key.AsFloat()[4], 3.0f);
  EXPECT_FLOAT_EQ(r.present_key.AsFloat()[5], 4.0f);
  EXPECT_FLOAT_EQ(r.present_value.AsFloat()[4], 30.0f);
  EXPECT_FLOAT_EQ(r.present_value.AsFloat()[5], 40.0f);
  // Y has the expected shape (B, Hq, Lq, Dv).
  EXPECT_EQ(r.Y.shape, (std::vector<int64_t>{1, 1, 1, 2}));
}

TEST(KernelClass, AttentionQkMatmulOutputModes) {
  // (B=1, H=1, Lq=1, Lk=2, D=Dv=2). scale=2.0 makes the raw scores
  // easy to verify by hand.
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor mask = Tensor::FromFloat("", {1, 1, 1, 2}, {0.0f, -0.5f});

  const KernelContext ctx = AttentionKernelContext();
  const Attention attention{ctx};

  // Mode 0: raw QK^T * scale, no bias.
  {
    Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 2.0f;
    attrs.qk_matmul_output_mode = 0;
    auto r = attention(Q, K, V, attrs, &mask);
    ASSERT_EQ(r.qk_matmul_output.shape, (std::vector<int64_t>{1, 1, 1, 2}));
    EXPECT_FLOAT_EQ(r.qk_matmul_output.AsFloat()[0], 2.0f); // 1*1*2
    EXPECT_FLOAT_EQ(r.qk_matmul_output.AsFloat()[1], 0.0f); // 1*0*2
  }
  // Mode 1: with bias = mask added.
  {
    Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 2.0f;
    attrs.qk_matmul_output_mode = 1;
    auto r = attention(Q, K, V, attrs, &mask);
    EXPECT_FLOAT_EQ(r.qk_matmul_output.AsFloat()[0], 2.0f + 0.0f);
    EXPECT_FLOAT_EQ(r.qk_matmul_output.AsFloat()[1], 0.0f + -0.5f);
  }
  // Mode 3: softmax probabilities sum to 1.
  {
    Attention::Attributes attrs;
    attrs.has_scale = true;
    attrs.scale = 2.0f;
    attrs.qk_matmul_output_mode = 3;
    auto r = attention(Q, K, V, attrs, &mask);
    const double sum =
        static_cast<double>(r.qk_matmul_output.AsFloat()[0]) + r.qk_matmul_output.AsFloat()[1];
    EXPECT_NEAR(sum, 1.0, 1e-6);
  }
}

TEST(KernelClass, AttentionRank3FusedLayoutRoundTripMatchesRank4) {
  // Same numerical data as the rank-4 case but expressed in the
  // ``(batch, seq, num_heads * head_size)`` fused layout, with
  // ``q_num_heads`` and ``kv_num_heads`` set on the attributes.
  const Tensor Q4 = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K4 = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V4 = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  const Tensor Q3 = Tensor::FromFloat("", {1, 1, 2}, {1.0f, 0.0f});
  const Tensor K3 = Tensor::FromFloat("", {1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V3 = Tensor::FromFloat("", {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  const KernelContext ctx = AttentionKernelContext();
  const Attention attention{ctx};
  const Tensor Y4 = attention(Q4, K4, V4);
  Attention::Attributes attrs;
  attrs.q_num_heads = 1;
  attrs.kv_num_heads = 1;
  const Tensor Y3 = attention(Q3, K3, V3, attrs).Y;

  ASSERT_EQ(Y4.shape, (std::vector<int64_t>{1, 1, 1, 2}));
  ASSERT_EQ(Y3.shape, (std::vector<int64_t>{1, 1, 2}));
  for (int64_t i = 0; i < Y4.element_count(); ++i) {
    EXPECT_NEAR(Y3.AsFloat()[i], Y4.AsFloat()[i], 1e-6);
  }
}

TEST(KernelClass, AttentionInPlaceOverloadWritesIntoOutput) {
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  const KernelContext ctx = AttentionKernelContext();
  const Attention attention{ctx};
  const Tensor expected = attention(Q, K, V, 1.0f / std::sqrt(2.0f));
  Tensor out("", onnx_kernels::DataType::FLOAT, {1, 1, 1, 2},
             std::vector<uint8_t>(2 * sizeof(float)));
  attention(Q, K, V, 1.0f / std::sqrt(2.0f), /*attn_mask=*/nullptr, out);
  for (int64_t i = 0; i < expected.element_count(); ++i) {
    EXPECT_FLOAT_EQ(out.AsFloat()[i], expected.AsFloat()[i]);
  }

  // Mismatched output shape is rejected.
  Tensor bad_shape("", onnx_kernels::DataType::FLOAT, {1, 1, 1, 1},
                   std::vector<uint8_t>(sizeof(float)));
  EXPECT_THROW(attention(Q, K, V, 1.0f / std::sqrt(2.0f), /*attn_mask=*/nullptr, bad_shape),
               std::invalid_argument);
  // Non-FLOAT output buffer is rejected.
  Tensor bad_type("", onnx_kernels::DataType::INT32, {1, 1, 1, 2},
                  std::vector<uint8_t>(2 * sizeof(int32_t)));
  EXPECT_THROW(attention(Q, K, V, 1.0f / std::sqrt(2.0f), /*attn_mask=*/nullptr, bad_type),
               std::invalid_argument);
}

TEST(KernelClass, AttentionRejectsInvalidInputs) {
  const KernelContext ctx = AttentionKernelContext();
  const Attention attention{ctx};

  // Mixed ranks across Q/K/V are rejected.
  {
    const Tensor Q3 = Tensor::FromFloat("", {1, 1, 2}, {1.0f, 0.0f});
    const Tensor K4 = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
    const Tensor V4 = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
    Attention::Attributes attrs;
    attrs.q_num_heads = 1;
    attrs.kv_num_heads = 1;
    EXPECT_THROW(attention(Q3, K4, V4, attrs), std::invalid_argument);
  }
  // Non rank-3/4 inputs are rejected.
  {
    const Tensor bad = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Attention::Attributes attrs;
    EXPECT_THROW(attention(bad, bad, bad, attrs), std::invalid_argument);
  }
  // Non-FLOAT input is rejected.
  {
    Tensor int_Q("", onnx_kernels::DataType::INT32, {1, 1, 1, 2},
                 std::vector<uint8_t>(2 * sizeof(int32_t)));
    const Tensor K = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
    const Tensor V = Tensor::FromFloat("", {1, 1, 1, 1}, {1.0f});
    EXPECT_THROW(attention(int_Q, K, V), std::invalid_argument);
  }
  // q_num_heads not a multiple of kv_num_heads is rejected.
  {
    const Tensor Q3h = Tensor::FromFloat("", {1, 3, 1, 2}, {1, 0, 0, 1, 1, 1});
    const Tensor K2h = Tensor::FromFloat("", {1, 2, 1, 2}, {1, 0, 0, 1});
    const Tensor V2h = Tensor::FromFloat("", {1, 2, 1, 1}, {1, 1});
    EXPECT_THROW(attention(Q3h, K2h, V2h), std::invalid_argument);
  }
  // Mismatched batch is rejected.
  {
    const Tensor Q1b = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
    const Tensor K2b = Tensor::FromFloat("", {2, 1, 1, 2}, {1, 0, 0, 1});
    const Tensor V2b = Tensor::FromFloat("", {2, 1, 1, 1}, {1, 1});
    EXPECT_THROW(attention(Q1b, K2b, V2b), std::invalid_argument);
  }
  // Mismatched head_size between Q and K is rejected.
  {
    const Tensor Qh2 = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
    const Tensor Kh3 = Tensor::FromFloat("", {1, 1, 1, 3}, {1.0f, 0.0f, 0.0f});
    const Tensor Vh1 = Tensor::FromFloat("", {1, 1, 1, 1}, {1.0f});
    EXPECT_THROW(attention(Qh2, Kh3, Vh1), std::invalid_argument);
  }
  // attn_mask that is not broadcastable is rejected.
  {
    const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
    const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1, 0, 0, 1});
    const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {1, 2, 3, 4});
    // mask q_seq dim = 2 but actual q_seq_len = 1 and 2 != 1.
    const Tensor bad_mask = Tensor::FromFloat("", {2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});
    Attention::Attributes attrs;
    EXPECT_THROW(attention(Q, K, V, attrs, &bad_mask).Y, std::invalid_argument);
  }
}

TEST(KernelClass, MaxPool2DDefault) {
  const KernelContext ctx{DefaultOpset(22)};
  MaxPool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
  Tensor y = pool(x, /*kernel_shape=*/{2, 2});
  const std::vector<int64_t> expected_shape = {1, 1, 3, 3};
  EXPECT_EQ(y.shape, expected_shape);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 6.0f);
  EXPECT_FLOAT_EQ(py[1], 7.0f);
  EXPECT_FLOAT_EQ(py[2], 8.0f);
  EXPECT_FLOAT_EQ(py[3], 10.0f);
  EXPECT_FLOAT_EQ(py[4], 11.0f);
  EXPECT_FLOAT_EQ(py[5], 12.0f);
  EXPECT_FLOAT_EQ(py[6], 14.0f);
  EXPECT_FLOAT_EQ(py[7], 15.0f);
  EXPECT_FLOAT_EQ(py[8], 16.0f);
}

TEST(KernelClass, MaxPool2DCeil) {
  // mirrors test_maxpool_2d_ceil: kernel (3, 3), stride (2, 2), ceil_mode=1.
  const KernelContext ctx{DefaultOpset(22)};
  MaxPool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
  Tensor y = pool(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2}, /*pads=*/{},
                  /*ceil_mode=*/true);
  const std::vector<int64_t> expected_shape = {1, 1, 2, 2};
  EXPECT_EQ(y.shape, expected_shape);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 11.0f);
  EXPECT_FLOAT_EQ(py[1], 12.0f);
  EXPECT_FLOAT_EQ(py[2], 15.0f);
  EXPECT_FLOAT_EQ(py[3], 16.0f);
}

TEST(KernelClass, MaxPool2DDilations) {
  // mirrors test_maxpool_2d_dilations: kernel (2, 2), dilations (2, 2).
  const KernelContext ctx{DefaultOpset(22)};
  MaxPool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
  Tensor y = pool(x, /*kernel_shape=*/{2, 2}, /*strides=*/{1, 1}, /*pads=*/{},
                  /*ceil_mode=*/false, /*dilations=*/{2, 2});
  const std::vector<int64_t> expected_shape = {1, 1, 2, 2};
  EXPECT_EQ(y.shape, expected_shape);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 11.0f);
  EXPECT_FLOAT_EQ(py[1], 12.0f);
  EXPECT_FLOAT_EQ(py[2], 15.0f);
  EXPECT_FLOAT_EQ(py[3], 16.0f);
}

TEST(KernelClass, MaxPoolWithIndices) {
  // mirrors test_maxpool_with_argmax_2d_precomputed_pads.
  const KernelContext ctx{DefaultOpset(22)};
  MaxPool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 5, 5},
                               {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
                                10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f});
  auto yz = pool.WithIndices(x, /*kernel_shape=*/{5, 5}, /*strides=*/{1, 1},
                             /*pads=*/{2, 2, 2, 2});
  const Tensor &y = yz.first;
  const Tensor &indices = yz.second;
  const std::vector<int64_t> expected_shape = {1, 1, 5, 5};
  EXPECT_EQ(y.shape, expected_shape);
  EXPECT_EQ(indices.shape, expected_shape);
  ASSERT_EQ(indices.data_type, static_cast<int32_t>(onnx_kernels::DataType::INT64));
  const float *py = y.AsFloat();
  const int64_t *pi = indices.AsInt64();
  // Top-left: window contains 13, located at flat index 12.
  EXPECT_FLOAT_EQ(py[0], 13.0f);
  EXPECT_EQ(pi[0], 12);
  // Bottom-right: window contains 25 at flat index 24.
  EXPECT_FLOAT_EQ(py[24], 25.0f);
  EXPECT_EQ(pi[24], 24);
}

TEST(KernelClass, MaxPoolRejectsNonRowMajorStorageOrder) {
  const KernelContext ctx{DefaultOpset(22)};
  MaxPool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  EXPECT_THROW(pool(x, /*kernel_shape=*/{2, 2}, /*strides=*/{}, /*pads=*/{},
                    /*ceil_mode=*/false, /*dilations=*/{}, /*storage_order=*/1),
               std::invalid_argument);
}

TEST(KernelClass, MaxUnpoolWithoutOutputShape) {
  // mirrors test_maxunpool_export_without_output_shape.
  const KernelContext ctx{DefaultOpset(22)};
  MaxUnpool unpool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor indices = Tensor::FromInt64("", {1, 1, 2, 2}, {5, 7, 13, 15});
  Tensor y = unpool(x, indices, /*kernel_shape=*/{2, 2}, /*strides=*/{2, 2});
  const std::vector<int64_t> expected_shape = {1, 1, 4, 4};
  EXPECT_EQ(y.shape, expected_shape);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[5], 1.0f);
  EXPECT_FLOAT_EQ(py[7], 2.0f);
  EXPECT_FLOAT_EQ(py[13], 3.0f);
  EXPECT_FLOAT_EQ(py[15], 4.0f);
  for (int i = 0; i < 16; ++i) {
    if (i != 5 && i != 7 && i != 13 && i != 15) {
      EXPECT_FLOAT_EQ(py[i], 0.0f);
    }
  }
}

TEST(KernelClass, MaxUnpoolWithOutputShape) {
  // mirrors test_maxunpool_export_with_output_shape.
  const KernelContext ctx{DefaultOpset(22)};
  MaxUnpool unpool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  Tensor indices = Tensor::FromInt64("", {1, 1, 2, 2}, {5, 7, 13, 15});
  Tensor output_shape = Tensor::FromInt64("", {4}, {1, 1, 5, 5});
  Tensor y = unpool(x, indices, output_shape, /*kernel_shape=*/{2, 2}, /*strides=*/{2, 2});
  const std::vector<int64_t> expected_shape = {1, 1, 5, 5};
  EXPECT_EQ(y.shape, expected_shape);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[6], 5.0f);
  EXPECT_FLOAT_EQ(py[8], 6.0f);
  EXPECT_FLOAT_EQ(py[16], 7.0f);
  EXPECT_FLOAT_EQ(py[18], 8.0f);
}

} // namespace Test
