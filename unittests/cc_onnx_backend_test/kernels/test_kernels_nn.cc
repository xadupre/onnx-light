// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::AveragePool;
using onnx_backend_test::kernel::KernelContext;

namespace Test {

TEST(BackendKernelClass, AveragePool2DDefault) {
  const KernelContext ctx{DefaultOpset(19)};
  AveragePool pool{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 4, 4},
                               {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                                12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
  Tensor y = pool(x, /*kernel_shape=*/{2, 2});
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
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

TEST(BackendKernelClass, AveragePool2DStrides) {
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

TEST(BackendKernelClass, AveragePool2DPadsCountIncludePad) {
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

TEST(BackendKernelClass, AveragePool2DPadsCountExcludePad) {
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

TEST(BackendKernelClass, AveragePoolCeilMode) {
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

TEST(BackendKernelClass, AveragePool1D) {
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

TEST(BackendKernelClass, AveragePoolRejectsBadInputs) {
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

TEST(BackendKernelClass, AveragePool2DDilations) {
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

TEST(BackendKernelClass, AveragePool2DAutoPadSameUpperPrecomputed) {
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

TEST(BackendKernelClass, AveragePoolAutoPadAndPadsAreMutuallyExclusive) {
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

} // namespace Test
