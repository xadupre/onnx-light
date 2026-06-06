// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::Conv;
using onnx_kernels::kernel::KernelContext;

namespace Test {

namespace {

void ExpectNear(const Tensor &y, const std::vector<float> &expected, float tol = 1e-4f) {
  ASSERT_EQ(y.data.size(), expected.size() * sizeof(float));
  const float *py = y.AsFloat();
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(py[i], expected[i], tol) << "index " << i;
  }
}

} // namespace

// Mirrors upstream ``test_basic_conv_without_padding``: 1x1x5x5 input with
// 1x1x3x3 kernel of ones, default stride 1, no padding. Output is the
// sum of each 3x3 window.
TEST(KernelClass, ConvBasicWithoutPaddingMatchesUpstream) {
  const KernelContext ctx{DefaultOpset(22)};
  const Conv conv{ctx};
  std::vector<float> X(25);
  for (int i = 0; i < 25; ++i) {
    X[i] = static_cast<float>(i);
  }
  Tensor x = Tensor::FromFloat("", {1, 1, 5, 5}, X);
  Tensor w = Tensor::FromFloat("", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  Tensor b;
  Conv::Attributes attrs;
  attrs.kernel_shape = {3, 3};
  Tensor y = conv(x, w, b, attrs);
  // Sums of 3x3 windows over [0..24].
  ExpectNear(y, {54.f, 63.f, 72.f, 99.f, 108.f, 117.f, 144.f, 153.f, 162.f});
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 3, 3}));
}

// Mirrors upstream ``test_basic_conv_with_padding``: same data, pads=[1,1,1,1].
TEST(KernelClass, ConvBasicWithPaddingMatchesUpstream) {
  const KernelContext ctx{DefaultOpset(22)};
  const Conv conv{ctx};
  std::vector<float> X(25);
  for (int i = 0; i < 25; ++i) {
    X[i] = static_cast<float>(i);
  }
  Tensor x = Tensor::FromFloat("", {1, 1, 5, 5}, X);
  Tensor w = Tensor::FromFloat("", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  Tensor b;
  Conv::Attributes attrs;
  attrs.kernel_shape = {3, 3};
  attrs.pads = {1, 1, 1, 1};
  Tensor y = conv(x, w, b, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 5, 5}));
}

// SAME_UPPER auto_pad keeps the output spatial size equal to the input
// (ceil(in/stride) = in when stride == 1).
TEST(KernelClass, ConvSameUpperMatchesInputShape) {
  const KernelContext ctx{DefaultOpset(22)};
  const Conv conv{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 4, 4}, std::vector<float>(16, 1.0f));
  Tensor w = Tensor::FromFloat("", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  Tensor b;
  Conv::Attributes attrs;
  attrs.kernel_shape = {3, 3};
  attrs.auto_pad = "SAME_UPPER";
  Tensor y = conv(x, w, b, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 4, 4}));
}

TEST(KernelClass, ConvCanRunInPlaceIsFalse) { EXPECT_FALSE(Conv::CanRunInPlace()); }

} // namespace Test
