// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::DefaultOpset;
using core::runtime::Tensor;
using onnx_kernels::kernel::AutoPad;
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
  attrs.auto_pad = AutoPad::kSameUpper;
  Tensor y = conv(x, w, b, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 4, 4}));
}

// SAME_UPPER auto_pad with stride=2: asymmetric padding (pad_begin=0,
// pad_end=1). Reproduces the scenario from microsoft/onnxruntime#26734 where
// stride > 1 with SAME_UPPER produced incorrect values in some ORT backends.
// Input [1,1,4,4], kernel 3x3, stride=2 → pads resolved to [0,0,1,1].
TEST(KernelClass, ConvSameUpperStride2AsymmetricPadding) {
  const KernelContext ctx{DefaultOpset(22)};
  const Conv conv{ctx};
  std::vector<float> Xv(16);
  for (int i = 0; i < 16; ++i) {
    Xv[i] = static_cast<float>(i);
  }
  Tensor x = Tensor::FromFloat("", {1, 1, 4, 4}, Xv);
  Tensor w = Tensor::FromFloat("", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  Tensor b;
  Conv::Attributes attrs;
  attrs.kernel_shape = {3, 3};
  attrs.auto_pad = AutoPad::kSameUpper;
  attrs.strides = {2, 2};
  Tensor y = conv(x, w, b, attrs);
  // pads resolved to [begin_h=0, begin_w=0, end_h=1, end_w=1].
  // Output shape [1, 1, 2, 2]:
  //   y[0,0] = X[0..2, 0..2].sum()        = 0+1+2+4+5+6+8+9+10 = 45
  //   y[0,1] = X[0..2, 2..3+pad].sum()    = 2+3+0+6+7+0+10+11+0 = 39
  //   y[1,0] = X[2..3+pad, 0..2].sum()    = 8+9+10+12+13+14+0+0+0 = 66
  //   y[1,1] = X[2..3+pad, 2..3+pad].sum()= 10+11+0+14+15+0+0+0+0 = 50
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 2, 2}));
  ExpectNear(y, {45.f, 39.f, 66.f, 50.f});
}

TEST(KernelClass, ConvCanRunInPlaceIsFalse) { EXPECT_FALSE(Conv::CanRunInPlace()); }

// When ``kernel_shape`` is omitted it is derived from ``W.shape[2:]``. This
// exercises ``Shape::assign(first, last)`` used to copy the weight spatial
// dimensions into the (now ``Shape``-typed) attribute.
TEST(KernelClass, ConvDerivesKernelShapeFromWeights) {
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
  // kernel_shape intentionally left empty; derived from W.shape[2:] = {3, 3}.
  Tensor y = conv(x, w, b, attrs);
  ExpectNear(y, {54.f, 63.f, 72.f, 99.f, 108.f, 117.f, 144.f, 153.f, 162.f});
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 3, 3}));
}

} // namespace Test
