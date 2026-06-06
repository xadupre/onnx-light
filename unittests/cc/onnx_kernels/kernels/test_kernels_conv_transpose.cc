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
using onnx_kernels::kernel::ConvTranspose;
using onnx_kernels::kernel::KernelContext;

namespace Test {

// Mirrors upstream ``test_convtranspose``: 1x1x3x3 input, 1x2x3x3 weight of
// ones, default attrs. Output shape is (1, 2, 5, 5).
TEST(KernelClass, ConvTransposeBasicMatchesUpstream) {
  const KernelContext ctx{DefaultOpset(22)};
  const ConvTranspose ct{ctx};
  std::vector<float> X(9);
  for (int i = 0; i < 9; ++i) {
    X[i] = static_cast<float>(i);
  }
  Tensor x = Tensor::FromFloat("", {1, 1, 3, 3}, X);
  Tensor w = Tensor::FromFloat("", {1, 2, 3, 3}, std::vector<float>(18, 1.0f));
  Tensor b;
  ConvTranspose::Attributes attrs;
  attrs.kernel_shape = {3, 3};
  Tensor y = ct(x, w, b, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 2, 5, 5}));
  // Spot-check a few values: y[0,0,0,0] = x[0,0,0,0] * w[0,0,0,0] = 0.
  // y[0,0,2,2] is the center where all 9 inputs contribute.
  const float *py = y.AsFloat();
  // Sum of all input elements: 0+1+...+8 = 36.
  EXPECT_NEAR(py[2 * 5 + 2], 36.0f, 1e-4f);
}

// ConvTranspose with explicit pads cropping the output.
TEST(KernelClass, ConvTransposeWithPadsCropsOutput) {
  const KernelContext ctx{DefaultOpset(22)};
  const ConvTranspose ct{ctx};
  std::vector<float> X(9);
  for (int i = 0; i < 9; ++i) {
    X[i] = static_cast<float>(i);
  }
  Tensor x = Tensor::FromFloat("", {1, 1, 3, 3}, X);
  Tensor w = Tensor::FromFloat("", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  Tensor b;
  ConvTranspose::Attributes attrs;
  attrs.kernel_shape = {3, 3};
  attrs.pads = {1, 1, 1, 1};
  Tensor y = ct(x, w, b, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 3, 3}));
}

// ConvTranspose with output_shape and stride.
TEST(KernelClass, ConvTransposeOutputShapeHonored) {
  const KernelContext ctx{DefaultOpset(22)};
  const ConvTranspose ct{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  Tensor w = Tensor::FromFloat("", {1, 1, 3, 3}, std::vector<float>(9, 1.0f));
  Tensor b;
  ConvTranspose::Attributes attrs;
  attrs.kernel_shape = {3, 3};
  attrs.strides = {2, 2};
  attrs.output_shape = {6, 6};
  Tensor y = ct(x, w, b, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 6, 6}));
}

TEST(KernelClass, ConvTransposeCanRunInPlaceIsFalse) {
  EXPECT_FALSE(ConvTranspose::CanRunInPlace());
}

} // namespace Test
