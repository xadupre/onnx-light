// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::DefaultOpset;
using core::runtime::Tensor;
using onnx_kernels::kernel::ConvTranspose;
using onnx_kernels::kernel::KernelContext;

namespace Test {

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
