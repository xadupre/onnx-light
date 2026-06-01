// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::DeformConv;
using onnx_backend_test::kernel::KernelContext;

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

// Mirrors upstream ``test_basic_deform_conv_without_padding``.
TEST(BackendKernelClass, DeformConvBasicWithoutPaddingMatchesUpstream) {
  const KernelContext ctx{DefaultOpset(19)};
  const DeformConv dc{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8});
  Tensor w = Tensor::FromFloat("", {1, 1, 2, 2}, {1, 1, 1, 1});
  std::vector<float> off(1 * 8 * 2 * 2, 0.0f);
  off[(0 * 2 + 0) * 2 + 0] = 0.5f;  // chan 0 (kh=0,kw=0,y), pos (0,0)
  off[(5 * 2 + 0) * 2 + 1] = -0.1f; // chan 5 (kh=1,kw=0,x), pos (0,1)
  Tensor offset = Tensor::FromFloat("", {1, 8, 2, 2}, off);
  Tensor b;
  Tensor mask;
  DeformConv::Attributes attrs;
  attrs.kernel_shape = {2, 2};
  attrs.pads = {0, 0, 0, 0};
  Tensor y = dc(x, w, offset, b, mask, attrs);
  ExpectNear(y, {9.5f, 11.9f, 20.0f, 24.0f});
}

// Mirrors upstream ``test_basic_deform_conv_with_padding``.
TEST(BackendKernelClass, DeformConvBasicWithPaddingMatchesUpstream) {
  const KernelContext ctx{DefaultOpset(19)};
  const DeformConv dc{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8});
  Tensor w = Tensor::FromFloat("", {1, 1, 2, 2}, {1, 1, 1, 1});
  std::vector<float> off(1 * 8 * 4 * 4, 0.0f);
  off[(0 * 4 + 0) * 4 + 0] = 0.5f;  // chan 0 (kh=0,kw=0,y), pos (0,0)
  off[(5 * 4 + 1) * 4 + 2] = -0.1f; // chan 5 (kh=1,kw=0,x), pos (1,2)
  Tensor offset = Tensor::FromFloat("", {1, 8, 4, 4}, off);
  Tensor b;
  Tensor mask;
  DeformConv::Attributes attrs;
  attrs.kernel_shape = {2, 2};
  attrs.pads = {1, 1, 1, 1};
  Tensor y = dc(x, w, offset, b, mask, attrs);
  ExpectNear(y, {0.0f, 1.0f, 3.0f, 2.0f, 3.0f, 8.0f, 11.9f, 7.0f, 9.0f, 20.0f, 24.0f, 13.0f, 6.0f,
                 13.0f, 15.0f, 8.0f});
}

// Mirrors upstream ``test_deform_conv_with_mask_bias``.
TEST(BackendKernelClass, DeformConvWithMaskBiasMatchesUpstream) {
  const KernelContext ctx{DefaultOpset(19)};
  const DeformConv dc{ctx};
  Tensor x = Tensor::FromFloat("", {1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8});
  Tensor w = Tensor::FromFloat("", {1, 1, 2, 2}, {1, 1, 1, 1});
  std::vector<float> off(1 * 8 * 2 * 2, 0.0f);
  off[(0 * 2 + 0) * 2 + 0] = 0.5f;
  off[(5 * 2 + 0) * 2 + 1] = -0.1f;
  Tensor offset = Tensor::FromFloat("", {1, 8, 2, 2}, off);
  Tensor b = Tensor::FromFloat("", {1}, {1.0f});
  std::vector<float> mvec(1 * 4 * 2 * 2, 1.0f);
  mvec[(2 * 2 + 1) * 2 + 1] = 0.2f; // chan 2 (kh=1,kw=0), pos (1,1)
  Tensor mask = Tensor::FromFloat("", {1, 4, 2, 2}, mvec);
  DeformConv::Attributes attrs;
  attrs.kernel_shape = {2, 2};
  attrs.pads = {0, 0, 0, 0};
  Tensor y = dc(x, w, offset, b, mask, attrs);
  ExpectNear(y, {10.5f, 12.9f, 21.0f, 19.4f});
}

// Mirrors upstream ``test_deform_conv_with_multiple_offset_groups``.
TEST(BackendKernelClass, DeformConvWithMultipleOffsetGroupsMatchesUpstream) {
  const KernelContext ctx{DefaultOpset(19)};
  const DeformConv dc{ctx};
  std::vector<float> X(1 * 2 * 3 * 3, 0.0f);
  for (int k = 0; k < 9; ++k) {
    X[k] = static_cast<float>(k);
    X[9 + k] = static_cast<float>(8 - k);
  }
  Tensor x = Tensor::FromFloat("", {1, 2, 3, 3}, X);
  Tensor w = Tensor::FromFloat("", {1, 2, 2, 2}, std::vector<float>(8, 1.0f));
  std::vector<float> off(1 * 16 * 2 * 2, 0.0f);
  off[(0 * 2 + 0) * 2 + 0] = 0.5f;   // chan 0 (og=0,kh=0,kw=0,y), pos (0,0)
  off[(13 * 2 + 0) * 2 + 1] = -0.1f; // chan 13 (og=1,kh=1,kw=0,x), pos (0,1)
  Tensor offset = Tensor::FromFloat("", {1, 16, 2, 2}, off);
  Tensor b;
  Tensor mask;
  DeformConv::Attributes attrs;
  attrs.kernel_shape = {2, 2};
  attrs.pads = {0, 0, 0, 0};
  attrs.offset_group = 2;
  Tensor y = dc(x, w, offset, b, mask, attrs);
  ExpectNear(y, {33.5f, 32.1f, 32.0f, 32.0f});
}

TEST(BackendKernelClass, DeformConvCanRunInPlaceIsFalse) {
  EXPECT_FALSE(DeformConv::CanRunInPlace());
}

} // namespace Test
