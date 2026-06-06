// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::Col2Im;
using onnx_kernels::kernel::KernelContext;

namespace Test {

namespace {

void ExpectFloatEq(const Tensor &y, const std::vector<float> &expected, float tol = 1e-5f) {
  ASSERT_EQ(y.data.size(), expected.size() * sizeof(float));
  const float *py = y.AsFloat();
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(py[i], expected[i], tol) << "index " << i;
  }
}

} // namespace

// Mirrors upstream ``test_col2im``: 1-D-like column folding back into a
// 5x5 image with block-shape (1, 5), default stride / dilation / pads.
// Each row of the 5x5 input contains a full block placed at row r.
TEST(KernelClass, Col2ImBasic2DMatchesUpstream) {
  const KernelContext ctx{DefaultOpset(18)};
  const Col2Im op{ctx};
  std::vector<float> in_v(25);
  for (int i = 0; i < 25; ++i) {
    in_v[i] = static_cast<float>(i + 1);
  }
  Tensor input = Tensor::FromFloat("", {1, 5, 5}, in_v);
  Tensor image_shape = Tensor::FromInt64("", {2}, {5, 5});
  Tensor block_shape = Tensor::FromInt64("", {2}, {1, 5});
  Col2Im::Attributes attrs;
  Tensor y = op(input, image_shape, block_shape, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 5, 5}));
  // With block_shape (1, 5), n_blocks = (5, 1) so L = 5 (5 row-blocks).
  // Block l corresponds to row l. Within block l, kernel positions
  // (0, 0..4) cover all 5 columns. The flat input is indexed as
  // input[0, k_flat, l] where k_flat is the column 0..4. So row r's
  // pixel at column k is in_v[k * 5 + r].
  std::vector<float> expected(25);
  for (int r = 0; r < 5; ++r) {
    for (int k = 0; k < 5; ++k) {
      expected[r * 5 + k] = in_v[k * 5 + r];
    }
  }
  ExpectFloatEq(y, expected);
}

// Overlap test: two adjacent 1x3 blocks contribute to the same column,
// values must be summed.
TEST(KernelClass, Col2ImOverlappingBlocksSum) {
  const KernelContext ctx{DefaultOpset(18)};
  const Col2Im op{ctx};
  // block_shape (1, 3), stride 1, image_shape (1, 4) → L = 2 blocks.
  // Channels = 1 → input shape (1, 3, 2).
  // Block 0 covers cols 0..2, block 1 covers cols 1..3.
  std::vector<float> in_v = {// k=0 (leftmost in each block) row → values at col block_l + 0
                             1.0f, 4.0f,
                             // k=1 → col block_l + 1
                             2.0f, 5.0f,
                             // k=2 → col block_l + 2
                             3.0f, 6.0f};
  Tensor input = Tensor::FromFloat("", {1, 3, 2}, in_v);
  Tensor image_shape = Tensor::FromInt64("", {2}, {1, 4});
  Tensor block_shape = Tensor::FromInt64("", {2}, {1, 3});
  Col2Im::Attributes attrs;
  Tensor y = op(input, image_shape, block_shape, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 1, 4}));
  // col 0: only block0 k0 = 1
  // col 1: block0 k1 (2) + block1 k0 (4) = 6
  // col 2: block0 k2 (3) + block1 k1 (5) = 8
  // col 3: only block1 k2 = 6
  ExpectFloatEq(y, {1.0f, 6.0f, 8.0f, 6.0f});
}

// Out-of-bounds samples from padding are dropped (padded values
// contributed by blocks falling outside the image must not appear in the
// output).
TEST(KernelClass, Col2ImPadsDropOutOfBoundsContributions) {
  const KernelContext ctx{DefaultOpset(18)};
  const Col2Im op{ctx};
  // image (1, 2), block (1, 3), pads (left=1, right=0) → padded width 3,
  // stride 1, dilation 1 → n_blocks = (3 - 3) / 1 + 1 = 1 → L = 1.
  // Wait: padded width = 2 + 1 + 0 = 3 → n_blocks = (3 - 3) + 1 = 1.
  // Block 0 starts at -1; kernel positions 0..2 map to image cols -1, 0, 1.
  // Only positions 1 and 2 of the block should contribute (to image cols
  // 0 and 1 respectively).
  std::vector<float> in_v = {7.0f, 11.0f, 13.0f}; // k=0 (col -1, dropped), k=1, k=2
  Tensor input = Tensor::FromFloat("", {1, 3, 1}, in_v);
  Tensor image_shape = Tensor::FromInt64("", {2}, {1, 2});
  Tensor block_shape = Tensor::FromInt64("", {2}, {1, 3});
  Col2Im::Attributes attrs;
  attrs.pads = {0, 1, 0, 0};
  Tensor y = op(input, image_shape, block_shape, attrs);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1, 1, 2}));
  ExpectFloatEq(y, {11.0f, 13.0f});
}

TEST(KernelClass, Col2ImRejectsNonFloatInput) {
  const KernelContext ctx{DefaultOpset(18)};
  const Col2Im op{ctx};
  Tensor input("", static_cast<int32_t>(onnx_kernels::DataType::INT32),
               std::vector<int64_t>{1, 5, 5}, std::vector<uint8_t>(25 * sizeof(int32_t)));
  Tensor image_shape = Tensor::FromInt64("", {2}, {5, 5});
  Tensor block_shape = Tensor::FromInt64("", {2}, {1, 5});
  Col2Im::Attributes attrs;
  EXPECT_THROW(op(input, image_shape, block_shape, attrs), std::invalid_argument);
}

TEST(KernelClass, Col2ImRejectsInconsistentL) {
  const KernelContext ctx{DefaultOpset(18)};
  const Col2Im op{ctx};
  // image (5, 5) + block (1, 5) → expected L = 5. We pass L = 4.
  Tensor input = Tensor::FromFloat("", {1, 5, 4}, std::vector<float>(20, 0.0f));
  Tensor image_shape = Tensor::FromInt64("", {2}, {5, 5});
  Tensor block_shape = Tensor::FromInt64("", {2}, {1, 5});
  Col2Im::Attributes attrs;
  EXPECT_THROW(op(input, image_shape, block_shape, attrs), std::invalid_argument);
}

} // namespace Test
