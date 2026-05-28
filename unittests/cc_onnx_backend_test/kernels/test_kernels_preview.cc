// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/preview/include_preview_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::OpsetId;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::FlexAttention;
using onnx_backend_test::kernel::KernelContext;

namespace Test {

namespace {

KernelContext PreviewKernelContext() { return KernelContext(OpsetId("ai.onnx.preview", 1)); }

} // namespace

TEST(BackendKernelClass, FlexAttentionMatchesHandComputedSingleHead) {
  // batch=1, heads=1, q_len=1, k_len=2, head_size=2, v_head_size=2.
  // Q = [1, 0]; K = [[1,0],[0,1]]; V = [[1,2],[3,4]]; scale = 1/sqrt(2).
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  const KernelContext ctx = PreviewKernelContext();
  const FlexAttention flex{ctx};
  const Tensor Y = flex(Q, K, V);
  ASSERT_EQ(Y.shape, (std::vector<int64_t>{1, 1, 1, 2}));

  const double s = 1.0 / std::sqrt(2.0);
  const double e0 = std::exp(s);
  const double e1 = 1.0;
  const double p0 = e0 / (e0 + e1);
  const double p1 = e1 / (e0 + e1);
  EXPECT_NEAR(Y.AsFloat()[0], static_cast<float>(p0 * 1.0 + p1 * 3.0), 1e-6);
  EXPECT_NEAR(Y.AsFloat()[1], static_cast<float>(p0 * 2.0 + p1 * 4.0), 1e-6);
}

TEST(BackendKernelClass, FlexAttentionExplicitScaleMatchesDefault) {
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  const KernelContext ctx = PreviewKernelContext();
  const FlexAttention flex{ctx};
  const Tensor Y_default = flex(Q, K, V);
  const Tensor Y_explicit = flex(Q, K, V, 1.0f / std::sqrt(2.0f));
  ASSERT_EQ(Y_default.shape, Y_explicit.shape);
  for (int64_t i = 0; i < Y_default.element_count(); ++i) {
    EXPECT_NEAR(Y_default.AsFloat()[i], Y_explicit.AsFloat()[i], 1e-6);
  }
}

TEST(BackendKernelClass, FlexAttentionSupportsGQAHeadSharing) {
  // batch=1, q_num_heads=2, kv_num_heads=1 (group_size=2), q_len=1, k_len=1,
  // head_size=v_head_size=2. With a single key/value position softmax yields
  // probability 1, so both query heads must produce the V[0] row exactly.
  const Tensor Q = Tensor::FromFloat("", {1, 2, 1, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 1, 2}, {0.5f, -0.5f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 1, 2}, {7.0f, -3.0f});

  const KernelContext ctx = PreviewKernelContext();
  const FlexAttention flex{ctx};
  const Tensor Y = flex(Q, K, V);
  ASSERT_EQ(Y.shape, (std::vector<int64_t>{1, 2, 1, 2}));
  for (int64_t h = 0; h < 2; ++h) {
    EXPECT_FLOAT_EQ(Y.AsFloat()[h * 2 + 0], 7.0f);
    EXPECT_FLOAT_EQ(Y.AsFloat()[h * 2 + 1], -3.0f);
  }
}

TEST(BackendKernelClass, FlexAttentionRejectsInvalidInputs) {
  const KernelContext ctx = PreviewKernelContext();
  const FlexAttention flex{ctx};

  // Non rank-4 input is rejected.
  const Tensor bad = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 1, 1}, {1.0f});
  EXPECT_THROW(flex(bad, K, V), std::invalid_argument);

  // Non-FLOAT input is rejected.
  Tensor int_Q("", TensorProto::DataType::INT32, {1, 1, 1, 2},
               std::vector<uint8_t>(2 * sizeof(int32_t)));
  EXPECT_THROW(flex(int_Q, K, V), std::invalid_argument);

  // q_num_heads not a multiple of kv_num_heads is rejected.
  const Tensor Q3h = Tensor::FromFloat("", {1, 3, 1, 2}, {1, 0, 0, 1, 1, 1});
  const Tensor K2h = Tensor::FromFloat("", {1, 2, 1, 2}, {1, 0, 0, 1});
  const Tensor V2h = Tensor::FromFloat("", {1, 2, 1, 1}, {1, 1});
  EXPECT_THROW(flex(Q3h, K2h, V2h), std::invalid_argument);

  // Mismatched batch is rejected.
  const Tensor Q1b = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K2b = Tensor::FromFloat("", {2, 1, 1, 2}, {1, 0, 0, 1});
  const Tensor V2b = Tensor::FromFloat("", {2, 1, 1, 1}, {1, 1});
  EXPECT_THROW(flex(Q1b, K2b, V2b), std::invalid_argument);

  // Mismatched head_size between Q and K is rejected.
  const Tensor Qh2 = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor Kh3 = Tensor::FromFloat("", {1, 1, 1, 3}, {1.0f, 0.0f, 0.0f});
  const Tensor Vh1 = Tensor::FromFloat("", {1, 1, 1, 1}, {1.0f});
  EXPECT_THROW(flex(Qh2, Kh3, Vh1), std::invalid_argument);
}

} // namespace Test
