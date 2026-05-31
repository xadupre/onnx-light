// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/controlflow/include_controlflow_kernels.h"
#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::If;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::Loop;

namespace Test {

TEST(BackendKernelClass, IfClassSelectsThenBranchWhenCondTrue) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor cond("", onnx_backend_test::DataType::BOOL, {}, {1});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor out = if_kernel(cond, then_v, else_v);
  ASSERT_EQ(out.element_count(), 2);
  EXPECT_EQ(out.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  EXPECT_FLOAT_EQ(out.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(out.AsFloat()[1], 2.0f);
}

TEST(BackendKernelClass, IfClassSelectsElseBranchWhenCondFalse) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor cond("", onnx_backend_test::DataType::BOOL, {}, {0});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor out = if_kernel(cond, then_v, else_v);
  ASSERT_EQ(out.element_count(), 2);
  EXPECT_FLOAT_EQ(out.AsFloat()[0], 3.0f);
  EXPECT_FLOAT_EQ(out.AsFloat()[1], 4.0f);
}

TEST(BackendKernelClass, IfClassRejectsInvalidInputs) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor cond_bool("", onnx_backend_test::DataType::BOOL, {}, {1});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // Non-bool cond is rejected.
  Tensor cond_float = Tensor::FromFloat("", {}, {1.0f});
  EXPECT_THROW((void)if_kernel(cond_float, then_v, else_v), std::invalid_argument);

  // Multi-element cond is rejected.
  Tensor cond_vec("", onnx_backend_test::DataType::BOOL, {2}, {1, 0});
  EXPECT_THROW((void)if_kernel(cond_vec, then_v, else_v), std::invalid_argument);

  // Mismatched branch types are rejected.
  Tensor else_int = Tensor::From<int32_t>("", {2}, {3, 4});
  EXPECT_THROW((void)if_kernel(cond_bool, then_v, else_int), std::invalid_argument);

  // Mismatched branch shapes are rejected.
  Tensor else_short = Tensor::FromFloat("", {1}, {3.0f});
  EXPECT_THROW((void)if_kernel(cond_bool, then_v, else_short), std::invalid_argument);
}

TEST(BackendKernelClass, IfInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // cond = true → then-branch.
  {
    Tensor cond("", onnx_backend_test::DataType::BOOL, {}, {1});
    Tensor out("", onnx_backend_test::DataType::FLOAT, {2},
               std::vector<uint8_t>(2 * sizeof(float)));
    if_kernel(cond, then_v, else_v, out);
    EXPECT_FLOAT_EQ(out.AsFloat()[0], 1.0f);
    EXPECT_FLOAT_EQ(out.AsFloat()[1], 2.0f);
  }

  // cond = false → else-branch.
  {
    Tensor cond("", onnx_backend_test::DataType::BOOL, {}, {0});
    Tensor out("", onnx_backend_test::DataType::FLOAT, {2},
               std::vector<uint8_t>(2 * sizeof(float)));
    if_kernel(cond, then_v, else_v, out);
    EXPECT_FLOAT_EQ(out.AsFloat()[0], 3.0f);
    EXPECT_FLOAT_EQ(out.AsFloat()[1], 4.0f);
  }
}

TEST(BackendKernelClass, IfInPlaceRejectsBadOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor cond("", onnx_backend_test::DataType::BOOL, {}, {1});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // Wrong dtype.
  Tensor bad_dtype("", onnx_backend_test::DataType::INT32, {2},
                   std::vector<uint8_t>(2 * sizeof(int32_t)));
  EXPECT_THROW(if_kernel(cond, then_v, else_v, bad_dtype), std::invalid_argument);

  // Wrong shape.
  Tensor bad_shape("", onnx_backend_test::DataType::FLOAT, {3},
                   std::vector<uint8_t>(3 * sizeof(float)));
  EXPECT_THROW(if_kernel(cond, then_v, else_v, bad_shape), std::invalid_argument);

  // Wrong buffer byte count.
  Tensor bad_bytes("", onnx_backend_test::DataType::FLOAT, {2},
                   std::vector<uint8_t>(1 * sizeof(float)));
  EXPECT_THROW(if_kernel(cond, then_v, else_v, bad_bytes), std::invalid_argument);
}

} // namespace Test
namespace Test {

namespace {
// Builds an INT64 scalar tensor carrying ``v``.
Tensor Int64Scalar(int64_t v) {
  std::vector<uint8_t> bytes(sizeof(int64_t));
  std::memcpy(bytes.data(), &v, sizeof(int64_t));
  return Tensor("", onnx_backend_test::DataType::INT64, {}, std::move(bytes));
}
} // namespace

TEST(BackendKernelClass, LoopStacksScanOutputsAcrossIterations) {
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  // M = 3, no cond, no carried deps, K = 1 scan output of shape [2] per iter.
  Tensor M = Int64Scalar(3);
  Tensor cond_undef;
  Tensor s0 = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor s1 = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor s2 = Tensor::FromFloat("", {2}, {5.0f, 6.0f});
  std::vector<Tensor> out =
      loop_kernel(M, cond_undef, /*v_initial=*/{}, /*final_state=*/{}, {{s0, s1, s2}});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(out[0].shape.size(), 2u);
  EXPECT_EQ(out[0].shape[0], 3);
  EXPECT_EQ(out[0].shape[1], 2);
  ASSERT_EQ(out[0].element_count(), 6);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[4], 5.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[5], 6.0f);
}

TEST(BackendKernelClass, LoopReturnsInitialStateWhenTripCountIsZero) {
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  Tensor M = Int64Scalar(0);
  Tensor cond_undef;
  Tensor v0 = Tensor::FromFloat("", {2}, {7.0f, 8.0f});
  Tensor v0_final = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  std::vector<Tensor> out = loop_kernel(M, cond_undef, {v0}, {v0_final}, {});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 7.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 8.0f);
}

TEST(BackendKernelClass, LoopHonorsCondFalseEvenWhenMIsLarge) {
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  Tensor M = Int64Scalar(5);
  Tensor cond_false("", onnx_backend_test::DataType::BOOL, {}, {0});
  Tensor scan = Tensor::FromFloat("", {1}, {1.0f});
  std::vector<Tensor> out = loop_kernel(M, cond_false, /*v_initial=*/{}, /*final_state=*/{},
                                        {{scan, scan, scan, scan, scan}});
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].shape.size(), 2u);
  EXPECT_EQ(out[0].shape[0], 0);
  EXPECT_EQ(out[0].shape[1], 1);
  EXPECT_TRUE(out[0].data.empty());
}

TEST(BackendKernelClass, LoopUsesPerIterRowLengthWhenMIsAbsent) {
  // No M and no cond → trip count is the per-iteration row length (2).
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  Tensor M_undef;
  Tensor cond_undef;
  Tensor s0 = Tensor::FromFloat("", {1}, {1.0f});
  Tensor s1 = Tensor::FromFloat("", {1}, {2.0f});
  std::vector<Tensor> out =
      loop_kernel(M_undef, cond_undef, /*v_initial=*/{}, /*final_state=*/{}, {{s0, s1}});
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].shape.size(), 2u);
  EXPECT_EQ(out[0].shape[0], 2);
  EXPECT_EQ(out[0].shape[1], 1);
}

TEST(BackendKernelClass, LoopRejectsMismatchedFinalStateAndVInitial) {
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  Tensor M = Int64Scalar(1);
  Tensor cond_undef;
  Tensor v0_float = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  // Mismatched dtype: float vs int32.
  Tensor v0_final_int = Tensor::From<int32_t>("", {2}, {0, 0});
  EXPECT_THROW((void)loop_kernel(M, cond_undef, {v0_float}, {v0_final_int}, {}),
               std::invalid_argument);
}

TEST(BackendKernelClass, LoopRejectsScanRowsOfDifferentLengths) {
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  Tensor M = Int64Scalar(2);
  Tensor cond_undef;
  Tensor s = Tensor::FromFloat("", {1}, {1.0f});
  EXPECT_THROW(
      (void)loop_kernel(M, cond_undef, /*v_initial=*/{}, /*final_state=*/{}, {{s, s}, {s}}),
      std::invalid_argument);
}

} // namespace Test
