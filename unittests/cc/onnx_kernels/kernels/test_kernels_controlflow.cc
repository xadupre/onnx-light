// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/controlflow/include_controlflow_kernels.h"
#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/runtime_context.h"
#include "onnx_proto/onnx.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_kernels::RuntimeContext;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::If;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::Loop;
using onnx_kernels::kernel::Scan;

namespace Test {

TEST(KernelClass, IfClassSelectsThenBranchWhenCondTrue) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor cond("", onnx_kernels::DataType::BOOL, {}, {1});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor out = if_kernel(cond, then_v, else_v);
  ASSERT_EQ(out.element_count(), 2);
  EXPECT_EQ(out.data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  EXPECT_FLOAT_EQ(out.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(out.AsFloat()[1], 2.0f);
}

TEST(KernelClass, IfClassSelectsElseBranchWhenCondFalse) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor cond("", onnx_kernels::DataType::BOOL, {}, {0});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor out = if_kernel(cond, then_v, else_v);
  ASSERT_EQ(out.element_count(), 2);
  EXPECT_FLOAT_EQ(out.AsFloat()[0], 3.0f);
  EXPECT_FLOAT_EQ(out.AsFloat()[1], 4.0f);
}

TEST(KernelClass, IfClassRejectsInvalidInputs) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor cond_bool("", onnx_kernels::DataType::BOOL, {}, {1});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // Non-bool cond is rejected.
  Tensor cond_float = Tensor::FromFloat("", {}, {1.0f});
  EXPECT_THROW((void)if_kernel(cond_float, then_v, else_v), std::invalid_argument);

  // Multi-element cond is rejected.
  Tensor cond_vec("", onnx_kernels::DataType::BOOL, {2}, {1, 0});
  EXPECT_THROW((void)if_kernel(cond_vec, then_v, else_v), std::invalid_argument);

  // Mismatched branch types are rejected.
  Tensor else_int = Tensor::From<int32_t>("", {2}, {3, 4});
  EXPECT_THROW((void)if_kernel(cond_bool, then_v, else_int), std::invalid_argument);

  // Mismatched branch shapes are rejected.
  Tensor else_short = Tensor::FromFloat("", {1}, {3.0f});
  EXPECT_THROW((void)if_kernel(cond_bool, then_v, else_short), std::invalid_argument);
}

TEST(KernelClass, IfInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // cond = true → then-branch.
  {
    Tensor cond("", onnx_kernels::DataType::BOOL, {}, {1});
    Tensor out("", onnx_kernels::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
    if_kernel(cond, then_v, else_v, out);
    EXPECT_FLOAT_EQ(out.AsFloat()[0], 1.0f);
    EXPECT_FLOAT_EQ(out.AsFloat()[1], 2.0f);
  }

  // cond = false → else-branch.
  {
    Tensor cond("", onnx_kernels::DataType::BOOL, {}, {0});
    Tensor out("", onnx_kernels::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
    if_kernel(cond, then_v, else_v, out);
    EXPECT_FLOAT_EQ(out.AsFloat()[0], 3.0f);
    EXPECT_FLOAT_EQ(out.AsFloat()[1], 4.0f);
  }
}

TEST(KernelClass, IfInPlaceRejectsBadOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  If if_kernel{ctx};
  Tensor cond("", onnx_kernels::DataType::BOOL, {}, {1});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // Wrong dtype.
  Tensor bad_dtype("", onnx_kernels::DataType::INT32, {2},
                   std::vector<uint8_t>(2 * sizeof(int32_t)));
  EXPECT_THROW(if_kernel(cond, then_v, else_v, bad_dtype), std::invalid_argument);

  // Wrong shape.
  Tensor bad_shape("", onnx_kernels::DataType::FLOAT, {3}, std::vector<uint8_t>(3 * sizeof(float)));
  EXPECT_THROW(if_kernel(cond, then_v, else_v, bad_shape), std::invalid_argument);

  // Wrong buffer byte count.
  Tensor bad_bytes("", onnx_kernels::DataType::FLOAT, {2}, std::vector<uint8_t>(1 * sizeof(float)));
  EXPECT_THROW(if_kernel(cond, then_v, else_v, bad_bytes), std::invalid_argument);
}

} // namespace Test
namespace Test {

namespace {
// Builds an INT64 scalar tensor carrying ``v``.
Tensor Int64Scalar(int64_t v) {
  std::vector<uint8_t> bytes(sizeof(int64_t));
  std::memcpy(bytes.data(), &v, sizeof(int64_t));
  return Tensor("", onnx_kernels::DataType::INT64, {}, std::move(bytes));
}
} // namespace

TEST(KernelClass, LoopStacksScanOutputsAcrossIterations) {
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
  EXPECT_EQ(out[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  ASSERT_EQ(out[0].shape.size(), 2u);
  EXPECT_EQ(out[0].shape[0], 3);
  EXPECT_EQ(out[0].shape[1], 2);
  ASSERT_EQ(out[0].element_count(), 6);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[4], 5.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[5], 6.0f);
}

TEST(KernelClass, LoopReturnsInitialStateWhenTripCountIsZero) {
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

TEST(KernelClass, LoopHonorsCondFalseEvenWhenMIsLarge) {
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  Tensor M = Int64Scalar(5);
  Tensor cond_false("", onnx_kernels::DataType::BOOL, {}, {0});
  Tensor scan = Tensor::FromFloat("", {1}, {1.0f});
  std::vector<Tensor> out = loop_kernel(M, cond_false, /*v_initial=*/{}, /*final_state=*/{},
                                        {{scan, scan, scan, scan, scan}});
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].shape.size(), 2u);
  EXPECT_EQ(out[0].shape[0], 0);
  EXPECT_EQ(out[0].shape[1], 1);
  EXPECT_TRUE(out[0].data.empty());
}

TEST(KernelClass, LoopUsesPerIterRowLengthWhenMIsAbsent) {
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

TEST(KernelClass, LoopRejectsMismatchedFinalStateAndVInitial) {
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

TEST(KernelClass, LoopRejectsScanRowsOfDifferentLengths) {
  const KernelContext ctx{DefaultOpset(13)};
  Loop loop_kernel{ctx};
  Tensor M = Int64Scalar(2);
  Tensor cond_undef;
  Tensor s = Tensor::FromFloat("", {1}, {1.0f});
  EXPECT_THROW(
      (void)loop_kernel(M, cond_undef, /*v_initial=*/{}, /*final_state=*/{}, {{s, s}, {s}}),
      std::invalid_argument);
}

TEST(KernelClass, ScanStacksPerIterAlongLeadingAxisByDefault) {
  const KernelContext ctx{DefaultOpset(18)};
  Scan scan_kernel{ctx};
  // T = 3, no state vars, K = 1 scan output of shape [2] per iter.
  Tensor s0 = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
  Tensor s1 = Tensor::FromFloat("", {2}, {2.0f, 3.0f});
  Tensor s2 = Tensor::FromFloat("", {2}, {4.0f, 5.0f});
  std::vector<Tensor> out =
      scan_kernel(3, /*initial_state=*/{}, /*final_state=*/{}, {{s0, s1, s2}});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].data_type, static_cast<int32_t>(onnx_kernels::DataType::FLOAT));
  ASSERT_EQ(out[0].shape, (std::vector<int64_t>{3, 2}));
  ASSERT_EQ(out[0].element_count(), 6);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 0.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 1.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[2], 2.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[3], 3.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[4], 4.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[5], 5.0f);
}

TEST(KernelClass, ScanReturnsInitialStateWhenTripCountIsZero) {
  const KernelContext ctx{DefaultOpset(18)};
  Scan scan_kernel{ctx};
  Tensor initial = Tensor::FromFloat("", {2}, {7.0f, 8.0f});
  Tensor final_ignored = Tensor::FromFloat("", {2}, {9.0f, 10.0f});
  Tensor s = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  std::vector<Tensor> out = scan_kernel(0, {initial}, {final_ignored}, {{s}});
  ASSERT_EQ(out.size(), 2u);
  // State output equals the initial value when T = 0.
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 7.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 8.0f);
  // Scan output has shape [0, 2] (empty along the new leading axis).
  EXPECT_EQ(out[1].shape, (std::vector<int64_t>{0, 2}));
  EXPECT_EQ(out[1].element_count(), 0);
}

TEST(KernelClass, ScanReversesPerIterWhenDirectionPrepend) {
  const KernelContext ctx{DefaultOpset(18)};
  Scan scan_kernel{ctx};
  Tensor s0 = Tensor::FromFloat("", {1}, {10.0f});
  Tensor s1 = Tensor::FromFloat("", {1}, {20.0f});
  Tensor s2 = Tensor::FromFloat("", {1}, {30.0f});
  // direction = 1 → prepend = reverse before stacking.
  std::vector<Tensor> out = scan_kernel(3, {}, {}, {{s0, s1, s2}},
                                        /*scan_output_axes=*/{},
                                        /*scan_output_directions=*/{1});
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].shape, (std::vector<int64_t>{3, 1}));
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 30.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 20.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[2], 10.0f);
}

TEST(KernelClass, ScanStacksAlongNonLeadingAxisWhenRequested) {
  const KernelContext ctx{DefaultOpset(18)};
  Scan scan_kernel{ctx};
  // Per-iter element shape [3] → stacking along axis=1 yields shape [3, T].
  Tensor s0 = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor s1 = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
  std::vector<Tensor> out = scan_kernel(2, {}, {}, {{s0, s1}}, /*scan_output_axes=*/{1});
  ASSERT_EQ(out.size(), 1u);
  ASSERT_EQ(out[0].shape, (std::vector<int64_t>{3, 2}));
  // Memory layout (row-major, axis 1 = trip):
  //   [s0[0], s1[0], s0[1], s1[1], s0[2], s1[2]] = [1, 4, 2, 5, 3, 6].
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 4.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[2], 2.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[3], 5.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[4], 3.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[5], 6.0f);
}

TEST(KernelClass, ScanRejectsMismatchedInitialAndFinalState) {
  const KernelContext ctx{DefaultOpset(18)};
  Scan scan_kernel{ctx};
  Tensor initial = Tensor::FromFloat("", {1}, {1.0f});
  // Different number of state tensors than initial_state.
  EXPECT_THROW((void)scan_kernel(1, {initial}, {}, {}), std::invalid_argument);
}

TEST(KernelClass, ScanRejectsNegativeTripCount) {
  const KernelContext ctx{DefaultOpset(18)};
  Scan scan_kernel{ctx};
  EXPECT_THROW((void)scan_kernel(-1, {}, {}, {}), std::invalid_argument);
}

namespace {

// Builds a minimal Scan body subgraph:
//   inputs : (sum_in [FLOAT], next [FLOAT])
//   nodes  : sum_out = Add(sum_in, next)
//   outputs: (sum_out, sum_out)  // state out and scan out share the same value
GraphProto MakeSumScanBody() {
  GraphProto body;
  body.set_name("scan_body");
  body.add_input()->set_name("sum_in");
  body.add_input()->set_name("next");
  NodeProto *add = body.add_node();
  add->set_op_type("Add");
  add->add_input("sum_in");
  add->add_input("next");
  add->add_output("sum_out");
  body.add_output()->set_name("sum_out");
  body.add_output()->set_name("sum_out");
  return body;
}

} // namespace

TEST(KernelClass, ScanBodyAwareOverloadRunsBodyAndAccumulatesState) {
  const KernelContext ctx{DefaultOpset(18)};
  Scan scan_kernel{ctx};
  const GraphProto body = MakeSumScanBody();

  RuntimeContext rt(ctx);
  Tensor initial = Tensor::FromFloat("initial", {2}, {0.0f, 0.0f});
  Tensor x = Tensor::FromFloat("x", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  std::vector<Tensor> out = scan_kernel(body, {initial}, {x}, rt);
  ASSERT_EQ(out.size(), 2u);
  // Final state = [1+3+5, 2+4+6] = [9, 12].
  ASSERT_EQ(out[0].shape, (std::vector<int64_t>{2}));
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 9.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 12.0f);
  // Scan output = stacked per-iter sums: [[1,2],[4,6],[9,12]].
  ASSERT_EQ(out[1].shape, (std::vector<int64_t>{3, 2}));
  const float *z = out[1].AsFloat();
  EXPECT_FLOAT_EQ(z[0], 1.0f);
  EXPECT_FLOAT_EQ(z[1], 2.0f);
  EXPECT_FLOAT_EQ(z[2], 4.0f);
  EXPECT_FLOAT_EQ(z[3], 6.0f);
  EXPECT_FLOAT_EQ(z[4], 9.0f);
  EXPECT_FLOAT_EQ(z[5], 12.0f);
}

TEST(KernelClass, ScanBodyAwareOverloadHonorsScanInputDirectionReverse) {
  const KernelContext ctx{DefaultOpset(18)};
  Scan scan_kernel{ctx};
  const GraphProto body = MakeSumScanBody();

  RuntimeContext rt(ctx);
  // Iterating x = [[1,2],[3,4],[5,6]] in reverse yields per-iter sums
  // [[5,6],[8,10],[9,12]] and the same final state [9, 12].
  Tensor initial = Tensor::FromFloat("initial", {2}, {0.0f, 0.0f});
  Tensor x = Tensor::FromFloat("x", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  std::vector<Tensor> out = scan_kernel(body, {initial}, {x}, rt, /*scan_input_axes=*/{0},
                                        /*scan_input_directions=*/{1});
  ASSERT_EQ(out.size(), 2u);
  ASSERT_EQ(out[0].shape, (std::vector<int64_t>{2}));
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 9.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 12.0f);
  ASSERT_EQ(out[1].shape, (std::vector<int64_t>{3, 2}));
  const float *z = out[1].AsFloat();
  EXPECT_FLOAT_EQ(z[0], 5.0f);
  EXPECT_FLOAT_EQ(z[1], 6.0f);
  EXPECT_FLOAT_EQ(z[2], 8.0f);
  EXPECT_FLOAT_EQ(z[3], 10.0f);
  EXPECT_FLOAT_EQ(z[4], 9.0f);
  EXPECT_FLOAT_EQ(z[5], 12.0f);
}

TEST(KernelClass, ScanBodyAwareOverloadHonorsScanOutputDirectionReverse) {
  const KernelContext ctx{DefaultOpset(18)};
  Scan scan_kernel{ctx};
  const GraphProto body = MakeSumScanBody();

  RuntimeContext rt(ctx);
  Tensor initial = Tensor::FromFloat("initial", {2}, {0.0f, 0.0f});
  Tensor x = Tensor::FromFloat("x", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  // Per-iter sums are [[1,2],[4,6],[9,12]]; reversed → [[9,12],[4,6],[1,2]].
  std::vector<Tensor> out = scan_kernel(body, {initial}, {x}, rt, /*scan_input_axes=*/{},
                                        /*scan_input_directions=*/{}, /*scan_output_axes=*/{},
                                        /*scan_output_directions=*/{1});
  ASSERT_EQ(out.size(), 2u);
  ASSERT_EQ(out[1].shape, (std::vector<int64_t>{3, 2}));
  const float *z = out[1].AsFloat();
  EXPECT_FLOAT_EQ(z[0], 9.0f);
  EXPECT_FLOAT_EQ(z[1], 12.0f);
  EXPECT_FLOAT_EQ(z[2], 4.0f);
  EXPECT_FLOAT_EQ(z[3], 6.0f);
  EXPECT_FLOAT_EQ(z[4], 1.0f);
  EXPECT_FLOAT_EQ(z[5], 2.0f);
}

TEST(KernelClass, ScanBodyAwareOverloadReturnsInitialStateWhenTripCountIsZero) {
  const KernelContext ctx{DefaultOpset(18)};
  Scan scan_kernel{ctx};
  const GraphProto body = MakeSumScanBody();

  RuntimeContext rt(ctx);
  Tensor initial = Tensor::FromFloat("initial", {2}, {7.0f, 8.0f});
  Tensor x = Tensor::FromFloat("x", {0, 2}, {});

  std::vector<Tensor> out = scan_kernel(body, {initial}, {x}, rt);
  ASSERT_EQ(out.size(), 2u);
  // Final state = initial when T = 0 (body never ran).
  EXPECT_FLOAT_EQ(out[0].AsFloat()[0], 7.0f);
  EXPECT_FLOAT_EQ(out[0].AsFloat()[1], 8.0f);
  // Scan output is empty.
  EXPECT_EQ(out[1].element_count(), 0);
}

TEST(KernelClass, ScanBodyAwareOverloadRejectsRank0ScanInput) {
  const KernelContext ctx{DefaultOpset(18)};
  Scan scan_kernel{ctx};
  const GraphProto body = MakeSumScanBody();

  RuntimeContext rt(ctx);
  Tensor initial = Tensor::FromFloat("initial", {2}, {0.0f, 0.0f});
  Tensor scalar = Tensor::FromFloat("x", {}, {1.0f});
  EXPECT_THROW((void)scan_kernel(body, {initial}, {scalar}, rt), std::invalid_argument);
}

TEST(KernelClass, ScanBodyAwareOverloadRejectsMismatchedScanInputTripCounts) {
  const KernelContext ctx{DefaultOpset(18)};
  Scan scan_kernel{ctx};
  // Body with N=0, M=2, K=0 (two scan inputs, no state, no scan outputs).
  GraphProto body;
  body.set_name("scan_body");
  body.add_input()->set_name("a");
  body.add_input()->set_name("b");
  // No nodes needed — body just declares formal I/O. We still need at
  // least one node so RunGraph has work to do; use an Identity that
  // doesn't contribute to the (empty) output set.
  NodeProto *id = body.add_node();
  id->set_op_type("Identity");
  id->add_input("a");
  id->add_output("a_id");

  RuntimeContext rt(ctx);
  Tensor a = Tensor::FromFloat("a", {3}, {0.f, 0.f, 0.f});
  Tensor b = Tensor::FromFloat("b", {2}, {0.f, 0.f});
  EXPECT_THROW((void)scan_kernel(body, {}, {a, b}, rt), std::invalid_argument);
}

} // namespace Test
