// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/tuning/cpu_executor.h"
#include "onnx_extensions/kernels/kernels/preview/include_preview_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::OpsetId;
using core::runtime::DataType;
using core::runtime::Tensor;
using onnx_kernels::kernel::FlexAttention;
using onnx_kernels::kernel::KernelContext;

namespace Test {

namespace {

KernelContext PreviewKernelContext() { return KernelContext(OpsetId("ai.onnx.preview", 1)); }

} // namespace

TEST(KernelClass, FlexAttentionMatchesHandComputedSingleHead) {
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

TEST(KernelClass, FlexAttentionExplicitScaleMatchesDefault) {
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

TEST(KernelClass, FlexAttentionSupportsGQAHeadSharing) {
  // batch=1, q_num_heads=2, kv_num_heads=1 (group_size=2), q_len=1, k_len=1,
  // head_size=v_head_size=2. With a single key/value position softmax yields
  // probability 1, so both query heads must produce the V[0] row exactly.
  const Tensor Q = Tensor::FromFloat("", {1, 2, 1, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 1, 2}, {0.5f, -0.5f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 1, 2}, {7.0f, -3.0f});

  const KernelContext ctx = PreviewKernelContext();
  FlexAttention flex{ctx};
  flex.Configure({flex.TuningKey(DataType::FLOAT), {{"parallel.minimum_elements", int64_t{1}}}});
  core::runtime::ParallelRegionCollector collector(3);
  core::runtime::ParallelRegionCollectorScope collector_scope(&collector);
  const Tensor Y = flex(Q, K, V);
  ASSERT_EQ(Y.shape, (std::vector<int64_t>{1, 2, 1, 2}));
  for (int64_t h = 0; h < 2; ++h) {
    EXPECT_FLOAT_EQ(Y.AsFloat()[h * 2 + 0], 7.0f);
    EXPECT_FLOAT_EQ(Y.AsFloat()[h * 2 + 1], -3.0f);
  }
  ASSERT_EQ(collector.events().size(), 3u);
  EXPECT_EQ(collector.events()[0].label, "FlexAttentionScores");
  EXPECT_EQ(collector.events()[1].label, "FlexAttentionSoftmax");
  EXPECT_EQ(collector.events()[2].label, "FlexAttentionOutput");
  if (core::runtime::ParallelForThreadCount() > 1) {
    for (const auto &event : collector.events()) {
      EXPECT_GT(event.admitted_threads, 1) << event.label;
    }
  }
}

TEST(KernelClass, FlexAttentionSerialPolicyReportsOneEventPerPhase) {
  core::runtime::CpuExecutionPolicy policy;
  policy.num_threads = 1;
  policy.affinity_policy = core::runtime::CpuAffinityPolicy::kNone;
  const auto executor = core::runtime::GlobalCpuExecutorRegistry().Acquire(policy);
  const core::runtime::CpuExecutorScope executor_scope(executor.get());
  const Tensor Q = Tensor::FromFloat("", {1, 2, 1, 1}, {0.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 1, 1}, {0.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 1, 1}, {1.0f});
  FlexAttention flex{PreviewKernelContext()};
  flex.Configure({flex.TuningKey(DataType::FLOAT), {{"parallel.minimum_elements", int64_t{1}}}});
  core::runtime::ParallelRegionCollector collector(3);
  core::runtime::ParallelRegionCollectorScope collector_scope(&collector);

  const Tensor Y = flex(Q, K, V);

  EXPECT_EQ(std::vector<float>(Y.AsFloat(), Y.AsFloat() + 2), (std::vector<float>{1.0f, 1.0f}));
  ASSERT_EQ(collector.events().size(), 3u);
  EXPECT_EQ(collector.events()[0].label, "FlexAttentionScores");
  EXPECT_EQ(collector.events()[1].label, "FlexAttentionSoftmax");
  EXPECT_EQ(collector.events()[2].label, "FlexAttentionOutput");
  for (const auto &event : collector.events()) {
    EXPECT_EQ(event.admitted_threads, 1) << event.label;
  }
}

TEST(KernelClass, FlexAttentionSoftmaxAdmissionUsesTotalTaskCount) {
  core::runtime::CpuExecutionPolicy policy;
  policy.num_threads = 2;
  policy.affinity_policy = core::runtime::CpuAffinityPolicy::kNone;
  const auto executor = core::runtime::GlobalCpuExecutorRegistry().Acquire(policy);
  const core::runtime::CpuExecutorScope executor_scope(executor.get());
  const Tensor Q = Tensor::FromFloat("", {1, 4, 1, 1}, std::vector<float>(4, 0.0f));
  const Tensor K = Tensor::FromFloat("", {1, 1, 1, 1}, {0.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 1, 1}, {1.0f});
  FlexAttention flex{PreviewKernelContext()};
  flex.Configure({flex.TuningKey(DataType::FLOAT), {{"parallel.minimum_elements", int64_t{6}}}});
  core::runtime::ParallelRegionCollector collector(3);
  core::runtime::ParallelRegionCollectorScope collector_scope(&collector);

  const Tensor Y = flex(Q, K, V);

  EXPECT_EQ(std::vector<float>(Y.AsFloat(), Y.AsFloat() + 4),
            (std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}));
  ASSERT_EQ(collector.events().size(), 3u);
  EXPECT_EQ(collector.events()[1].label, "FlexAttentionSoftmax");
  EXPECT_EQ(collector.events()[1].admitted_threads, 2);
}

TEST(KernelClass, FlexAttentionScratchIsBoundedByAvailableParticipants) {
  class RecordingAllocator : public core::runtime::RawBufferAllocator {
  public:
    core::runtime::RawBuffer *Allocate(size_t n_bytes) override {
      requests.push_back(n_bytes);
      return pool.Allocate(n_bytes);
    }
    void Free(core::runtime::RawBuffer *buffer) override { pool.Free(buffer); }
    size_t TotalAllocatedSize() const override { return pool.TotalAllocatedSize(); }
    size_t PeakAllocatedSize() const override { return pool.PeakAllocatedSize(); }
    void ResetPeak() override { pool.ResetPeak(); }

    std::vector<size_t> requests;
    core::runtime::SimpleRawBufferAllocator pool{4};
  };

  const int64_t participants = core::runtime::ParallelForThreadCount();
  const int64_t task_count = participants + 1;
  const int64_t kv_seq_len = 4;
  const Tensor Q =
      Tensor::FromFloat("", {1, task_count, 1, 1}, std::vector<float>(task_count, 0.0f));
  const Tensor K =
      Tensor::FromFloat("", {1, 1, kv_seq_len, 1}, std::vector<float>(kv_seq_len, 0.0f));
  const Tensor V =
      Tensor::FromFloat("", {1, 1, kv_seq_len, 1}, std::vector<float>(kv_seq_len, 1.0f));
  Tensor output =
      Tensor::FromFloat("", {1, task_count, 1, 1}, std::vector<float>(task_count, 0.0f));
  RecordingAllocator allocator;
  core::runtime::RuntimeContext rt(core::runtime::RuntimeContextOptions{.allocator = &allocator});
  FlexAttention flex{PreviewKernelContext()};
  flex.Configure({flex.TuningKey(DataType::FLOAT), {{"parallel.minimum_elements", int64_t{1}}}});

  flex(Q, K, V, 1.0f, output, &rt);
  EXPECT_EQ(output.shape, (std::vector<int64_t>{1, task_count, 1, 1}));
  const size_t unbounded_scratch = static_cast<size_t>(task_count * kv_seq_len) * sizeof(double);
  const size_t bounded_scratch = static_cast<size_t>(participants * kv_seq_len) * sizeof(double);
  EXPECT_EQ(std::count(allocator.requests.begin(), allocator.requests.end(), unbounded_scratch), 0);
  EXPECT_GE(std::count(allocator.requests.begin(), allocator.requests.end(), bounded_scratch), 1);
}

TEST(KernelClass, FlexAttentionRejectsInvalidInputs) {
  const KernelContext ctx = PreviewKernelContext();
  const FlexAttention flex{ctx};

  // Non rank-4 input is rejected.
  const Tensor bad = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 1, 1}, {1.0f});
  EXPECT_THROW(flex(bad, K, V), std::invalid_argument);

  // Non-FLOAT input is rejected.
  Tensor int_Q("", core::runtime::DataType::INT32, {1, 1, 1, 2},
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

TEST(KernelClass, FlexAttentionEmptyProbModMatchesBaseline) {
  // An empty std::function for ``prob_mod`` must reproduce exactly the
  // un-modified baseline produced by the overload without a callback.
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  const KernelContext ctx = PreviewKernelContext();
  const FlexAttention flex{ctx};
  const float scale = 1.0f / std::sqrt(2.0f);
  const Tensor Y_baseline = flex(Q, K, V, scale);
  const Tensor Y_empty_cb = flex(Q, K, V, scale, FlexAttention::ProbModFn{});
  ASSERT_EQ(Y_baseline.shape, Y_empty_cb.shape);
  for (int64_t i = 0; i < Y_baseline.element_count(); ++i) {
    EXPECT_FLOAT_EQ(Y_baseline.AsFloat()[i], Y_empty_cb.AsFloat()[i]);
  }
}

TEST(KernelClass, FlexAttentionIdentityProbModMatchesBaseline) {
  // A no-op ``prob_mod`` callback must reproduce the baseline output and
  // observe a probability tensor with the expected shape, dtype and
  // softmax row-sums of 1.
  const Tensor Q = Tensor::FromFloat("", {1, 2, 1, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor K = Tensor::FromFloat("", {1, 2, 2, 2}, {1, 0, 0, 1, 1, 1, -1, 1});
  const Tensor V = Tensor::FromFloat("", {1, 2, 2, 2}, {1, 2, 3, 4, -1, 0, 0, 1});

  const KernelContext ctx = PreviewKernelContext();
  const FlexAttention flex{ctx};
  const float scale = 1.0f / std::sqrt(2.0f);
  const Tensor Y_baseline = flex(Q, K, V, scale);

  bool observed = false;
  auto identity = [&observed](Tensor &probs) {
    observed = true;
    EXPECT_EQ(probs.data_type, core::runtime::DataType::FLOAT);
    EXPECT_EQ(probs.shape, (std::vector<int64_t>{1, 2, 1, 2}));
    // Softmax probabilities sum to 1 along the last axis.
    for (int64_t b = 0; b < probs.shape[0]; ++b) {
      for (int64_t h = 0; h < probs.shape[1]; ++h) {
        for (int64_t i = 0; i < probs.shape[2]; ++i) {
          double row_sum = 0.0;
          for (int64_t j = 0; j < probs.shape[3]; ++j) {
            row_sum +=
                probs.AsFloat()[((b * probs.shape[1] + h) * probs.shape[2] + i) * probs.shape[3] +
                                j];
          }
          EXPECT_NEAR(row_sum, 1.0, 1e-5);
        }
      }
    }
  };

  const Tensor Y_identity = flex(Q, K, V, scale, identity);
  EXPECT_TRUE(observed);
  ASSERT_EQ(Y_baseline.shape, Y_identity.shape);
  for (int64_t i = 0; i < Y_baseline.element_count(); ++i) {
    EXPECT_FLOAT_EQ(Y_baseline.AsFloat()[i], Y_identity.AsFloat()[i]);
  }
}

TEST(KernelClass, FlexAttentionScalingProbModRescalesOutput) {
  // Rescaling every probability by 0.5 must scale the final output by the
  // same factor (since Y = probs @ V is linear in probs).
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  const KernelContext ctx = PreviewKernelContext();
  const FlexAttention flex{ctx};
  const float scale = 1.0f / std::sqrt(2.0f);
  const Tensor Y_baseline = flex(Q, K, V, scale);

  auto half = [](Tensor &probs) {
    float *p = probs.AsFloat();
    for (int64_t i = 0; i < probs.element_count(); ++i) {
      p[i] *= 0.5f;
    }
  };

  const Tensor Y_half = flex(Q, K, V, scale, half);
  ASSERT_EQ(Y_baseline.shape, Y_half.shape);
  for (int64_t i = 0; i < Y_baseline.element_count(); ++i) {
    EXPECT_FLOAT_EQ(Y_half.AsFloat()[i], 0.5f * Y_baseline.AsFloat()[i]);
  }
}

TEST(KernelClass, FlexAttentionProbModMustPreserveShapeAndType) {
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  const KernelContext ctx = PreviewKernelContext();
  const FlexAttention flex{ctx};
  const float scale = 1.0f / std::sqrt(2.0f);

  auto reshape = [](Tensor &probs) { probs.shape = {1, 1, 2, 1}; };
  EXPECT_THROW(flex(Q, K, V, scale, reshape), std::invalid_argument);

  auto retype = [](Tensor &probs) { probs.data_type = core::runtime::DataType::DOUBLE; };
  EXPECT_THROW(flex(Q, K, V, scale, retype), std::invalid_argument);
}

// Verifies that ``kernel::FlexAttention`` supports FLOAT16 / BFLOAT16 Q, K, V
// and returns a half-precision output whose values match the FLOAT path
// rounded through the same dtype.
TEST(KernelClass, FlexAttentionHalfPrecisionMatchesFloatReference) {
  const Tensor Q = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor K = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor V = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  const KernelContext ctx = PreviewKernelContext();
  const FlexAttention flex{ctx};
  const Tensor ref = flex(Q, K, V);

  for (int32_t target : {static_cast<int32_t>(core::runtime::DataType::FLOAT16),
                         static_cast<int32_t>(core::runtime::DataType::BFLOAT16)}) {
    const Tensor Qh = onnx_kernels::DemoteFromFloat32(Q, target);
    const Tensor Kh = onnx_kernels::DemoteFromFloat32(K, target);
    const Tensor Vh = onnx_kernels::DemoteFromFloat32(V, target);
    const Tensor Yh = flex(Qh, Kh, Vh);
    ASSERT_EQ(Yh.data_type, target);
    ASSERT_EQ(Yh.shape, ref.shape);
    const Tensor Yh_decoded = onnx_kernels::PromoteToFloat32(Yh);
    const float tol =
        target == static_cast<int32_t>(core::runtime::DataType::FLOAT16) ? 1e-2f : 5e-2f;
    for (int64_t i = 0; i < ref.element_count(); ++i) {
      EXPECT_NEAR(Yh_decoded.AsFloat()[i], ref.AsFloat()[i], tol) << "i=" << i;
    }
  }
}

TEST(KernelClass, FlexAttentionDoubleMatchesFloatReference) {
  // The DOUBLE code path must reproduce the FLOAT reference (same input
  // values) while storing Q/K/V/Y as ``double``.
  const Tensor Qf = Tensor::FromFloat("", {1, 1, 1, 2}, {1.0f, 0.0f});
  const Tensor Kf = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  const Tensor Vf = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  const KernelContext ctx = PreviewKernelContext();
  const FlexAttention flex{ctx};
  const Tensor ref = flex(Qf, Kf, Vf);

  const Tensor Qd = Tensor::FromDouble("", {1, 1, 1, 2}, {1.0, 0.0});
  const Tensor Kd = Tensor::FromDouble("", {1, 1, 2, 2}, {1.0, 0.0, 0.0, 1.0});
  const Tensor Vd = Tensor::FromDouble("", {1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});
  const Tensor Yd = flex(Qd, Kd, Vd);
  ASSERT_EQ(Yd.data_type, static_cast<int32_t>(core::runtime::DataType::DOUBLE));
  ASSERT_EQ(Yd.shape, ref.shape);
  for (int64_t i = 0; i < ref.element_count(); ++i) {
    EXPECT_NEAR(Yd.AsDouble()[i], static_cast<double>(ref.AsFloat()[i]), 1e-6) << "i=" << i;
  }
}

} // namespace Test