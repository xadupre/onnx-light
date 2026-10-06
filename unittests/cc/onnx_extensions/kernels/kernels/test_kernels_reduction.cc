// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/float16_promote.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/parallel_for.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/runtime/tuning/cpu_executor.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_extensions/kernels/kernels/reduction/include_reduction_kernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using core::backend_test::DefaultOpset;
using core::runtime::RuntimeContext;
using core::runtime::Tensor;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::ReduceL1;
using onnx_kernels::kernel::ReduceL2;
using onnx_kernels::kernel::ReduceLogSumExp;
using onnx_kernels::kernel::ReduceMax;
using onnx_kernels::kernel::ReduceMean;
using onnx_kernels::kernel::ReduceMin;
using onnx_kernels::kernel::ReduceProd;
using onnx_kernels::kernel::ReduceSum;
using onnx_kernels::kernel::ReduceSumSquare;

namespace Test {

TEST(KernelClass, ReduceLogSumExpNonFiniteInputs) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor data =
      Tensor::FromFloat("", {5, 2}, {inf, -inf, -inf, -inf, -inf, 2.0f, inf, nan, 0.0f, 0.0f});
  const Tensor axes = Tensor::FromInt64("", {1}, {1});
  for (int64_t opset : {13, 18}) {
    const ReduceLogSumExp kernel{KernelContext{DefaultOpset(opset)}};
    const Tensor result = kernel(data, axes, /*keepdims=*/false);
    ASSERT_EQ(result.shape, (std::vector<int64_t>{5}));
    EXPECT_EQ(result.AsFloat()[0], inf);
    EXPECT_EQ(result.AsFloat()[1], -inf);
    EXPECT_FLOAT_EQ(result.AsFloat()[2], 2.0f);
    EXPECT_TRUE(std::isnan(result.AsFloat()[3]));
    EXPECT_FLOAT_EQ(result.AsFloat()[4], std::log(2.0f));
  }
}

TEST(KernelClass, NativeReductionDoublePrecisionAndEmptyIdentities) {
  const KernelContext ctx{DefaultOpset(18)};
  const Tensor data = Tensor::FromDouble("", {2, 2}, {1, 1 + 1e-12, 2, 2 + 1e-12});
  const Tensor axes = Tensor::FromInt64("", {1}, {-1});
  const Tensor empty_axes = Tensor::FromInt64("", {0}, {});
  ReduceMin minimum{ctx};
  ReduceMax maximum{ctx};
  ReduceMean mean{ctx};
  Tensor out = Tensor::FromDouble("", {2}, {0, 0});
  minimum(data, axes, false, false, out);
  EXPECT_EQ(out.AsDouble()[0], 1);
  maximum(data, axes, false, false, out);
  EXPECT_EQ(out.AsDouble()[0], 1 + 1e-12);
  mean(data, axes, false, false, out);
  EXPECT_EQ(out.AsDouble()[0], (1 + (1 + 1e-12)) / 2);
  EXPECT_EQ(mean(data, empty_axes, false, true).data, data.data);
  const Tensor empty = Tensor::FromDouble("", {0, 3}, {});
  const Tensor axis0 = Tensor::FromInt64("", {1}, {0});
  Tensor low = minimum(empty, axis0, false);
  Tensor high = maximum(empty, axis0, false);
  Tensor average = mean(empty, axis0, false);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(low.AsDouble()[i], std::numeric_limits<double>::infinity());
    EXPECT_EQ(high.AsDouble()[i], -std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isnan(average.AsDouble()[i]));
  }
  const Tensor integers = Tensor::FromInt64("", {2}, {9007199254740993LL, 9007199254740992LL});
  EXPECT_EQ(maximum(integers).AsInt64()[0], 9007199254740993LL);
  const Tensor nan_data = Tensor::FromDouble("", {2}, {NAN, 1});
  EXPECT_TRUE(std::isnan(minimum(nan_data).AsDouble()[0]));
  EXPECT_TRUE(std::isnan(maximum(nan_data).AsDouble()[0]));
}

TEST(KernelClass, ReduceSumDefaultAxesReducesAll) {
  const KernelContext ctx{DefaultOpset(13)};
  ReduceSum reduce_sum{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor y = reduce_sum(data); // keepdims=true, noop_with_empty_axes=false
  ASSERT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 21.0f);
}

TEST(KernelClass, ReduceSumParallelLeadingSlicesMatchSerial) {
  core::runtime::CpuExecutionPolicy policy;
  policy.num_threads = 2;
  policy.affinity_policy = core::runtime::CpuAffinityPolicy::kNone;
  const auto executor = core::runtime::GlobalCpuExecutorRegistry().Acquire(policy);
  const core::runtime::CpuExecutorScope executor_scope(executor.get());
  onnx_kernels::RegisterKernelFunctions();
  ReduceSum sum{KernelContext{DefaultOpset(18)}};
  for (int32_t dtype : {core::runtime::DataType::FLOAT, core::runtime::DataType::DOUBLE,
                        core::runtime::DataType::INT64}) {
    ASSERT_NE(core::runtime::GetKernelTuningRegistry().FindSchema(sum.TuningKey(dtype)), nullptr);
  }

  std::vector<float> values(2 * 4 * 257);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 7;
  const Tensor data = Tensor::FromFloat("", {2, 4, 257}, values);
  const Tensor last_axis = Tensor::FromInt64("", {1}, {-1});
  const Tensor middle_axis = Tensor::FromInt64("", {1}, {1});
  const Tensor first_axis = Tensor::FromInt64("", {1}, {0});
  const Tensor serial = sum(data, last_axis, false);
  const Tensor serial_middle = sum(data, middle_axis, false);
  const Tensor serial_first = sum(data, first_axis, false);
  sum.Configure(
      {sum.TuningKey(core::runtime::DataType::FLOAT), {{"parallel.minimum_elements", int64_t{1}}}});
  core::runtime::ParallelRegionCollector collector(8);
  const core::runtime::ParallelRegionCollectorScope collector_scope(&collector);
  EXPECT_EQ(sum(data, last_axis, false).data, serial.data);
  ASSERT_EQ(collector.events().size(), 1u);
  EXPECT_EQ(collector.events()[0].admitted_threads, 2);
  EXPECT_EQ(sum(data, middle_axis, false).data, serial_middle.data);
  ASSERT_EQ(collector.events().size(), 2u);
  EXPECT_EQ(collector.events()[1].admitted_threads, 2);
  EXPECT_EQ(sum(data, first_axis, false).data, serial_first.data);
  ASSERT_EQ(collector.events().size(), 3u);
  EXPECT_EQ(collector.events()[2].admitted_threads, 1);

  const Tensor empty = Tensor::FromFloat("", {2, 0, 257}, {});
  EXPECT_EQ(sum(empty, last_axis, false).element_count(), 0);
  std::vector<int64_t> integers(values.size());
  for (size_t i = 0; i < integers.size(); ++i)
    integers[i] = static_cast<int64_t>(i % 17) - 8;
  const Tensor int_data = Tensor::FromInt64("", {2, 4, 257}, integers);
  sum.Configure({sum.TuningKey(core::runtime::DataType::INT64),
                 {{"parallel.minimum_elements", std::numeric_limits<int64_t>::max()}}});
  const Tensor int_serial = sum(int_data, last_axis, false);
  sum.Configure(
      {sum.TuningKey(core::runtime::DataType::INT64), {{"parallel.minimum_elements", int64_t{1}}}});
  EXPECT_EQ(sum(int_data, last_axis, false).data, int_serial.data);
  std::vector<double> doubles(values.begin(), values.end());
  const Tensor double_data = Tensor::FromDouble("", {2, 4, 257}, doubles);
  sum.Configure({sum.TuningKey(core::runtime::DataType::DOUBLE),
                 {{"parallel.minimum_elements", std::numeric_limits<int64_t>::max()}}});
  const Tensor double_serial = sum(double_data, middle_axis, false);
  sum.Configure({sum.TuningKey(core::runtime::DataType::DOUBLE),
                 {{"parallel.minimum_elements", int64_t{1}}}});
  EXPECT_EQ(sum(double_data, middle_axis, false).data, double_serial.data);
}

TEST(KernelClass, ReduceSumInt64DefaultAndPreallocatedAxes) {
  const ReduceSum reduce_sum{KernelContext{DefaultOpset(13)}};
  const Tensor data = Tensor::FromInt64("", {2, 2}, {9007199254740993LL, -2, 4, 5});
  const Tensor axes = Tensor::FromInt64("", {1}, {1});

  const Tensor total = reduce_sum(data, /*keepdims=*/false);
  EXPECT_EQ(total.data_type, static_cast<int32_t>(core::runtime::DataType::INT64));
  EXPECT_EQ(total.shape, (std::vector<int64_t>{}));
  EXPECT_EQ(total.AsInt64()[0], 9007199254741000LL);

  Tensor out = Tensor::FromInt64("", {2}, {0, 0});
  reduce_sum(data, axes, /*keepdims=*/false, /*noop_with_empty_axes=*/false, out);
  EXPECT_EQ(out.AsInt64()[0], 9007199254740991LL);
  EXPECT_EQ(out.AsInt64()[1], 9);

  const Tensor noop = reduce_sum(data, /*keepdims=*/true, /*noop_with_empty_axes=*/true);
  EXPECT_EQ(noop.data_type, data.data_type);
  EXPECT_EQ(noop.data, data.data);

  const Tensor overflow = Tensor::FromInt64("", {2}, {std::numeric_limits<int64_t>::max(), 1});
  EXPECT_EQ(reduce_sum(overflow, /*keepdims=*/false).AsInt64()[0],
            std::numeric_limits<int64_t>::min());

  Tensor wrong_type = Tensor::FromDouble("", {2}, {0, 0});
  EXPECT_THROW(reduce_sum(data, axes, false, false, wrong_type), std::invalid_argument);
}

TEST(KernelClass, ReduceSumDefaultAxesNoKeepdimsProducesScalar) {
  const KernelContext ctx{DefaultOpset(13)};
  ReduceSum reduce_sum{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = reduce_sum(data, /*keepdims=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{}));
  ASSERT_EQ(y.element_count(), 1);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 10.0f);
}

TEST(KernelClass, ReduceSumNoopWithEmptyAxesIsIdentity) {
  const KernelContext ctx{DefaultOpset(13)};
  ReduceSum reduce_sum{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = reduce_sum(data, /*keepdims=*/true, /*noop_with_empty_axes=*/true);
  ASSERT_EQ(y.shape, data.shape);
  EXPECT_EQ(y.data, data.data);
}

TEST(KernelClass, ReduceSumExplicitAxisReducesAlongAxis) {
  const KernelContext ctx{DefaultOpset(13)};
  ReduceSum reduce_sum{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {1});
  Tensor y = reduce_sum(data, axes, /*keepdims=*/false,
                        /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 6.0f);
  EXPECT_FLOAT_EQ(py[1], 15.0f);
}

TEST(KernelClass, ReduceSumNegativeAxisKeepdims) {
  const KernelContext ctx{DefaultOpset(13)};
  ReduceSum reduce_sum{ctx};
  Tensor data = Tensor::FromFloat(
      "", {3, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {-2});
  Tensor y = reduce_sum(data, axes, /*keepdims=*/true,
                        /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3, 1, 2}));
  const float *py = y.AsFloat();
  // sum along axis 1 (middle dim of size 2):
  // batch 0: rows (1,2) + (3,4) = (4,6)
  // batch 1: rows (5,6) + (7,8) = (12,14)
  // batch 2: rows (9,10) + (11,12) = (20,22)
  EXPECT_FLOAT_EQ(py[0], 4.0f);
  EXPECT_FLOAT_EQ(py[1], 6.0f);
  EXPECT_FLOAT_EQ(py[2], 12.0f);
  EXPECT_FLOAT_EQ(py[3], 14.0f);
  EXPECT_FLOAT_EQ(py[4], 20.0f);
  EXPECT_FLOAT_EQ(py[5], 22.0f);
}

TEST(KernelClass, ReduceSumInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  ReduceSum reduce_sum{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {0});
  Tensor out("", static_cast<int32_t>(core::runtime::DataType::FLOAT), {1, 3},
             std::vector<uint8_t>(3 * sizeof(float), 0u));
  reduce_sum(data, axes, /*keepdims=*/true, /*noop_with_empty_axes=*/false, out);
  const float *po = out.AsFloat();
  EXPECT_FLOAT_EQ(po[0], 5.0f);
  EXPECT_FLOAT_EQ(po[1], 7.0f);
  EXPECT_FLOAT_EQ(po[2], 9.0f);
}

TEST(KernelClass, ReduceSumRejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(13)};
  ReduceSum reduce_sum{ctx};
  Tensor data = Tensor::FromFloat("", {2}, {1.0f, 2.0f});

  // Non-FLOAT data is rejected.
  Tensor bad_data = Tensor::FromInt32("", {2}, {1, 2});
  EXPECT_THROW(reduce_sum(bad_data), std::invalid_argument);

  // Non-INT64 axes is rejected.
  Tensor bad_axes = Tensor::FromInt32("", {1}, {0});
  EXPECT_THROW(reduce_sum(data, bad_axes), std::invalid_argument);

  // Out-of-range axis is rejected.
  Tensor oob_axes = Tensor::FromInt64("", {1}, {5});
  EXPECT_THROW(reduce_sum(data, oob_axes), std::invalid_argument);

  // In-place overload with mismatched output shape is rejected.
  Tensor axes = Tensor::FromInt64("", {1}, {0});
  Tensor bad_shape("", static_cast<int32_t>(core::runtime::DataType::FLOAT), {2},
                   std::vector<uint8_t>(2 * sizeof(float), 0u));
  EXPECT_THROW(reduce_sum(data, axes, /*keepdims=*/true,
                          /*noop_with_empty_axes=*/false, bad_shape),
               std::invalid_argument);
}

TEST(KernelClass, ReduceMaxExplicitAxisNoKeepdims) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceMax reduce_max{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 9.0f, 3.0f, 4.0f, 2.0f, 6.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {1});
  Tensor y = reduce_max(data, axes, /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 9.0f);
  EXPECT_FLOAT_EQ(py[1], 6.0f);
}

TEST(KernelClass, ReduceMinNegativeAxisKeepdims) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceMin reduce_min{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2, 2}, {4.0f, 2.0f, 3.0f, 7.0f, 1.0f, 9.0f, 6.0f, 5.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {-1});
  Tensor y = reduce_min(data, axes, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 2, 1}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 2.0f);
  EXPECT_FLOAT_EQ(py[1], 3.0f);
  EXPECT_FLOAT_EQ(py[2], 1.0f);
  EXPECT_FLOAT_EQ(py[3], 5.0f);
}

TEST(KernelClass, ReduceMaxAndMinNoopWithEmptyAxesIsIdentity) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceMax reduce_max{ctx};
  ReduceMin reduce_min{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, 5.0f, 3.0f, 4.0f});
  Tensor empty_axes = Tensor::FromInt64("", {0}, {});
  Tensor y_max = reduce_max(data, empty_axes, /*keepdims=*/true, /*noop_with_empty_axes=*/true);
  Tensor y_min = reduce_min(data, empty_axes, /*keepdims=*/true, /*noop_with_empty_axes=*/true);
  EXPECT_EQ(y_max.shape, data.shape);
  EXPECT_EQ(y_min.shape, data.shape);
  EXPECT_EQ(y_max.data, data.data);
  EXPECT_EQ(y_min.data, data.data);
}

TEST(KernelClass, ReduceMinMaxRejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceMax reduce_max{ctx};
  Tensor data = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor bad_data = Tensor::FromStrings("", {2}, {"a", "b"});
  EXPECT_THROW(reduce_max(bad_data), std::invalid_argument);

  Tensor bad_axes = Tensor::FromInt32("", {1}, {0});
  EXPECT_THROW(reduce_max(data, bad_axes), std::invalid_argument);

  Tensor oob_axes = Tensor::FromInt64("", {1}, {5});
  EXPECT_THROW(reduce_max(data, oob_axes), std::invalid_argument);
}

TEST(KernelClass, ReduceMaxBoolInputsIsLogicalOr) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceMax reduce_max{ctx};
  // shape (4, 2): [[T,T],[T,F],[F,T],[F,F]]
  Tensor data = Tensor::FromBool("", {4, 2}, {1, 1, 1, 0, 0, 1, 0, 0});
  Tensor axes = Tensor::FromInt64("", {1}, {1});
  Tensor y = reduce_max(data, axes, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{4, 1}));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::BOOL));
  const uint8_t *py = y.AsBool();
  EXPECT_EQ(py[0], 1); // T OR T = T
  EXPECT_EQ(py[1], 1); // T OR F = T
  EXPECT_EQ(py[2], 1); // F OR T = T
  EXPECT_EQ(py[3], 0); // F OR F = F
}

TEST(KernelClass, ReduceMinBoolInputsIsLogicalAnd) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceMin reduce_min{ctx};
  Tensor data = Tensor::FromBool("", {4, 2}, {1, 1, 1, 0, 0, 1, 0, 0});
  Tensor axes = Tensor::FromInt64("", {1}, {1});
  Tensor y = reduce_min(data, axes, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{4, 1}));
  ASSERT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::BOOL));
  const uint8_t *py = y.AsBool();
  EXPECT_EQ(py[0], 1); // T AND T = T
  EXPECT_EQ(py[1], 0); // T AND F = F
  EXPECT_EQ(py[2], 0); // F AND T = F
  EXPECT_EQ(py[3], 0); // F AND F = F
}

// ── ReduceL1 / ReduceL2 kernels ───────────────────────────────────────────

TEST(KernelClass, ReduceL1ExplicitAxisSumsAbsoluteValues) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceL1 reduce_l1{ctx};
  // Mix of positives and negatives so the result differs from ReduceSum.
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {1});
  Tensor y = reduce_l1(data, axes, /*keepdims=*/false, /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 6.0f);  // |1| + |-2| + |3|
  EXPECT_FLOAT_EQ(py[1], 15.0f); // |-4| + |5| + |-6|
}

TEST(KernelClass, ReduceL2ExplicitAxisIsSqrtOfSumOfSquares) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceL2 reduce_l2{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {3.0f, 4.0f, -6.0f, 8.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {1});
  Tensor y = reduce_l2(data, axes, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 1}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 5.0f);  // sqrt(9 + 16)
  EXPECT_FLOAT_EQ(py[1], 10.0f); // sqrt(36 + 64)
}

TEST(KernelClass, ReduceL1DefaultAxesReducesAll) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceL1 reduce_l1{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, -2.0f, 3.0f, -4.0f});
  Tensor y = reduce_l1(data); // keepdims=true, noop_with_empty_axes=false
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 10.0f);
}

TEST(KernelClass, ReduceL2NoopWithEmptyAxesAppliesElementwiseAbs) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceL2 reduce_l2{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, -2.0f, 3.0f, -4.0f});
  Tensor empty_axes = Tensor::FromInt64("", {0}, {});
  Tensor y = reduce_l2(data, empty_axes, /*keepdims=*/true, /*noop_with_empty_axes=*/true);
  EXPECT_EQ(y.shape, data.shape);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 2.0f);
  EXPECT_FLOAT_EQ(py[2], 3.0f);
  EXPECT_FLOAT_EQ(py[3], 4.0f);
}

TEST(KernelClass, ReduceL1L2RejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceL1 reduce_l1{ctx};
  Tensor bad_data = Tensor::FromInt32("", {2}, {1, 2});
  EXPECT_THROW(reduce_l1(bad_data), std::invalid_argument);

  Tensor data = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor bad_axes = Tensor::FromInt32("", {1}, {0});
  EXPECT_THROW(reduce_l1(data, bad_axes), std::invalid_argument);

  Tensor oob_axes = Tensor::FromInt64("", {1}, {5});
  EXPECT_THROW(reduce_l1(data, oob_axes), std::invalid_argument);
}

// ── ReduceSumSquare kernel ────────────────────────────────────────────────

TEST(KernelClass, ReduceSumSquareExplicitAxisIsSumOfSquares) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceSumSquare reduce_sum_square{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {3.0f, 4.0f, -6.0f, 8.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {1});
  Tensor y = reduce_sum_square(data, axes, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 1}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 25.0f);  // 9 + 16
  EXPECT_FLOAT_EQ(py[1], 100.0f); // 36 + 64
}

TEST(KernelClass, ReduceSumSquareDefaultAxesReducesAll) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceSumSquare reduce_sum_square{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, -2.0f, 3.0f, -4.0f});
  Tensor y = reduce_sum_square(data); // keepdims=true, noop_with_empty_axes=false
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 30.0f); // 1 + 4 + 9 + 16
}

TEST(KernelClass, ReduceSumSquareNoopWithEmptyAxesAppliesElementwiseSquare) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceSumSquare reduce_sum_square{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, -2.0f, 3.0f, -4.0f});
  Tensor empty_axes = Tensor::FromInt64("", {0}, {});
  Tensor y = reduce_sum_square(data, empty_axes, /*keepdims=*/true,
                               /*noop_with_empty_axes=*/true);
  EXPECT_EQ(y.shape, data.shape);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 4.0f);
  EXPECT_FLOAT_EQ(py[2], 9.0f);
  EXPECT_FLOAT_EQ(py[3], 16.0f);
}

TEST(KernelClass, ReduceSumSquareNegativeAxisAndNoKeepdims) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceSumSquare reduce_sum_square{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {-1});
  Tensor y = reduce_sum_square(data, axes, /*keepdims=*/false,
                               /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2}));
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 14.0f); // 1 + 4 + 9
  EXPECT_FLOAT_EQ(py[1], 77.0f); // 16 + 25 + 36
}

// ── ArgMax / ArgMin kernels ────────────────────────────────────────────────

using onnx_kernels::kernel::ArgMax;
using onnx_kernels::kernel::ArgMin;

TEST(KernelClass, ArgMaxAlongAxisKeepdims) {
  const KernelContext ctx{DefaultOpset(13)};
  ArgMax argmax{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 3.0f, 10.0f});
  Tensor y = argmax(data, /*axis=*/1, /*keepdims=*/true,
                    /*select_last_index=*/false);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(core::runtime::DataType::INT64));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 1}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 1);
}

TEST(KernelClass, ArgMaxDefaultAxisNoKeepdims) {
  const KernelContext ctx{DefaultOpset(13)};
  ArgMax argmax{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 3.0f, 10.0f});
  Tensor y = argmax(data, /*axis=*/0, /*keepdims=*/false,
                    /*select_last_index=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 1);
  EXPECT_EQ(py[1], 1);
}

TEST(KernelClass, ArgMaxNegativeAxisSelectLastIndex) {
  const KernelContext ctx{DefaultOpset(13)};
  ArgMax argmax{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 3.0f, 10.0f});
  Tensor y = argmax(data, /*axis=*/-1, /*keepdims=*/true,
                    /*select_last_index=*/true);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 1}));
  const int64_t *py = y.AsInt64();
  // Row 0 ties at 2 -> last index 1; row 1 unique max at col 1.
  EXPECT_EQ(py[0], 1);
  EXPECT_EQ(py[1], 1);
}

TEST(KernelClass, ArgMinAlongAxisSelectLastIndex) {
  const KernelContext ctx{DefaultOpset(13)};
  ArgMin argmin{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 3.0f, 10.0f});
  Tensor y = argmin(data, /*axis=*/1, /*keepdims=*/false,
                    /*select_last_index=*/true);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2}));
  const int64_t *py = y.AsInt64();
  // Row 0 ties at 2 -> last index 1; row 1 unique min at col 0.
  EXPECT_EQ(py[0], 1);
  EXPECT_EQ(py[1], 0);
}

TEST(KernelClass, ArgReduceInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  ArgMax argmax{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 5.0f, 2.0f, 4.0f, 0.0f, 9.0f});
  Tensor out("", static_cast<int32_t>(core::runtime::DataType::INT64), {2, 1},
             std::vector<uint8_t>(2 * sizeof(int64_t), 0u));
  argmax(data, /*axis=*/1, /*keepdims=*/true, /*select_last_index=*/false, out);
  const int64_t *po = out.AsInt64();
  EXPECT_EQ(po[0], 1);
  EXPECT_EQ(po[1], 2);
}

TEST(KernelClass, ArgReduceRejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(13)};
  ArgMax argmax{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  // Wrong data dtype.
  Tensor bad_data("", static_cast<int32_t>(core::runtime::DataType::INT32), {2, 3},
                  std::vector<uint8_t>(6 * sizeof(int32_t)));
  EXPECT_THROW(argmax(bad_data, /*axis=*/0), std::invalid_argument);

  // Out-of-range axis.
  EXPECT_THROW(argmax(data, /*axis=*/5), std::invalid_argument);

  // Scalar input.
  Tensor scalar("", static_cast<int32_t>(core::runtime::DataType::FLOAT), {},
                std::vector<uint8_t>(sizeof(float), 0u));
  EXPECT_THROW(argmax(scalar, /*axis=*/0), std::invalid_argument);

  // Mismatched preallocated output shape.
  Tensor bad_out("", static_cast<int32_t>(core::runtime::DataType::INT64), {2, 3},
                 std::vector<uint8_t>(6 * sizeof(int64_t), 0u));
  EXPECT_THROW(argmax(data, /*axis=*/0, /*keepdims=*/true, /*select_last_index=*/false, bad_out),
               std::invalid_argument);

  // Wrong output dtype.
  Tensor wrong_dtype_out("", static_cast<int32_t>(core::runtime::DataType::FLOAT), {1, 3},
                         std::vector<uint8_t>(3 * sizeof(float), 0u));
  EXPECT_THROW(
      argmax(data, /*axis=*/0, /*keepdims=*/true, /*select_last_index=*/false, wrong_dtype_out),
      std::invalid_argument);
}

// ── ReduceProd ────────────────────────────────────────────────────────────

TEST(KernelClass, ReduceProdDefaultAxesReducesAll) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceProd reduce_prod{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor y = reduce_prod(data); // keepdims=true, noop_with_empty_axes=false
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 720.0f);
}

TEST(KernelClass, ReduceProdExplicitAxisReducesAlongAxis) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceProd reduce_prod{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {1});
  Tensor y = reduce_prod(data, axes, /*keepdims=*/false,
                         /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 6.0f);   // 1*2*3
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 120.0f); // 4*5*6
}

TEST(KernelClass, ReduceProdNegativeAxisKeepdims) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceProd reduce_prod{ctx};
  Tensor data = Tensor::FromFloat(
      "", {3, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {-2});
  Tensor y = reduce_prod(data, axes, /*keepdims=*/true,
                         /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3, 1, 2}));
  const float *py = y.AsFloat();
  // prod along axis 1 (middle dim of size 2):
  // batch 0: rows (1,2) * (3,4) = (3, 8)
  // batch 1: rows (5,6) * (7,8) = (35, 48)
  // batch 2: rows (9,10) * (11,12) = (99, 120)
  EXPECT_FLOAT_EQ(py[0], 3.0f);
  EXPECT_FLOAT_EQ(py[1], 8.0f);
  EXPECT_FLOAT_EQ(py[2], 35.0f);
  EXPECT_FLOAT_EQ(py[3], 48.0f);
  EXPECT_FLOAT_EQ(py[4], 99.0f);
  EXPECT_FLOAT_EQ(py[5], 120.0f);
}

TEST(KernelClass, ReduceProdNoopWithEmptyAxesIsIdentity) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceProd reduce_prod{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = reduce_prod(data, /*keepdims=*/true, /*noop_with_empty_axes=*/true);
  ASSERT_EQ(y.shape, data.shape);
  EXPECT_EQ(y.data, data.data);
}

TEST(KernelClass, ReduceProdEmptySetIdentityIsOne) {
  // Reducing over an axis of size 0 must yield the identity (1).
  const KernelContext ctx{DefaultOpset(18)};
  ReduceProd reduce_prod{ctx};
  Tensor data("", static_cast<int32_t>(core::runtime::DataType::FLOAT), {2, 0, 4}, {});
  Tensor axes = Tensor::FromInt64("", {1}, {1});
  Tensor y = reduce_prod(data, axes, /*keepdims=*/true, /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 1, 4}));
  const float *py = y.AsFloat();
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(py[i], 1.0f) << "at index " << i;
  }
}

TEST(KernelClass, ReduceProdRejectsNonFloatData) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceProd reduce_prod{ctx};
  Tensor data = Tensor::FromInt64("", {2}, {1, 2});
  EXPECT_THROW(reduce_prod(data), std::invalid_argument);
}

// ── ReduceMean ────────────────────────────────────────────────────────────

TEST(KernelClass, ReduceMeanParallelLeadingSlicesMatchSerial) {
  core::runtime::CpuExecutionPolicy policy;
  policy.num_threads = 2;
  policy.affinity_policy = core::runtime::CpuAffinityPolicy::kNone;
  const auto executor = core::runtime::GlobalCpuExecutorRegistry().Acquire(policy);
  const core::runtime::CpuExecutorScope executor_scope(executor.get());
  onnx_kernels::RegisterKernelFunctions();
  ReduceMean mean{KernelContext{DefaultOpset(18)}};
  for (int32_t dtype : {core::runtime::DataType::FLOAT, core::runtime::DataType::DOUBLE,
                        core::runtime::DataType::FLOAT16, core::runtime::DataType::BFLOAT16}) {
    ASSERT_NE(core::runtime::GetKernelTuningRegistry().FindSchema(mean.TuningKey(dtype)), nullptr);
  }

  std::vector<float> values(2 * 4 * 257);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 7;
  const Tensor data = Tensor::FromFloat("", {2, 4, 257}, values);
  const Tensor last_axis = Tensor::FromInt64("", {1}, {-1});
  const Tensor middle_axis = Tensor::FromInt64("", {1}, {1});
  const Tensor first_axis = Tensor::FromInt64("", {1}, {0});
  const Tensor serial = mean(data, last_axis, false);
  const Tensor serial_middle = mean(data, middle_axis, true);
  const Tensor serial_first = mean(data, first_axis, false);
  mean.Configure({mean.TuningKey(core::runtime::DataType::FLOAT),
                  {{"parallel.minimum_elements", int64_t{1}}}});
  core::runtime::ParallelRegionCollector collector(32);
  const core::runtime::ParallelRegionCollectorScope collector_scope(&collector);
  EXPECT_EQ(mean(data, last_axis, false).data, serial.data);
  ASSERT_EQ(collector.events().size(), 1u);
  EXPECT_EQ(collector.events()[0].admitted_threads, 2);
  EXPECT_EQ(mean(data, middle_axis, true).data, serial_middle.data);
  ASSERT_EQ(collector.events().size(), 2u);
  EXPECT_EQ(collector.events()[1].admitted_threads, 2);
  EXPECT_EQ(mean(data, first_axis, false).data, serial_first.data);
  ASSERT_EQ(collector.events().size(), 3u);
  EXPECT_EQ(collector.events()[2].admitted_threads, 1);

  std::vector<double> doubles(values.begin(), values.end());
  const Tensor double_data = Tensor::FromDouble("", {2, 4, 257}, doubles);
  mean.Configure({mean.TuningKey(core::runtime::DataType::DOUBLE),
                  {{"parallel.minimum_elements", std::numeric_limits<int64_t>::max()}}});
  const Tensor double_serial = mean(double_data, last_axis, false);
  mean.Configure({mean.TuningKey(core::runtime::DataType::DOUBLE),
                  {{"parallel.minimum_elements", int64_t{1}}}});
  EXPECT_EQ(mean(double_data, last_axis, false).data, double_serial.data);
  for (int32_t dtype : {core::runtime::DataType::FLOAT16, core::runtime::DataType::BFLOAT16}) {
    const Tensor half_data = core::runtime::DemoteFromFloat32(data, dtype);
    mean.Configure({mean.TuningKey(dtype),
                    {{"parallel.minimum_elements", std::numeric_limits<int64_t>::max()}}});
    const Tensor half_serial = mean(half_data, last_axis, false);
    mean.Configure({mean.TuningKey(dtype), {{"parallel.minimum_elements", int64_t{1}}}});
    EXPECT_EQ(mean(half_data, last_axis, false).data, half_serial.data);
    const auto parallel_event =
        std::find_if(collector.events().rbegin(), collector.events().rend(),
                     [](const auto &event) { return event.label == "ReduceMean"; });
    ASSERT_NE(parallel_event, collector.events().rend());
    EXPECT_EQ(parallel_event->admitted_threads, 2);
  }
  const Tensor empty = Tensor::FromFloat("", {2, 0, 257}, {});
  EXPECT_EQ(mean(empty, last_axis, false).element_count(), 0);
  const Tensor empty_reduced = Tensor::FromFloat("", {2, 257, 0}, {});
  EXPECT_TRUE(std::isnan(mean(empty_reduced, last_axis, false).AsFloat()[0]));
}

TEST(KernelClass, ReduceMeanSessionTunesForIntermediateDataType) {
  onnx_kernels::RegisterKernelFunctions();
  const KernelContext ctx{DefaultOpset(18)};
  ReduceMean mean{ctx};
  const auto key = mean.TuningKey(static_cast<int32_t>(core::runtime::DataType::FLOAT));
  const core::runtime::KernelTuningParameters profile{key,
                                                      {{"parallel.minimum_elements", int64_t{1}}}};
  core::runtime::GetKernelTuningRegistry().PublishProfiles(
      std::span<const core::runtime::KernelTuningParameters>(&profile, 1));

  GraphProto graph;
  graph.add_input()->set_name("source");
  graph.add_output()->set_name("output");
  ValueInfoProto *data_info = graph.add_value_info();
  data_info->set_name("data");
  data_info->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  TensorProto *axes = graph.add_initializer();
  axes->set_name("axes");
  axes->set_data_type(TensorProto::INT64);
  axes->add_dims(1);
  axes->add_int64_data(-1);
  NodeProto *identity = graph.add_node();
  identity->set_op_type("Identity");
  identity->add_input("source");
  identity->add_output("data");
  NodeProto *reduce = graph.add_node();
  reduce->set_op_type("ReduceMean");
  reduce->add_input("data");
  reduce->add_input("axes");
  reduce->add_output("output");
  AttributeProto *keepdims = reduce->add_attribute();
  keepdims->set_name("keepdims");
  keepdims->set_type(AttributeProto::INT);
  keepdims->set_i(0);

  constexpr int64_t count = 2 * 4 * 257;
  RuntimeContext rt(ctx);
  rt.Set("source", Tensor::FromFloat("source", {2, 4, 257}, std::vector<float>(count, 1.0f)));
  rt.Set("axes", Tensor::FromInt64("axes", {1}, {-1}));
  auto collector = std::make_shared<core::runtime::ParallelRegionCollector>(4);
  core::runtime::RuntimeSession session(rt.GetExecutionPlan(graph),
                                        core::runtime::RuntimeSessionOptions{
                                            .parameters = core::runtime::RuntimeParameters(2),
                                            .parallel_region_collector = collector,
                                        });
  session.SetDeclaredShapes(graph);
  session.Run(rt);

  core::runtime::GetKernelTuningRegistry().PublishProfiles(
      {}, std::span<const core::runtime::KernelTuningKey>(&key, 1));
  EXPECT_EQ(session.tuning_resolution_statistics().resolved_profiles, 1u);
  ASSERT_EQ(collector->events().size(), 1u);
  EXPECT_EQ(collector->events()[0].label, "ReduceMean");
  EXPECT_GT(collector->events()[0].admitted_threads, 1);
  EXPECT_FLOAT_EQ(rt.Get("output").AsFloat()[0], 1.0f);
}

TEST(KernelClass, ReduceMeanDefaultAxesReducesAll) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceMean reduce_mean{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor y = reduce_mean(data); // keepdims=true, noop_with_empty_axes=false
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 3.5f);
}

TEST(KernelClass, ReduceMeanExplicitAxisReducesAlongAxis) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceMean reduce_mean{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {1});
  Tensor y = reduce_mean(data, axes, /*keepdims=*/false,
                         /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 2.0f); // (1+2+3)/3
  EXPECT_FLOAT_EQ(y.AsFloat()[1], 5.0f); // (4+5+6)/3
}

TEST(KernelClass, ReduceMeanNegativeAxisKeepdims) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceMean reduce_mean{ctx};
  Tensor data = Tensor::FromFloat(
      "", {3, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {-2});
  Tensor y = reduce_mean(data, axes, /*keepdims=*/true,
                         /*noop_with_empty_axes=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{3, 1, 2}));
  const float *py = y.AsFloat();
  // mean along axis 1 (middle dim of size 2):
  // batch 0: rows (1,2) and (3,4) -> ((1+3)/2, (2+4)/2) = (2, 3)
  // batch 1: rows (5,6) and (7,8) -> (6, 7)
  // batch 2: rows (9,10) and (11,12) -> (10, 11)
  EXPECT_FLOAT_EQ(py[0], 2.0f);
  EXPECT_FLOAT_EQ(py[1], 3.0f);
  EXPECT_FLOAT_EQ(py[2], 6.0f);
  EXPECT_FLOAT_EQ(py[3], 7.0f);
  EXPECT_FLOAT_EQ(py[4], 10.0f);
  EXPECT_FLOAT_EQ(py[5], 11.0f);
}

TEST(KernelClass, ReduceMeanNoopWithEmptyAxesIsIdentity) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceMean reduce_mean{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = reduce_mean(data, /*keepdims=*/true, /*noop_with_empty_axes=*/true);
  ASSERT_EQ(y.shape, data.shape);
  EXPECT_EQ(y.data, data.data);
}

TEST(KernelClass, ReduceMeanRejectsNonFloatData) {
  const KernelContext ctx{DefaultOpset(18)};
  ReduceMean reduce_mean{ctx};
  Tensor data = Tensor::FromInt64("", {2}, {1, 2});
  EXPECT_THROW(reduce_mean(data), std::invalid_argument);
}

} // namespace Test
