// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/reduction/include_reduction_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::ReduceSum;

namespace Test {

TEST(BackendKernelClass, ReduceSumDefaultAxesReducesAll) {
  const KernelContext ctx{DefaultOpset(13)};
  ReduceSum reduce_sum{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor y = reduce_sum(data); // keepdims=true, noop_with_empty_axes=false
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{1, 1}));
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 21.0f);
}

TEST(BackendKernelClass, ReduceSumDefaultAxesNoKeepdimsProducesScalar) {
  const KernelContext ctx{DefaultOpset(13)};
  ReduceSum reduce_sum{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = reduce_sum(data, /*keepdims=*/false);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{}));
  ASSERT_EQ(y.element_count(), 1);
  EXPECT_FLOAT_EQ(y.AsFloat()[0], 10.0f);
}

TEST(BackendKernelClass, ReduceSumNoopWithEmptyAxesIsIdentity) {
  const KernelContext ctx{DefaultOpset(13)};
  ReduceSum reduce_sum{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = reduce_sum(data, /*keepdims=*/true, /*noop_with_empty_axes=*/true);
  ASSERT_EQ(y.shape, data.shape);
  EXPECT_EQ(y.data, data.data);
}

TEST(BackendKernelClass, ReduceSumExplicitAxisReducesAlongAxis) {
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

TEST(BackendKernelClass, ReduceSumNegativeAxisKeepdims) {
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

TEST(BackendKernelClass, ReduceSumInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  ReduceSum reduce_sum{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor axes = Tensor::FromInt64("", {1}, {0});
  Tensor out("", static_cast<int32_t>(TensorProto::DataType::FLOAT), {1, 3},
             std::vector<uint8_t>(3 * sizeof(float), 0u));
  reduce_sum(data, axes, /*keepdims=*/true, /*noop_with_empty_axes=*/false, out);
  const float *po = out.AsFloat();
  EXPECT_FLOAT_EQ(po[0], 5.0f);
  EXPECT_FLOAT_EQ(po[1], 7.0f);
  EXPECT_FLOAT_EQ(po[2], 9.0f);
}

TEST(BackendKernelClass, ReduceSumRejectsBadInputs) {
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
  Tensor bad_shape("", static_cast<int32_t>(TensorProto::DataType::FLOAT), {2},
                   std::vector<uint8_t>(2 * sizeof(float), 0u));
  EXPECT_THROW(reduce_sum(data, axes, /*keepdims=*/true,
                          /*noop_with_empty_axes=*/false, bad_shape),
               std::invalid_argument);
}

// ── ArgMax / ArgMin kernels ────────────────────────────────────────────────

using onnx_backend_test::kernel::ArgMax;
using onnx_backend_test::kernel::ArgMin;

TEST(BackendKernelClass, ArgMaxAlongAxisKeepdims) {
  const KernelContext ctx{DefaultOpset(13)};
  ArgMax argmax{ctx};
  Tensor data = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 3.0f, 10.0f});
  Tensor y = argmax(data, /*axis=*/1, /*keepdims=*/true,
                    /*select_last_index=*/false);
  ASSERT_EQ(y.data_type, static_cast<int32_t>(TensorProto::DataType::INT64));
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 1}));
  const int64_t *py = y.AsInt64();
  EXPECT_EQ(py[0], 0);
  EXPECT_EQ(py[1], 1);
}

TEST(BackendKernelClass, ArgMaxDefaultAxisNoKeepdims) {
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

TEST(BackendKernelClass, ArgMaxNegativeAxisSelectLastIndex) {
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

TEST(BackendKernelClass, ArgMinAlongAxisSelectLastIndex) {
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

TEST(BackendKernelClass, ArgReduceInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  ArgMax argmax{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 5.0f, 2.0f, 4.0f, 0.0f, 9.0f});
  Tensor out("", static_cast<int32_t>(TensorProto::DataType::INT64), {2, 1},
             std::vector<uint8_t>(2 * sizeof(int64_t), 0u));
  argmax(data, /*axis=*/1, /*keepdims=*/true, /*select_last_index=*/false, out);
  const int64_t *po = out.AsInt64();
  EXPECT_EQ(po[0], 1);
  EXPECT_EQ(po[1], 2);
}

TEST(BackendKernelClass, ArgReduceRejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(13)};
  ArgMax argmax{ctx};
  Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  // Wrong data dtype.
  Tensor bad_data("", static_cast<int32_t>(TensorProto::DataType::INT32), {2, 3},
                  std::vector<uint8_t>(6 * sizeof(int32_t)));
  EXPECT_THROW(argmax(bad_data, /*axis=*/0), std::invalid_argument);

  // Out-of-range axis.
  EXPECT_THROW(argmax(data, /*axis=*/5), std::invalid_argument);

  // Scalar input.
  Tensor scalar("", static_cast<int32_t>(TensorProto::DataType::FLOAT), {},
                std::vector<uint8_t>(sizeof(float), 0u));
  EXPECT_THROW(argmax(scalar, /*axis=*/0), std::invalid_argument);

  // Mismatched preallocated output shape.
  Tensor bad_out("", static_cast<int32_t>(TensorProto::DataType::INT64), {2, 3},
                 std::vector<uint8_t>(6 * sizeof(int64_t), 0u));
  EXPECT_THROW(argmax(data, /*axis=*/0, /*keepdims=*/true, /*select_last_index=*/false, bad_out),
               std::invalid_argument);

  // Wrong output dtype.
  Tensor wrong_dtype_out("", static_cast<int32_t>(TensorProto::DataType::FLOAT), {1, 3},
                         std::vector<uint8_t>(3 * sizeof(float), 0u));
  EXPECT_THROW(
      argmax(data, /*axis=*/0, /*keepdims=*/true, /*select_last_index=*/false, wrong_dtype_out),
      std::invalid_argument);
}

} // namespace Test
