// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/controlflow/include_controlflow_kernels.h"
#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::If;
using onnx_backend_test::kernel::KernelContext;

namespace Test {

TEST(BackendKernelClass, IfClassSelectsThenBranchWhenCondTrue) {
  If if_kernel{KernelContext(DefaultOpset(13))};
  Tensor cond("", TensorProto::DataType::BOOL, {}, {1});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor out = if_kernel(cond, then_v, else_v);
  ASSERT_EQ(out.element_count(), 2);
  EXPECT_EQ(out.data_type, static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_FLOAT_EQ(out.AsFloat()[0], 1.0f);
  EXPECT_FLOAT_EQ(out.AsFloat()[1], 2.0f);
}

TEST(BackendKernelClass, IfClassSelectsElseBranchWhenCondFalse) {
  If if_kernel{KernelContext(DefaultOpset(13))};
  Tensor cond("", TensorProto::DataType::BOOL, {}, {0});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
  Tensor out = if_kernel(cond, then_v, else_v);
  ASSERT_EQ(out.element_count(), 2);
  EXPECT_FLOAT_EQ(out.AsFloat()[0], 3.0f);
  EXPECT_FLOAT_EQ(out.AsFloat()[1], 4.0f);
}

TEST(BackendKernelClass, IfClassRejectsInvalidInputs) {
  If if_kernel{KernelContext(DefaultOpset(13))};
  Tensor cond_bool("", TensorProto::DataType::BOOL, {}, {1});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // Non-bool cond is rejected.
  Tensor cond_float = Tensor::FromFloat("", {}, {1.0f});
  EXPECT_THROW((void)if_kernel(cond_float, then_v, else_v), std::invalid_argument);

  // Multi-element cond is rejected.
  Tensor cond_vec("", TensorProto::DataType::BOOL, {2}, {1, 0});
  EXPECT_THROW((void)if_kernel(cond_vec, then_v, else_v), std::invalid_argument);

  // Mismatched branch types are rejected.
  Tensor else_int = Tensor::From<int32_t>("", {2}, {3, 4});
  EXPECT_THROW((void)if_kernel(cond_bool, then_v, else_int), std::invalid_argument);

  // Mismatched branch shapes are rejected.
  Tensor else_short = Tensor::FromFloat("", {1}, {3.0f});
  EXPECT_THROW((void)if_kernel(cond_bool, then_v, else_short), std::invalid_argument);
}

TEST(BackendKernelClass, IfInPlaceWritesToPreallocatedOutput) {
  If if_kernel{KernelContext(DefaultOpset(13))};
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // cond = true → then-branch.
  {
    Tensor cond("", TensorProto::DataType::BOOL, {}, {1});
    Tensor out("", TensorProto::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
    if_kernel(cond, then_v, else_v, out);
    EXPECT_FLOAT_EQ(out.AsFloat()[0], 1.0f);
    EXPECT_FLOAT_EQ(out.AsFloat()[1], 2.0f);
  }

  // cond = false → else-branch.
  {
    Tensor cond("", TensorProto::DataType::BOOL, {}, {0});
    Tensor out("", TensorProto::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
    if_kernel(cond, then_v, else_v, out);
    EXPECT_FLOAT_EQ(out.AsFloat()[0], 3.0f);
    EXPECT_FLOAT_EQ(out.AsFloat()[1], 4.0f);
  }
}

TEST(BackendKernelClass, IfInPlaceRejectsBadOutput) {
  If if_kernel{KernelContext(DefaultOpset(13))};
  Tensor cond("", TensorProto::DataType::BOOL, {}, {1});
  Tensor then_v = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor else_v = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

  // Wrong dtype.
  Tensor bad_dtype("", TensorProto::DataType::INT32, {2},
                   std::vector<uint8_t>(2 * sizeof(int32_t)));
  EXPECT_THROW(if_kernel(cond, then_v, else_v, bad_dtype), std::invalid_argument);

  // Wrong shape.
  Tensor bad_shape("", TensorProto::DataType::FLOAT, {3}, std::vector<uint8_t>(3 * sizeof(float)));
  EXPECT_THROW(if_kernel(cond, then_v, else_v, bad_shape), std::invalid_argument);

  // Wrong buffer byte count.
  Tensor bad_bytes("", TensorProto::DataType::FLOAT, {2}, std::vector<uint8_t>(1 * sizeof(float)));
  EXPECT_THROW(if_kernel(cond, then_v, else_v, bad_bytes), std::invalid_argument);
}

} // namespace Test
