// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/controlflow/include_controlflow_kernels.h"
#include "onnx_backend_test/kernels/generator/include_generator_kernels.h"
#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::OpsetId;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::Abs;
using onnx_backend_test::kernel::Add;
using onnx_backend_test::kernel::And;
using onnx_backend_test::kernel::BlackmanWindow;
using onnx_backend_test::kernel::Concat;
using onnx_backend_test::kernel::Constant;
using onnx_backend_test::kernel::If;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::Or;
using onnx_backend_test::kernel::Xor;

namespace Test {

TEST(BackendKernelClass, KernelContextStoresOpset) {
  KernelContext ctx(DefaultOpset(13));
  EXPECT_EQ(ctx.opset.domain, std::string());
  EXPECT_EQ(ctx.opset.version, 13);
}

TEST(BackendKernelClass, AbsClassMatchesReference) {
  Abs abs_kernel{KernelContext(DefaultOpset(13))};

  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});
  Tensor y = abs_kernel(x);
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 2.5f);
}

TEST(BackendKernelClass, AddClassBroadcastsScalar) {
  Add add_kernel{KernelContext(DefaultOpset(14))};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {0.5f});
  Tensor z = add_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 1.5f);
  EXPECT_FLOAT_EQ(pz[1], 2.5f);
  EXPECT_FLOAT_EQ(pz[2], 3.5f);
  EXPECT_FLOAT_EQ(pz[3], 4.5f);
}

TEST(BackendKernelClass, BlackmanWindowPeriodicLength) {
  BlackmanWindow blackman_kernel{KernelContext(DefaultOpset(17))};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y = blackman_kernel(size, /*periodic=*/true);
  EXPECT_EQ(y.element_count(), 8);
  // First sample of the Blackman window is 0 by construction.
  EXPECT_NEAR(y.AsFloat()[0], 0.0f, 1e-6f);
}

TEST(BackendKernelClass, ConstantClassMatchesReference) {
  Constant constant_kernel{KernelContext(DefaultOpset(13))};
  Tensor value = Tensor::FromFloat("", {2, 2}, {1.0f, -2.0f, 3.5f, 0.0f});
  Tensor y = constant_kernel(value);
  ASSERT_EQ(y.data_type, value.data_type);
  EXPECT_EQ(y.shape, value.shape);
  ASSERT_EQ(y.element_count(), value.element_count());
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], -2.0f);
  EXPECT_FLOAT_EQ(py[2], 3.5f);
  EXPECT_FLOAT_EQ(py[3], 0.0f);
}

TEST(BackendKernelClass, ConstantRejectsMismatchedOutput) {
  Constant constant_kernel{KernelContext(DefaultOpset(13))};
  Tensor value = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  Tensor bad_shape("", TensorProto::DataType::FLOAT, {3}, std::vector<uint8_t>(3 * sizeof(float)));
  EXPECT_THROW(constant_kernel(value, bad_shape), std::invalid_argument);
  Tensor bad_type("", TensorProto::DataType::INT32, {2}, std::vector<uint8_t>(2 * sizeof(int32_t)));
  EXPECT_THROW(constant_kernel(value, bad_type), std::invalid_argument);
}

TEST(BackendKernelClass, AndClassMatchesReference) {
  And and_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z = and_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, AndClassBroadcastsScalar) {
  And and_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {}, {1});
  Tensor z = and_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, OrClassMatchesReference) {
  Or or_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z = or_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, XorClassMatchesReference) {
  Xor xor_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z = xor_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(z.data[0], 0);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, LogicalRejectsNonBoolTensors) {
  And and_kernel{KernelContext(DefaultOpset(7))};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 0.0f});
  Tensor y("", TensorProto::DataType::BOOL, {2}, {1, 0});
  EXPECT_THROW((void)and_kernel(x, y), std::invalid_argument);
}

TEST(BackendKernelClass, ConcatClassConcatenatesAxis0) {
  Concat concat_kernel{KernelContext(DefaultOpset(13))};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  Tensor y = concat_kernel({x0, x1}, /*axis=*/0);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{4, 2}));
  ASSERT_EQ(y.element_count(), 8);
  const float *py = y.AsFloat();
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(py[i], static_cast<float>(i + 1));
  }
}

TEST(BackendKernelClass, ConcatClassConcatenatesNegativeAxis) {
  Concat concat_kernel{KernelContext(DefaultOpset(13))};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {2, 3}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
  Tensor y = concat_kernel({x0, x1}, /*axis=*/-1);
  ASSERT_EQ(y.shape, (std::vector<int64_t>{2, 5}));
  ASSERT_EQ(y.element_count(), 10);
  const float *py = y.AsFloat();
  const std::vector<float> expected{1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 3.0f, 4.0f, 8.0f, 9.0f, 10.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]);
  }
}

TEST(BackendKernelClass, ConcatClassRejectsMismatchedShape) {
  Concat concat_kernel{KernelContext(DefaultOpset(13))};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {3, 2}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
  EXPECT_THROW((void)concat_kernel({x0, x1}, /*axis=*/1), std::invalid_argument);
}

TEST(BackendKernelClass, ConcatClassRejectsScalar) {
  Concat concat_kernel{KernelContext(DefaultOpset(13))};
  Tensor x = Tensor::FromFloat("", {}, {1.0f});
  EXPECT_THROW((void)concat_kernel({x}, /*axis=*/0), std::invalid_argument);
}

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

// ---------------------------------------------------------------------------
// In-place overloads — the caller pre-allocates the output Tensor and the
// kernel writes results into it.
// ---------------------------------------------------------------------------

TEST(BackendKernelClass, AbsInPlaceWritesToPreallocatedOutput) {
  Abs abs_kernel{KernelContext(DefaultOpset(13))};
  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});
  Tensor y("out", TensorProto::DataType::FLOAT, {3}, std::vector<uint8_t>(3 * sizeof(float), 0xFF));
  abs_kernel(x, y);
  EXPECT_EQ(y.name, "out");
  ASSERT_EQ(y.element_count(), 3);
  const float *py = y.AsFloat();
  EXPECT_FLOAT_EQ(py[0], 1.0f);
  EXPECT_FLOAT_EQ(py[1], 0.0f);
  EXPECT_FLOAT_EQ(py[2], 2.5f);
}

TEST(BackendKernelClass, AddInPlaceWritesToPreallocatedOutput) {
  Add add_kernel{KernelContext(DefaultOpset(14))};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {0.5f});
  Tensor z("", TensorProto::DataType::FLOAT, {2, 2}, std::vector<uint8_t>(4 * sizeof(float)));
  add_kernel(x, y, z);
  ASSERT_EQ(z.element_count(), 4);
  const float *pz = z.AsFloat();
  EXPECT_FLOAT_EQ(pz[0], 1.5f);
  EXPECT_FLOAT_EQ(pz[1], 2.5f);
  EXPECT_FLOAT_EQ(pz[2], 3.5f);
  EXPECT_FLOAT_EQ(pz[3], 4.5f);
}

TEST(BackendKernelClass, BlackmanWindowInPlaceWritesToPreallocatedOutput) {
  BlackmanWindow blackman_kernel{KernelContext(DefaultOpset(17))};
  Tensor size = Tensor::FromInt32("", {}, {8});
  Tensor y("", TensorProto::DataType::FLOAT, {8}, std::vector<uint8_t>(8 * sizeof(float)));
  blackman_kernel(size, /*periodic=*/true, y);
  EXPECT_EQ(y.element_count(), 8);
  EXPECT_NEAR(y.AsFloat()[0], 0.0f, 1e-6f);
}

TEST(BackendKernelClass, ConcatInPlaceWritesToPreallocatedOutput) {
  Concat concat_kernel{KernelContext(DefaultOpset(13))};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {2, 3}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
  Tensor y("out", TensorProto::DataType::FLOAT, {2, 5}, std::vector<uint8_t>(10 * sizeof(float)));
  concat_kernel({x0, x1}, /*axis=*/-1, y);
  EXPECT_EQ(y.name, "out");
  ASSERT_EQ(y.element_count(), 10);
  const float *py = y.AsFloat();
  const std::vector<float> expected{1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 3.0f, 4.0f, 8.0f, 9.0f, 10.0f};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(py[i], expected[i]);
  }
}

TEST(BackendKernelClass, ConcatInPlaceRejectsMismatchedShape) {
  Concat concat_kernel{KernelContext(DefaultOpset(13))};
  Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor x1 = Tensor::FromFloat("", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  Tensor bad_shape("", TensorProto::DataType::FLOAT, {3, 2},
                   std::vector<uint8_t>(6 * sizeof(float)));
  EXPECT_THROW(concat_kernel({x0, x1}, /*axis=*/0, bad_shape), std::invalid_argument);
}

TEST(BackendKernelClass, AndInPlaceWritesToPreallocatedOutput) {
  And and_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z("", TensorProto::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4, 9));
  and_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, OrInPlaceWritesToPreallocatedOutput) {
  Or or_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z("", TensorProto::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4));
  or_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, XorInPlaceWritesToPreallocatedOutput) {
  Xor xor_kernel{KernelContext(DefaultOpset(7))};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z("", TensorProto::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4));
  xor_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 0);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
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

TEST(BackendKernelClass, InPlaceRejectsMismatchedShapeOrType) {
  Abs abs_kernel{KernelContext(DefaultOpset(13))};
  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});

  // Wrong dtype.
  Tensor bad_dtype("", TensorProto::DataType::INT32, {3},
                   std::vector<uint8_t>(3 * sizeof(int32_t)));
  EXPECT_THROW(abs_kernel(x, bad_dtype), std::invalid_argument);

  // Wrong shape.
  Tensor bad_shape("", TensorProto::DataType::FLOAT, {2}, std::vector<uint8_t>(2 * sizeof(float)));
  EXPECT_THROW(abs_kernel(x, bad_shape), std::invalid_argument);

  // Wrong buffer byte count.
  Tensor bad_bytes("", TensorProto::DataType::FLOAT, {3}, std::vector<uint8_t>(1 * sizeof(float)));
  EXPECT_THROW(abs_kernel(x, bad_bytes), std::invalid_argument);
}

TEST(BackendKernelClass, CanRunInPlaceReportsKernelCapability) {
  // Element-wise unary/binary kernels can write their output into an input
  // buffer (shape and dtype match by construction or when no broadcasting
  // expansion is needed for that input).
  EXPECT_TRUE(Abs::CanRunInPlace());
  EXPECT_TRUE(Add::CanRunInPlace());
  EXPECT_TRUE(And::CanRunInPlace());
  EXPECT_TRUE(Or::CanRunInPlace());
  EXPECT_TRUE(Xor::CanRunInPlace());

  // If just copies the selected branch into the output.
  EXPECT_TRUE(If::CanRunInPlace());

  // Output buffer fundamentally cannot equal any input buffer for these.
  EXPECT_FALSE(BlackmanWindow::CanRunInPlace());
  EXPECT_FALSE(Concat::CanRunInPlace());
}

TEST(BackendKernelClass, AbsInPlaceAliasingInputAndOutput) {
  // Demonstrates that Abs::CanRunInPlace() is honored by the implementation:
  // pass the same Tensor object as both input and output and verify the
  // result is written correctly in-place.
  ASSERT_TRUE(Abs::CanRunInPlace());
  Abs abs_kernel{KernelContext(DefaultOpset(13))};
  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});
  abs_kernel(x, x);
  const float *px = x.AsFloat();
  EXPECT_FLOAT_EQ(px[0], 1.0f);
  EXPECT_FLOAT_EQ(px[1], 0.0f);
  EXPECT_FLOAT_EQ(px[2], 2.5f);
}

} // namespace Test
