// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/kernels/optional/include_optional_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::KernelContext;
using OptionalKernel = onnx_kernels::kernel::Optional;

namespace Test {

TEST(KernelClass, OptionalPassthroughCopiesInput) {
  const KernelContext ctx{DefaultOpset(15)};
  OptionalKernel opt{ctx};
  Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
  Tensor y = opt(x);
  ASSERT_EQ(y.element_count(), 6);
  EXPECT_EQ(y.data_type, x.data_type);
  EXPECT_EQ(y.shape, x.shape);
  EXPECT_EQ(y.data, x.data);
  // Distinct buffers (returning overload allocates a fresh tensor).
  EXPECT_NE(y.data.data(), x.data.data());
}

TEST(KernelClass, OptionalRejectsBadInputsAndMismatchedOutput) {
  const KernelContext ctx{DefaultOpset(15)};
  OptionalKernel opt{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 2.0f});

  // Undefined input element type is rejected.
  Tensor bad_input;
  EXPECT_THROW(opt(bad_input), std::invalid_argument);

  // In-place overload with a mismatched output buffer is rejected.
  Tensor bad_dtype("", static_cast<int32_t>(onnx_kernels::DataType::INT32), x.shape,
                   std::vector<uint8_t>(x.element_count() * sizeof(int32_t)));
  EXPECT_THROW(opt(x, bad_dtype), std::invalid_argument);

  Tensor bad_shape("", x.data_type, {3}, std::vector<uint8_t>(3 * sizeof(float)));
  EXPECT_THROW(opt(x, bad_shape), std::invalid_argument);
}

TEST(KernelClass, OptionalInPlaceAliasingInputAndOutput) {
  // Optional::CanRunInPlace() should be honored: passing the same Tensor as
  // both input and output must succeed and leave the bytes untouched (since
  // Optional is a passthrough).
  ASSERT_TRUE(OptionalKernel::CanRunInPlace());
  const KernelContext ctx{DefaultOpset(15)};
  OptionalKernel opt{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 2.5f});
  const std::vector<uint8_t> before = x.data;
  opt(x, x);
  EXPECT_EQ(x.data, before);
}

} // namespace Test
