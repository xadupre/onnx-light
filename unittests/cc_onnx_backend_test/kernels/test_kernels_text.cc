// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/text/include_text_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::StringConcat;

namespace Test {

TEST(BackendKernelClass, StringConcatEqualShapeMatchesReference) {
  const KernelContext ctx{DefaultOpset(20)};
  StringConcat string_concat{ctx};
  Tensor x = Tensor::FromStrings("", {3}, {"abc", "", "hello "});
  Tensor y = Tensor::FromStrings("", {3}, {"def", "xyz", "world"});
  Tensor z = string_concat(x, y);
  ASSERT_EQ(z.element_count(), 3);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::STRING));
  EXPECT_EQ(z.shape, x.shape);
  const auto &out = z.AsStrings();
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0], "abcdef");
  EXPECT_EQ(out[1], "xyz");
  EXPECT_EQ(out[2], "hello world");
}

TEST(BackendKernelClass, StringConcatBroadcastsScalar) {
  const KernelContext ctx{DefaultOpset(20)};
  StringConcat string_concat{ctx};
  Tensor x = Tensor::FromStrings("", {2, 2}, {"a", "b", "c", "d"});
  Tensor y = Tensor::FromStrings("", {}, {"!"});
  Tensor z = string_concat(x, y);
  EXPECT_EQ(z.shape, x.shape);
  const auto &out = z.AsStrings();
  ASSERT_EQ(out.size(), 4u);
  EXPECT_EQ(out[0], "a!");
  EXPECT_EQ(out[1], "b!");
  EXPECT_EQ(out[2], "c!");
  EXPECT_EQ(out[3], "d!");

  // Symmetric: scalar on the left side.
  Tensor z2 = string_concat(y, x);
  const auto &out2 = z2.AsStrings();
  ASSERT_EQ(out2.size(), 4u);
  EXPECT_EQ(out2[0], "!a");
  EXPECT_EQ(out2[3], "!d");
}

TEST(BackendKernelClass, StringConcatRejectsBadInputsAndMismatchedOutput) {
  const KernelContext ctx{DefaultOpset(20)};
  StringConcat string_concat{ctx};
  Tensor x = Tensor::FromStrings("", {2}, {"a", "b"});
  Tensor y = Tensor::FromStrings("", {2}, {"x", "y"});

  // Non-STRING input is rejected.
  Tensor bad_dtype = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  EXPECT_THROW(string_concat(bad_dtype, y), std::invalid_argument);
  EXPECT_THROW(string_concat(x, bad_dtype), std::invalid_argument);

  // Incompatible shapes are rejected (neither equal nor scalar broadcast).
  Tensor mismatched = Tensor::FromStrings("", {3}, {"u", "v", "w"});
  EXPECT_THROW(string_concat(x, mismatched), std::invalid_argument);

  // In-place overload with wrong-dtype / shape / size preallocated output is
  // rejected.
  Tensor bad_out_dtype = Tensor::FromFloat("", {2}, {0.0f, 0.0f});
  EXPECT_THROW(string_concat(x, y, bad_out_dtype), std::invalid_argument);

  Tensor bad_out_shape = Tensor::MakeString("", {3}, std::vector<std::string>(3));
  EXPECT_THROW(string_concat(x, y, bad_out_shape), std::invalid_argument);

  Tensor bad_out_size = Tensor::MakeString("", {2}, std::vector<std::string>(1));
  EXPECT_THROW(string_concat(x, y, bad_out_size), std::invalid_argument);
}

} // namespace Test
