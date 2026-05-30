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
using onnx_backend_test::kernel::StringNormalizer;
using onnx_backend_test::kernel::StringSplit;

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

TEST(BackendKernelClass, StringSplitBasicMatchesReference) {
  const KernelContext ctx{DefaultOpset(20)};
  StringSplit string_split{ctx};
  Tensor x = Tensor::FromStrings("", {2}, {"abc.com", "def.net"});
  auto [y, z] = string_split(x, ".");

  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(z.shape, x.shape);
  EXPECT_EQ(y.AsStrings(), (std::vector<std::string>{"abc", "com", "def", "net"}));
  const int64_t *counts = z.AsInt64();
  ASSERT_NE(counts, nullptr);
  EXPECT_EQ(counts[0], 2);
  EXPECT_EQ(counts[1], 2);
}

TEST(BackendKernelClass, StringSplitHandlesWhitespaceAndPadding) {
  const KernelContext ctx{DefaultOpset(20)};
  StringSplit string_split{ctx};
  Tensor x =
      Tensor::FromStrings("", {2, 2}, {"hello world", "def.net", "o n n x", "the quick brown fox"});
  auto [y, z] = string_split(x, "", 2);

  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 2, 3}));
  EXPECT_EQ(z.shape, x.shape);
  EXPECT_EQ(y.AsStrings(), (std::vector<std::string>{"hello", "world", "", "def.net", "", "", "o",
                                                     "n", "n x", "the", "quick", "brown fox"}));
  const int64_t *counts = z.AsInt64();
  ASSERT_NE(counts, nullptr);
  EXPECT_EQ(counts[0], 2);
  EXPECT_EQ(counts[1], 1);
  EXPECT_EQ(counts[2], 3);
  EXPECT_EQ(counts[3], 3);
}

TEST(BackendKernelClass, StringSplitConsecutiveDelimitersAndEmptyTensor) {
  const KernelContext ctx{DefaultOpset(20)};
  StringSplit string_split{ctx};

  Tensor x = Tensor::FromStrings("", {2}, {"o-n-n--x-", "o-n----nx"});
  auto [y, z] = string_split(x, "-");
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2, 6}));
  EXPECT_EQ(y.AsStrings(),
            (std::vector<std::string>{"o", "n", "n", "", "x", "", "o", "n", "", "", "", "nx"}));
  const int64_t *counts = z.AsInt64();
  ASSERT_NE(counts, nullptr);
  EXPECT_EQ(counts[0], 6);
  EXPECT_EQ(counts[1], 6);

  Tensor empty = Tensor::FromStrings("", {0}, std::vector<std::string>{});
  auto [empty_y, empty_z] = string_split(empty);
  EXPECT_EQ(empty_y.shape, (std::vector<int64_t>{0, 0}));
  EXPECT_TRUE(empty_y.AsStrings().empty());
  EXPECT_EQ(empty_z.shape, (std::vector<int64_t>{0}));
}

TEST(BackendKernelClass, StringSplitRejectsNonStringInput) {
  const KernelContext ctx{DefaultOpset(20)};
  StringSplit string_split{ctx};
  Tensor bad_dtype = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  EXPECT_THROW(string_split(bad_dtype), std::invalid_argument);
}

TEST(BackendKernelClass, StringNormalizerLowercases1D) {
  const KernelContext ctx{DefaultOpset(10)};
  StringNormalizer normalizer{ctx};
  Tensor x = Tensor::FromStrings("", {3}, {"Hello", "World", "FOO"});
  Tensor y = normalizer(x, StringNormalizer::CaseChangeAction::kLower);
  EXPECT_EQ(y.shape, x.shape);
  const auto &out = y.AsStrings();
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0], "hello");
  EXPECT_EQ(out[1], "world");
  EXPECT_EQ(out[2], "foo");
}

TEST(BackendKernelClass, StringNormalizerDropsCaseInsensitiveStopwords2D) {
  const KernelContext ctx{DefaultOpset(10)};
  StringNormalizer normalizer{ctx};
  Tensor x = Tensor::FromStrings("", {1, 4}, {"A", "hello", "a", "world"});
  Tensor y = normalizer(x, StringNormalizer::CaseChangeAction::kUpper,
                        /*is_case_sensitive=*/false, {"a"});
  EXPECT_EQ(y.shape, (std::vector<int64_t>{1, 2}));
  const auto &out = y.AsStrings();
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], "HELLO");
  EXPECT_EQ(out[1], "WORLD");
}

TEST(BackendKernelClass, StringNormalizerCaseSensitiveStopwords) {
  const KernelContext ctx{DefaultOpset(10)};
  StringNormalizer normalizer{ctx};
  Tensor x = Tensor::FromStrings("", {3}, {"The", "the", "cat"});
  Tensor y = normalizer(x, StringNormalizer::CaseChangeAction::kNone,
                        /*is_case_sensitive=*/true, {"the"});
  EXPECT_EQ(y.shape, (std::vector<int64_t>{2}));
  const auto &out = y.AsStrings();
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], "The");
  EXPECT_EQ(out[1], "cat");
}

TEST(BackendKernelClass, StringNormalizerAllDroppedEmitsEmpty) {
  const KernelContext ctx{DefaultOpset(10)};
  StringNormalizer normalizer{ctx};
  Tensor x1d = Tensor::FromStrings("", {2}, {"a", "b"});
  Tensor y1d = normalizer(x1d, StringNormalizer::CaseChangeAction::kNone, false, {"a", "b"});
  EXPECT_EQ(y1d.shape, (std::vector<int64_t>{1}));
  ASSERT_EQ(y1d.AsStrings().size(), 1u);
  EXPECT_EQ(y1d.AsStrings()[0], "");

  Tensor x2d = Tensor::FromStrings("", {1, 2}, {"a", "b"});
  Tensor y2d = normalizer(x2d, StringNormalizer::CaseChangeAction::kNone, false, {"a", "b"});
  EXPECT_EQ(y2d.shape, (std::vector<int64_t>{1, 1}));
  ASSERT_EQ(y2d.AsStrings().size(), 1u);
  EXPECT_EQ(y2d.AsStrings()[0], "");
}

TEST(BackendKernelClass, StringNormalizerParseCaseChangeAction) {
  EXPECT_EQ(StringNormalizer::ParseCaseChangeAction("NONE"),
            StringNormalizer::CaseChangeAction::kNone);
  EXPECT_EQ(StringNormalizer::ParseCaseChangeAction("LOWER"),
            StringNormalizer::CaseChangeAction::kLower);
  EXPECT_EQ(StringNormalizer::ParseCaseChangeAction("UPPER"),
            StringNormalizer::CaseChangeAction::kUpper);
  EXPECT_THROW(StringNormalizer::ParseCaseChangeAction("lower"), std::invalid_argument);
  EXPECT_THROW(StringNormalizer::ParseCaseChangeAction(""), std::invalid_argument);
}

TEST(BackendKernelClass, StringNormalizerRejectsBadInputs) {
  const KernelContext ctx{DefaultOpset(10)};
  StringNormalizer normalizer{ctx};

  // Non-string input.
  Tensor bad_dtype = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
  EXPECT_THROW(normalizer(bad_dtype), std::invalid_argument);

  // Unsupported rank (rank 3).
  Tensor bad_rank = Tensor::MakeString("", {1, 1, 2}, std::vector<std::string>(2));
  EXPECT_THROW(normalizer(bad_rank), std::invalid_argument);

  // 2-D shape with leading dim != 1 is rejected.
  Tensor bad_2d = Tensor::MakeString("", {2, 2}, std::vector<std::string>(4));
  EXPECT_THROW(normalizer(bad_2d), std::invalid_argument);
}

} // namespace Test
