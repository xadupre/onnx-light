// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"

#include <gtest/gtest.h>

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {
namespace {

TEST(TestCaseNames, ReportsUnknownEnumValues) {
  constexpr auto unknown_kind = static_cast<TestCaseKind>(-1);
  constexpr auto unknown_tag = static_cast<TestCaseTag>(-1);

  try {
    TestCaseKindName(unknown_kind);
    FAIL() << "Expected an invalid test case kind to throw.";
  } catch (const std::invalid_argument &error) {
    EXPECT_STREQ(error.what(), "Unknown TestCaseKind value -1.");
  }

  try {
    TestCaseTagName(unknown_tag);
    FAIL() << "Expected an invalid test case tag to throw.";
  } catch (const std::invalid_argument &error) {
    EXPECT_STREQ(error.what(), "Unknown TestCaseTag value -1.");
  }
}

} // namespace
} // namespace ONNX_LIGHT_NAMESPACE::core::backend_test
