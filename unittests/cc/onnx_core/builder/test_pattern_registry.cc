// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/pattern_registry.h"

#include <memory>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

class CustomPattern final : public core::builder::PatternOptimization {
public:
  core::builder::MatchResult Match(core::builder::GraphGraph &, const NodeProto &) const override {
    return {};
  }

  utils::RepeatedProtoField<NodeProto>
  Apply(core::builder::GraphGraph &, const std::vector<const NodeProto *> &) const override {
    return utils::RepeatedProtoField<NodeProto>();
  }
};

} // namespace

TEST(PatternRegistry, StartsEmptyWithoutPatternExtension) {
  EXPECT_TRUE(core::builder::RegisteredPatternNames().empty());
  EXPECT_TRUE(core::builder::CreateRegisteredPatterns().empty());
}

TEST(PatternRegistry, RegistersAndCreatesCustomPattern) {
  const std::size_t initial_size = core::builder::RegisteredPatternNames().size();
  core::builder::RegisterPattern("test.CustomPattern",
                                 []() { return std::make_unique<CustomPattern>(); });

  const std::vector<std::string> names = core::builder::RegisteredPatternNames();
  ASSERT_EQ(names.size(), initial_size + 1);
  EXPECT_EQ(names.back(), "test.CustomPattern");

  std::vector<std::unique_ptr<core::builder::PatternOptimization>> patterns =
      core::builder::CreateRegisteredPatterns();
  ASSERT_EQ(patterns.size(), names.size());
  EXPECT_NE(dynamic_cast<CustomPattern *>(patterns.back().get()), nullptr);
}

TEST(PatternRegistry, RejectsDuplicateName) {
  core::builder::RegisterPattern("test.DuplicatePattern",
                                 []() { return std::make_unique<CustomPattern>(); });
  EXPECT_THROW(core::builder::RegisterPattern("test.DuplicatePattern",
                                              []() { return std::make_unique<CustomPattern>(); }),
               core::builder::PatternRegistrationError);
}

TEST(PatternRegistry, RejectsInvalidRegistration) {
  EXPECT_THROW(
      core::builder::RegisterPattern("", []() { return std::make_unique<CustomPattern>(); }),
      core::builder::PatternRegistrationError);
  EXPECT_THROW(core::builder::RegisterPattern("test.EmptyFactory", {}),
               core::builder::PatternRegistrationError);
}

} // namespace Test
