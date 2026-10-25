// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/builder/pattern_registry.h"

#include <algorithm>
#include <memory>

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

class CustomPattern final : public core::builder::PatternOptimization {
public:
  explicit CustomPattern(std::string name = "test.CustomPattern",
                         core::symbolic::Device device = core::symbolic::Device::kUndefined)
      : PatternOptimization(1, std::move(name), device) {}

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

  std::unique_ptr<core::builder::PatternOptimization> by_name =
      core::builder::CreateRegisteredPattern("test.CustomPattern");
  ASSERT_NE(by_name, nullptr);
  EXPECT_EQ(by_name->Name(), "test.CustomPattern");

  std::unique_ptr<core::builder::PatternOptimization> by_name_with_priority =
      core::builder::CreateRegisteredPattern("test.CustomPattern", 7);
  ASSERT_NE(by_name_with_priority, nullptr);
  EXPECT_EQ(by_name_with_priority->Name(), "test.CustomPattern");
  EXPECT_EQ(by_name_with_priority->priority, 7);
}

TEST(PatternRegistry, RegistersExtendsAndRemovesPatternSets) {
  core::builder::RegisterPattern("test.PatternSetMember", []() {
    return std::make_unique<CustomPattern>("test.PatternSetMember");
  });
  core::builder::RegisterPatternSet("test.PatternSet", {"test.PatternSetMember"});
  core::builder::ExtendPatternSet("test.PatternSet", {"test.PatternSetMember", "test.OtherMember"});

  const auto sets = core::builder::RegisteredPatternSets();
  const auto set = sets.find("test.PatternSet");
  ASSERT_NE(set, sets.end());
  EXPECT_EQ(set->second, (std::vector<std::string>{"test.PatternSetMember", "test.OtherMember"}));
  EXPECT_NE(std::find(sets.at("all").begin(), sets.at("all").end(), "test.OtherMember"),
            sets.at("all").end());
  EXPECT_FALSE(core::builder::UnregisterPatternSet("L1"));
  EXPECT_TRUE(core::builder::UnregisterPatternSet("test.PatternSet"));
  EXPECT_FALSE(core::builder::UnregisterPatternSet("test.PatternSet"));
}

TEST(PatternRegistry, RejectsInvalidPatternSets) {
  EXPECT_THROW(core::builder::RegisterPatternSet("", {}), core::builder::PatternRegistrationError);
  EXPECT_THROW(core::builder::RegisterPatternSet("all", {}),
               core::builder::PatternRegistrationError);
  EXPECT_THROW(core::builder::RegisterPatternSet("test.EmptyName", {""}),
               core::builder::PatternRegistrationError);
  EXPECT_THROW(core::builder::ExtendPatternSet("test.UnknownSet", {"Pattern"}),
               core::builder::PatternRegistrationError);
}

TEST(PatternRegistry, FiltersDeviceSpecificPatterns) {
  using core::symbolic::Device;
  core::builder::RegisterPattern("test.CPUOnly", []() {
    return std::make_unique<CustomPattern>("test.CPUOnly", Device::kCPU);
  });
  core::builder::RegisterPattern("test.GPUOnly", []() {
    return std::make_unique<CustomPattern>("test.GPUOnly", Device::kGPU0);
  });
  const auto all = core::builder::CreateRegisteredPatterns();
  const auto generic = core::builder::CreateRegisteredPatterns(Device::kUndefined);
  EXPECT_EQ(all.size(), generic.size() + 2);
  for (const auto &pattern : generic)
    EXPECT_EQ(pattern->device, Device::kUndefined);
  for (Device device : {Device::kCPU, Device::kGPU0}) {
    const auto selected = core::builder::CreateRegisteredPatterns(device);
    EXPECT_EQ(selected.size(), generic.size() + 1);
    EXPECT_EQ(std::count_if(selected.begin(), selected.end(),
                            [device](const auto &pattern) { return pattern->device == device; }),
              1);
    core::builder::GraphBuilder builder("device_patterns");
    builder.set_device(device);
    core::builder::GraphGraph graph(builder);
    EXPECT_EQ(graph.Patterns().size(), generic.size());
    EXPECT_EQ(builder.device(), device);
  }
  const auto explicit_pattern = core::builder::CreateRegisteredPattern("test.GPUOnly");
  EXPECT_EQ(explicit_pattern->device, Device::kGPU0);
}

TEST(PatternRegistry, RecursiveOptimizationInheritsOnlyUndefinedDevices) {
  using core::symbolic::Device;
  core::builder::GraphBuilder builder("device_inheritance");
  builder.set_device(Device::kGPU0);
  auto &inherited = builder.MakeSubgraph("inherited");
  auto &explicit_cpu = builder.MakeSubgraph("explicit_cpu");
  explicit_cpu.set_device(Device::kCPU);
  auto &grandchild = explicit_cpu.MakeSubgraph("grandchild");
  core::builder::GraphGraph graph(
      builder, std::vector<std::shared_ptr<core::builder::PatternOptimization>>{});
  graph.Optimize();
  EXPECT_EQ(inherited.device(), Device::kGPU0);
  EXPECT_EQ(explicit_cpu.device(), Device::kCPU);
  EXPECT_EQ(grandchild.device(), Device::kCPU);
}

TEST(PatternRegistry, RejectsDuplicateName) {
  core::builder::RegisterPattern("test.DuplicatePattern", []() {
    return std::make_unique<CustomPattern>("test.DuplicatePattern");
  });
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
  EXPECT_THROW(core::builder::CreateRegisteredPattern(""), core::builder::PatternRegistrationError);
  EXPECT_THROW(core::builder::CreateRegisteredPattern("test.UnknownPattern"),
               core::builder::PatternRegistrationError);
}

TEST(PatternRegistry, ExplicitOnlyPatternsAreExcludedFromDefaults) {
  core::builder::RegisterPattern(
      "test.ExplicitOnly", []() { return std::make_unique<CustomPattern>("test.ExplicitOnly"); },
      false);
  const auto names = core::builder::RegisteredPatternNames();
  EXPECT_NE(std::find(names.begin(), names.end(), "test.ExplicitOnly"), names.end());
  const auto defaults = core::builder::RegisteredPatternNames(false);
  EXPECT_EQ(std::find(defaults.begin(), defaults.end(), "test.ExplicitOnly"), defaults.end());
  for (const auto &pattern : core::builder::CreateRegisteredPatterns())
    EXPECT_NE(pattern->Name(), "test.ExplicitOnly");
  EXPECT_EQ(core::builder::CreateRegisteredPattern("test.ExplicitOnly")->Name(),
            "test.ExplicitOnly");
}

} // namespace Test
