#include "onnx_proto/simple_string.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <type_traits>
#include <unordered_set>

using namespace ONNX_LIGHT_NAMESPACE;

static_assert(std::is_convertible_v<utils::OptionalString &, std::optional<std::string> &>);
static_assert(
    std::is_convertible_v<const utils::OptionalString &, const std::optional<std::string> &>);
static_assert(std::is_convertible_v<utils::OptionalString &&, std::optional<std::string> &&>);
static_assert(!std::is_convertible_v<utils::OptionalString &&, std::optional<std::string> &>);
static_assert(!utils::OptionalString().has_value());

TEST(onnx_string, OptionalString_ComparisonMatrix) {
  const std::vector<std::optional<std::string>> values = {std::nullopt, "", "abc", "abd",
                                                          std::string("abc\0def", 7)};
  for (const auto &left : values) {
    for (const auto &right : values) {
      const utils::OptionalString lhs(left);
      const utils::OptionalString rhs(right);
      EXPECT_EQ(lhs == rhs, left == right);
      EXPECT_EQ(lhs != rhs, left != right);
      EXPECT_EQ(lhs < rhs, left < right);
      EXPECT_EQ(lhs <= rhs, left <= right);
      EXPECT_EQ(lhs > rhs, left > right);
      EXPECT_EQ(lhs >= rhs, left >= right);
      EXPECT_EQ(lhs <=> rhs, left <=> right);
      EXPECT_EQ(lhs == right, left == right);
      EXPECT_EQ(left == rhs, left == right);
      EXPECT_EQ(lhs != right, left != right);
      EXPECT_EQ(left != rhs, left != right);
      EXPECT_EQ(lhs < right, left < right);
      EXPECT_EQ(left < rhs, left < right);
      EXPECT_EQ(lhs <= right, left <= right);
      EXPECT_EQ(left <= rhs, left <= right);
      EXPECT_EQ(lhs > right, left > right);
      EXPECT_EQ(left > rhs, left > right);
      EXPECT_EQ(lhs >= right, left >= right);
      EXPECT_EQ(left >= rhs, left >= right);
    }
  }
}

TEST(onnx_string, OptionalString_HeterogeneousComparisons) {
  const utils::OptionalString absent;
  const utils::OptionalString empty("");
  const utils::OptionalString value("abc");
  const std::string text("abc");
  const char *null = nullptr;
  EXPECT_TRUE(absent == null);
  EXPECT_TRUE(null == absent);
  EXPECT_FALSE(empty == null);
  EXPECT_FALSE(null == empty);
  EXPECT_TRUE(value != null);
  EXPECT_TRUE(null != value);
  EXPECT_FALSE(absent == "");
  EXPECT_FALSE("" == absent);
  EXPECT_TRUE(empty == "");
  EXPECT_TRUE("" == empty);
  EXPECT_TRUE(value == text);
  EXPECT_TRUE(text == value);
  EXPECT_TRUE(value == "abc");
  EXPECT_TRUE("abc" == value);
  EXPECT_TRUE(absent != text);
  EXPECT_TRUE(text != absent);
  EXPECT_TRUE(absent < text);
  EXPECT_TRUE(value <= text);
  EXPECT_FALSE(value > text);
  EXPECT_TRUE(value >= text);
  EXPECT_FALSE(text < absent);
  EXPECT_TRUE(text > absent);
  EXPECT_TRUE(text <= value);
  EXPECT_TRUE(text >= value);
  EXPECT_TRUE("abc" > absent);
  EXPECT_TRUE("abc" <= value);
  EXPECT_TRUE("abc" >= value);
  EXPECT_TRUE(absent == std::nullopt);
  EXPECT_TRUE(std::nullopt == absent);
  EXPECT_TRUE(value != std::nullopt);
  EXPECT_TRUE(std::nullopt < value);
  EXPECT_TRUE(absent <= std::nullopt);
  EXPECT_TRUE(std::nullopt >= absent);
  EXPECT_TRUE(absent <= null);
  EXPECT_TRUE(absent >= null);
  EXPECT_TRUE(value > null);
  EXPECT_FALSE(empty <= null);
}

TEST(onnx_string, OptionalString_OptionalAccess) {
  utils::OptionalString value;
  EXPECT_FALSE(static_cast<bool>(value));
  EXPECT_EQ(value.value_or("fallback"), "fallback");
  EXPECT_EQ(value.value(), "");
  EXPECT_EQ(value.emplace(3, 'x'), "xxx");
  EXPECT_EQ(value.emplace({'x', 'x', 'x'}), "xxx");
  EXPECT_TRUE(static_cast<bool>(value));
  EXPECT_EQ(*value, "xxx");
  EXPECT_EQ(value->size(), 3u);
  const auto &constant = value;
  EXPECT_EQ(*constant, "xxx");
  EXPECT_EQ(constant->size(), 3u);
  const std::optional<std::string> &optional = constant;
  EXPECT_EQ(optional.value(), "xxx");
  std::optional<std::string> &mutable_optional = value;
  mutable_optional = "changed";
  EXPECT_EQ(value, "changed");
  utils::OptionalString absent;
  value.swap(absent);
  EXPECT_FALSE(value.has_value());
  EXPECT_EQ(absent, "changed");
  value = std::nullopt;
  EXPECT_FALSE(value.has_value());
  value = std::optional<std::string>("restored");
  EXPECT_EQ(value, "restored");
  value.reset();
  EXPECT_TRUE(value.null());
  std::optional<std::string> other("swapped");
  value.swap(other);
  EXPECT_EQ(value, "swapped");
  EXPECT_FALSE(other.has_value());
}

TEST(onnx_string, OptionalString_Containers) {
  std::vector<utils::OptionalString> values = {"b", "", std::nullopt, "a", "a"};
  std::sort(values.begin(), values.end());
  EXPECT_FALSE(values[0].has_value());
  EXPECT_TRUE(values[1].has_value());
  EXPECT_EQ(values[1], "");
  EXPECT_EQ(values[2], "a");
  EXPECT_EQ(values[3], "a");
  EXPECT_EQ(values[4], "b");
  const std::unordered_set<utils::OptionalString> unique(values.begin(), values.end());
  EXPECT_EQ(unique.size(), 4u);
  EXPECT_EQ(unique.count(utils::OptionalString()), 1u);
  EXPECT_EQ(unique.count(utils::OptionalString("")), 1u);
}
