// Unit tests for the write_as_string family of helpers declared in
// onnx_light/onnx_proto/stream_class_print.hpp. They cover the scalar overloads,
// the vector/optional/RepeatedField helpers, and the special hex/quoted
// formatting for byte and string fields.

#include "onnx.h"
#include "stream_class_print.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

utils::PrintOptions make_options() { return utils::PrintOptions(); }

} // namespace

TEST(stream_class_print, WriteAsStringScalar) {
  utils::PrintOptions options = make_options();
  EXPECT_EQ(write_as_string(options, static_cast<int32_t>(42)), "42");
  EXPECT_EQ(write_as_string(options, static_cast<int64_t>(-7)), "-7");
  EXPECT_EQ(write_as_string(options, static_cast<uint64_t>(123)), "123");
  EXPECT_EQ(write_as_string(options, 1.5f), "1.5");
  EXPECT_EQ(write_as_string(options, 2.25), "2.25");
}

TEST(stream_class_print, WriteAsStringStringQuoted) {
  utils::PrintOptions options = make_options();
  utils::String value("hello", 5);
  EXPECT_EQ(write_as_string(options, value), "\"hello\"");

  utils::String empty;
  EXPECT_EQ(write_as_string(options, empty), "\"\"");
}

TEST(stream_class_print, WriteAsStringVectorUint8AsHex) {
  utils::PrintOptions options = make_options();
  std::vector<uint8_t> bytes{0x00, 0x0F, 0xAB, 0xFF};
  EXPECT_EQ(write_as_string(options, bytes), "000FABFF");

  std::vector<uint8_t> empty;
  EXPECT_EQ(write_as_string(options, empty), "");
}

TEST(stream_class_print, WriteAsStringByteSpanAsHex) {
  utils::PrintOptions options = make_options();
  std::vector<uint8_t> bytes{0x01, 0x23, 0x45};
  utils::ByteSpan span(bytes);
  EXPECT_EQ(write_as_string(options, span), "012345");

  std::vector<uint8_t> empty;
  utils::ByteSpan empty_span(empty);
  EXPECT_EQ(write_as_string(options, empty_span), "");
}

TEST(stream_class_print, WriteAsStringVectorHelper) {
  utils::PrintOptions options = make_options();
  std::vector<int32_t> values{1, 2, 3};
  EXPECT_EQ(write_as_string_vector(options, values), "[1, 2, 3]");

  std::vector<int32_t> single{9};
  EXPECT_EQ(write_as_string_vector(options, single), "[9]");

  std::vector<int32_t> empty;
  EXPECT_EQ(write_as_string_vector(options, empty), "[]");
}

TEST(stream_class_print, WriteAsStringVectorOverloads) {
  utils::PrintOptions options = make_options();
  EXPECT_EQ(write_as_string(options, std::vector<float>{1.5f, 2.5f}), "[1.5, 2.5]");
  EXPECT_EQ(write_as_string(options, std::vector<int64_t>{-1, 0, 1}), "[-1, 0, 1]");
  EXPECT_EQ(write_as_string(options, std::vector<uint64_t>{4, 5}), "[4, 5]");
  EXPECT_EQ(write_as_string(options, std::vector<double>{0.25}), "[0.25]");
  EXPECT_EQ(write_as_string(options, std::vector<int32_t>{}), "[]");
}

TEST(stream_class_print, WriteAsStringOptionalHelper) {
  utils::PrintOptions options = make_options();
  std::optional<int64_t> value = 5;
  EXPECT_EQ(write_as_string_optional(options, value), "5");

  std::optional<int64_t> empty;
  EXPECT_EQ(write_as_string_optional(options, empty), "null");
}

TEST(stream_class_print, WriteAsStringOptionalOverloads) {
  utils::PrintOptions options = make_options();
  EXPECT_EQ(write_as_string(options, std::optional<float>(1.5f)), "1.5");
  EXPECT_EQ(write_as_string(options, std::optional<float>()), "null");
  EXPECT_EQ(write_as_string(options, std::optional<int64_t>(-3)), "-3");
  EXPECT_EQ(write_as_string(options, std::optional<int64_t>()), "null");
  EXPECT_EQ(write_as_string(options, std::optional<uint64_t>(7)), "7");
  EXPECT_EQ(write_as_string(options, std::optional<uint64_t>()), "null");
  EXPECT_EQ(write_as_string(options, std::optional<double>(2.5)), "2.5");
  EXPECT_EQ(write_as_string(options, std::optional<double>()), "null");
  EXPECT_EQ(write_as_string(options, std::optional<int32_t>(11)), "11");
  EXPECT_EQ(write_as_string(options, std::optional<int32_t>()), "null");
}

TEST(stream_class_print, WriteAsRepeatedFieldHelper) {
  utils::PrintOptions options = make_options();
  utils::RepeatedField<int64_t> values;
  values.push_back(1);
  values.push_back(2);
  EXPECT_EQ(write_as_repeated_field(options, values), "[1, 2]");

  utils::RepeatedField<int64_t> empty;
  EXPECT_EQ(write_as_repeated_field(options, empty), "[]");
}

TEST(stream_class_print, WriteAsStringRepeatedFieldOverloads) {
  utils::PrintOptions options = make_options();

  utils::RepeatedField<float> floats;
  floats.push_back(1.5f);
  floats.push_back(2.5f);
  EXPECT_EQ(write_as_string(options, floats), "[1.5, 2.5]");

  utils::RepeatedField<int64_t> ints;
  ints.push_back(-1);
  ints.push_back(3);
  EXPECT_EQ(write_as_string(options, ints), "[-1, 3]");

  utils::RepeatedField<utils::String> strings;
  strings.push_back(utils::String("aa", 2));
  strings.push_back(utils::String("bb", 2));
  EXPECT_EQ(write_as_string(options, strings), "[\"aa\", \"bb\"]");
}
