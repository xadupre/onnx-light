// Unit tests for the protobuf-compatible message API surface that onnx-light
// exposes on generated proto classes: ParseFromString / SerializeToString now
// return bool (true on success) and ByteSizeLong() reports the serialized size
// without performing a real serialization (it delegates to SerializeSize()).
#include "onnx.h"
#include <gtest/gtest.h>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

// Builds a small AttributeProto carrying an int payload.
AttributeProto MakeIntAttr() {
  AttributeProto attr;
  attr.set_name("i_attr");
  attr.ref_i() = 123456789;
  return attr;
}

} // namespace

TEST(proto_stream_api, SerializeToStringReturnsTrue) {
  AttributeProto attr = MakeIntAttr();
  std::string serialized;
  EXPECT_TRUE(attr.SerializeToString(serialized));
  EXPECT_FALSE(serialized.empty());
}

TEST(proto_stream_api, ParseFromStringReturnsTrue) {
  AttributeProto attr = MakeIntAttr();
  std::string serialized;
  ASSERT_TRUE(attr.SerializeToString(serialized));

  AttributeProto parsed;
  EXPECT_TRUE(parsed.ParseFromString(serialized));
  EXPECT_TRUE(parsed.has_i());
  EXPECT_EQ(parsed.ref_i(), 123456789);
  EXPECT_EQ(parsed.ref_name(), "i_attr");
}

TEST(proto_stream_api, SerializeToStringWithOptionsReturnsTrue) {
  AttributeProto attr = MakeIntAttr();
  SerializeOptions sopts;
  std::string serialized;
  EXPECT_TRUE(attr.SerializeToString(serialized, sopts));

  AttributeProto parsed;
  ParseOptions popts;
  EXPECT_TRUE(parsed.ParseFromString(serialized, popts));
  EXPECT_EQ(parsed.ref_i(), 123456789);
}

TEST(proto_stream_api, ByteSizeLongMatchesSerializedSize) {
  AttributeProto attr = MakeIntAttr();
  std::string serialized;
  ASSERT_TRUE(attr.SerializeToString(serialized));

  // ByteSizeLong() must equal the number of bytes an actual serialization
  // produces, and equal SerializeSize().size(), without serializing itself.
  EXPECT_EQ(attr.ByteSizeLong(), serialized.size());
  EXPECT_EQ(attr.ByteSizeLong(), static_cast<size_t>(attr.SerializeSize().size()));
}

TEST(proto_stream_api, ByteSizeLongOnModelProto) {
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("onnx-light");
  std::string serialized;
  ASSERT_TRUE(model.SerializeToString(serialized));
  EXPECT_EQ(model.ByteSizeLong(), serialized.size());
}
