#include "onnx_lib/common/proto_utils.h"
#include "onnx_lib/common/string_utils.h"
#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

TEST(onnx_compatibility, StringUtilsMakeString) {
  EXPECT_EQ(ONNX_LIGHT_NAMESPACE::MakeString("ab", 3, 'c'), "ab3c");
  EXPECT_EQ(ONNX_LIGHT_NAMESPACE::MakeString(std::string("xyz")), "xyz");
}

TEST(onnx_compatibility, ProtoUtilsParseAndRetrieve) {
  ModelProto model;
  GraphProto *graph = model.add_graph();
  graph->set_name("g1");
  std::string serialized;
  model.SerializeToString(serialized);

  ModelProto parsed;
  EXPECT_TRUE(
      ONNX_LIGHT_NAMESPACE::ParseProtoFromBytes(&parsed, serialized.data(), serialized.size()));
  EXPECT_TRUE(parsed.has_graph());
  EXPECT_EQ(parsed.ref_graph().ref_name(), "g1");
  EXPECT_FALSE(ONNX_LIGHT_NAMESPACE::ParseProtoFromBytes<ModelProto>(nullptr, serialized.data(),
                                                                     serialized.size()));

  ModelProto unchanged;
  unchanged.mutable_graph()->set_name("keep");
  ASSERT_GT(serialized.size(), 1u);
  EXPECT_FALSE(ONNX_LIGHT_NAMESPACE::ParseProtoFromBytes(&unchanged, serialized.data(),
                                                         serialized.size() - 1));
  ASSERT_TRUE(unchanged.has_graph());
  EXPECT_EQ(unchanged.ref_graph().ref_name(), "keep");

  for (const std::string &trailer : {std::string("\0\xff", 2), std::string("\x0c\xff", 2)}) {
    const std::string invalid = serialized + trailer;
    SCOPED_TRACE(static_cast<int>(static_cast<unsigned char>(trailer[0])));
    EXPECT_FALSE(
        ONNX_LIGHT_NAMESPACE::ParseProtoFromBytes(&unchanged, invalid.data(), invalid.size()));
    EXPECT_EQ(unchanged.ref_graph().ref_name(), "keep");
    ModelProto array_parsed;
    EXPECT_FALSE(array_parsed.ParseFromArray(invalid.data(), static_cast<int>(invalid.size())));
  }
  EXPECT_FALSE(
      ONNX_LIGHT_NAMESPACE::ParseProtoFromBytes(&unchanged, serialized.data(), (2048ULL << 20)));
  EXPECT_EQ(unchanged.ref_graph().ref_name(), "keep");

  AttributeProto attr;
  attr.set_name("attr");
  attr.ref_ints().push_back(3);
  attr.ref_ints().push_back(4);
  attr.ref_floats().push_back(1.5f);
  attr.ref_strings().push_back(utils::String("aa", 2));
  attr.ref_strings().push_back(utils::String("bb", 2));

  EXPECT_EQ((ONNX_LIGHT_NAMESPACE::RetrieveValues<int64_t>(attr)), (std::vector<int64_t>{3, 4}));
  EXPECT_EQ((ONNX_LIGHT_NAMESPACE::RetrieveValues<float>(attr)), (std::vector<float>{1.5f}));
  EXPECT_EQ((ONNX_LIGHT_NAMESPACE::RetrieveValues<std::string>(attr)),
            (std::vector<std::string>{"aa", "bb"}));

  const std::string debug = ONNX_LIGHT_NAMESPACE::ProtoDebugString(attr);
  EXPECT_FALSE(debug.empty());
}
