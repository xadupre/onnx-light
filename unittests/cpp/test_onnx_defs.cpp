#include "../defs/attr_proto_util.h"
#include "../defs/data_type_utils.h"
#include "../defs/parser.h"
#include "../defs/tensor_util.h"
#include "onnx.h"
#include <gtest/gtest.h>
#include <type_traits>

using namespace onnx;

TEST(onnx_defs, AttrProtoUtilAndTensorUtilSignatures) {
  EXPECT_TRUE(
      (std::is_same<decltype(MakeAttribute(std::string("a"), 1.0f)), AttributeProto>::value));
  EXPECT_TRUE(
      (std::is_same<decltype(MakeAttribute(std::string("a"), int64_t{1})), AttributeProto>::value));
  EXPECT_TRUE((
      std::is_same<decltype(MakeRefAttribute(std::string("a"), AttributeProto::AttributeType::INT)),
                   AttributeProto>::value));
  EXPECT_TRUE((std::is_same<decltype(MakeRefAttribute(std::string("a"), std::string("b"),
                                                      AttributeProto::AttributeType::INT)),
                            AttributeProto>::value));
  EXPECT_TRUE((std::is_same<decltype(ParseData<float>(std::declval<const Tensor *>())),
                            std::vector<float>>::value));
}

TEST(onnx_defs, DataTypeAndParserMaps) {
  EXPECT_TRUE((std::is_same<DataType, const std::string *>::value));
  EXPECT_EQ(PrimitiveTypeNameMap::Lookup("float"),
            static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_EQ(AttributeTypeNameMap::Lookup("tensor"),
            static_cast<int32_t>(AttributeProto::AttributeType::TENSOR));
  EXPECT_TRUE(PrimitiveTypeNameMap::IsTypeName("int64"));
  EXPECT_FALSE(PrimitiveTypeNameMap::IsTypeName("not_a_type"));
}
