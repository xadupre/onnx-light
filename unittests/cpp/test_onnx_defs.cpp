#include "../common/platform_helpers.h"
#include "../common/tensor.h"
#include "../defs/attr_proto_util.h"
#include "../defs/data_type_utils.h"
#include "../defs/parser.h"
#include "../defs/tensor_util.h"
#include "onnx.h"
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>

using namespace ONNX_LIGHT_NAMESPACE;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Parse a single ONNX-text-format entity and assert the parse succeeded and
// the entire input was consumed.
template <typename T> static void ParseIt(T &parsedData, const char *input) {
  OnnxParser parser(input);
  auto status = parser.Parse(parsedData);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";
}

} // namespace

// ===========================================================================
// attr_proto_util.cc tests
// ===========================================================================

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

TEST(onnx_defs, MakeAttribute_Float) {
  auto attr = MakeAttribute(std::string("alpha"), 0.5f);
  EXPECT_EQ(attr.ref_name(), "alpha");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::FLOAT);
  EXPECT_FLOAT_EQ(attr.ref_f(), 0.5f);
}

TEST(onnx_defs, MakeAttribute_Int64) {
  auto attr = MakeAttribute(std::string("axis"), int64_t{3});
  EXPECT_EQ(attr.ref_name(), "axis");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::INT);
  EXPECT_EQ(attr.ref_i(), int64_t{3});
}

TEST(onnx_defs, MakeAttribute_Int) {
  auto attr = MakeAttribute(std::string("mode"), int{7});
  EXPECT_EQ(attr.ref_name(), "mode");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::INT);
  EXPECT_EQ(attr.ref_i(), int64_t{7});
}

TEST(onnx_defs, MakeAttribute_String) {
  auto attr = MakeAttribute(std::string("direction"), std::string("FORWARD"));
  EXPECT_EQ(attr.ref_name(), "direction");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::STRING);
  EXPECT_EQ(attr.ref_s(), "FORWARD");
}

TEST(onnx_defs, MakeAttribute_VectorFloat) {
  std::vector<float> vals{1.0f, 2.0f, 3.0f};
  auto attr = MakeAttribute(std::string("weights"), vals);
  EXPECT_EQ(attr.ref_name(), "weights");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::FLOATS);
  ASSERT_EQ(attr.ref_floats().size(), 3u);
  EXPECT_FLOAT_EQ(attr.ref_floats()[0], 1.0f);
  EXPECT_FLOAT_EQ(attr.ref_floats()[2], 3.0f);
}

TEST(onnx_defs, MakeAttribute_VectorInt64) {
  std::vector<int64_t> vals{10, 20, 30};
  auto attr = MakeAttribute(std::string("pads"), vals);
  EXPECT_EQ(attr.ref_name(), "pads");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::INTS);
  ASSERT_EQ(attr.ref_ints().size(), 3u);
  EXPECT_EQ(attr.ref_ints()[1], int64_t{20});
}

TEST(onnx_defs, MakeAttribute_VectorString) {
  std::vector<std::string> vals{"a", "b", "c"};
  auto attr = MakeAttribute(std::string("keys"), vals);
  EXPECT_EQ(attr.ref_name(), "keys");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::STRINGS);
  ASSERT_EQ(attr.ref_strings().size(), 3u);
  EXPECT_EQ(attr.ref_strings()[0], "a");
  EXPECT_EQ(attr.ref_strings()[2], "c");
}

TEST(onnx_defs, MakeRefAttribute_SameName) {
  auto attr = MakeRefAttribute(std::string("alpha"), AttributeProto::AttributeType::FLOAT);
  EXPECT_EQ(attr.ref_name(), "alpha");
  EXPECT_EQ(attr.ref_ref_attr_name(), "alpha");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::FLOAT);
}

TEST(onnx_defs, MakeRefAttribute_DifferentName) {
  auto attr = MakeRefAttribute(std::string("local_alpha"), std::string("parent_alpha"),
                               AttributeProto::AttributeType::INT);
  EXPECT_EQ(attr.ref_name(), "local_alpha");
  EXPECT_EQ(attr.ref_ref_attr_name(), "parent_alpha");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::INT);
}

TEST(onnx_defs, MakeAttribute_TensorProto) {
  TensorProto tp;
  tp.set_name("w");
  tp.set_data_type(TensorProto::DataType::FLOAT);
  tp.ref_dims().push_back(2);
  tp.ref_float_data().push_back(1.0f);
  tp.ref_float_data().push_back(2.0f);
  auto attr = MakeAttribute(std::string("value"), tp);
  EXPECT_EQ(attr.ref_name(), "value");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::TENSOR);
  EXPECT_EQ(attr.ref_t().ref_name(), "w");
}

// ===========================================================================
// data_type_utils.cc tests
// ===========================================================================

TEST(onnx_defs, DataTypeAndParserMaps) {
  EXPECT_TRUE((std::is_same<DataType, const std::string *>::value));
  EXPECT_EQ(PrimitiveTypeNameMap::Lookup("float"),
            static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_EQ(AttributeTypeNameMap::Lookup("tensor"),
            static_cast<int32_t>(AttributeProto::AttributeType::TENSOR));
  EXPECT_TRUE(PrimitiveTypeNameMap::IsTypeName("int64"));
  EXPECT_FALSE(PrimitiveTypeNameMap::IsTypeName("not_a_type"));
}

TEST(onnx_defs, DataTypeUtils_ToDataTypeString) {
  EXPECT_EQ(
      Utils::DataTypeUtils::ToDataTypeString(static_cast<int32_t>(TensorProto::DataType::FLOAT)),
      "float");
  EXPECT_EQ(
      Utils::DataTypeUtils::ToDataTypeString(static_cast<int32_t>(TensorProto::DataType::INT64)),
      "int64");
  EXPECT_EQ(
      Utils::DataTypeUtils::ToDataTypeString(static_cast<int32_t>(TensorProto::DataType::DOUBLE)),
      "double");
  EXPECT_EQ(
      Utils::DataTypeUtils::ToDataTypeString(static_cast<int32_t>(TensorProto::DataType::BOOL)),
      "bool");
  // Invalid type should throw
  EXPECT_THROW(Utils::DataTypeUtils::ToDataTypeString(9999), std::invalid_argument);
}

TEST(onnx_defs, DataTypeUtils_ToType_String) {
  // "float" scalar maps to "tensor(float)" in the canonical form used by DataTypeUtils
  DataType dt = Utils::DataTypeUtils::ToType("float");
  ASSERT_NE(dt, nullptr);
  EXPECT_EQ(*dt, "tensor(float)");

  DataType dt64 = Utils::DataTypeUtils::ToType("int64");
  ASSERT_NE(dt64, nullptr);
  EXPECT_EQ(*dt64, "tensor(int64)");

  // The same DataType pointer is returned for equivalent inputs
  DataType dt2 = Utils::DataTypeUtils::ToType("float");
  EXPECT_EQ(dt, dt2); // pointer equality — same singleton entry
}

TEST(onnx_defs, DataTypeUtils_ToType_TensorProto) {
  TypeProto tp;
  tp.ref_tensor_type().set_elem_type(static_cast<int32_t>(TensorProto::DataType::FLOAT));
  tp.ref_tensor_type().ref_shape(); // create zero-dim shape (scalar)
  DataType dt = Utils::DataTypeUtils::ToType(tp);
  ASSERT_NE(dt, nullptr);
  // The string representation of a scalar float is "tensor(float)"
  EXPECT_EQ(*dt, "tensor(float)");
}

TEST(onnx_defs, DataTypeUtils_ToTypeProto) {
  DataType dt = Utils::DataTypeUtils::ToType("float");
  const TypeProto &proto = Utils::DataTypeUtils::ToTypeProto(dt);
  EXPECT_TRUE(proto.has_tensor_type());
  EXPECT_EQ(static_cast<int32_t>(proto.ref_tensor_type().ref_elem_type()),
            static_cast<int32_t>(TensorProto::DataType::FLOAT));
}

// ===========================================================================
// parser.cc tests
// ===========================================================================

TEST(onnx_defs, Parser_TypeProto_Scalar) {
  TypeProto type;
  ParseIt(type, "float");
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(static_cast<int32_t>(type.ref_tensor_type().ref_elem_type()),
            static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_TRUE(type.ref_tensor_type().has_shape());
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim().size(), 0u);
}

TEST(onnx_defs, Parser_TypeProto_1D_Symbolic) {
  TypeProto type;
  ParseIt(type, "float[N]");
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_TRUE(type.ref_tensor_type().has_shape());
  ASSERT_EQ(type.ref_tensor_type().ref_shape().ref_dim().size(), 1u);
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_param(), "N");
}

TEST(onnx_defs, Parser_TypeProto_1D_Value) {
  TypeProto type;
  ParseIt(type, "int64[3]");
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(static_cast<int32_t>(type.ref_tensor_type().ref_elem_type()),
            static_cast<int32_t>(TensorProto::DataType::INT64));
  ASSERT_EQ(type.ref_tensor_type().ref_shape().ref_dim().size(), 1u);
  EXPECT_TRUE(type.ref_tensor_type().ref_shape().ref_dim()[0].has_dim_value());
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), int64_t{3});
}

TEST(onnx_defs, Parser_TypeProto_UnknownRank) {
  TypeProto type;
  ParseIt(type, "float[]");
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_FALSE(type.ref_tensor_type().has_shape());
}

TEST(onnx_defs, Parser_TypeProto_MultiDim) {
  TypeProto type;
  ParseIt(type, "float[N,M,K]");
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim().size(), 3u);
}

TEST(onnx_defs, Parser_TypeProto_Sequence) {
  TypeProto type;
  ParseIt(type, "seq(float[])");
  EXPECT_TRUE(type.has_sequence_type());
  const auto &elem = type.ref_sequence_type().ref_elem_type();
  EXPECT_TRUE(elem.has_tensor_type());
  EXPECT_EQ(static_cast<int32_t>(elem.ref_tensor_type().ref_elem_type()),
            static_cast<int32_t>(TensorProto::DataType::FLOAT));
}

TEST(onnx_defs, Parser_TypeProto_Optional) {
  TypeProto type;
  ParseIt(type, "optional(float)");
  EXPECT_TRUE(type.has_optional_type());
  const auto &elem = type.ref_optional_type().ref_elem_type();
  EXPECT_TRUE(elem.has_tensor_type());
}

TEST(onnx_defs, Parser_TypeProto_Map) {
  TypeProto type;
  ParseIt(type, "map(int32, float[N])");
  EXPECT_TRUE(type.has_map_type());
  EXPECT_EQ(type.ref_map_type().ref_key_type(), static_cast<int32_t>(TensorProto::DataType::INT32));
  const auto &valtype = type.ref_map_type().ref_value_type();
  EXPECT_TRUE(valtype.has_tensor_type());
  EXPECT_EQ(static_cast<int32_t>(valtype.ref_tensor_type().ref_elem_type()),
            static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_EQ(valtype.ref_tensor_type().ref_shape().ref_dim().size(), 1u);
}

TEST(onnx_defs, Parser_AttributeProto_Int) {
  AttributeProto attr;
  ParseIt(attr, "x = 2");
  EXPECT_EQ(attr.ref_name(), "x");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::INT);
  EXPECT_EQ(attr.ref_i(), int64_t{2});
}

TEST(onnx_defs, Parser_AttributeProto_Float) {
  AttributeProto attr;
  ParseIt(attr, "x = 0.625");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::FLOAT);
  EXPECT_FLOAT_EQ(attr.ref_f(), 0.625f);
}

TEST(onnx_defs, Parser_AttributeProto_String) {
  AttributeProto attr;
  ParseIt(attr, R"(x = "astring")");
  EXPECT_EQ(attr.ref_name(), "x");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::STRING);
  EXPECT_EQ(attr.ref_s(), "astring");
}

TEST(onnx_defs, Parser_AttributeProto_Ints) {
  AttributeProto attr;
  ParseIt(attr, "x = [2, 4, 6]");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::INTS);
  ASSERT_EQ(attr.ref_ints().size(), 3u);
  EXPECT_EQ(attr.ref_ints()[0], int64_t{2});
  EXPECT_EQ(attr.ref_ints()[2], int64_t{6});
}

TEST(onnx_defs, Parser_AttributeProto_Floats) {
  AttributeProto attr;
  ParseIt(attr, "x = [0.125, 0.625]");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::FLOATS);
  ASSERT_EQ(attr.ref_floats().size(), 2u);
  EXPECT_FLOAT_EQ(attr.ref_floats()[0], 0.125f);
}

TEST(onnx_defs, Parser_AttributeProto_TypeAnnotatedEmptyInts) {
  AttributeProto attr;
  ParseIt(attr, "x : ints = []");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::INTS);
  EXPECT_EQ(attr.ref_ints().size(), 0u);
}

TEST(onnx_defs, Parser_NodeProto_Basic) {
  NodeProto n;
  ParseIt(n, "x = foo(y, z)");
  ASSERT_EQ(n.ref_input().size(), 2u);
  EXPECT_EQ(n.ref_input()[0], "y");
  EXPECT_EQ(n.ref_input()[1], "z");
  ASSERT_EQ(n.ref_output().size(), 1u);
  EXPECT_EQ(n.ref_output()[0], "x");
  EXPECT_EQ(n.ref_op_type(), "foo");
}

TEST(onnx_defs, Parser_NodeProto_QualifiedDomain) {
  NodeProto n;
  ParseIt(n, "x = com.example.foo(y, z)");
  EXPECT_EQ(n.ref_domain(), "com.example");
  EXPECT_EQ(n.ref_op_type(), "foo");
}

TEST(onnx_defs, Parser_NodeProto_WithLabel) {
  NodeProto n;
  ParseIt(n, "[node1] x = foo(y, z)");
  EXPECT_EQ(n.ref_name(), "node1");
}

TEST(onnx_defs, Parser_NodeProto_WithAttributes) {
  NodeProto n;
  ParseIt(n, R"(x = foo <a = 100, b = 200.5, c = "astring"> (y, z))");
  ASSERT_EQ(n.ref_attribute().size(), 3u);
  EXPECT_EQ(n.ref_attribute()[0].ref_name(), "a");
  EXPECT_EQ(n.ref_attribute()[1].ref_name(), "b");
  EXPECT_EQ(n.ref_attribute()[2].ref_name(), "c");
}

TEST(onnx_defs, Parser_NodeProto_OptionalInput) {
  NodeProto n;
  ParseIt(n, "x = SomeOp(y, , z)");
  ASSERT_EQ(n.ref_input().size(), 3u);
  EXPECT_EQ(n.ref_input()[0], "y");
  EXPECT_EQ(n.ref_input()[1], "");
  EXPECT_EQ(n.ref_input()[2], "z");
}

TEST(onnx_defs, Parser_GraphProto_Basic) {
  const char *code = R"ONNX(
agraph (float[N] y, float[N] z) => (float[N] w)
{
    x = foo(y, z)
    w = bar(x, y)
}
)ONNX";
  GraphProto graph;
  ParseIt(graph, code);
  EXPECT_EQ(graph.ref_name(), "agraph");
  EXPECT_EQ(graph.ref_input().size(), 2u);
  EXPECT_EQ(graph.ref_output().size(), 1u);
  EXPECT_EQ(graph.ref_node().size(), 2u);
  EXPECT_EQ(graph.ref_node()[0].ref_op_type(), "foo");
  EXPECT_EQ(graph.ref_node()[1].ref_op_type(), "bar");
}

TEST(onnx_defs, Parser_GraphProto_WithInitializers) {
  const char *code = R"ONNX(
agraph (float[N] y, float[N] z) => (float[N] w)
<float[2] w1 = {1.0, 2.0}, float[3] w2 = {4.0, 5.0, 6.0}, float[N] x>
{
    x = foo(y, z, w1)
    w = bar(x, y, w2)
}
)ONNX";
  GraphProto graph;
  ParseIt(graph, code);
  EXPECT_EQ(graph.ref_name(), "agraph");
  EXPECT_EQ(graph.ref_input().size(), 2u);
  EXPECT_EQ(graph.ref_output().size(), 1u);
  EXPECT_EQ(graph.ref_node().size(), 2u);
  EXPECT_EQ(graph.ref_initializer().size(), 2u);
  EXPECT_EQ(graph.ref_value_info().size(), 1u);
}

TEST(onnx_defs, Parser_ModelProto_Basic) {
  const char *code = R"ONNX(
<
  ir_version: 7,
  opset_import: [ "ai.onnx.ml" : 10 ],
  producer_name: "ParserTest",
  producer_version: "1.0",
  domain: "ai.onnx.ml",
  model_version: 1,
  doc_string: "A parser test case model."
>
agraph (float[N] y, float[N] z) => (float[N] w)
{
    x = foo(y, z)
    w = bar(x, y)
}
)ONNX";
  ModelProto model;
  ParseIt(model, code);
  EXPECT_EQ(model.ref_graph().ref_input().size(), 2u);
  EXPECT_EQ(model.ref_graph().ref_output().size(), 1u);
  EXPECT_EQ(model.ref_graph().ref_node().size(), 2u);
  EXPECT_EQ(model.ref_opset_import().size(), 1u);
  EXPECT_EQ(model.ref_producer_name(), "ParserTest");
}

TEST(onnx_defs, Parser_TensorProto_Int32) {
  TensorProto tp;
  ParseIt(tp, "int32[5] {1, 2, 3, 4, 5}");
  EXPECT_EQ(static_cast<int32_t>(tp.ref_data_type()),
            static_cast<int32_t>(TensorProto::DataType::INT32));
  ASSERT_EQ(tp.ref_dims().size(), 1u);
  EXPECT_EQ(tp.ref_dims()[0], uint64_t{5});
  EXPECT_EQ(tp.ref_int32_data().size(), 5u);
}

TEST(onnx_defs, Parser_TensorProto_Float) {
  TensorProto tp;
  ParseIt(tp, "float[3] {1.0, 2.0, 3.0}");
  EXPECT_EQ(static_cast<int32_t>(tp.ref_data_type()),
            static_cast<int32_t>(TensorProto::DataType::FLOAT));
  ASSERT_EQ(tp.ref_float_data().size(), 3u);
  EXPECT_FLOAT_EQ(tp.ref_float_data()[0], 1.0f);
}

TEST(onnx_defs, Parser_TensorProto_Named) {
  TensorProto tp;
  ParseIt(tp, "int32[5] T {1, 2, 3, 4, 5}");
  EXPECT_EQ(tp.ref_name(), "T");
}

TEST(onnx_defs, Parser_EscapeStringLiteral) {
  OnnxParser parser(R"("123\"56\\89")");
  std::string s;
  auto status = parser.ParserBase::Parse(s);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";
  EXPECT_EQ(s, std::string("123\"56\\89"));
}

TEST(onnx_defs, Parser_FunctionProto_Basic) {
  const char *code = R"ONNX(
<
  opset_import: [ "" : 10 ],
  domain: "ai.onnx.ml"
>
f (y, z) => (w)
{
    x = Add(y, z)
    w = Mul(x, y)
}
)ONNX";
  FunctionProto fn;
  ParseIt(fn, code);
  EXPECT_EQ(fn.ref_name(), "f");
  EXPECT_EQ(fn.ref_input().size(), 2u);
  EXPECT_EQ(fn.ref_output().size(), 1u);
  EXPECT_EQ(fn.ref_node().size(), 2u);
  EXPECT_EQ(fn.ref_opset_import().size(), 1u);
}

// ===========================================================================
// tensor_util.cc tests
// ===========================================================================

TEST(onnx_defs, ParseData_Float_FromField) {
  Tensor tensor;
  tensor.floats() = {1.0f, 2.0f, 3.0f};

  auto data = ParseData<float>(&tensor);
  ASSERT_EQ(data.size(), 3u);
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[1], 2.0f);
  EXPECT_FLOAT_EQ(data[2], 3.0f);
}

TEST(onnx_defs, ParseData_Float_FromRawData) {
  float raw_vals[3] = {1.5f, 2.5f, 3.5f};
  std::string raw_str(reinterpret_cast<const char *>(raw_vals), 3 * sizeof(float));

  Tensor tensor;
  tensor.set_raw_data(raw_str);

  auto data = ParseData<float>(&tensor);
  ASSERT_EQ(data.size(), 3u);
  EXPECT_FLOAT_EQ(data[0], 1.5f);
  EXPECT_FLOAT_EQ(data[1], 2.5f);
  EXPECT_FLOAT_EQ(data[2], 3.5f);
}

TEST(onnx_defs, ParseData_Int32_FromField) {
  Tensor tensor;
  tensor.int32s() = {10, 20, 30};

  auto data = ParseData<int32_t>(&tensor);
  ASSERT_EQ(data.size(), 3u);
  EXPECT_EQ(data[0], 10);
  EXPECT_EQ(data[1], 20);
  EXPECT_EQ(data[2], 30);
}

TEST(onnx_defs, ParseData_Int64_FromField) {
  Tensor tensor;
  tensor.int64s() = {100LL, 200LL, 300LL};

  auto data = ParseData<int64_t>(&tensor);
  ASSERT_EQ(data.size(), 3u);
  EXPECT_EQ(data[0], 100LL);
  EXPECT_EQ(data[2], 300LL);
}

TEST(onnx_defs, ParseData_Double_FromField) {
  Tensor tensor;
  tensor.doubles() = {1.1, 2.2};

  auto data = ParseData<double>(&tensor);
  ASSERT_EQ(data.size(), 2u);
  EXPECT_DOUBLE_EQ(data[0], 1.1);
  EXPECT_DOUBLE_EQ(data[1], 2.2);
}

TEST(onnx_defs, ParseData_Int64_FromRawData) {
  int64_t raw_vals[2] = {42LL, 99LL};
  std::string raw_str(reinterpret_cast<const char *>(raw_vals), 2 * sizeof(int64_t));

  Tensor tensor;
  tensor.set_raw_data(raw_str);

  auto data = ParseData<int64_t>(&tensor);
  ASSERT_EQ(data.size(), 2u);
  EXPECT_EQ(data[0], 42LL);
  EXPECT_EQ(data[1], 99LL);
}
