#include "../common/platform_helpers.h"
#include "../common/tensor.h"
#include "../defs/attr_proto_util.h"
#include "../defs/data_propagators.h"
#include "../defs/data_type_utils.h"
#include "../defs/doc_strings.h"
#include "../defs/parser.h"
#include "../defs/schema.h"
#include "../defs/tensor_util.h"
#include "onnx.h"
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

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

template <typename T> static void ExpectParseFailure(T &result, const char *input) {
  auto status = OnnxParser::Parse(result, input);
  EXPECT_FALSE(status.IsOK());
}

class TestDataPropagationContext final : public DataPropagationContext {
public:
  const AttributeProto *getAttribute(const std::string &name) const override {
    if (name == "axis") {
      return axis_attr_;
    }
    return nullptr;
  }

  size_t getNumInputs() const override { return input_types_.size(); }

  const TypeProto *getInputType(size_t index) const override {
    return (index < input_types_.size()) ? input_types_[index] : nullptr;
  }

  size_t getNumOutputs() const override { return output_types_.size(); }

  const TypeProto *getOutputType(size_t index) const override {
    return (index < output_types_.size()) ? output_types_[index] : nullptr;
  }

  const TensorShapeProto *getInputData(size_t index) override {
    return (index < input_data_.size()) ? input_data_[index] : nullptr;
  }

  void addOutputData(size_t index, TensorShapeProto &&tp) override {
    output_data_[index] = std::move(tp);
  }

  const AttributeProto *axis_attr_{nullptr};
  std::vector<const TypeProto *> input_types_;
  std::vector<const TypeProto *> output_types_;
  std::vector<const TensorShapeProto *> input_data_;
  std::unordered_map<size_t, TensorShapeProto> output_data_;
};

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

TEST(onnx_defs, DataPropagators_GatherAndPropagate) {
  TensorShapeProto input_data;
  input_data.add_dim().set_dim_value(7);
  input_data.add_dim().set_dim_value(8);
  input_data.add_dim().set_dim_value(9);

  TensorShapeProto input_indices;
  input_indices.add_dim().set_dim_value(2);
  input_indices.add_dim().set_dim_value(0);

  TestDataPropagationContext ctx;
  ctx.input_data_ = {&input_data, &input_indices};
  ctx.output_types_.push_back(nullptr);

  GatherOp13DataPropagator(ctx);
  ASSERT_EQ(ctx.output_data_.count(0), 1u);
  ASSERT_EQ(ctx.output_data_.at(0).ref_dim().size(), 2u);
  EXPECT_EQ(ctx.output_data_.at(0).ref_dim()[0].ref_dim_value(), int64_t{9});
  EXPECT_EQ(ctx.output_data_.at(0).ref_dim()[1].ref_dim_value(), int64_t{7});

  TestDataPropagationContext ctx_copy;
  ctx_copy.input_data_ = {&input_data};
  ctx_copy.output_types_.push_back(nullptr);
  PropagateShapeDataFromInputToOutput(ctx_copy, 0);
  ASSERT_EQ(ctx_copy.output_data_.count(0), 1u);
  ASSERT_EQ(ctx_copy.output_data_.at(0).ref_dim().size(), 3u);
  EXPECT_EQ(ctx_copy.output_data_.at(0).ref_dim()[2].ref_dim_value(), int64_t{9});
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
// schema.cc tests
// ===========================================================================

TEST(onnx_defs, Schema_FormalParameter_Getters) {
  DataTypeSet allowed{Utils::DataTypeUtils::ToType("float")};
  OpSchema::FormalParameter parameter("X", allowed, "T", "input", OpSchema::Single, true, 1,
                                      OpSchema::Differentiable);

  EXPECT_EQ(parameter.GetName(), "X");
  EXPECT_EQ(parameter.GetTypeStr(), "T");
  EXPECT_EQ(parameter.GetDescription(), "input");
  EXPECT_EQ(parameter.GetOption(), OpSchema::Single);
  EXPECT_TRUE(parameter.GetIsHomogeneous());
  EXPECT_EQ(parameter.GetMinArity(), 1);
  EXPECT_EQ(parameter.GetDifferentiationCategory(), OpSchema::Differentiable);
  EXPECT_EQ(parameter.GetTypes().size(), 1u);
}

TEST(onnx_defs, Schema_OpSchemaBasicRegistration) {
  const std::string op_name = "UnitTestSchemaAdd";
  DeregisterSchema(op_name, 1, ONNX_DOMAIN);

  OpSchema schema(op_name, __FILE__, __LINE__);
  schema.SinceVersion(1)
      .Input(0, "X", "input", "T")
      .Output(0, "Y", "output", "T")
      .TypeConstraint("T", {"float", "double"}, "type constraint")
      .Attr(std::string("alpha"), std::string("alpha coefficient"), AttributeProto::FLOAT, 1.0f)
      .Finalize();

  RegisterSchema(std::move(schema), 0, true, true);

  const OpSchema *registered = OpSchemaRegistry::Schema(op_name, 1, ONNX_DOMAIN);
  ASSERT_NE(registered, nullptr);
  EXPECT_EQ(registered->Name(), op_name);
  EXPECT_EQ(registered->since_version(), 1);
  EXPECT_EQ(registered->inputs().size(), 1u);
  EXPECT_EQ(registered->outputs().size(), 1u);
  EXPECT_EQ(registered->attributes().count("alpha"), 1u);
  EXPECT_EQ(registered->typeConstraintMap().count("T"), 1u);
}

TEST(onnx_defs, Schema_DomainToVersionRange_CustomDomain) {
  const std::string domain = "com.onnxlight.unittest.schema";
  auto &ranges = OpSchemaRegistry::DomainToVersionRange::Instance();

  if (ranges.Map().count(domain) == 0) {
    ranges.AddDomainToVersion(domain, 1, 3, 2);
  } else {
    ranges.UpdateDomainToVersion(domain, 1, 3, 2);
  }
  ranges.UpdateDomainToVersion(domain, 2, 5, 4);

  ASSERT_EQ(ranges.Map().count(domain), 1u);
  EXPECT_EQ(ranges.Map().at(domain).first, 2);
  EXPECT_EQ(ranges.Map().at(domain).second, 5);
  ASSERT_EQ(ranges.LastReleaseVersionMap().count(domain), 1u);
  EXPECT_EQ(ranges.LastReleaseVersionMap().at(domain), 4);
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

TEST(onnx_defs, Parser_TypeProto_UnspecifiedDim) {
  TypeProto type;
  ParseIt(type, "float[N,?,K]");
  ASSERT_EQ(type.ref_tensor_type().ref_shape().ref_dim().size(), 3u);
  EXPECT_FALSE(type.ref_tensor_type().ref_shape().ref_dim()[1].has_dim_param());
  EXPECT_FALSE(type.ref_tensor_type().ref_shape().ref_dim()[1].has_dim_value());
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

TEST(onnx_defs, Parser_TypeProto_SparseTensor) {
  TypeProto type;
  ParseIt(type, "sparse_tensor(float[1000])");
  EXPECT_TRUE(type.has_sparse_tensor_type());
  EXPECT_EQ(static_cast<int32_t>(type.ref_sparse_tensor_type().ref_elem_type()),
            static_cast<int32_t>(TensorProto::DataType::FLOAT));
  EXPECT_EQ(type.ref_sparse_tensor_type().ref_shape().ref_dim().size(), 1u);
}

TEST(onnx_defs, Parser_TypeProto_QuotedSymbolicDim) {
  TypeProto type;
  ParseIt(type, R"(float["M + N"])");
  ASSERT_EQ(type.ref_tensor_type().ref_shape().ref_dim().size(), 1u);
  EXPECT_TRUE(type.ref_tensor_type().ref_shape().ref_dim()[0].has_dim_param());
  EXPECT_FALSE(type.ref_tensor_type().ref_shape().ref_dim()[0].has_dim_value());
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_param(), "M + N");
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

TEST(onnx_defs, Parser_AttributeProto_Tensor) {
  AttributeProto attr;
  ParseIt(attr, "x = float[3] {2.1, 4.1, 6.1}");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::TENSOR);
  ASSERT_EQ(attr.ref_t().ref_float_data().size(), 3u);
  EXPECT_FLOAT_EQ(attr.ref_t().ref_float_data()[0], 2.1f);
  EXPECT_FLOAT_EQ(attr.ref_t().ref_float_data()[1], 4.1f);
  EXPECT_FLOAT_EQ(attr.ref_t().ref_float_data()[2], 6.1f);
}

TEST(onnx_defs, Parser_AttributeProto_Strings) {
  AttributeProto attr;
  ParseIt(attr, R"(x = ["abc", "def"])");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::STRINGS);
  ASSERT_EQ(attr.ref_strings().size(), 2u);
  EXPECT_EQ(attr.ref_strings()[0], "abc");
  EXPECT_EQ(attr.ref_strings()[1], "def");
}

TEST(onnx_defs, Parser_AttributeProto_RefAttr) {
  AttributeProto attr;
  ParseIt(attr, "x : ints = @xyz");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::INTS);
  EXPECT_EQ(attr.ref_ref_attr_name(), "xyz");
}

TEST(onnx_defs, Parser_AttributeProto_Graph) {
  AttributeProto attr;
  ParseIt(attr, R"ONNX(
body = somegraph (float[N] y, float[N] z) => (float[N] w)
{
    x = foo(y, z)
    w = bar(x, y)
}
)ONNX");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::GRAPH);
  EXPECT_EQ(attr.ref_g().ref_node().size(), 2u);
}

TEST(onnx_defs, Parser_AttributeProto_TypeProto) {
  AttributeProto attr;
  ParseIt(attr, "type = float[3]");
  EXPECT_EQ(attr.ref_type(), AttributeProto::AttributeType::TYPE_PROTO);
  EXPECT_TRUE(attr.ref_tp().has_tensor_type());
  EXPECT_EQ(static_cast<int32_t>(attr.ref_tp().ref_tensor_type().ref_elem_type()),
            static_cast<int32_t>(TensorProto::DataType::FLOAT));
}

TEST(onnx_defs, Parser_AttrList_Basic) {
  AttrList attrs;
  ParseIt(attrs, "<x = 2, w = 3>");
  ASSERT_EQ(attrs.size(), 2u);
  EXPECT_EQ(attrs[0].ref_name(), "x");
  EXPECT_EQ(attrs[1].ref_name(), "w");
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

TEST(onnx_defs, Parser_NodeProto_DomainOpCall) {
  NodeProto n;
  ParseIt(n, "x = somedomain.foo(y, z)");
  EXPECT_EQ(n.ref_domain(), "somedomain");
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

TEST(onnx_defs, Parser_NodeProto_LeadingOptionalInput) {
  NodeProto n;
  ParseIt(n, "x = SomeOp( , z)");
  ASSERT_EQ(n.ref_input().size(), 2u);
  EXPECT_EQ(n.ref_input()[0], "");
  EXPECT_EQ(n.ref_input()[1], "z");
}

TEST(onnx_defs, Parser_NodeProto_QuotedOptionalInput) {
  NodeProto n;
  ParseIt(n, "x = SomeOp(y, \"\", z)");
  ASSERT_EQ(n.ref_input().size(), 3u);
  EXPECT_EQ(n.ref_input()[0], "y");
  EXPECT_EQ(n.ref_input()[1], "");
  EXPECT_EQ(n.ref_input()[2], "z");
}

TEST(onnx_defs, Parser_NodeProto_LeadingQuotedOptionalInput) {
  NodeProto n;
  ParseIt(n, "x = SomeOp(\"\", z)");
  ASSERT_EQ(n.ref_input().size(), 2u);
  EXPECT_EQ(n.ref_input()[0], "");
  EXPECT_EQ(n.ref_input()[1], "z");
}

TEST(onnx_defs, Parser_NodeList_Basic) {
  NodeList nodes;
  ParseIt(nodes, R"ONNX(
{
    x = foo(y, z)
    w = bar(x, y)
}
)ONNX");
  ASSERT_EQ(nodes.size(), 2u);
  EXPECT_EQ(nodes[0].ref_op_type(), "foo");
  EXPECT_EQ(nodes[1].ref_op_type(), "bar");
}

TEST(onnx_defs, Parser_NodeList_WithLabels) {
  NodeList nodes;
  ParseIt(nodes, R"ONNX(
{
    [node1] x = foo(y, z)
    [node2] w = bar(x, y)
    s = foobar(x, w)
}
)ONNX");
  ASSERT_EQ(nodes.size(), 3u);
  EXPECT_EQ(nodes[0].ref_name(), "node1");
  EXPECT_EQ(nodes[1].ref_name(), "node2");
  EXPECT_TRUE(nodes[2].ref_name().empty());
}

TEST(onnx_defs, Parser_NodeProto_ListValuedAttributes) {
  NodeProto n;
  ParseIt(n, R"(x = foo <d = [5, 10], e = [0.55, 0.66], f = ["str1", "str2"]> (y, z))");
  ASSERT_EQ(n.ref_attribute().size(), 3u);
  EXPECT_EQ(n.ref_attribute()[0].ref_type(), AttributeProto::AttributeType::INTS);
  EXPECT_EQ(n.ref_attribute()[1].ref_type(), AttributeProto::AttributeType::FLOATS);
  EXPECT_EQ(n.ref_attribute()[2].ref_type(), AttributeProto::AttributeType::STRINGS);
}

TEST(onnx_defs, Parser_NodeList_ComplexBlock) {
  NodeList nodes;
  ParseIt(nodes, R"ONNX(
{
  sub_result = Sub(limit, start)
  sub_result_casted = Cast<to = 1>(sub_result)
  delta_casted = Cast<to = 1>(delta)
  div_result = Div(sub_result_casted, delta_casted)
  ceil_result = Ceil(div_result)
  ceil_result_relu = Relu(ceil_result)
  ceil_result_relu_int = Cast<to = 7>(ceil_result_relu)
  ceil_result_relu_bool = Cast<to = 9>(ceil_result_relu)
  variadic_output, output = Loop (ceil_result_relu_int, ceil_result_relu_bool, start)
}
)ONNX");
  EXPECT_EQ(nodes.size(), 9u);
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

TEST(onnx_defs, Parser_GraphProto_WithComments) {
  const char *code = R"ONNX(
agraph (float[N] y, float[N] z) => (float[N] w)
<float[2] w1 = {1.0, 2.0}, float[3] w2 = {4.0, 5.0, 6.0}, float[N] x>
{
    # This is a comment.
    x = foo(y, z, w1) # More comments.
    w = bar(x, y, w2)
}
)ONNX";
  GraphProto graph;
  ParseIt(graph, code);
  EXPECT_EQ(graph.ref_node().size(), 2u);
  EXPECT_EQ(graph.ref_initializer().size(), 2u);
}

TEST(onnx_defs, Parser_GraphProto_PartialType) {
  const char *code = R"ONNX(
agraph (float[N] y, z) => (float[N] w)
{
    x = foo(y, z)
    w = bar(x, y)
}
)ONNX";
  GraphProto graph;
  ParseIt(graph, code);
  EXPECT_EQ(graph.ref_input().size(), 2u);
  EXPECT_EQ(graph.ref_output().size(), 1u);
}

TEST(onnx_defs, Parser_GraphProto_InitializerInInput) {
  const char *code = R"ONNX(
agraph (float y = {1.0}, float[N] z) => (float[N] w)
<float[2] w1 = {1.0, 2.0}, float[3] w2 = {4.0, 5.0, 6.0}, float[N] x>
{
    x = foo(y, z, w1)
    w = bar(x, y, w2)
}
)ONNX";
  GraphProto graph;
  ParseIt(graph, code);
  EXPECT_EQ(graph.ref_input().size(), 2u);
  EXPECT_EQ(graph.ref_initializer().size(), 3u);
  EXPECT_EQ(graph.ref_value_info().size(), 1u);
}

TEST(onnx_defs, Parser_NodeProto_IfNodeAttributes) {
  const char *code = R"ONNX(
z = If (b) <
    then_branch = g1 () => (float[N] z_then)
      {
        z_then = foo(y)
      },
    else_branch = g2 () => (float[N] z_else)
      {
        z_else = bar(x)
      }
    >
)ONNX";
  NodeProto node;
  ParseIt(node, code);
  EXPECT_EQ(node.ref_input().size(), 1u);
  EXPECT_EQ(node.ref_output().size(), 1u);
  EXPECT_EQ(node.ref_attribute().size(), 2u);
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

TEST(onnx_defs, Parser_ModelProto_MetadataProps) {
  const char *code = R"ONNX(
<
  ir_version: 7,
  opset_import: [ "ai.onnx.ml" : 10 ],
  metadata_props: [ "somekey" : "somevalue", "key2" : "value2" ]
>
agraph (float[N] y, float[N] z) => (float[N] w)
{
    x = foo(y, z)
    w = bar(x, y)
}
)ONNX";
  ModelProto model;
  ParseIt(model, code);
  ASSERT_EQ(model.ref_metadata_props().size(), 2u);
  EXPECT_EQ(model.ref_metadata_props()[0].ref_key(), "somekey");
  EXPECT_EQ(model.ref_metadata_props()[0].ref_value(), "somevalue");
  EXPECT_EQ(model.ref_metadata_props()[1].ref_key(), "key2");
  EXPECT_EQ(model.ref_metadata_props()[1].ref_value(), "value2");
}

TEST(onnx_defs, Parser_ModelProto_WithFunctions) {
  const char *code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 10, "local" : 1 ]
>
agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
{
  T = local.foo (X, W, B)
  C = local.square(T)
}

<
  opset_import: [ "" : 10 ],
  domain: "local",
  doc_string: "Function foo."
>
foo (x, w, b) => (c) {
  T = MatMul(x, w)
  S = Add(T, b)
  c = Softmax(S)
}

<
  opset_import: [ "" : 10 ],
  domain: "local",
  doc_string: "Function square."
>
square (x) => (y) {
  y = Mul (x, x)
}
)ONNX";
  ModelProto model;
  ParseIt(model, code);
  EXPECT_EQ(model.ref_graph().ref_node().size(), 2u);
  EXPECT_EQ(model.ref_functions().size(), 2u);
  EXPECT_EQ(model.ref_functions()[0].ref_name(), "foo");
  EXPECT_EQ(model.ref_functions()[1].ref_name(), "square");
}

TEST(onnx_defs, Parser_GraphProto_QuotedIdentifiers) {
  const char *code = R"ONNX(
"a graph name" (float[N, 128] "input/X", float[128,10] "input W", float[10] B) => (float[N] C)
{
    "some/temp" = MatMul("input/X", "input W")
    S = Add("some/temp", B)
    C = Softmax(S)
}
)ONNX";
  GraphProto graph;
  ParseIt(graph, code);
  EXPECT_EQ(graph.ref_name(), "a graph name");
  EXPECT_EQ(graph.ref_input().size(), 3u);
  EXPECT_EQ(graph.ref_node().size(), 3u);
  EXPECT_EQ(graph.ref_node()[0].ref_input()[0], "input/X");
  EXPECT_EQ(graph.ref_node()[0].ref_output()[0], "some/temp");
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

TEST(onnx_defs, Parser_TensorProto_InvalidShapeFailures) {
  TensorProto tp;
  // TensorProto parsing requires concrete numeric dimensions in the tensor shape.
  // Unknown-rank and symbolic dimensions are valid in TypeProto, but not in TensorProto literals.
  ExpectParseFailure(tp, "int32[] {1, 2, 3, 4, 5}");
  ExpectParseFailure(tp, "int32[N] {1, 2, 3, 4, 5}");
}

TEST(onnx_defs, Parser_TensorProto_ScientificAndStringLiterals) {
  TensorProto tp;
  ParseIt(tp, "float[5] {1e1, 2.0e-1, 3.1E-1, 4E+1, 5.5e-10}");
  ASSERT_EQ(tp.ref_float_data().size(), 5u);
  EXPECT_FLOAT_EQ(tp.ref_float_data()[0], 10.0f);
  ParseIt(tp, R"(string[2] { "Hello", "World" })");
  ASSERT_EQ(tp.ref_string_data().size(), 2u);
  EXPECT_EQ(tp.ref_string_data()[0], "Hello");
  ParseIt(tp, R"(string[2] { "Use a \"quoted\" word", "Use a backslash \\ like this." })");
  ASSERT_EQ(tp.ref_string_data().size(), 2u);
  EXPECT_EQ(tp.ref_string_data()[1], "Use a backslash \\ like this.");
}

TEST(onnx_defs, Parser_TensorProto_ExternalData) {
  const char *code = R"ONNX(
agraph (float y = {1.0}, float[N] z) => (w) <
    float[3, 2] m1 = ["location": "weight_1.bin", "offset": "17"],
    float[2, 1] m2 = {1.0, 2.0}
>
{
    x = Add(y, z)
    m = Mul(m1, m1)
}
)ONNX";
  GraphProto graph;
  ParseIt(graph, code);
  ASSERT_EQ(graph.ref_initializer().size(), 3u);
  EXPECT_EQ(graph.ref_initializer()[1].ref_data_location(), TensorProto::DataLocation::EXTERNAL);
  ASSERT_EQ(graph.ref_initializer()[1].ref_external_data().size(), 2u);
  EXPECT_EQ(graph.ref_initializer()[1].ref_external_data()[0].ref_key(), "location");
  EXPECT_EQ(graph.ref_initializer()[1].ref_external_data()[0].ref_value(), "weight_1.bin");
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

TEST(onnx_defs, Parser_FunctionProto_ValueInfo) {
  const char *code = R"ONNX(
<
  opset_import: [ "" : 10 ],
  domain: "ai.onnx.ml"
>
f (float[N] y, float[N] z) => (float[N] w)
<float[N] x>
{
    x = Add(y, z)
    w = Mul(x, y)
}
)ONNX";
  FunctionProto fn;
  ParseIt(fn, code);
  ASSERT_EQ(fn.ref_value_info().size(), 4u);
  EXPECT_EQ(fn.ref_value_info()[0].ref_name(), "y");
  EXPECT_EQ(fn.ref_value_info()[1].ref_name(), "z");
  EXPECT_EQ(fn.ref_value_info()[2].ref_name(), "w");
  EXPECT_EQ(fn.ref_value_info()[3].ref_name(), "x");
}

TEST(onnx_defs, Parser_FunctionProto_ValueInfoTypedIO) {
  const char *code = R"ONNX(
<
  opset_import: [ "" : 10 ],
  domain: "ai.onnx.ml"
>
f (float[N] y, float[N] z) => (float[N] w)
{
    x = Add(y, z)
    w = Mul(x, y)
}
)ONNX";
  FunctionProto fn;
  ParseIt(fn, code);
  ASSERT_EQ(fn.ref_value_info().size(), 3u);
  EXPECT_EQ(fn.ref_value_info()[0].ref_name(), "y");
  EXPECT_EQ(fn.ref_value_info()[1].ref_name(), "z");
  EXPECT_EQ(fn.ref_value_info()[2].ref_name(), "w");
  for (const auto &vi : fn.ref_value_info()) {
    EXPECT_NE(vi.ref_name(), "x");
  }
}

TEST(onnx_defs, Parser_FunctionProto_ValueInfoPartialTypedAndLocals) {
  const char *code = R"ONNX(
<
  opset_import: [ "" : 10 ],
  domain: "ai.onnx.ml"
>
f (float[N] y, z) => (float[N] w)
<float[N] x, float[N] t>
{
    x = Add(y, z)
    t = Add(x, x)
    w = Mul(t, y)
}
)ONNX";
  FunctionProto fn;
  ParseIt(fn, code);
  ASSERT_EQ(fn.ref_value_info().size(), 4u);
  EXPECT_EQ(fn.ref_value_info()[0].ref_name(), "y");
  EXPECT_EQ(fn.ref_value_info()[1].ref_name(), "w");
  EXPECT_EQ(fn.ref_value_info()[2].ref_name(), "x");
  EXPECT_EQ(fn.ref_value_info()[3].ref_name(), "t");
}

TEST(onnx_defs, Parser_FunctionProto_QuotedIdentifiers) {
  const char *code = R"ONNX(
<
  opset_import: [ "" : 10 ],
  domain: "ai.onnx.ml",
  doc_string: "A function test case."
>
"a function name" (float[N] "#y", "$z") => (float[N] "!w")
<float[N] "/layer/x", float[N] t>
{
    "/layer/x" = Add("#y", "$z")
    t = Add("/layer/x", "/layer/x")
    "!w" = Mul(t, "#y")
}
)ONNX";
  FunctionProto fn;
  ParseIt(fn, code);
  EXPECT_EQ(fn.ref_name(), "a function name");
  ASSERT_EQ(fn.ref_input().size(), 2u);
  EXPECT_EQ(fn.ref_input()[0], "#y");
  EXPECT_EQ(fn.ref_input()[1], "$z");
  ASSERT_EQ(fn.ref_output().size(), 1u);
  EXPECT_EQ(fn.ref_output()[0], "!w");
  ASSERT_EQ(fn.ref_value_info().size(), 4u);
  EXPECT_EQ(fn.ref_value_info()[0].ref_name(), "#y");
  EXPECT_EQ(fn.ref_value_info()[1].ref_name(), "!w");
  EXPECT_EQ(fn.ref_value_info()[2].ref_name(), "/layer/x");
  EXPECT_EQ(fn.ref_value_info()[3].ref_name(), "t");
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

// ===========================================================================
// doc_strings.cc tests
// ===========================================================================

TEST(onnx_defs, DocStrings_NonEmpty) {
  EXPECT_GT(strlen(kDoc_Relu_ver6), 0u);
  EXPECT_GT(strlen(kDoc_Sigmoid_ver6), 0u);
  EXPECT_GT(strlen(kDoc_Tanh_ver6), 0u);
  EXPECT_GT(strlen(kDoc_GRU_ver14), 0u);
  EXPECT_GT(strlen(kDoc_LSTM_ver14), 0u);
  EXPECT_GT(strlen(kDoc_RNN_ver14), 0u);
  EXPECT_GT(strlen(kDoc_MatMul_ver9), 0u);
  EXPECT_GT(strlen(kDoc_Cast_ver24), 0u);
  EXPECT_GT(strlen(kDoc_Reshape_ver24), 0u);
  EXPECT_GT(strlen(kDoc_Squeeze_ver24), 0u);
  EXPECT_GT(strlen(kDoc_Unsqueeze_ver24), 0u);
  EXPECT_GT(strlen(kDoc_Pad_ver24), 0u);
  EXPECT_GT(strlen(kDoc_Loop_ver23), 0u);
  EXPECT_GT(strlen(kDoc_scan_24), 0u);
  EXPECT_GT(strlen(kDoc_BitCast_ver26), 0u);
}

TEST(onnx_defs, DocStrings_ContainExpectedContent) {
  EXPECT_NE(strstr(kDoc_Relu_ver6, "max(0, x)"), nullptr);
  EXPECT_NE(strstr(kDoc_Sigmoid_ver6, "sigmoid"), nullptr);
  EXPECT_NE(strstr(kDoc_Tanh_ver6, "hyperbolic tangent"), nullptr);
  EXPECT_NE(strstr(kDoc_MatMul_ver9, "matmul"), nullptr);
  EXPECT_NE(strstr(kDoc_GRU_ver14, "GRU"), nullptr);
  EXPECT_NE(strstr(kDoc_LSTM_ver14, "LSTM"), nullptr);
}
