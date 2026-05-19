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
#include <type_traits>
#include <unordered_map>

using namespace ONNX_LIGHT_NAMESPACE;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

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
  input_data.add_dim()->set_dim_value(7);
  input_data.add_dim()->set_dim_value(8);
  input_data.add_dim()->set_dim_value(9);

  TensorShapeProto input_indices;
  input_indices.add_dim()->set_dim_value(2);
  input_indices.add_dim()->set_dim_value(0);

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

TEST(onnx_defs, MathOpDataPropagator_InvalidBroadcastRank) {
  const OpSchema *add_schema = OpSchemaRegistry::Schema("Add", 14, ONNX_DOMAIN);
  ASSERT_NE(add_schema, nullptr);

  TensorShapeProto lhs_data;
  lhs_data.add_dim()->set_dim_value(1);
  lhs_data.add_dim()->set_dim_value(2);
  TensorShapeProto rhs_data;
  rhs_data.add_dim()->set_dim_value(1);
  rhs_data.add_dim()->set_dim_value(2);
  rhs_data.add_dim()->set_dim_value(3);

  TestDataPropagationContext ctx;
  ctx.input_data_ = {&lhs_data, &rhs_data};
  ctx.output_types_.push_back(nullptr);

  EXPECT_THROW(add_schema->GetDataPropagationFunction()(ctx), InferenceError);
  EXPECT_TRUE(ctx.output_data_.empty());
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

TEST(onnx_defs, DataTypeUtils_ToType_SequenceOfMap) {
  DataType dt = Utils::DataTypeUtils::ToType("seq(map(string, float))");
  ASSERT_NE(dt, nullptr);
  EXPECT_EQ(*dt, "seq(map(string,tensor(float)))");

  const TypeProto &proto = Utils::DataTypeUtils::ToTypeProto(dt);
  ASSERT_TRUE(proto.has_sequence_type());
  const TypeProto &elem = proto.ref_sequence_type().ref_elem_type();
  ASSERT_TRUE(elem.has_map_type());
  EXPECT_EQ(elem.ref_map_type().ref_key_type(),
            static_cast<int32_t>(TensorProto::DataType::STRING));
  ASSERT_TRUE(elem.ref_map_type().ref_value_type().has_tensor_type());
  EXPECT_EQ(
      static_cast<int32_t>(elem.ref_map_type().ref_value_type().ref_tensor_type().ref_elem_type()),
      static_cast<int32_t>(TensorProto::DataType::FLOAT));
}

TEST(onnx_defs, DataTypeUtils_ToType_OptionalMap) {
  DataType dt = Utils::DataTypeUtils::ToType("optional(map(string, float))");
  ASSERT_NE(dt, nullptr);
  EXPECT_EQ(*dt, "optional(map(string,tensor(float)))");

  const TypeProto &proto = Utils::DataTypeUtils::ToTypeProto(dt);
  ASSERT_TRUE(proto.has_optional_type());
  const TypeProto &elem = proto.ref_optional_type().ref_elem_type();
  ASSERT_TRUE(elem.has_map_type());
  EXPECT_EQ(elem.ref_map_type().ref_key_type(),
            static_cast<int32_t>(TensorProto::DataType::STRING));
  ASSERT_TRUE(elem.ref_map_type().ref_value_type().has_tensor_type());
  EXPECT_EQ(
      static_cast<int32_t>(elem.ref_map_type().ref_value_type().ref_tensor_type().ref_elem_type()),
      static_cast<int32_t>(TensorProto::DataType::FLOAT));
}

TEST(onnx_defs, DataTypeUtils_ToType_MapOfMap) {
  DataType dt = Utils::DataTypeUtils::ToType("map(string, map(int64, float))");
  ASSERT_NE(dt, nullptr);
  EXPECT_EQ(*dt, "map(string,map(int64,tensor(float)))");

  const TypeProto &proto = Utils::DataTypeUtils::ToTypeProto(dt);
  ASSERT_TRUE(proto.has_map_type());
  EXPECT_EQ(proto.ref_map_type().ref_key_type(),
            static_cast<int32_t>(TensorProto::DataType::STRING));
  const TypeProto &value = proto.ref_map_type().ref_value_type();
  ASSERT_TRUE(value.has_map_type());
  EXPECT_EQ(value.ref_map_type().ref_key_type(),
            static_cast<int32_t>(TensorProto::DataType::INT64));
  ASSERT_TRUE(value.ref_map_type().ref_value_type().has_tensor_type());
  EXPECT_EQ(
      static_cast<int32_t>(value.ref_map_type().ref_value_type().ref_tensor_type().ref_elem_type()),
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
  DeregisterSchema(op_name, 1, ONNX_DOMAIN);
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
