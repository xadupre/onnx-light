// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "onnx_lib/checker.h"
#include "onnx_lib/defs/function.h"
#include "onnx_lib/defs/operator_sets.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_lib/version_converter/convert.h"
#include "onnx_op/operator_sets_nn.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

void AddCausalTensor(ValueInfoProto *value, const std::string &name,
                     const std::vector<int64_t> &dimensions, int64_t type = TensorProto::FLOAT) {
  value->set_name(name);
  auto *tensor = value->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(type);
  for (const auto dimension : dimensions) {
    auto *dim = tensor->mutable_shape()->add_dim();
    if (dimension >= 0)
      dim->set_dim_value(dimension);
  }
}

ModelProto MakeCausalModel(int version, std::optional<int64_t> dilation = std::nullopt,
                           int64_t kernel = 3, std::optional<int64_t> history = std::nullopt,
                           int64_t length = 7, bool bias = false,
                           const std::string &activation = "none",
                           int64_t type = TensorProto::FLOAT) {
  ModelProto model;
  model.set_ir_version(IR_VERSION);
  model.add_opset_import()->set_version(version);
  auto *graph = model.mutable_graph();
  graph->set_name("causal_convolution");
  AddCausalTensor(graph->add_input(), "X", {2, 3, length}, type);
  AddCausalTensor(graph->add_input(), "W", {3, 1, kernel}, type);
  if (bias)
    AddCausalTensor(graph->add_input(), "B", {3}, type);
  if (history)
    AddCausalTensor(graph->add_input(), "P", {2, 3, *history}, type);
  graph->add_output()->set_name("Y");
  graph->add_output()->set_name("S");
  auto *node = graph->add_node();
  node->set_op_type("CausalConvWithState");
  node->add_input("X");
  node->add_input("W");
  if (bias || history)
    node->add_input(bias ? "B" : "");
  if (history)
    node->add_input("P");
  node->add_output("Y");
  node->add_output("S");
  if (dilation) {
    auto *attribute = node->add_attribute();
    attribute->set_name("dilation");
    attribute->set_type(AttributeProto::INT);
    attribute->set_i(*dilation);
  }
  if (activation != "none") {
    auto *attribute = node->add_attribute();
    attribute->set_name("activation");
    attribute->set_type(AttributeProto::STRING);
    attribute->set_s(activation);
  }
  return model;
}

const AttributeProto *CausalAttribute(const NodeProto &node, const std::string &name) {
  for (const auto &attribute : node.attribute())
    if (attribute.name() == name)
      return &attribute;
  return nullptr;
}

const NodeProto *CausalProducer(const FunctionProto &function, const std::string &name) {
  for (const auto &node : function.node())
    if (node.output_size() == 1 && node.output(0) == name)
      return &node;
  return nullptr;
}

bool BuildCausalFunction(const ModelProto &model, FunctionProto &function) {
  const int version = static_cast<int>(model.opset_import(0).version());
  const auto *schema = OpSchemaRegistry::Schema("CausalConvWithState", version, ONNX_DOMAIN);
  std::vector<TypeProto> types;
  for (const auto &input : model.graph().input())
    types.push_back(input.type());
  if (model.graph().node(0).input_size() == 4 && model.graph().node(0).input(2).empty())
    types.insert(types.begin() + 2, TypeProto{});
  FunctionBodyBuildContextImpl context(model.graph().node(0), types);
  return schema->BuildContextDependentFunction(context, function, version);
}

void InferCausalShapes(ModelProto &model) {
  shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                               ShapeInferenceOptions(true, 1, false));
}

class CausalConvWithStateSchemaTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { RegisterOnnxOperatorSetSchema(0, false); }
};

TEST_F(CausalConvWithStateSchemaTest, KeepsLegacySchemaAndMatchesLightMetadata) {
  const auto light = onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("CausalConvWithState");
  ASSERT_EQ(light.size(), 2u);
  for (const int version : {27, 28, 29}) {
    const auto *schema = OpSchemaRegistry::Schema("CausalConvWithState", version, ONNX_DOMAIN);
    ASSERT_NE(schema, nullptr);
    EXPECT_EQ(schema->SinceVersion(), version == 29 ? 29 : 27);
    EXPECT_EQ(schema->attributes().count("dilation"), version == 29 ? 1u : 0u);
    EXPECT_EQ(schema->attributes().at("activation").default_value.s(), "none");
    const auto &matching = light[version == 29 ? 1 : 0];
    EXPECT_EQ(matching.since_version(), schema->SinceVersion());
    if (version == 29) {
      EXPECT_EQ(matching.doc(), schema->doc());
    }
    ASSERT_EQ(matching.attributes().size(), schema->attributes().size());
    for (const auto &attribute : matching.attributes()) {
      const auto &full = schema->attributes().at(attribute.name);
      EXPECT_EQ(attribute.description, full.description);
      EXPECT_EQ(static_cast<int>(attribute.type), static_cast<int>(full.type));
      EXPECT_EQ(attribute.required, full.required);
      if (attribute.name == "dilation") {
        EXPECT_EQ(std::get<int64_t>(attribute.default_value), int64_t{1});
      }
    }
    for (size_t i = 0; i < matching.inputs().size(); ++i)
      EXPECT_EQ(matching.inputs()[i].description, schema->inputs()[i].GetDescription());
    for (size_t i = 0; i < matching.outputs().size(); ++i)
      EXPECT_EQ(matching.outputs()[i].description, schema->outputs()[i].GetDescription());
  }
  auto invalid_legacy = MakeCausalModel(27, 1);
  checker::CheckerContext context;
  context.set_ir_version(IR_VERSION);
  context.set_opset_imports({{ONNX_DOMAIN, 27}});
  EXPECT_THROW(
      checker::check_node(invalid_legacy.graph().node(0), context, checker::LexicalScopeContext{}),
      checker::ValidationError);
}

TEST_F(CausalConvWithStateSchemaTest, InfersDilatedAndLegacyStatesIncludingEmptyChunks) {
  for (const int version : {27, 28, 29}) {
    for (const auto dilation :
         {std::optional<int64_t>{}, std::optional<int64_t>{1}, std::optional<int64_t>{3}}) {
      if (version < 29 && dilation)
        continue;
      for (const int64_t kernel : {1, 4}) {
        const int64_t history = (kernel - 1) * dilation.value_or(1);
        for (const int64_t length : {0, 1, 7}) {
          for (const bool with_state : {false, true}) {
            auto model = MakeCausalModel(
                version, dilation, kernel,
                with_state ? std::optional<int64_t>{history} : std::nullopt, length);
            ASSERT_NO_THROW(InferCausalShapes(model));
            const auto &output = model.graph().output(0).type().tensor_type();
            const auto &state = model.graph().output(1).type().tensor_type();
            ASSERT_EQ(output.shape().dim_size(), 3);
            ASSERT_EQ(state.shape().dim_size(), 3);
            EXPECT_EQ(output.shape().dim(2).dim_value(), length);
            EXPECT_EQ(state.shape().dim(0).dim_value(), 2);
            EXPECT_EQ(state.shape().dim(1).dim_value(), 3);
            EXPECT_EQ(state.shape().dim(2).dim_value(), history);
            EXPECT_EQ(state.elem_type(), TensorProto::FLOAT);
            EXPECT_NO_THROW(checker::check_model(model, true));
          }
        }
      }
    }
  }
}

TEST_F(CausalConvWithStateSchemaTest, RejectsInvalidDilationKernelStateAndActivation) {
  for (const int64_t dilation : {0, -1}) {
    auto model = MakeCausalModel(29, dilation);
    EXPECT_THROW(InferCausalShapes(model), InferenceError);
    FunctionProto function;
    EXPECT_FALSE(BuildCausalFunction(model, function));
    EXPECT_EQ(function.node_size(), 0);
  }
  for (const int64_t kernel : {0, -1}) {
    auto model = MakeCausalModel(29, 2, kernel);
    model.mutable_graph()
        ->mutable_input(1)
        ->mutable_type()
        ->mutable_tensor_type()
        ->mutable_shape()
        ->mutable_dim(2)
        ->set_dim_value(kernel);
    EXPECT_THROW(InferCausalShapes(model), InferenceError);
    FunctionProto function;
    EXPECT_FALSE(BuildCausalFunction(model, function));
  }
  for (const int version : {27, 29}) {
    auto model = MakeCausalModel(version, std::nullopt, 3, 3);
    EXPECT_THROW(InferCausalShapes(model), InferenceError);
    for (const std::string activation : {"relu", ""}) {
      model = MakeCausalModel(version, std::nullopt, 3, std::nullopt, 7, false, activation);
      EXPECT_THROW(InferCausalShapes(model), InferenceError);
      FunctionProto function;
      EXPECT_FALSE(BuildCausalFunction(model, function));
    }
  }
}

TEST_F(CausalConvWithStateSchemaTest, GuardsEffectiveKernelOverflowAndAllowsUnitKernel) {
  constexpr auto max = std::numeric_limits<int64_t>::max();
  for (const auto &test :
       {std::pair<int64_t, int64_t>{2, max}, {3, max / 2 + 1}, {2, max - 1}, {1, max}}) {
    auto model = MakeCausalModel(29, test.second, test.first, std::nullopt, 0);
    const bool valid = test.first == 1 || test.second == max - 1;
    if (valid) {
      ASSERT_NO_THROW(InferCausalShapes(model));
      EXPECT_EQ(model.graph().output(1).type().tensor_type().shape().dim(2).dim_value(),
                test.first == 1 ? 0 : max - 1);
    } else {
      EXPECT_THROW(InferCausalShapes(model), InferenceError);
    }
    FunctionProto function;
    EXPECT_EQ(BuildCausalFunction(model, function), valid);
  }
}

TEST_F(CausalConvWithStateSchemaTest, ValidatesKnownHistoryWithUnknownKernel) {
  constexpr auto max = std::numeric_limits<int64_t>::max();
  for (const int64_t history : std::vector<int64_t>{-1, 0, 3, 4, max}) {
    auto model = MakeCausalModel(29, 2, -1, history);
    model.mutable_graph()
        ->mutable_input(2)
        ->mutable_type()
        ->mutable_tensor_type()
        ->mutable_shape()
        ->mutable_dim(2)
        ->set_dim_value(history);
    if (history == 0 || history == 4) {
      ASSERT_NO_THROW(InferCausalShapes(model));
      EXPECT_EQ(model.graph().output(1).type().tensor_type().shape().dim(2).dim_value(), history);
    } else {
      EXPECT_THROW(InferCausalShapes(model), InferenceError);
    }
  }
}

TEST_F(CausalConvWithStateSchemaTest, BuildsDilatedFunctionsWithRawStateAndEmptyChunkSupport) {
  for (const int version : {27, 29}) {
    for (const int64_t dilation : {1, 3}) {
      if (version == 27 && dilation != 1)
        continue;
      for (const int64_t type : {TensorProto::FLOAT, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
        for (const int64_t kernel : {1, 4}) {
          for (const int64_t length : {0, 7}) {
            for (const bool with_state : {false, true}) {
              for (const bool with_bias : {false, true}) {
                for (const std::string activation : {"none", "silu", "swish"}) {
                  auto model = MakeCausalModel(
                      version, version == 29 ? std::optional<int64_t>{dilation} : std::nullopt,
                      kernel,
                      with_state ? std::optional<int64_t>{(kernel - 1) * dilation} : std::nullopt,
                      length, with_bias, activation, type);
                  FunctionProto function;
                  ASSERT_TRUE(BuildCausalFunction(model, function));
                  const auto *history = CausalProducer(function, "History");
                  ASSERT_NE(history, nullptr);
                  EXPECT_EQ(history->op_type(), "Mul");
                  EXPECT_EQ(history->input(1), "Dilation");
                  const auto *conv = CausalProducer(function, "ConvAll");
                  ASSERT_NE(conv, nullptr);
                  EXPECT_EQ(conv->input(0), "ConvInput");
                  EXPECT_EQ(conv->input_size(), with_bias ? 3 : 2);
                  ASSERT_NE(CausalAttribute(*conv, "dilations"), nullptr);
                  ASSERT_EQ(CausalAttribute(*conv, "dilations")->ints_size(), 1);
                  EXPECT_EQ(CausalAttribute(*conv, "dilations")->ints(0), dilation);
                  EXPECT_EQ(CausalAttribute(*conv, "group")->i(), 3);
                  const auto *state = CausalProducer(function, "present_state");
                  ASSERT_NE(state, nullptr);
                  EXPECT_EQ(state->op_type(), "Slice");
                  EXPECT_EQ(state->input(0), "PaddedInput");
                  const auto *conv_output = CausalProducer(function, "ConvOut");
                  ASSERT_NE(conv_output, nullptr);
                  EXPECT_EQ(conv_output->op_type(), "Slice");
                  EXPECT_EQ(conv_output->input(2), "InputLength");
                  ASSERT_NO_THROW(InferCausalShapes(model));
                  const auto node = model.graph().node(0);
                  model.mutable_graph()->clear_node();
                  ASSERT_NO_THROW(FunctionExpandHelper(node, function, *model.mutable_graph()));
                  EXPECT_NO_THROW(checker::check_model(model, true));
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST_F(CausalConvWithStateSchemaTest, UsesWeightChannelsWhenInputChannelsAreUnknown) {
  for (const int version : {27, 29}) {
    auto model = MakeCausalModel(version);
    auto *input = model.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type();
    input->mutable_shape()->mutable_dim(1)->clear_dim_value();
    FunctionProto function;
    ASSERT_TRUE(BuildCausalFunction(model, function));
    const auto *conv = CausalProducer(function, "ConvAll");
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(CausalAttribute(*conv, "group")->i(), 3);

    input->clear_shape();
    function.Clear();
    EXPECT_TRUE(BuildCausalFunction(model, function));
    auto *weight = model.mutable_graph()->mutable_input(1)->mutable_type()->mutable_tensor_type();
    weight->mutable_shape()->mutable_dim(0)->clear_dim_value();
    function.Clear();
    EXPECT_FALSE(BuildCausalFunction(model, function));
    EXPECT_EQ(function.node_size(), 0);
  }
}

TEST_F(CausalConvWithStateSchemaTest, ConvertsOnlyUnitDilationAndPreservesLegacyDefaults) {
  for (const auto dilation : {std::optional<int64_t>{}, std::optional<int64_t>{1}}) {
    for (const int target : {27, 28}) {
      auto model = MakeCausalModel(29, dilation, 3, 2, 7, true, "swish");
      ASSERT_NO_THROW(InferCausalShapes(model));
      const auto serialized = model.SerializeAsString();
      const auto converted = version_conversion::ConvertVersion(model, target);
      EXPECT_EQ(model.SerializeAsString(), serialized);
      EXPECT_EQ(converted.opset_import(0).version(), target);
      ASSERT_EQ(converted.graph().node_size(), 1);
      const auto &node = converted.graph().node(0);
      EXPECT_EQ(node.op_type(), "CausalConvWithState");
      EXPECT_EQ(CausalAttribute(node, "dilation"), nullptr);
      ASSERT_NE(CausalAttribute(node, "activation"), nullptr);
      EXPECT_EQ(CausalAttribute(node, "activation")->s(), "swish");
      EXPECT_NO_THROW(checker::check_model(converted, true));
      const auto upgraded = version_conversion::ConvertVersion(converted, 29);
      EXPECT_NO_THROW(checker::check_model(upgraded, true));
    }
  }
  for (const int64_t kernel : {1, 3}) {
    auto dilated = MakeCausalModel(29, 2, kernel, (kernel - 1) * 2);
    ASSERT_NO_THROW(InferCausalShapes(dilated));
    EXPECT_THROW(version_conversion::ConvertVersion(dilated, 28), std::runtime_error);
  }
}

} // namespace
