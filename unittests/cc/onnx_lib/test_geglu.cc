// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/checker.h"
#include "onnx_lib/defs/function.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_lib/version_converter/convert.h"
#include "onnx_op/operator_sets_math.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ModelProto MakeGeGLUModel(TensorProto::DataType dtype, const std::vector<int64_t> &a_shape,
                          const std::vector<int64_t> &b_shape,
                          const std::string &approximate = "") {
  RegisterAllOnnxOperatorSchemas();
  ModelProto model;
  model.set_ir_version(13);
  model.add_opset_import()->set_version(29);
  auto *graph = model.mutable_graph();
  graph->set_name("geglu");
  for (int i = 0; i < 2; ++i) {
    auto *input = graph->add_input();
    input->set_name(i == 0 ? "A" : "B");
    auto *tensor = input->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(dtype);
    auto *shape = tensor->mutable_shape();
    for (auto dim : i == 0 ? a_shape : b_shape) {
      shape->add_dim()->set_dim_value(dim);
    }
  }
  graph->add_output()->set_name("Y");
  auto *node = graph->add_node();
  node->set_op_type("GeGLU");
  node->add_input("A");
  node->add_input("B");
  node->add_output("Y");
  if (!approximate.empty()) {
    auto *attribute = node->add_attribute();
    attribute->set_name("approximate");
    attribute->set_type(AttributeProto::STRING);
    attribute->set_s(approximate);
  }
  return model;
}

void InferGeGLU(ModelProto &model) {
  shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                               ShapeInferenceOptions(true, 1, false));
}

} // namespace

TEST(GeGLUSchema, RegistrationAndShapeInference) {
  RegisterAllOnnxOperatorSchemas();
  EXPECT_EQ(OpSchemaRegistry::Schema("GeGLU", 28, ""), nullptr);
  const auto *schema = OpSchemaRegistry::Schema("GeGLU", 29, "");
  ASSERT_NE(schema, nullptr);
  EXPECT_TRUE(schema->HasContextDependentFunction());
  EXPECT_EQ(schema->attributes().at("approximate").default_value.s(), "none");
  const auto light = onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("GeGLU");
  ASSERT_EQ(light.size(), 1u);
  EXPECT_EQ(light[0].since_version(), 29);
  EXPECT_TRUE(light[0].has_function_implementation());
  EXPECT_EQ(light[0].doc(), schema->doc());
  for (auto dtype :
       {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    for (const auto &dims :
         {std::vector<int64_t>{}, std::vector<int64_t>{2, 3, 4}, std::vector<int64_t>{2, 0, 3}}) {
      for (const auto *approximate : {"", "none", "tanh"}) {
        auto model = MakeGeGLUModel(dtype, dims, dims, approximate);
        InferGeGLU(model);
        const auto &output = model.graph().output(0).type().tensor_type();
        EXPECT_EQ(output.elem_type(), dtype);
        ASSERT_EQ(output.shape().dim_size(), static_cast<int>(dims.size()));
        for (size_t i = 0; i < dims.size(); ++i) {
          EXPECT_EQ(output.shape().dim(i).dim_value(), dims[i]);
        }
        EXPECT_NO_THROW(checker::check_model(model, true));
      }
    }
  }
}

TEST(GeGLUSchema, SymbolicAndUnknownShapes) {
  auto model = MakeGeGLUModel(TensorProto::FLOAT, {2, 3}, {2, 3});
  auto *shape = model.mutable_graph()
                    ->mutable_input(0)
                    ->mutable_type()
                    ->mutable_tensor_type()
                    ->mutable_shape();
  shape->mutable_dim(0)->clear_dim_value();
  shape->mutable_dim(0)->set_dim_param("N");
  shape->mutable_dim(1)->clear_dim_value();
  InferGeGLU(model);
  const auto &output = model.graph().output(0).type().tensor_type().shape();
  ASSERT_EQ(output.dim_size(), 2);
  EXPECT_EQ(output.dim(0).dim_value(), 2);
  EXPECT_EQ(output.dim(1).dim_value(), 3);

  model = MakeGeGLUModel(TensorProto::FLOAT, {2, 3}, {2, 3});
  model.mutable_graph()->mutable_input(1)->mutable_type()->mutable_tensor_type()->clear_shape();
  InferGeGLU(model);
  EXPECT_EQ(model.graph().output(0).type().tensor_type().elem_type(), TensorProto::FLOAT);
  EXPECT_FALSE(model.graph().output(0).type().tensor_type().has_shape());
}

TEST(GeGLUSchema, RejectsBroadcastingAndInvalidAttributesAndTypes) {
  for (const auto &shape : {std::vector<int64_t>{}, std::vector<int64_t>{3},
                            std::vector<int64_t>{2, 1}, std::vector<int64_t>{2, 4}}) {
    auto model = MakeGeGLUModel(TensorProto::FLOAT, {2, 3}, shape);
    EXPECT_THROW(InferGeGLU(model), InferenceError);
  }
  auto invalid_attribute = MakeGeGLUModel(TensorProto::FLOAT, {2, 3}, {2, 3}, "sigmoid");
  EXPECT_THROW(InferGeGLU(invalid_attribute), InferenceError);
  auto integer = MakeGeGLUModel(TensorProto::INT32, {2, 3}, {2, 3});
  EXPECT_THROW(InferGeGLU(integer), InferenceError);
  auto mismatched = MakeGeGLUModel(TensorProto::FLOAT, {2, 3}, {2, 3});
  mismatched.mutable_graph()
      ->mutable_input(1)
      ->mutable_type()
      ->mutable_tensor_type()
      ->set_elem_type(TensorProto::DOUBLE);
  EXPECT_THROW(InferGeGLU(mismatched), InferenceError);
}

TEST(GeGLUSchema, FunctionExpansionPreservesPrecisionAndAttribute) {
  RegisterAllOnnxOperatorSchemas();
  const auto *schema = OpSchemaRegistry::Schema("GeGLU", 29, "");
  ASSERT_NE(schema, nullptr);
  for (auto dtype :
       {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    for (const auto *approximate : {"", "none", "tanh"}) {
      auto model = MakeGeGLUModel(dtype, {2, 3}, {2, 3}, approximate);
      auto type = model.graph().input(0).type();
      type.mutable_tensor_type()->clear_shape();
      FunctionBodyBuildContextImpl context(model.graph().node(0),
                                           std::vector<TypeProto>{type, type});
      FunctionProto function;
      ASSERT_TRUE(schema->BuildContextDependentFunction(context, function));
      const bool half = dtype == TensorProto::FLOAT16 || dtype == TensorProto::BFLOAT16;
      ASSERT_EQ(function.node_size(), half ? 5 : 2);
      const auto &gelu = function.node(half ? 2 : 0);
      EXPECT_EQ(gelu.op_type(), "Gelu");
      ASSERT_EQ(gelu.attribute_size(), 1);
      EXPECT_EQ(gelu.attribute(0).ref_attr_name(), "approximate");
      EXPECT_EQ(gelu.attribute(0).type(), AttributeProto::STRING);
      EXPECT_EQ(function.node(half ? 3 : 1).op_type(), "Mul");
      if (half) {
        for (int i : {0, 1}) {
          EXPECT_EQ(function.node(i).op_type(), "Cast");
          EXPECT_EQ(function.node(i).attribute(0).i(), TensorProto::FLOAT);
        }
        EXPECT_EQ(function.node(4).op_type(), "Cast");
        EXPECT_EQ(function.node(4).attribute(0).i(), dtype);
      }
      checker::CheckerContext checker_context;
      checker_context.set_ir_version(13);
      checker::LexicalScopeContext lexical_scope;
      EXPECT_NO_THROW(checker::check_function(function, checker_context, lexical_scope));
    }
  }
  auto model = MakeGeGLUModel(TensorProto::FLOAT, {2}, {2});
  FunctionBodyBuildContextImpl missing_context(model.graph().node(0));
  FunctionProto function;
  EXPECT_FALSE(schema->BuildContextDependentFunction(missing_context, function));
}

TEST(GeGLUSchema, VersionConversion) {
  const auto model = MakeGeGLUModel(TensorProto::FLOAT, {2, 3}, {2, 3}, "tanh");
  EXPECT_EQ(version_conversion::ConvertVersion(model, 29).opset_import(0).version(), 29);
  EXPECT_THROW(version_conversion::ConvertVersion(model, 28), std::runtime_error);
}
