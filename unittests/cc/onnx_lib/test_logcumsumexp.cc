// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/checker.h"
#include "onnx_lib/defs/function.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_lib/version_converter/convert.h"
#include "onnx_op/operator_sets_math.h"

#include <gtest/gtest.h>
#include <limits>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ModelProto MakeLogCumSumExpModel(TensorProto::DataType dtype, const std::vector<int64_t> &dims,
                                 int64_t axis = 0) {
  RegisterAllOnnxOperatorSchemas();
  ModelProto model;
  model.set_ir_version(13);
  model.add_opset_import()->set_version(29);
  auto *graph = model.mutable_graph();
  graph->set_name("logcumsumexp");
  auto *input = graph->add_input();
  input->set_name("X");
  auto *tensor = input->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(dtype);
  auto *shape = tensor->mutable_shape();
  for (auto dim : dims) {
    shape->add_dim()->set_dim_value(dim);
  }
  graph->add_output()->set_name("Y");
  auto *node = graph->add_node();
  node->set_op_type("LogCumSumExp");
  node->add_input("X");
  node->add_output("Y");
  auto *attribute = node->add_attribute();
  attribute->set_name("axis");
  attribute->set_type(AttributeProto::INT);
  attribute->set_i(axis);
  return model;
}

void InferLogCumSumExp(ModelProto &model) {
  shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                               ShapeInferenceOptions(true, 1, false));
}

} // namespace

TEST(LogCumSumExpSchema, RegistrationAndShapeInference) {
  RegisterAllOnnxOperatorSchemas();
  EXPECT_EQ(OpSchemaRegistry::Schema("LogCumSumExp", 28, ""), nullptr);
  const auto *schema = OpSchemaRegistry::Schema("LogCumSumExp", 29, "");
  ASSERT_NE(schema, nullptr);
  EXPECT_TRUE(schema->HasContextDependentFunction());
  const auto light = onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("LogCumSumExp");
  ASSERT_EQ(light.size(), 1u);
  EXPECT_EQ(light[0].since_version(), 29);
  EXPECT_TRUE(light[0].has_function_implementation());
  for (auto dtype :
       {TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::FLOAT, TensorProto::DOUBLE}) {
    for (const auto &dims : {std::vector<int64_t>{2, 3, 4}, std::vector<int64_t>{2, 0, 3}}) {
      for (int64_t axis : {0, 1, 2, -1, -3}) {
        auto model = MakeLogCumSumExpModel(dtype, dims, axis);
        InferLogCumSumExp(model);
        const auto &output = model.graph().output(0).type().tensor_type();
        EXPECT_EQ(output.elem_type(), dtype);
        ASSERT_EQ(output.shape().dim_size(), dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
          EXPECT_EQ(output.shape().dim(i).dim_value(), dims[i]);
        }
        EXPECT_NO_THROW(checker::check_model(model, true));
      }
    }
  }
}

TEST(LogCumSumExpSchema, SymbolicAndUnknownShape) {
  auto model = MakeLogCumSumExpModel(TensorProto::FLOAT, {2, 3}, -1);
  auto *shape = model.mutable_graph()
                    ->mutable_input(0)
                    ->mutable_type()
                    ->mutable_tensor_type()
                    ->mutable_shape();
  shape->mutable_dim(0)->clear_dim_value();
  shape->mutable_dim(0)->set_dim_param("N");
  shape->mutable_dim(1)->clear_dim_value();
  InferLogCumSumExp(model);
  const auto &output = model.graph().output(0).type().tensor_type().shape();
  ASSERT_EQ(output.dim_size(), 2);
  EXPECT_EQ(output.dim(0).dim_param(), "N");
  EXPECT_FALSE(output.dim(1).has_dim_value());
  model = MakeLogCumSumExpModel(TensorProto::FLOAT, {2, 3}, -1);
  model.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->clear_shape();
  InferLogCumSumExp(model);
  EXPECT_EQ(model.graph().output(0).type().tensor_type().elem_type(), TensorProto::FLOAT);
  EXPECT_FALSE(model.graph().output(0).type().tensor_type().has_shape());
}

TEST(LogCumSumExpSchema, RejectsInvalidRankAxisAndType) {
  auto scalar = MakeLogCumSumExpModel(TensorProto::FLOAT, {});
  EXPECT_THROW(InferLogCumSumExp(scalar), InferenceError);
  for (const auto &dims : {std::vector<int64_t>{2, 3}, std::vector<int64_t>{0, 3}}) {
    for (int64_t axis : {int64_t{-3}, int64_t{2}, std::numeric_limits<int64_t>::min(),
                         std::numeric_limits<int64_t>::max()}) {
      auto model = MakeLogCumSumExpModel(TensorProto::FLOAT, dims, axis);
      EXPECT_THROW(InferLogCumSumExp(model), InferenceError);
    }
  }
  auto integer = MakeLogCumSumExpModel(TensorProto::INT32, {2, 3});
  EXPECT_THROW(InferLogCumSumExp(integer), InferenceError);
}

TEST(LogCumSumExpSchema, FunctionBuildAndCheck) {
  RegisterAllOnnxOperatorSchemas();
  const auto *schema = OpSchemaRegistry::Schema("LogCumSumExp", 29, "");
  ASSERT_NE(schema, nullptr);
  for (auto dtype :
       {TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::FLOAT, TensorProto::DOUBLE}) {
    for (bool explicit_axis : {false, true}) {
      auto model = MakeLogCumSumExpModel(dtype, {2, 3}, -1);
      auto node = model.graph().node(0);
      if (!explicit_axis) {
        node.clear_attribute();
      }
      auto type = model.graph().input(0).type();
      type.mutable_tensor_type()->clear_shape();
      FunctionBodyBuildContextImpl context(node, std::vector<TypeProto>{type});
      FunctionProto function;
      ASSERT_TRUE(schema->BuildContextDependentFunction(context, function));
      ASSERT_GT(function.node_size(), 0);
      EXPECT_EQ(function.node(0).op_type(), "Cast");
      ASSERT_EQ(function.node(0).attribute_size(), 1);
      EXPECT_EQ(function.node(0).attribute(0).i(),
                dtype == TensorProto::DOUBLE ? TensorProto::DOUBLE : TensorProto::FLOAT);
      checker::CheckerContext checker_context;
      checker_context.set_ir_version(13);
      checker::LexicalScopeContext lexical_scope;
      EXPECT_NO_THROW(checker::check_function(function, checker_context, lexical_scope));
      GraphProto expanded;
      EXPECT_NO_THROW(FunctionExpandHelper(node, function, expanded));
      EXPECT_EQ(expanded.node_size(), function.node_size());
    }
  }
  auto model = MakeLogCumSumExpModel(TensorProto::INT32, {2});
  FunctionBodyBuildContextImpl invalid_context(
      model.graph().node(0), std::vector<TypeProto>{model.graph().input(0).type()});
  FunctionProto function;
  EXPECT_FALSE(schema->BuildContextDependentFunction(invalid_context, function));
  FunctionBodyBuildContextImpl missing_context(model.graph().node(0));
  EXPECT_FALSE(schema->BuildContextDependentFunction(missing_context, function));
}

TEST(LogCumSumExpSchema, VersionConversion) {
  const auto model = MakeLogCumSumExpModel(TensorProto::FLOAT, {3});
  EXPECT_EQ(version_conversion::ConvertVersion(model, 29).opset_import(0).version(), 29);
  EXPECT_THROW(version_conversion::ConvertVersion(model, 28), std::runtime_error);
}
