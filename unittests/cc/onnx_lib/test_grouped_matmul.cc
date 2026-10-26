// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/defs/function.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_lib/version_converter/convert.h"
#include "onnx_op/operator_sets_math.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ModelProto MakeGroupedMatMulModel(const std::vector<std::vector<int64_t>> &shapes = {{4, 3},
                                                                                     {2, 3, 5},
                                                                                     {4, 2}},
                                  TensorProto::DataType dtype = TensorProto::FLOAT,
                                  TensorProto::DataType index_dtype = TensorProto::INT64) {
  RegisterAllOnnxOperatorSchemas();
  ModelProto model;
  model.set_ir_version(13);
  model.add_opset_import()->set_version(29);
  auto *graph = model.mutable_graph();
  graph->set_name("grouped_matmul");
  auto *node = graph->add_node();
  node->set_op_type("GroupedMatMul");
  const std::vector<std::string> names{"input", "weights", "group_indices", "bias"};
  for (size_t i = 0; i < shapes.size(); ++i) {
    node->add_input(names[i]);
    auto *value = graph->add_input();
    value->set_name(names[i]);
    auto *tensor = value->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(i == 2 ? index_dtype : dtype);
    auto *shape = tensor->mutable_shape();
    for (int64_t dim : shapes[i]) {
      shape->add_dim()->set_dim_value(dim);
    }
  }
  node->add_output("output");
  auto *output = graph->add_output();
  output->set_name("output");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(dtype);
  return model;
}

void Infer(ModelProto &model) {
  shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                               ShapeInferenceOptions(true, 1, false));
}

} // namespace

TEST(GroupedMatMul, SchemaAndTypes) {
  RegisterAllOnnxOperatorSchemas();
  EXPECT_EQ(OpSchemaRegistry::Schema("GroupedMatMul", 28, ""), nullptr);
  const auto *schema = OpSchemaRegistry::Schema("GroupedMatMul", 29, "");
  ASSERT_NE(schema, nullptr);
  EXPECT_TRUE(schema->HasContextDependentFunction());
  const auto light = onnx_op::math::GetAllOnnxOpMathSchemasWithHistory("GroupedMatMul");
  ASSERT_EQ(light.size(), 1u);
  EXPECT_EQ(light[0].since_version(), 29);
  EXPECT_TRUE(light[0].has_function_implementation());
  EXPECT_EQ(light[0].inputs().size(), 4u);
  EXPECT_EQ(light[0].type_constraints().size(), 2u);
  for (auto dtype :
       {TensorProto::FLOAT, TensorProto::FLOAT16, TensorProto::BFLOAT16, TensorProto::DOUBLE}) {
    for (auto index_dtype : {TensorProto::INT32, TensorProto::INT64}) {
      auto model = MakeGroupedMatMulModel({{4, 3}, {2, 3, 5}, {4, 2}}, dtype, index_dtype);
      Infer(model);
      const auto &output = model.graph().output(0).type().tensor_type();
      EXPECT_EQ(output.elem_type(), dtype);
      ASSERT_EQ(output.shape().dim_size(), 3);
      EXPECT_EQ(output.shape().dim(0).dim_value(), 4);
      EXPECT_EQ(output.shape().dim(1).dim_value(), 2);
      EXPECT_EQ(output.shape().dim(2).dim_value(), 5);
    }
  }
}

TEST(GroupedMatMul, SymbolicShapeWithBias) {
  auto model = MakeGroupedMatMulModel({{4, 3}, {2, 3, 5}, {4, 2}, {2, 5}});
  const std::vector<std::vector<std::string>> dims{
      {"M", "K"}, {"G", "K", "N"}, {"M", "S"}, {"G", "N"}};
  for (size_t i = 0; i < dims.size(); ++i) {
    auto *shape = model.mutable_graph()
                      ->mutable_input(i)
                      ->mutable_type()
                      ->mutable_tensor_type()
                      ->mutable_shape();
    for (size_t j = 0; j < dims[i].size(); ++j) {
      shape->mutable_dim(j)->clear_dim_value();
      shape->mutable_dim(j)->set_dim_param(dims[i][j]);
    }
  }
  Infer(model);
  const auto &shape = model.graph().output(0).type().tensor_type().shape();
  ASSERT_EQ(shape.dim_size(), 3);
  EXPECT_EQ(shape.dim(0).dim_param(), "M");
  EXPECT_EQ(shape.dim(1).dim_param(), "S");
  EXPECT_EQ(shape.dim(2).dim_param(), "N");
}

TEST(GroupedMatMul, RejectsInvalidShapes) {
  const std::vector<std::vector<std::vector<int64_t>>> invalid{
      {{4, 3, 1}, {2, 3, 5}, {4, 2}},
      {{4, 3}, {2, 3}, {4, 2}},
      {{4, 3}, {2, 3, 5}, {4}},
      {{4, 3}, {2, 4, 5}, {4, 2}},
      {{4, 3}, {2, 3, 5}, {3, 2}},
      {{4, 3}, {2, 3, 5}, {4, 2}, {3, 5}},
      {{4, 3}, {2, 3, 5}, {4, 2}, {2, 4}},
      {{4, 3}, {2, 3, 5}, {4, 2}, {2, 5, 1}}};
  for (const auto &shapes : invalid) {
    auto model = MakeGroupedMatMulModel(shapes);
    EXPECT_THROW(Infer(model), InferenceError);
  }
}

TEST(GroupedMatMul, OptionalBiasFunctionExpansion) {
  RegisterAllOnnxOperatorSchemas();
  const auto *schema = OpSchemaRegistry::Schema("GroupedMatMul", 29, "");
  ASSERT_NE(schema, nullptr);
  for (int bias_mode : {0, 1, 2}) {
    auto model = MakeGroupedMatMulModel();
    NodeProto node = model.graph().node(0);
    if (bias_mode != 0) {
      node.add_input(bias_mode == 1 ? "" : "bias");
    }
    FunctionBodyBuildContextImpl context(node);
    FunctionProto function;
    ASSERT_TRUE(schema->BuildContextDependentFunction(context, function));
    EXPECT_EQ(function.input_size(), 4);
    ASSERT_GT(function.node_size(), 0);
    EXPECT_EQ(function.node(function.node_size() - 1).op_type(),
              bias_mode == 2 ? "Add" : "Identity");
    GraphProto expanded;
    FunctionExpandHelper(node, function, expanded);
    EXPECT_EQ(expanded.node_size(), function.node_size());
    int reshapes = 0;
    for (const auto &inner : function.node()) {
      if (inner.op_type() == "Reshape") {
        ++reshapes;
        ASSERT_EQ(inner.attribute_size(), 1);
        EXPECT_EQ(inner.attribute(0).name(), "allowzero");
        EXPECT_EQ(inner.attribute(0).i(), 1);
      }
    }
    EXPECT_EQ(reshapes, bias_mode == 2 ? 4 : 3);
  }
}

TEST(GroupedMatMul, VersionConversion) {
  const auto model = MakeGroupedMatMulModel();
  EXPECT_EQ(version_conversion::ConvertVersion(model, 29).opset_import(0).version(), 29);
  EXPECT_THROW(version_conversion::ConvertVersion(model, 28), std::runtime_error);
}
