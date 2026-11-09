// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/checker.h"
#include "onnx_lib/defs/function.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_lib/version_converter/convert.h"
#include "onnx_op/operator_sets.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

ValueInfoProto GemmValueInfo(const char *name, const std::vector<int64_t> &dims) {
  ValueInfoProto value;
  value.set_name(name);
  auto *type = value.mutable_type()->mutable_tensor_type();
  type->set_elem_type(TensorProto::FLOAT);
  auto *shape = type->mutable_shape();
  for (const auto dim : dims) {
    shape->add_dim()->set_dim_value(dim);
  }
  return value;
}

ModelProto MakeGemmModel(const std::vector<int64_t> &a, const std::vector<int64_t> &b,
                         const std::vector<int64_t> *c = nullptr, int64_t transA = 0,
                         int64_t transB = 0, int64_t opset = 29) {
  ModelProto model;
  model.set_ir_version(IR_VERSION);
  model.add_opset_import()->set_version(opset);
  auto *graph = model.mutable_graph();
  graph->set_name("gemm");
  *graph->add_input() = GemmValueInfo("A", a);
  *graph->add_input() = GemmValueInfo("B", b);
  if (c != nullptr) {
    *graph->add_input() = GemmValueInfo("C", *c);
  }
  auto *output = graph->add_output();
  output->set_name("Y");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  *graph->add_node() = MakeNode(
      "Gemm", c ? std::vector<std::string>{"A", "B", "C"} : std::vector<std::string>{"A", "B"},
      {"Y"});
  AddAttribute<int64_t>(*graph->mutable_node(0), "transA", transA);
  AddAttribute<int64_t>(*graph->mutable_node(0), "transB", transB);
  return model;
}

std::vector<int64_t> InferGemmShape(ModelProto &model) {
  shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                               ShapeInferenceOptions(true, 1, false));
  std::vector<int64_t> dims;
  for (const auto &dim : model.graph().output(0).type().tensor_type().shape().dim()) {
    dims.push_back(dim.dim_value());
  }
  return dims;
}

} // namespace

TEST(onnx_shape_inference, Gemm29LightSchemaVerifiesAttributes) {
  const auto schemas = onnx_op::GetAllOnnxOpSchemasWithHistory("Gemm");
  ASSERT_FALSE(schemas.empty());
  const auto &schema = schemas.front();
  ASSERT_EQ(schema.since_version(), 29);
  auto model = MakeGemmModel({2, 3, 5}, {7, 5}, nullptr, 0, 1);
  auto node = model.graph().node(0);
  AddAttribute<float>(node, "alpha", 0.25f);
  AddAttribute<float>(node, "beta", 0.0f);
  EXPECT_NO_THROW(schema.Verify(node));
  auto invalid = node;
  invalid.mutable_attribute(0)->set_type(AttributeProto::FLOAT);
  invalid.mutable_attribute(0)->clear_i();
  invalid.mutable_attribute(0)->set_f(1.0f);
  EXPECT_THROW(schema.Verify(invalid), core::schema::SchemaError);
  AddAttribute<int64_t>(node, "unknown", 1);
  EXPECT_THROW(schema.Verify(node), core::schema::SchemaError);
}

TEST(onnx_shape_inference, Gemm29RanksAndTranspose) {
  RegisterAllOnnxOperatorSchemas();
  for (const std::vector<int64_t> &a :
       {std::vector<int64_t>{5}, {2, 3, 5}, {2, 3, 4, 5}, {0, 3, 5}, {2, 3, 0}}) {
    for (const bool transpose_a : {false, true}) {
      for (const bool transpose_b : {false, true}) {
        auto input_shape = a;
        if (transpose_a && a.size() > 1) {
          std::swap(input_shape[input_shape.size() - 2], input_shape.back());
        }
        const std::vector<int64_t> b =
            transpose_b ? std::vector<int64_t>{7, a.back()} : std::vector<int64_t>{a.back(), 7};
        const std::vector<int64_t> c{7};
        auto expected = a;
        expected.back() = 7;
        auto model = MakeGemmModel(input_shape, b, &c, transpose_a, transpose_b);
        EXPECT_EQ(InferGemmShape(model), expected);
        EXPECT_NO_THROW(checker::check_model(model, true));
      }
    }
  }
  EXPECT_EQ(OpSchemaRegistry::Schema("Gemm", 28, "")->SinceVersion(), 13);
  EXPECT_EQ(OpSchemaRegistry::Schema("Gemm", 29, "")->SinceVersion(), 29);
  const auto schemas = onnx_op::GetAllOnnxOpSchemasWithHistory("Gemm");
  ASSERT_FALSE(schemas.empty());
  EXPECT_EQ(schemas.front().since_version(), 29);
}

TEST(onnx_shape_inference, Gemm29RejectsInvalidShapesEvenWithZeroBeta) {
  RegisterAllOnnxOperatorSchemas();
  for (const auto &[a, b] :
       {std::pair{std::vector<int64_t>{}, std::vector<int64_t>{5, 7}},
        std::pair{std::vector<int64_t>{2, 3, 5}, std::vector<int64_t>{2, 5, 7}},
        std::pair{std::vector<int64_t>{2, 3, 4}, std::vector<int64_t>{5, 7}}}) {
    auto model = MakeGemmModel(a, b);
    EXPECT_THROW(InferGemmShape(model), std::runtime_error);
  }
  for (const std::vector<int64_t> &c : {std::vector<int64_t>{2, 3, 7, 1}, {2, 2, 7}, {8}}) {
    auto model = MakeGemmModel({2, 3, 5}, {5, 7}, &c);
    AddAttribute<float>(*model.mutable_graph()->mutable_node(0), "beta", 0.0f);
    EXPECT_THROW(InferGemmShape(model), std::runtime_error);
  }
  auto old_model = MakeGemmModel({2, 3, 5}, {5, 7}, nullptr, 0, 0, 28);
  EXPECT_THROW(InferGemmShape(old_model), std::runtime_error);
}

TEST(onnx_shape_inference, Gemm29SymbolicAndMissingShapes) {
  RegisterAllOnnxOperatorSchemas();
  auto model = MakeGemmModel({2, 3, 5}, {5, 7});
  auto *a_shape = model.mutable_graph()
                      ->mutable_input(0)
                      ->mutable_type()
                      ->mutable_tensor_type()
                      ->mutable_shape();
  a_shape->mutable_dim(0)->clear_dim_value();
  a_shape->mutable_dim(0)->set_dim_param("batch");
  a_shape->mutable_dim(2)->clear_dim_value();
  a_shape->mutable_dim(2)->set_dim_param("features");
  InferGemmShape(model);
  EXPECT_EQ(model.graph().output(0).type().tensor_type().shape().dim(0).dim_param(), "batch");
  model.mutable_graph()->mutable_input(1)->mutable_type()->mutable_tensor_type()->clear_shape();
  model.mutable_graph()->mutable_output(0)->clear_type();
  InferGemmShape(model);
  EXPECT_FALSE(model.graph().output(0).type().tensor_type().shape().dim(2).has_dim_value());
  model.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->clear_shape();
  model.mutable_graph()->mutable_output(0)->clear_type();
  InferGemmShape(model);
  EXPECT_FALSE(model.graph().output(0).type().tensor_type().has_shape());
}

TEST(onnx_shape_inference, Gemm29FunctionDecomposition) {
  RegisterAllOnnxOperatorSchemas();
  const auto *schema = OpSchemaRegistry::Schema("Gemm", 29, "");
  ASSERT_NE(schema, nullptr);
  for (const auto dtype :
       {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    for (const std::vector<int64_t> &a : {std::vector<int64_t>{5}, {2, 5, 3}}) {
      for (const bool has_bias : {false, true}) {
        for (const float beta : {0.0f, 0.5f}) {
          const std::vector<int64_t> c{7};
          auto model = MakeGemmModel(a, {7, 5}, has_bias ? &c : nullptr, 1, 1);
          auto *graph = model.mutable_graph();
          graph->mutable_output(0)->mutable_type()->mutable_tensor_type()->set_elem_type(dtype);
          std::vector<TypeProto> types;
          for (auto &input : *graph->mutable_input()) {
            input.mutable_type()->mutable_tensor_type()->set_elem_type(dtype);
            types.push_back(input.type());
          }
          NodeProto node = graph->node(0);
          AddAttribute<float>(node, "alpha", 0.25f);
          AddAttribute<float>(node, "beta", beta);
          FunctionBodyBuildContextImpl context(node, types);
          FunctionProto function;
          ASSERT_TRUE(schema->BuildContextDependentFunction(context, function, 29));
          graph->clear_node();
          FunctionExpandHelper(node, function, *graph);
          EXPECT_EQ(InferGemmShape(model),
                    a.size() == 1 ? std::vector<int64_t>{7} : std::vector<int64_t>({2, 3, 7}));
          EXPECT_NO_THROW(checker::check_model(model, true));
          if (!has_bias || beta == 0.0f) {
            for (const auto &expanded : graph->node()) {
              for (const auto &input : expanded.input()) {
                EXPECT_NE(input, "C");
              }
            }
          }
        }
      }
    }
  }
  auto model = MakeGemmModel({2, 5, 3}, {7, 5}, nullptr, 1, 1);
  auto type = model.graph().input(0).type();
  type.mutable_tensor_type()->clear_shape();
  FunctionBodyBuildContextImpl unknown_rank(model.graph().node(0), std::vector<TypeProto>{type});
  FunctionProto function;
  EXPECT_FALSE(schema->BuildContextDependentFunction(unknown_rank, function, 29));
  type.mutable_tensor_type()->set_elem_type(TensorProto::INT32);
  FunctionBodyBuildContextImpl integer_context(model.graph().node(0), std::vector<TypeProto>{type});
  EXPECT_FALSE(schema->BuildContextDependentFunction(integer_context, function, 29));
}

TEST(onnx_version_converter, Gemm29UpgradeAndMatrixDowngrade) {
  for (const int target : {28, 29}) {
    auto model = MakeGemmModel({3, 5}, {5, 7}, nullptr, 0, 0, target == 28 ? 29 : 28);
    const auto converted = version_conversion::ConvertVersion(model, target);
    EXPECT_EQ(converted.opset_import(0).version(), target);
    EXPECT_TRUE(converted.graph().node(0).Equals(model.graph().node(0)));
  }
  for (const std::vector<int64_t> &a : {std::vector<int64_t>{}, {5}, {2, 3, 5}}) {
    auto model = MakeGemmModel(a, {5, 7});
    EXPECT_THROW(version_conversion::ConvertVersion(model, 28), std::runtime_error);
  }
  auto model = MakeGemmModel({3, 5}, {5, 7});
  model.mutable_graph()->mutable_input(0)->mutable_type()->mutable_tensor_type()->clear_shape();
  EXPECT_THROW(version_conversion::ConvertVersion(model, 28), std::runtime_error);
}
