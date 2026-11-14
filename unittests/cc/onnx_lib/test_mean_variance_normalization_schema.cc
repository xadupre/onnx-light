// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "onnx_lib/checker.h"
#include "onnx_lib/defs/function.h"
#include "onnx_lib/defs/operator_sets.h"
#include "onnx_lib/inliner/inliner.h"
#include "onnx_lib/shape_inference/implementation.h"
#include "onnx_op/operator_sets_nn.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace Test {
namespace {

const NodeProto *FindMvnOutput(const FunctionProto &function, const std::string &name) {
  for (const auto &node : function.node()) {
    if (node.output_size() == 1 && node.output(0) == name) {
      return &node;
    }
  }
  return nullptr;
}

const AttributeProto *FindMvnAttribute(const NodeProto &node, const std::string &name) {
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() == name) {
      return &attribute;
    }
  }
  return nullptr;
}

NodeProto MakeMvnNode(const std::vector<int64_t> &axes, bool with_epsilon) {
  NodeProto node;
  node.set_op_type("MeanVarianceNormalization");
  node.add_input("X");
  node.add_output("Y");
  auto *attribute = node.add_attribute();
  attribute->set_name("axes");
  attribute->set_type(AttributeProto::INTS);
  for (const int64_t axis : axes) {
    attribute->add_ints(axis);
  }
  if (with_epsilon) {
    attribute = node.add_attribute();
    attribute->set_name("epsilon");
    attribute->set_type(AttributeProto::FLOAT);
    attribute->set_f(1e-5f);
  }
  return node;
}

TypeProto MakeMvnType(int64_t elem_type) {
  TypeProto type;
  type.mutable_tensor_type()->set_elem_type(elem_type);
  return type;
}

class MeanVarianceNormalizationSchemaTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { RegisterOnnxOperatorSetSchema(0, false); }
};

TEST_F(MeanVarianceNormalizationSchemaTest, KeepsLegacyTypedFunctions) {
  for (const int version : {13, 17, 18, 21, 28}) {
    SCOPED_TRACE(version);
    const auto *schema =
        OpSchemaRegistry::Schema("MeanVarianceNormalization", version, ONNX_DOMAIN);
    ASSERT_NE(schema, nullptr);
    EXPECT_EQ(schema->SinceVersion(), 13);
    EXPECT_FALSE(schema->HasFunction());
    EXPECT_TRUE(schema->HasContextDependentFunction());
    EXPECT_EQ(schema->attributes().count("epsilon"), 0u);
    for (const int64_t elem_type :
         {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
      const auto node = MakeMvnNode({0}, false);
      FunctionBodyBuildContextImpl context(node, std::vector<TypeProto>{MakeMvnType(elem_type)});
      FunctionProto function;
      ASSERT_TRUE(schema->BuildContextDependentFunction(context, function, version));
      EXPECT_EQ(function.name(), "MeanVarianceNormalization");
      const auto *variance = FindMvnOutput(function, "Variance");
      ASSERT_NE(variance, nullptr);
      ASSERT_EQ(variance->input_size(), 2);
      EXPECT_EQ(variance->op_type(), "Sub");
      EXPECT_EQ(variance->input(0), "E_Xsquared");
      EXPECT_EQ(variance->input(1), "EX_squared");
      const auto *cast = FindMvnOutput(function, "XCompute");
      ASSERT_NE(cast, nullptr);
      const auto *compute_type = FindMvnAttribute(*cast, "to");
      ASSERT_NE(compute_type, nullptr);
      EXPECT_EQ(compute_type->i(),
                elem_type == TensorProto::DOUBLE ? TensorProto::DOUBLE : TensorProto::FLOAT);
      int reduction_count = 0;
      for (const auto &internal_node : function.node()) {
        if (internal_node.op_type() == "ReduceMean") {
          ++reduction_count;
          EXPECT_EQ(internal_node.input_size(), version < 18 ? 1 : 2);
        }
      }
      EXPECT_EQ(reduction_count, 2);
    }
  }
}

TEST_F(MeanVarianceNormalizationSchemaTest, DescribesVersion29AndLightSchema) {
  const auto *schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 29, ONNX_DOMAIN);
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->SinceVersion(), 29);
  EXPECT_FALSE(schema->HasFunction());
  EXPECT_TRUE(schema->HasContextDependentFunction());
  EXPECT_EQ(schema->context_dependent_function_opset_versions(), (std::vector<int>{13, 18}));
  EXPECT_EQ(schema->GetNodeDeterminism(), OpSchema::NodeDeterminism::Deterministic);
  EXPECT_FLOAT_EQ(schema->attributes().at("epsilon").default_value.f(), 1e-9f);
  const auto &axes = schema->attributes().at("axes").default_value.ints();
  EXPECT_EQ((std::vector<int64_t>(axes.begin(), axes.end())), (std::vector<int64_t>{0, 2, 3}));
  const auto schemas =
      onnx_op::nn::GetAllOnnxOpNnSchemasWithHistory("MeanVarianceNormalization", true);
  ASSERT_EQ(schemas.size(), 3u);
}

TEST_F(MeanVarianceNormalizationSchemaTest, BuildsChecksAndExpandsVersionedFunctions) {
  const auto *schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 29, ONNX_DOMAIN);
  ASSERT_NE(schema, nullptr);
  for (const int version : {13, 18, 29}) {
    for (const int64_t elem_type :
         {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
      for (const auto &axes : std::vector<std::vector<int64_t>>{{1, -1}, {}}) {
        for (const bool with_epsilon : {false, true}) {
          SCOPED_TRACE("opset=" + std::to_string(version) + " dtype=" + std::to_string(elem_type) +
                       " epsilon=" + std::to_string(with_epsilon) +
                       " axes_size=" + std::to_string(axes.size()));
          const auto node = MakeMvnNode(axes, with_epsilon);
          FunctionBodyBuildContextImpl context(node,
                                               std::vector<TypeProto>{MakeMvnType(elem_type)});
          FunctionProto function;
          ASSERT_TRUE(schema->BuildContextDependentFunction(context, function, version));
          EXPECT_EQ(function.name(), "MeanVarianceNormalization");
          ASSERT_EQ(function.opset_import_size(), 1);
          EXPECT_EQ(function.opset_import(0).version(), version);

          const int64_t compute_type =
              elem_type == TensorProto::FLOAT16 || elem_type == TensorProto::BFLOAT16
                  ? static_cast<int64_t>(TensorProto::FLOAT)
                  : elem_type;
          for (const std::string output : {"XCompute", "Epsilon", "Y"}) {
            const auto *cast = FindMvnOutput(function, output);
            ASSERT_NE(cast, nullptr);
            EXPECT_EQ(cast->op_type(), "Cast");
            const auto *to = FindMvnAttribute(*cast, "to");
            ASSERT_NE(to, nullptr);
            EXPECT_EQ(to->i(), output == "Y" ? elem_type : compute_type);
          }
          const auto *variance = FindMvnOutput(function, "Variance");
          ASSERT_NE(variance, nullptr);
          EXPECT_EQ(variance->op_type(), "ReduceMean");
          EXPECT_EQ(variance->input(0), "X_variance_squared");
          const auto *squared = FindMvnOutput(function, "X_variance_squared");
          ASSERT_NE(squared, nullptr);
          EXPECT_EQ(squared->op_type(), "Mul");
          EXPECT_EQ(squared->input(0), "X_variance");
          EXPECT_EQ(squared->input(1), "X_variance");
          int reductions = 0;
          for (const auto &internal_node : function.node()) {
            if (internal_node.op_type() == "ReduceMean") {
              ++reductions;
              EXPECT_EQ(internal_node.input_size(), version == 13 ? 1 : 2);
            }
          }
          EXPECT_EQ(reductions, 2);
          const auto *epsilon = FindMvnOutput(function, "EpsilonFloat");
          ASSERT_NE(epsilon, nullptr);
          const auto *value = FindMvnAttribute(*epsilon, "value_float");
          ASSERT_NE(value, nullptr);
          if (with_epsilon) {
            EXPECT_EQ(value->ref_attr_name(), "epsilon");
          } else {
            EXPECT_TRUE(value->has_f());
            EXPECT_FLOAT_EQ(value->f(), 1e-9f);
          }

          checker::CheckerContext checker_context;
          checker_context.set_ir_version(IR_VERSION);
          checker_context.set_opset_imports({{ONNX_DOMAIN, version}});
          checker::LexicalScopeContext lexical_scope;
          EXPECT_NO_THROW(checker::check_function(function, checker_context, lexical_scope));
          GraphProto expanded;
          ASSERT_NO_THROW(FunctionExpandHelper(node, function, expanded));
          int expanded_reductions = 0;
          bool found_epsilon = false;
          for (const auto &internal_node : expanded.node()) {
            if (internal_node.op_type() == "ReduceMean") {
              ++expanded_reductions;
              if (version == 13) {
                const auto *expanded_axes = FindMvnAttribute(internal_node, "axes");
                ASSERT_NE(expanded_axes, nullptr);
                EXPECT_EQ((std::vector<int64_t>(expanded_axes->ints().begin(),
                                                expanded_axes->ints().end())),
                          axes);
              }
            }
            if (internal_node.output(0).find("EpsilonFloat") != std::string::npos) {
              found_epsilon = true;
              const auto *expanded_value = FindMvnAttribute(internal_node, "value_float");
              ASSERT_NE(expanded_value, nullptr);
              EXPECT_TRUE(expanded_value->has_f());
              EXPECT_FLOAT_EQ(expanded_value->f(), with_epsilon ? 1e-5f : 1e-9f);
            }
          }
          EXPECT_TRUE(found_epsilon);
          EXPECT_EQ(expanded_reductions, 2);
        }
      }
    }
  }
}

TEST_F(MeanVarianceNormalizationSchemaTest, PreservesLinkedAttributesAndRejectsMissingTypes) {
  const auto *schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 29, ONNX_DOMAIN);
  ASSERT_NE(schema, nullptr);
  auto node = MakeMvnNode({0}, true);
  auto *epsilon = node.mutable_attribute(1);
  epsilon->clear_f();
  epsilon->set_ref_attr_name("outer_epsilon");
  auto *axes = node.mutable_attribute(0);
  axes->clear_ints();
  axes->set_ref_attr_name("outer_axes");
  for (const int version : {13, 18, 29}) {
    FunctionBodyBuildContextImpl context(node,
                                         std::vector<TypeProto>{MakeMvnType(TensorProto::FLOAT)});
    FunctionProto function;
    ASSERT_TRUE(schema->BuildContextDependentFunction(context, function, version));
    const auto *constant = FindMvnOutput(function, "EpsilonFloat");
    ASSERT_NE(constant, nullptr);
    const auto *value = FindMvnAttribute(*constant, "value_float");
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->ref_attr_name(), "epsilon");
    const auto *reduction = FindMvnOutput(function, "X_RM");
    ASSERT_NE(reduction, nullptr);
    const AttributeProto *axis_attribute = nullptr;
    if (version == 13) {
      axis_attribute = FindMvnAttribute(*reduction, "axes");
    } else {
      const auto *axis_constant = FindMvnOutput(function, "axes");
      ASSERT_NE(axis_constant, nullptr);
      axis_attribute = FindMvnAttribute(*axis_constant, "value_ints");
    }
    ASSERT_NE(axis_attribute, nullptr);
    EXPECT_EQ(axis_attribute->ref_attr_name(), "axes");
  }
  FunctionBodyBuildContextImpl missing_context(node);
  FunctionProto missing_function;
  EXPECT_FALSE(schema->BuildContextDependentFunction(missing_context, missing_function));
  for (const auto &type : std::vector<TypeProto>{TypeProto{}, MakeMvnType(TensorProto::UNDEFINED),
                                                 MakeMvnType(TensorProto::INT64)}) {
    FunctionBodyBuildContextImpl context(node, std::vector<TypeProto>{type});
    FunctionProto function;
    EXPECT_FALSE(schema->BuildContextDependentFunction(context, function));
    EXPECT_EQ(function.node_size(), 0);
  }
}

TEST_F(MeanVarianceNormalizationSchemaTest, MaterializesOmittedAxesForExpansionAndInlining) {
  const auto *schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 29, ONNX_DOMAIN);
  ASSERT_NE(schema, nullptr);
  const auto selected = inliner::FunctionIdSet::Create({{"", "MeanVarianceNormalization"}});
  const std::vector<int64_t> default_axes{0, 2, 3};
  for (const int64_t elem_type :
       {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    SCOPED_TRACE(elem_type);
    auto node = MakeMvnNode({}, false);
    node.clear_attribute();
    auto input_type = MakeMvnType(elem_type);
    for (const int64_t dimension : {2, 3, 4, 5}) {
      input_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);
    }
    ModelProto model;
    model.set_ir_version(IR_VERSION);
    model.add_opset("", 29);
    auto *graph = model.mutable_graph();
    graph->set_name("mvn_default_axes");
    auto *input = graph->add_input();
    input->set_name("X");
    *input->mutable_type() = input_type;
    auto *output = graph->add_output();
    output->set_name("Y");
    *output->mutable_type() = input_type;
    *graph->add_node() = node;
    for (const int version : {13, 18, 29}) {
      SCOPED_TRACE(version);
      FunctionBodyBuildContextImpl context(node, std::vector<TypeProto>{input_type});
      FunctionProto function;
      ASSERT_TRUE(schema->BuildContextDependentFunction(context, function, version));
      int axes_values = 0;
      for (const auto &internal_node : function.node()) {
        const AttributeProto *axes = nullptr;
        if (version == 13 && internal_node.op_type() == "ReduceMean") {
          axes = FindMvnAttribute(internal_node, "axes");
        } else if (version != 13 && internal_node.output(0) == "axes") {
          axes = FindMvnAttribute(internal_node, "value_ints");
        } else {
          continue;
        }
        ASSERT_NE(axes, nullptr);
        EXPECT_FALSE(axes->has_ref_attr_name());
        EXPECT_EQ((std::vector<int64_t>(axes->ints().begin(), axes->ints().end())), default_axes);
        ++axes_values;
      }
      EXPECT_EQ(axes_values, version == 13 ? 2 : 1);
      auto expanded = model;
      expanded.mutable_opset_import(0)->set_version(version);
      expanded.mutable_graph()->clear_node();
      ASSERT_NO_THROW(FunctionExpandHelper(node, function, *expanded.mutable_graph()));
      EXPECT_NO_THROW(checker::check_model(expanded, true));
    }
    ASSERT_NO_THROW(
        inliner::InlineSelectedFunctions(model, *selected, OpSchemaRegistry::Instance()));
    ASSERT_GT(model.graph().node_size(), 1);
    EXPECT_NO_THROW(checker::check_model(model, true));
  }
}

TEST_F(MeanVarianceNormalizationSchemaTest, PropagatesShapesAndValidatesAxes) {
  for (const int version : {28, 29}) {
    for (const int64_t elem_type :
         {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
      ModelProto model;
      model.set_ir_version(IR_VERSION);
      model.add_opset_import()->set_version(version);
      auto *graph = model.mutable_graph();
      graph->set_name("mvn_shape");
      auto *input = graph->add_input();
      input->set_name("X");
      *input->mutable_type() = MakeMvnType(elem_type);
      auto *shape = input->mutable_type()->mutable_tensor_type()->mutable_shape();
      shape->add_dim()->set_dim_value(2);
      shape->add_dim()->set_dim_param("channels");
      shape->add_dim()->set_dim_value(4);
      graph->add_output()->set_name("Y");
      *graph->add_node() = MakeMvnNode({0, -1}, version == 29);
      ASSERT_NO_THROW(shape_inference::InferShapes(model, OpSchemaRegistry::Instance(),
                                                   ShapeInferenceOptions(true, 1, false)));
      const auto &output = graph->output(0).type().tensor_type();
      EXPECT_EQ(output.elem_type(), elem_type);
      EXPECT_EQ(output.shape().SerializeAsString(), shape->SerializeAsString());
      if (version == 29) {
        for (const int64_t invalid_axis : {-4, 3}) {
          auto invalid = model;
          *invalid.mutable_graph()->mutable_node(0)->mutable_attribute(0)->mutable_ints(0) =
              invalid_axis;
          EXPECT_THROW(shape_inference::InferShapes(invalid, OpSchemaRegistry::Instance(),
                                                    ShapeInferenceOptions(true, 1, false)),
                       InferenceError);
        }
      }
    }
  }
}

} // namespace
} // namespace Test
} // namespace ONNX_LIGHT_NAMESPACE
