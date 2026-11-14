// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_lib/version_converter/convert.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "onnx_lib/checker.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

void AddMvnTensor(ValueInfoProto *value, const std::string &name, TensorProto::DataType type) {
  value->set_name(name);
  if (type == TensorProto::UNDEFINED)
    return;
  auto *tensor = value->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(type);
  for (int64_t dim : {2, 3, 4, 5})
    tensor->mutable_shape()->add_dim()->set_dim_value(dim);
}

ModelProto MakeMvnModel(int64_t version, TensorProto::DataType type) {
  ModelProto model;
  model.set_ir_version(13);
  model.add_opset_import()->set_version(version);
  auto *graph = model.mutable_graph();
  graph->set_name("mvn_conversion");
  AddMvnTensor(graph->add_input(), "X", type);
  AddMvnTensor(graph->add_output(), "Y", type);
  graph->mutable_output(0)->set_doc_string("normalized output");
  auto *node = graph->add_node();
  node->set_op_type("MeanVarianceNormalization");
  node->add_input("X");
  node->add_output("Y");
  return model;
}

const NodeProto *MvnProducer(const GraphProto &graph, const std::string &name) {
  for (const auto &node : graph.node())
    for (const auto &output : node.output())
      if (output == name)
        return &node;
  return nullptr;
}

std::vector<const NodeProto *> MvnNodes(const GraphProto &graph, const std::string &op) {
  std::vector<const NodeProto *> result;
  for (const auto &node : graph.node())
    if (node.op_type() == op)
      result.push_back(&node);
  return result;
}

void CheckMvnExpansion(const ModelProto &converted, bool upgrade, TensorProto::DataType type,
                       float epsilon = 1e-9f, const std::vector<int64_t> &axes = {0, 2, 3}) {
  const auto &graph = converted.graph();
  EXPECT_TRUE(MvnNodes(graph, "MeanVarianceNormalization").empty());
  const auto means = MvnNodes(graph, "ReduceMean");
  ASSERT_EQ(means.size(), 2u);
  EXPECT_TRUE(MvnNodes(graph, "Pow").empty());
  const auto squares = MvnNodes(graph, "Mul");
  ASSERT_EQ(squares.size(), upgrade ? 2u : 1u);
  for (const auto *square : squares)
    EXPECT_EQ(square->input(0), square->input(1));
  EXPECT_EQ(MvnNodes(graph, "Constant").size(), 2u);
  const auto adds = MvnNodes(graph, "Add");
  ASSERT_EQ(adds.size(), 1u);
  const auto *sqrt = MvnProducer(graph, adds[0]->input(0));
  ASSERT_NE(sqrt, nullptr);
  EXPECT_EQ(sqrt->op_type(), "Sqrt");
  const auto *variance = MvnProducer(graph, sqrt->input(0));
  ASSERT_NE(variance, nullptr);
  EXPECT_EQ(variance->op_type(), upgrade ? "Sub" : "ReduceMean");
  if (upgrade) {
    EXPECT_EQ(variance->input(0), means[1]->output(0));
    EXPECT_EQ(variance->input(1), squares[0]->output(0));
    EXPECT_EQ(squares[0]->input(0), means[0]->output(0));
    EXPECT_EQ(squares[1]->input(0), "X");
  } else {
    EXPECT_EQ(variance->input(0), squares[0]->output(0));
    const auto *centered = MvnProducer(graph, squares[0]->input(0));
    ASSERT_NE(centered, nullptr);
    EXPECT_EQ(centered->op_type(), "Sub");
    EXPECT_EQ(centered->input(0), means[0]->input(0));
    EXPECT_EQ(centered->input(1), means[0]->output(0));
  }
  const auto *axes_constant = MvnProducer(graph, means[0]->input(1));
  ASSERT_NE(axes_constant, nullptr);
  const auto &axes_tensor = axes_constant->attribute(0).t();
  EXPECT_EQ(std::vector<int64_t>(axes_tensor.int64_data().begin(), axes_tensor.int64_data().end()),
            axes);
  EXPECT_EQ(means[0]->input(1), means[1]->input(1));
  const auto *epsilon_constant = MvnProducer(graph, adds[0]->input(1));
  ASSERT_NE(epsilon_constant, nullptr);
  if (epsilon_constant->op_type() == "Cast")
    epsilon_constant = MvnProducer(graph, epsilon_constant->input(0));
  ASSERT_NE(epsilon_constant, nullptr);
  EXPECT_EQ(epsilon_constant->op_type(), "Constant");
  ASSERT_EQ(epsilon_constant->attribute(0).t().float_data_size(), 1);
  EXPECT_EQ(epsilon_constant->attribute(0).t().float_data(0), epsilon);
  const bool promoted = !upgrade && (type == TensorProto::FLOAT16 || type == TensorProto::BFLOAT16);
  const auto casts = MvnNodes(graph, "Cast");
  EXPECT_EQ(casts.size(), promoted ? 2u : type == TensorProto::FLOAT ? 0u : 1u);
  const auto *output = MvnProducer(graph, "Y");
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->op_type(), promoted ? "Cast" : "Div");
  const auto *division = promoted ? MvnProducer(graph, output->input(0)) : output;
  ASSERT_NE(division, nullptr);
  EXPECT_EQ(division->op_type(), "Div");
  EXPECT_EQ(division->input(1), adds[0]->output(0));
  if (promoted) {
    ASSERT_EQ(casts.size(), 2u);
    EXPECT_EQ(casts.front()->attribute(0).i(), TensorProto::FLOAT);
    EXPECT_EQ(casts.back()->attribute(0).i(), type);
    EXPECT_EQ(means[0]->input(0), casts.front()->output(0));
  }
  EXPECT_EQ(graph.output(0).name(), "Y");
  EXPECT_EQ(graph.output(0).type().tensor_type().elem_type(), type);
  EXPECT_EQ(graph.output(0).doc_string(), "normalized output");
  EXPECT_EQ(graph.output(0).type().tensor_type().shape().dim_size(), 4);
  EXPECT_NO_THROW(checker::check_model(converted));
}

TEST(onnx_version_converter, MvnExpandsBothDirectionsForAllFloatingTypes) {
  for (const auto type :
       {TensorProto::FLOAT, TensorProto::DOUBLE, TensorProto::FLOAT16, TensorProto::BFLOAT16}) {
    SCOPED_TRACE(type);
    for (const bool upgrade : {false, true}) {
      SCOPED_TRACE(upgrade);
      const auto original = MakeMvnModel(upgrade ? 28 : 29, type);
      const auto serialized = original.SerializeAsString();
      const auto converted = version_conversion::ConvertVersion(original, upgrade ? 29 : 28);
      EXPECT_EQ(original.SerializeAsString(), serialized);
      EXPECT_EQ(converted.opset_import(0).version(), upgrade ? 29 : 28);
      CheckMvnExpansion(converted, upgrade, type);
    }
  }
}

TEST(onnx_version_converter, MvnRejectsMalformedArityBeforeAccessingValues) {
  for (const bool upgrade : {false, true}) {
    for (const auto input_count : {0u, 1u, 2u}) {
      for (const auto output_count : {0u, 1u, 2u}) {
        if (input_count == 1 && output_count == 1)
          continue;
        auto graph = std::make_shared<Graph>();
        auto *node = graph->create(Symbol("MeanVarianceNormalization"), output_count);
        graph->appendNode(node);
        for (unsigned i = 0; i < input_count; ++i)
          node->addInput(graph->addInput()->setElemType(TensorProto::FLOAT));
        if (upgrade) {
          version_conversion::MeanVarianceNormalization_28_29 adapter;
          EXPECT_THROW(adapter.adapt(graph, node), std::runtime_error);
        } else {
          version_conversion::MeanVarianceNormalization_29_28 adapter;
          EXPECT_THROW(adapter.adapt(graph, node), std::runtime_error);
        }
      }
    }
  }
}

TEST(onnx_version_converter, MvnPreservesCustomAndEmptyAxesAndEpsilon) {
  for (const auto &axes : {std::vector<int64_t>{-1}, std::vector<int64_t>{}}) {
    for (const float epsilon : {0.0f, 0.125f}) {
      auto model = MakeMvnModel(29, TensorProto::DOUBLE);
      auto *node = model.mutable_graph()->mutable_node(0);
      auto *axes_attr = node->add_attribute();
      axes_attr->set_name("axes");
      axes_attr->set_type(AttributeProto::INTS);
      for (const auto axis : axes)
        axes_attr->add_ints(axis);
      auto *epsilon_attr = node->add_attribute();
      epsilon_attr->set_name("epsilon");
      epsilon_attr->set_type(AttributeProto::FLOAT);
      epsilon_attr->set_f(epsilon);
      CheckMvnExpansion(version_conversion::ConvertVersion(model, 28), false, TensorProto::DOUBLE,
                        epsilon, axes);
    }
  }
}

TEST(onnx_version_converter, MvnRejectsUnknownAndNonFloatingTypes) {
  for (const bool upgrade : {false, true}) {
    for (const auto type : {TensorProto::UNDEFINED, TensorProto::INT64}) {
      EXPECT_THROW(version_conversion::ConvertVersion(MakeMvnModel(upgrade ? 28 : 29, type),
                                                      upgrade ? 29 : 28),
                   std::runtime_error);
    }
  }
}

TEST(onnx_version_converter, MvnRejectsOmittedInputEvenWithAnnotatedOutput) {
  for (const bool upgrade : {false, true}) {
    auto model = MakeMvnModel(upgrade ? 28 : 29, TensorProto::FLOAT);
    *model.mutable_graph()->mutable_node(0)->mutable_input(0) = "";
    EXPECT_THROW(version_conversion::ConvertVersion(model, upgrade ? 29 : 28), std::runtime_error);
  }
}

TEST(onnx_version_converter, MvnUsesOutputTypeWhenCustomProducerTypeIsUnknown) {
  for (const bool upgrade : {false, true}) {
    auto model = MakeMvnModel(upgrade ? 28 : 29, TensorProto::FLOAT16);
    auto *graph = model.mutable_graph();
    graph->clear_input();
    auto mvn = graph->node(0);
    graph->clear_node();
    auto *custom = graph->add_node();
    custom->set_op_type("Unknown");
    custom->set_domain("custom");
    custom->add_output("X");
    graph->add_node()->CopyFrom(mvn);
    auto *opset = model.add_opset_import();
    opset->set_domain("custom");
    opset->set_version(1);
    const auto converted = version_conversion::ConvertVersion(model, upgrade ? 29 : 28);
    EXPECT_TRUE(MvnNodes(converted.graph(), "MeanVarianceNormalization").empty());
    EXPECT_EQ(MvnNodes(converted.graph(), "Cast").size(), upgrade ? 1u : 2u);
  }
}

TEST(onnx_version_converter, MvnConvertsAnnotatedFunctionBody) {
  auto model = MakeMvnModel(29, TensorProto::BFLOAT16);
  auto *function = model.add_functions();
  function->set_domain("local");
  function->set_name("Normalize");
  function->add_input("X");
  function->add_output("Y");
  function->add_opset_import()->set_version(29);
  function->add_node()->CopyFrom(model.graph().node(0));
  AddMvnTensor(function->add_value_info(), "X", TensorProto::BFLOAT16);
  AddMvnTensor(function->add_value_info(), "Y", TensorProto::BFLOAT16);
  auto *call = model.mutable_graph()->mutable_node(0);
  call->set_op_type("Normalize");
  call->set_domain("local");
  auto *opset = model.add_opset_import();
  opset->set_domain("local");
  opset->set_version(1);
  const auto converted = version_conversion::ConvertVersion(model, 28);
  ASSERT_EQ(converted.functions_size(), 1);
  EXPECT_EQ(converted.functions(0).opset_import(0).version(), 28);
  int casts = 0;
  for (const auto &node : converted.functions(0).node()) {
    EXPECT_NE(node.op_type(), "MeanVarianceNormalization");
    casts += node.op_type() == "Cast";
  }
  EXPECT_EQ(casts, 2);
}

void AddMvnBranch(NodeProto *node, const char *name, const std::string &input,
                  bool shadow_input = false) {
  auto *attribute = node->add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::GRAPH);
  auto *graph = attribute->mutable_g();
  graph->set_name(name);
  // The graph's output type must not supply the MVN input type.
  AddMvnTensor(graph->add_output(), "Z", TensorProto::FLOAT);
  if (shadow_input) {
    auto *constant = graph->add_node();
    constant->set_op_type("Constant");
    constant->add_output(input);
    auto *value = constant->add_attribute();
    value->set_name("value");
    value->set_type(AttributeProto::TENSOR);
    auto *tensor = value->mutable_t();
    tensor->set_data_type(TensorProto::FLOAT);
    for (int64_t dim : {2, 3, 4, 5})
      tensor->add_dims(dim);
    for (int i = 0; i < 120; ++i)
      tensor->add_float_data(1.0f);
  }
  auto *mvn = graph->add_node();
  mvn->set_op_type("MeanVarianceNormalization");
  mvn->add_input(input);
  mvn->add_output("normalized");
  auto *cast = graph->add_node();
  cast->set_op_type("Cast");
  cast->add_input("normalized");
  cast->add_output("Z");
  auto *to = cast->add_attribute();
  to->set_name("to");
  to->set_type(AttributeProto::INT);
  to->set_i(TensorProto::FLOAT);
}

TEST(onnx_version_converter, MvnInfersCapturedIntermediateWithoutChangingShadowedBindings) {
  for (const bool upgrade : {false, true}) {
    for (const std::string provider : {"Identity", "MeanVarianceNormalization"}) {
      SCOPED_TRACE(provider);
      auto model = MakeMvnModel(upgrade ? 28 : 29, TensorProto::DOUBLE);
      auto *graph = model.mutable_graph();
      graph->clear_node();
      auto *identity = graph->add_node();
      identity->set_op_type(provider);
      identity->add_input("X");
      identity->add_output("intermediate");
      AddMvnTensor(graph->add_input(), "condition", TensorProto::BOOL);
      graph->mutable_input(1)->mutable_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
      auto *branch = graph->add_node();
      branch->set_op_type("If");
      branch->add_input("condition");
      branch->add_output("Y");
      graph->mutable_output(0)->mutable_type()->mutable_tensor_type()->set_elem_type(
          TensorProto::FLOAT);
      AddMvnBranch(branch, "then_branch", "intermediate");
      AddMvnBranch(branch, "else_branch", "intermediate", true);
      const auto converted = version_conversion::ConvertVersion(model, upgrade ? 29 : 28);
      const auto ifs = MvnNodes(converted.graph(), "If");
      ASSERT_EQ(ifs.size(), 1u);
      const auto &converted_if = *ifs[0];
      ASSERT_EQ(converted_if.attribute_size(), 2);
      for (const auto &attribute : converted_if.attribute()) {
        const auto &body = attribute.g();
        EXPECT_TRUE(MvnNodes(body, "MeanVarianceNormalization").empty());
        EXPECT_EQ(MvnNodes(body, "Cast").size(), attribute.name() == "then_branch" ? 2u : 1u);
        const auto means = MvnNodes(body, "ReduceMean");
        ASSERT_EQ(means.size(), 2u);
        EXPECT_EQ(means[0]->input(0), "intermediate");
        EXPECT_NE(MvnProducer(body, "normalized"), nullptr);
      }
      // These deliberately shadowed bindings are accepted by the converter, not the checker.
      EXPECT_THROW(checker::check_model(model), checker::ValidationError);
    }
  }
}

} // namespace
