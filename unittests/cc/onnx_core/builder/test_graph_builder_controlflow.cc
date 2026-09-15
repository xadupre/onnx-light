// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder.h"
#include "onnx_helper.h"
#include "onnx_op/operator_sets.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

core::builder::GraphBuilder::SchemaLookupFn SchemaLookup() {
  return [](const std::string &op_type) {
    return onnx_op::GetAllOnnxOpSchemasWithHistory(op_type, /*init_doc=*/false);
  };
}

core::symbolic::SymShape MakeShape(std::initializer_list<int64_t> dims) {
  core::symbolic::SymShape shape;
  for (int64_t dim : dims) {
    shape.PushBack(core::symbolic::SymDim(dim));
  }
  return shape;
}

void AddGraphAttribute(NodeProto &node, const char *name, const GraphProto &graph) {
  auto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::GRAPH);
  attribute->set_g(graph);
}

GraphProto MakeScanBody(bool referenced_attribute = false) {
  GraphProto body;
  body.set_name("scan_body");
  body.add_input()->set_name("state");
  body.add_input()->set_name("item");
  AddNode(body, "Add", {"state", "item"}, {"next_state"});
  auto &scanned =
      AddNode(body, referenced_attribute ? "Cast" : "Identity", {"next_state"}, {"scan_item"});
  if (referenced_attribute) {
    auto *attribute = scanned.add_attribute();
    attribute->set_name("to");
    attribute->set_type(AttributeProto::AttributeType::INT);
    attribute->set_ref_attr_name("dtype");
  }
  body.add_output()->set_name("next_state");
  body.add_output()->set_name("scan_item");
  return body;
}

void ExpectTensorRoundTrip(core::builder::GraphBuilder &builder,
                           const std::vector<std::string> &outputs) {
  for (const auto &output : outputs) {
    builder.MakeOutput(output);
  }
  const ModelProto model = builder.ToModel();
  core::builder::GraphBuilder imported(model, SchemaLookup());
  const ModelProto exported = imported.ToModel();
  ASSERT_EQ(exported.graph().output().size(), outputs.size());
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto &name = outputs[i];
    SCOPED_TRACE(name);
    EXPECT_EQ(exported.graph().output(i).name(), name);
    ASSERT_TRUE(imported.HasShape(name));
    EXPECT_EQ(imported.GetShape(name).Dtype(), builder.GetShape(name).Dtype());
    EXPECT_EQ(imported.GetShape(name).Shape(), builder.GetShape(name).Shape());
  }
}

} // namespace

TEST(GraphBuilderControlflow, IfBuildsAndImportsMultipleOutputs) {
  for (int opset : {1, 11, 13, 22}) {
    SCOPED_TRACE(opset);
    core::builder::GraphBuilder builder("if_graph", SchemaLookup());
    builder.SetOpsetVersion("", opset);
    builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
    builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2, 3}));
    NodeProto node = MakeNode("If", {"condition"}, {"value", "dimensions"});
    for (const char *branch_name : {"then_branch", "else_branch"}) {
      GraphProto branch;
      branch.set_name(branch_name);
      AddNode(branch, "Identity", {"x"}, {"branch_value"});
      AddNode(branch, "Shape", {"x"}, {"branch_shape"});
      branch.add_output()->set_name("branch_value");
      branch.add_output()->set_name("branch_shape");
      AddGraphAttribute(node, branch_name, branch);
    }
    EXPECT_EQ(
        builder.MakeNode("If", {"condition"}, {"value", "dimensions"}, "", "", node.attribute()),
        (std::vector<std::string>{"value", "dimensions"}));
    EXPECT_EQ(builder.GetShape("value").Shape(), MakeShape({2, 3}));
    EXPECT_EQ(builder.GetShape("value").Dtype(), core::symbolic::TensorType::kFloat);
    EXPECT_EQ(builder.GetShape("dimensions").Shape(), MakeShape({2}));
    EXPECT_EQ(builder.GetShape("dimensions").Dtype(), core::symbolic::TensorType::kInt64);
    ExpectTensorRoundTrip(builder, {"value", "dimensions"});
  }
}

TEST(GraphBuilderControlflow, LoopBuildsAndImportsCarriedAndScanOutputs) {
  for (int opset : {1, 11, 13, 22}) {
    SCOPED_TRACE(opset);
    core::builder::GraphBuilder builder("loop_graph", SchemaLookup());
    builder.SetOpsetVersion("", opset);
    builder.MakeInput("trip_count", core::symbolic::TensorType::kInt64, MakeShape({}));
    builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
    builder.MakeInput("initial", core::symbolic::TensorType::kFloat, MakeShape({2}));
    GraphProto body;
    body.set_name("loop_body");
    for (const char *name : {"iteration", "keep_going", "state"}) {
      body.add_input()->set_name(name);
    }
    AddNode(body, "Identity", {"keep_going"}, {"next_condition"});
    AddNode(body, "Neg", {"state"}, {"next_state"});
    AddNode(body, "Identity", {"next_state"}, {"scan_item"});
    for (const char *name : {"next_condition", "next_state", "scan_item"}) {
      body.add_output()->set_name(name);
    }
    NodeProto node = MakeNode("Loop", {"trip_count", "condition", "initial"}, {"final", "scanned"});
    AddGraphAttribute(node, "body", body);
    builder.MakeNode("Loop", {"trip_count", "condition", "initial"}, {"final", "scanned"}, "", "",
                     node.attribute());
    EXPECT_EQ(builder.GetShape("final").Shape(), MakeShape({2}));
    EXPECT_EQ(builder.GetShape("final").Dtype(), core::symbolic::TensorType::kFloat);
    ASSERT_EQ(builder.GetShape("scanned").Shape().Rank(), 2u);
    EXPECT_EQ(builder.GetShape("scanned").Shape()[1], core::symbolic::SymDim(2));
    EXPECT_EQ(builder.GetShape("scanned").Dtype(), core::symbolic::TensorType::kFloat);
    ExpectTensorRoundTrip(builder, {"final", "scanned"});
  }
}

TEST(GraphBuilderControlflow, ScanBuildsAndImportsStateAndScanOutputs) {
  for (int opset : {8, 9, 11, 22}) {
    SCOPED_TRACE(opset);
    core::builder::GraphBuilder builder("scan_graph", SchemaLookup());
    builder.SetOpsetVersion("", opset);
    const auto state_shape = opset == 8 ? MakeShape({4, 2}) : MakeShape({2});
    const auto scan_shape = opset == 8 ? MakeShape({4, 3, 2}) : MakeShape({3, 2});
    builder.MakeInput("initial", core::symbolic::TensorType::kFloat, state_shape);
    builder.MakeInput("items", core::symbolic::TensorType::kFloat, scan_shape);
    const std::vector<std::string> inputs = opset == 8
                                                ? std::vector<std::string>{"", "initial", "items"}
                                                : std::vector<std::string>{"initial", "items"};
    NodeProto node = MakeNode("Scan", inputs, {"final", "scanned"});
    AddGraphAttribute(node, "body", MakeScanBody());
    AddAttribute<int64_t>(node, "num_scan_inputs", 1);
    builder.MakeNode("Scan", inputs, {"final", "scanned"}, "", "", node.attribute());
    EXPECT_EQ(builder.GetShape("final").Shape(), state_shape);
    EXPECT_EQ(builder.GetShape("scanned").Shape(), scan_shape);
    EXPECT_EQ(builder.GetShape("final").Dtype(), core::symbolic::TensorType::kFloat);
    EXPECT_EQ(builder.GetShape("scanned").Dtype(), core::symbolic::TensorType::kFloat);
    ExpectTensorRoundTrip(builder, {"final", "scanned"});
  }
}

TEST(GraphBuilderControlflow, SequenceMapBuildsAndImportsMultipleOutputTypes) {
  core::builder::GraphBuilder builder("sequence_graph", SchemaLookup());
  builder.SetOpsetVersion("", 22);
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({}));
  builder.MakeNode("SequenceConstruct", {"x"}, {"sequence"});
  GraphProto body;
  body.set_name("map_body");
  body.add_input()->set_name("item");
  AddNode(body, "Identity", {"item"}, {"body_value"});
  auto &cast = AddNode(body, "Cast", {"item"}, {"body_integer"});
  AddAttribute<int64_t>(cast, "to", TensorProto::DataType::INT64);
  body.add_output()->set_name("body_value");
  body.add_output()->set_name("body_integer");
  NodeProto node = MakeNode("SequenceMap", {"sequence"}, {"values", "integers"});
  AddGraphAttribute(node, "body", body);
  builder.MakeNode("SequenceMap", {"sequence"}, {"values", "integers"}, "", "", node.attribute());
  builder.MakeInitializer(MakeInitializer<int64_t>("index", {}, {0}));
  builder.MakeNode("SequenceAt", {"values", "index"}, {"value"});
  builder.MakeNode("SequenceAt", {"integers", "index"}, {"integer"});
  EXPECT_EQ(builder.GetShape("value").Dtype(), core::symbolic::TensorType::kFloat);
  EXPECT_EQ(builder.GetShape("integer").Dtype(), core::symbolic::TensorType::kInt64);
  ExpectTensorRoundTrip(builder, {"value", "integer"});
}

TEST(GraphBuilderControlflow, ImportsScanInLocalFunctionWithReferencedBodyAttribute) {
  for (const bool referenced_attribute : {false, true}) {
    SCOPED_TRACE(referenced_attribute);
    core::builder::GraphBuilder seed("function_graph", SchemaLookup());
    seed.SetOpsetVersion("", 22);
    seed.SetOpsetVersion("local", 1);
    seed.MakeInput("initial", core::symbolic::TensorType::kFloat, MakeShape({2}));
    seed.MakeInput("items", core::symbolic::TensorType::kFloat, MakeShape({3, 2}));
    ModelProto model = seed.ToModel();
    FunctionProto function;
    function.set_name("ScanFunction");
    function.set_domain("local");
    function.add_opset("", 22);
    AddInputs(function, {"a", "b"});
    AddOutputs(function, {"final_state", "scan_result"});
    if (referenced_attribute) {
      function.add_attribute("dtype");
    }
    NodeProto scan = MakeNode("Scan", {"a", "b"}, {"final_state", "scan_result"});
    AddGraphAttribute(scan, "body", MakeScanBody(referenced_attribute));
    AddAttribute<int64_t>(scan, "num_scan_inputs", 1);
    function.add_node(scan);
    model.add_function(function);
    NodeProto call = MakeNode("ScanFunction", {"initial", "items"}, {"final", "scanned"}, "local");
    if (referenced_attribute) {
      AddAttribute<int64_t>(call, "dtype", TensorProto::DataType::DOUBLE);
    }
    model.mutable_graph()->add_node(call);
    core::builder::GraphBuilder builder(model, SchemaLookup());
    EXPECT_EQ(builder.GetShape("final").Shape(), MakeShape({2}));
    EXPECT_EQ(builder.GetShape("final").Dtype(), core::symbolic::TensorType::kFloat);
    EXPECT_EQ(builder.GetShape("scanned").Shape(), MakeShape({3, 2}));
    EXPECT_EQ(builder.GetShape("scanned").Dtype(), referenced_attribute
                                                       ? core::symbolic::TensorType::kDouble
                                                       : core::symbolic::TensorType::kFloat);
    ExpectTensorRoundTrip(builder, {"final", "scanned"});
  }
}

TEST(GraphBuilderControlflow, RejectsIfOutputCountDifferentFromItsBranches) {
  core::builder::GraphBuilder builder("invalid_if", SchemaLookup());
  builder.SetOpsetVersion("", 22);
  builder.MakeInput("condition", core::symbolic::TensorType::kBool, MakeShape({}));
  builder.MakeInput("x", core::symbolic::TensorType::kFloat, MakeShape({2}));
  GraphProto branch;
  branch.set_name("branch");
  AddNode(branch, "Identity", {"x"}, {"branch_result"});
  branch.add_output()->set_name("branch_result");
  NodeProto node = MakeNode("If", {"condition"}, {"first", "second"});
  AddGraphAttribute(node, "then_branch", branch);
  AddGraphAttribute(node, "else_branch", branch);
  EXPECT_THROW(builder.MakeNode("If", {"condition"}, {"first", "second"}, "", "", node.attribute()),
               std::invalid_argument);
}

} // namespace Test
