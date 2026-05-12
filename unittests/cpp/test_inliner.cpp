// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Translated from https://github.com/onnx/onnx/blob/main/onnx/test/cpp/inliner_test.cc
// Adapted for onnx_light (onnx-light project).

#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "onnx/checker.h"
#include "onnx/defs/function.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/inliner/inliner.h"
#include "onnx/shape_inference/implementation.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

static void InlineFunctions(ModelProto &model, const char *input,
                             const inliner::FunctionIdSet *to_inline = nullptr,
                             const ISchemaRegistry *schema_registry = nullptr) {
  OnnxParser parser(input);
  auto status = parser.Parse(model);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  checker::check_model(model, false, true);
  shape_inference::InferShapes(model);

  if (schema_registry != nullptr)
    inliner::InlineSelectedFunctions(model, *to_inline, schema_registry);
  else if (to_inline != nullptr)
    inliner::InlineSelectedFunctions(model, *to_inline);
  else
    inliner::InlineLocalFunctions(model, false);

  // The following ensures basic safety checks hold after inlining, including
  // absence of duplicate names (multiple assignments to same name).
  checker::check_model(model, true, true);
}

} // namespace

TEST(FunctionInliner, BasicTest) {
  const char *code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 10, "local" : 1 ]
>
agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N, 10] C)
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
  InlineFunctions(model, code);
  auto num_nodes = model.graph().node_size();
  ASSERT_EQ(num_nodes, 4);
  auto num_functions = model.functions_size();
  ASSERT_EQ(num_functions, 0);
}

// Test that inlining processes subgraphs.
TEST(FunctionInliner, SubgraphTest) {
  const char *code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 10, "local" : 1 ]
>
agraph (bool cond, float[N] X) => (float[N] Y)
{
  Y = If (cond) <
    then_branch = then_graph () => (y) {
        y = local.square (X)
    },
    else_branch = else_graph () => (y) {
        y = local.square (X)
    }
  >
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
  InlineFunctions(model, code);
  const auto &if_node = model.graph().node(0);
  const auto &graph1 = if_node.attribute(0).g();
  ASSERT_EQ(graph1.node(0).op_type(), "Mul");
  const auto &graph2 = if_node.attribute(1).g();
  ASSERT_EQ(graph2.node(0).op_type(), "Mul");
  auto num_functions = model.functions_size();
  ASSERT_EQ(num_functions, 0);
}

TEST(FunctionInliner, Nested) {
  const char *code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  Y = local.foo (X)
}

<opset_import: [ "" : 17, "local" : 1 ], domain: "local">
foo (x) => (y) {
  temp = Add(x, x)
  y = local.bar(temp)
}

<opset_import: [ "" : 17 ], domain: "local">
bar (x) => (y) {
  y = Mul (x, x)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);
  auto num_nodes = model.graph().node_size();
  ASSERT_EQ(num_nodes, 2);
  auto num_functions = model.functions_size();
  ASSERT_EQ(num_functions, 0);
}

TEST(FunctionInliner, Renaming) {
  const char *code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  temp = local.foo (X)
  temp__1 = Mul (temp, temp)
  Y = Abs (temp__1)
}

<opset_import: [ "" : 17, "local" : 1 ], domain: "local">
foo (x) => (y) {
  temp = Add(x, x)
  y = Neg (temp)
}
)ONNX";

  ModelProto model;
  // Check that renaming handles accidental collision of names: when "temp" in "foo" is
  // inlined, it will be renamed to something distinct from "temp" and "temp__1" as
  // both these names occur in the main graph.
  InlineFunctions(model, code);
}

TEST(FunctionInliner, ValueInfoPropagation) {
  const char *code = R"ONNX(
<ir_version: 10, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  result = local.foo (X)
  Y = Abs (result)
}

<opset_import: [ "" : 17, "local" : 1 ], domain: "local">
foo (x) => (y)
<float[N] temp> {
  temp = Add(x, x)
  y = Neg (temp)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);
  // Check that valueinfo is propagated from function to main graph.
  const auto &graph = model.graph();
  const auto &temp_new_name = graph.node(0).output(0);
  const auto &valueinfos = graph.value_info();
  for (const auto &valueinfo : valueinfos) {
    if (valueinfo.name() == temp_new_name) {
      ASSERT_TRUE(valueinfo.has_type());
      ASSERT_TRUE(valueinfo.type().has_tensor_type());
      ASSERT_TRUE(valueinfo.type().tensor_type().has_shape());
      ASSERT_EQ(valueinfo.type().tensor_type().shape().dim_size(), 1);
      return;
    }
  }
  ASSERT_TRUE(false) << "ValueInfo not found";
}

TEST(FunctionInliner, TwoCallsToSameFunction) {
  const char *code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  temp = local.foo (X)
  Y = local.foo (temp)
}

<opset_import: [ "" : 17, "local" : 1 ], domain: "local">
foo (x) => (y) {
  temp = Add(x, x)
  y = Neg (temp)
}
)ONNX";

  ModelProto model;
  // The call below will check that multiple assignments to the same name do not
  // happen after inlining two calls to the same function.
  InlineFunctions(model, code);
}

TEST(FunctionInliner, OpsetMismatch) {
  // foo uses opset 18 while the model uses opset 17 — no version converter
  // available in onnx_light so foo is NOT inlined; bar (opset 17) IS inlined.
  const char *code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  temp = local.foo (X)
  Y = local.bar (temp)
}

<opset_import: [ "" : 18], domain: "local">
foo (x) => (y) {
  y = Add(x, x)
}

<opset_import: [ "" : 17], domain: "local">
bar (x) => (y) {
  y = Add(x, x)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);

  // bar (opset 17, compatible with model) must be inlined — first node is Add.
  const auto &first_node = model.graph().node(0);
  ASSERT_EQ(first_node.op_type(), "local.foo"); // call-site kept as-is

  const auto &second_node = model.graph().node(1);
  ASSERT_EQ(second_node.op_type(), "Add"); // bar was inlined

  // foo is kept because it requires version conversion; bar is removed.
  ASSERT_EQ(model.functions_size(), 1);
  ASSERT_EQ(model.functions(0).name(), "foo");
}

TEST(FunctionInliner, SelectiveInlining) {
  const char *code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  temp = local.foo (X)
  Y = local.bar (temp)
}

<opset_import: [ "" : 17], domain: "local">
foo (x) => (y) {
  y = Add(x, x)
}

<opset_import: [ "" : 17, "local" : 1], domain: "local">
bar (x) => (y) {
  y = local.foo(x)
}
)ONNX";

  ModelProto model;
  inliner::FunctionIdVector to_inline = {{"local", "foo"}};
  auto to_inline_set = inliner::FunctionIdSet::Create(std::move(to_inline));
  InlineFunctions(model, code, to_inline_set.get());

  // The first node's call to foo must be inlined.
  const auto &first_node = model.graph().node(0);
  ASSERT_EQ(first_node.op_type(), "Add");

  // The second node's call to bar must not be inlined.
  const auto &second_node = model.graph().node(1);
  ASSERT_EQ(second_node.op_type(), "bar");

  // foo will be removed, bar will remain in model.functions().
  ASSERT_EQ(model.functions_size(), 1);

  const auto &bar_node = model.functions(0).node(0);
  // bar's body was also processed: the call to foo inside bar is inlined.
  ASSERT_EQ(bar_node.op_type(), "Add");
}

// Tests that require opset version conversion are skipped in onnx_light because
// the version converter depends on the full ONNX IR (ir.h / ir_pb_converter.h)
// which is not available in this lightweight build.

TEST(FunctionInliner, VersionConversion) {
  GTEST_SKIP() << "Version conversion requires the full ONNX IR (not available in onnx_light).";
}

TEST(FunctionInliner, NestedVersionConversion) {
  GTEST_SKIP() << "Version conversion requires the full ONNX IR (not available in onnx_light).";
}

static bool ContainsOp(const ModelProto &model, const char *op_type) {
  for (const auto &node : model.graph().node()) {
    if (node.op_type() == op_type) {
      return true;
    }
  }
  return false;
}

TEST(SchemaFunctionInliner, BasicTest) {
  const char *code = R"ONNX(
<ir_version: 8, opset_import: ["" : 18]>
agraph (float[N, 128] X) => (float[N, 128] Y)
{
  Y = Softmax (X)
}
)ONNX";

  ModelProto model;
  inliner::FunctionIdVector to_inline = {{"", "Softmax"}};
  auto to_inline_set = inliner::FunctionIdSet::Create(std::move(to_inline));
  InlineFunctions(model, code, to_inline_set.get(), OpSchemaRegistry::Instance());
  auto num_nodes = model.graph().node_size();
  ASSERT_GT(num_nodes, 1);
}

TEST(SchemaFunctionInliner, NestedTest) {
  const char *code = R"ONNX(
<ir_version: 8, opset_import: ["" : 18]>
agraph (float[N, C] X, int32[N] expected) => (float Y)
{
  Y, log_prob = SoftmaxCrossEntropyLoss (X, expected)
}
)ONNX";

  ModelProto model;
  inliner::FunctionIdVector to_inline = {{"", "SoftmaxCrossEntropyLoss"}};
  auto to_inline_set = inliner::FunctionIdSet::Create(std::move(to_inline));
  InlineFunctions(model, code, to_inline_set.get(), OpSchemaRegistry::Instance());
  auto num_nodes = model.graph().node_size();
  ASSERT_GT(num_nodes, 1);
  // Nested call to LogSoftmax should not be inlined.
  ASSERT_TRUE(ContainsOp(model, "LogSoftmax"));

  inliner::FunctionIdVector to_inline2 = {{"", "SoftmaxCrossEntropyLoss"}, {"", "LogSoftmax"}};
  to_inline_set = inliner::FunctionIdSet::Create(std::move(to_inline2));
  InlineFunctions(model, code, to_inline_set.get(), OpSchemaRegistry::Instance());
  num_nodes = model.graph().node_size();
  ASSERT_GT(num_nodes, 1);
  // Nested call to LogSoftmax should be inlined.
  ASSERT_FALSE(ContainsOp(model, "LogSoftmax"));
}

TEST(FunctionBuilder, AddInlinedCallBasic) {
  // Test the AddInlinedCall functionality.
  GraphProto graph;

  const char *graph_text = R"ONNX(
test_graph (float x) => (float y)
<float const_val = {2.0}>
{
    y = Add(x, const_val)
}
)ONNX";

  auto status = OnnxParser::Parse(graph, graph_text);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  FunctionProto function;
  FunctionBuilder builder(function);

  builder.AddInlinedCall({"result"}, graph, {"input_x"}, "test");

  // Verify the function has the expected structure.
  ASSERT_EQ(function.node_size(), 2); // One Constant node + one Add node

  // Check the first node is a Constant.
  ASSERT_EQ(function.node(0).op_type(), "Constant");
  ASSERT_EQ(function.node(0).output_size(), 1);
  ASSERT_TRUE(function.node(0).output(0).find("test") != std::string::npos);

  // Check the second node is an Add.
  ASSERT_EQ(function.node(1).op_type(), "Add");
  ASSERT_EQ(function.node(1).input_size(), 2);
  ASSERT_EQ(function.node(1).output_size(), 1);
  ASSERT_EQ(function.node(1).input(0), "input_x");  // renamed to actual input
  ASSERT_EQ(function.node(1).output(0), "result");  // renamed to actual output
}

TEST(Renamer, BasicFunctionality) {
  // Test the Renamer class functionality.
  GraphProto graph;

  auto *input = graph.add_input();
  input->set_name("input");

  inliner::Renamer renamer("test", graph);

  // Test binding names.
  renamer.BindName("formal_input", "actual_input");

  // Test creating a unique name and binding.
  std::string unique_name = renamer.BindToUniqueName("temp");
  ASSERT_TRUE(unique_name.find("test") != std::string::npos);

  // Test renaming a node.
  NodeProto node;
  node.set_op_type("Add");
  node.add_input("formal_input");
  node.add_output("temp_output");

  renamer.RenameNode(node);

  // Verify renaming worked correctly.
  ASSERT_EQ(node.input(0), "actual_input");                        // bound to actual name
  ASSERT_TRUE(node.output(0).find("test") != std::string::npos); // has prefix
}
