// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Translated from file onnx/test/cpp/inliner_test.cc for onnx-light.
// Skipped tests: VersionConversion, NestedVersionConversion (require version_converter),
//                SchemaFunctionInliner.* (require op schema registration at startup).

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "onnx/defs/function.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/inliner/inliner.h"
#include "gtest/gtest.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace Test {

// Helper: parse an ONNX text model, optionally inline functions, and return
// the result via the model out-parameter.
//
// Note: checker::check_model and shape_inference::InferShapes are omitted here
// because the corresponding source files are not yet compiled into onnx-light.
static void InlineFunctions(ModelProto &model, const char *input,
                            const inliner::FunctionIdSet *to_inline = nullptr,
                            const ISchemaRegistry *schema_registry = nullptr) {
  auto status = OnnxParser::Parse(model, input);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  if (schema_registry != nullptr)
    inliner::InlineSelectedFunctions(model, *to_inline, schema_registry);
  else if (to_inline != nullptr)
    inliner::InlineSelectedFunctions(model, *to_inline);
  else
    inliner::InlineLocalFunctions(model, true);
}

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
  auto num_nodes = model.ref_graph().ref_node().size();
  ASSERT_EQ(num_nodes, 4U);
  auto num_functions = model.ref_functions().size();
  ASSERT_EQ(num_functions, 0U);
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
  const auto &if_node = model.ref_graph().ref_node()[0];
  const auto &graph1 = if_node.ref_attribute()[0].ref_g();
  ASSERT_EQ(graph1.ref_node()[0].ref_op_type(), "Mul");
  const auto &graph2 = if_node.ref_attribute()[1].ref_g();
  ASSERT_EQ(graph2.ref_node()[0].ref_op_type(), "Mul");
  auto num_functions = model.ref_functions().size();
  ASSERT_EQ(num_functions, 0U);
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
  auto num_nodes = model.ref_graph().ref_node().size();
  ASSERT_EQ(num_nodes, 2U);
  auto num_functions = model.ref_functions().size();
  ASSERT_EQ(num_functions, 0U);
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
  // Check that renaming handles accidental collision of names: when "temp" in "foo"
  // is inlined, it will be renamed into something distinct from "temp" and "temp__1"
  // as both these names occur in the main graph.
  InlineFunctions(model, code);
  // Verify no duplicate output names exist after inlining.
  std::unordered_set<std::string> output_names;
  for (const auto &node : model.ref_graph().ref_node()) {
    for (const auto &out : node.ref_output()) {
      if (!out.empty()) {
        ASSERT_TRUE(output_names.insert(out.as_string()).second)
            << "Duplicate output name: " << out.as_string();
      }
    }
  }
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
  const auto &graph = model.ref_graph();
  const std::string temp_new_name = graph.ref_node()[0].ref_output()[0].as_string();
  const auto &valueinfos = graph.ref_value_info();
  for (const auto &valueinfo : valueinfos) {
    if (valueinfo.ref_name() == temp_new_name) {
      ASSERT_TRUE(valueinfo.has_type());
      ASSERT_TRUE(valueinfo.ref_type().has_tensor_type());
      ASSERT_TRUE(valueinfo.ref_type().ref_tensor_type().has_shape());
      ASSERT_EQ(valueinfo.ref_type().ref_tensor_type().ref_shape().ref_dim().size(), 1U);
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
  // The call below will check that multiple assignments to same name does not happen
  // after inlining two calls to same function.
  InlineFunctions(model, code);
  // Verify no duplicate output names exist after inlining.
  std::unordered_set<std::string> output_names;
  for (const auto &node : model.ref_graph().ref_node()) {
    for (const auto &out : node.ref_output()) {
      if (!out.empty()) {
        ASSERT_TRUE(output_names.insert(out.as_string()).second)
            << "Duplicate output name: " << out.as_string();
      }
    }
  }
}

TEST(FunctionInliner, OpsetMismatch) {
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

  // Both foo and bar must be inlined.
  // foo uses opset 18 which mismatches model's opset 17 for "", but InlineLocalFunctions
  // with convert_version=true will still inline it since it only needs standard ONNX domain.
  const auto &first_node = model.ref_graph().ref_node()[0];
  ASSERT_EQ(first_node.ref_op_type(), "Add");

  const auto &second_node = model.ref_graph().ref_node()[1];
  ASSERT_EQ(second_node.ref_op_type(), "Add");

  ASSERT_EQ(model.ref_functions().size(), 0U);
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

  // The first node's call, to foo, must be inlined.
  const auto &first_node = model.ref_graph().ref_node()[0];
  ASSERT_EQ(first_node.ref_op_type(), "Add");

  // The second node's call, to bar, must not be inlined.
  const auto &second_node = model.ref_graph().ref_node()[1];
  ASSERT_EQ(second_node.ref_op_type(), "bar");

  // foo will be removed, bar will remain, in model.functions()
  ASSERT_EQ(model.ref_functions().size(), 1U);

  const auto &bar_node = model.ref_functions()[0].ref_node()[0];
  // Check that it is a call to Add, due to inlining the call to foo in bar.
  ASSERT_EQ(bar_node.ref_op_type(), "Add");
}

TEST(FunctionBuilder, AddInlinedCallBasic) {
  // Test the AddInlinedCall functionality.
  GraphProto graph;

  // Create a simple graph using parser for better readability.
  const char *graph_text = R"ONNX(
test_graph (float x) => (float y)
<float const_val = {2.0}>
{
    y = Add(x, const_val)
}
)ONNX";

  auto status = OnnxParser::Parse(graph, graph_text);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  // Create a function and use AddInlinedCall.
  FunctionProto function;
  FunctionBuilder builder(function);

  builder.AddInlinedCall({"result"}, graph, {"input_x"}, "test");

  // Verify the function has the expected structure:
  // One Constant node (for const_val initializer) + one Add node.
  ASSERT_EQ(function.ref_node().size(), 2U);

  // Check the first node is a Constant.
  ASSERT_EQ(function.ref_node()[0].ref_op_type(), "Constant");
  ASSERT_EQ(function.ref_node()[0].ref_output().size(), 1U);
  ASSERT_NE(function.ref_node()[0].ref_output()[0].as_string().find("test"), std::string::npos);

  // Check the second node is an Add.
  ASSERT_EQ(function.ref_node()[1].ref_op_type(), "Add");
  ASSERT_EQ(function.ref_node()[1].ref_input().size(), 2U);
  ASSERT_EQ(function.ref_node()[1].ref_output().size(), 1U);
  ASSERT_EQ(function.ref_node()[1].ref_input()[0], "input_x");
  ASSERT_EQ(function.ref_node()[1].ref_output()[0], "result");
}

TEST(Renamer, BasicFunctionality) {
  // Test the Renamer class functionality.
  GraphProto graph;

  // Add input to graph.
  auto *input = graph.add_input();
  input->set_name("input");

  // Create a Renamer instance.
  inliner::Renamer renamer("test", graph);

  // Test binding names.
  renamer.BindName("formal_input", "actual_input");

  // Test creating unique names and binding.
  std::string unique_name = renamer.BindToUniqueName("temp");
  ASSERT_NE(unique_name.find("test"), std::string::npos);

  // Test renaming a node.
  NodeProto node;
  node.set_op_type("Add");
  *node.add_input() = "formal_input";
  *node.add_output() = "temp_output";

  renamer.RenameNode(node);

  // Verify renaming worked correctly.
  ASSERT_EQ(node.ref_input()[0], "actual_input");
  ASSERT_NE(node.ref_output()[0].as_string().find("test"), std::string::npos);
}

// Tests for GetUsedVars (ComputeInputs).

// A plain node's inputs are all direct input names.
TEST(ComputeInputs, SimpleNode) {
  NodeProto node;
  node.set_op_type("Add");
  *node.add_input() = "x";
  *node.add_input() = "y";
  *node.add_output() = "z";

  auto used = inliner::GetUsedVars(node);
  ASSERT_EQ(used.size(), 2U);
  EXPECT_EQ(used[0], "x");
  EXPECT_EQ(used[1], "y");
}

// Empty (optional) inputs are skipped.
TEST(ComputeInputs, EmptyInputsSkipped) {
  NodeProto node;
  node.set_op_type("Add");
  *node.add_input() = "x";
  *node.add_input() = ""; // optional input, absent
  *node.add_output() = "z";

  auto used = inliner::GetUsedVars(node);
  ASSERT_EQ(used.size(), 1U);
  EXPECT_EQ(used[0], "x");
}

// Node with no inputs returns an empty result.
TEST(ComputeInputs, NoInputs) {
  NodeProto node;
  node.set_op_type("Constant");
  *node.add_output() = "c";

  auto used = inliner::GetUsedVars(node);
  EXPECT_TRUE(used.empty());
}

// A node with a subgraph attribute (e.g. If): variables from the outer scope
// that are referenced inside the subgraph are included; variables defined
// locally inside the subgraph are not.
TEST(ComputeInputs, SubgraphImplicitInputs) {
  // Build an If node whose then-branch references outer var "X".
  const char *model_text = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17 ]>
agraph (bool cond, float[N] X) => (float[N] Y)
{
  Y = If (cond) <
    then_branch = then_graph () => (y) {
        y = Abs (X)
    },
    else_branch = else_graph () => (y) {
        y = Abs (X)
    }
  >
}
)ONNX";

  ModelProto model;
  auto status = OnnxParser::Parse(model, model_text);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  // The If node is the only node in the graph.
  const NodeProto &if_node = model.ref_graph().ref_node()[0];
  ASSERT_EQ(if_node.ref_op_type(), "If");

  auto used = inliner::GetUsedVars(if_node);

  // "cond" is the direct input; "X" is an implicit input referenced in the
  // subgraph bodies.
  EXPECT_NE(std::find(used.begin(), used.end(), "cond"), used.end());
  EXPECT_NE(std::find(used.begin(), used.end(), "X"), used.end());
}

// Variables produced (defined) inside a subgraph are NOT treated as inputs.
TEST(ComputeInputs, SubgraphLocalVarsNotIncluded) {
  // Build an If node whose branches use only locally-defined intermediates.
  const char *model_text = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17 ]>
agraph (bool cond, float[N] X) => (float[N] Y)
{
  Y = If (cond) <
    then_branch = then_graph () => (y) {
        tmp = Abs (X)
        y = Neg (tmp)
    },
    else_branch = else_graph () => (y) {
        tmp = Abs (X)
        y = Neg (tmp)
    }
  >
}
)ONNX";

  ModelProto model;
  auto status = OnnxParser::Parse(model, model_text);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  const NodeProto &if_node = model.ref_graph().ref_node()[0];
  auto used = inliner::GetUsedVars(if_node);

  // "tmp" is produced inside the subgraph, so it must not appear in used vars.
  EXPECT_EQ(std::find(used.begin(), used.end(), "tmp"), used.end());
  // "cond" and "X" are outer-scope; they must appear.
  EXPECT_NE(std::find(used.begin(), used.end(), "cond"), used.end());
  EXPECT_NE(std::find(used.begin(), used.end(), "X"), used.end());
}

} // namespace Test
} // namespace ONNX_LIGHT_NAMESPACE
