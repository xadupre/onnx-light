// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Translated from file onnx/test/cpp/function_context_test.cc
// and adapted to work with onnx-light.
//
// Key differences from vanilla ONNX:
//   - Schemas are NOT auto-registered at startup in onnx-light.
//   - Use ONNX_LIGHT_NAMESPACE instead of ONNX_LIGHT_NAMESPACE.
//   - FunctionBody() only accepts std::vector<NodeProto>, not ONNX text.
//     For VersionedFunctionBodyTest, nodes are built programmatically.
//   - BuildContextDependentFunction() returns void in onnx-light;
//     success is verified by checking the resulting FunctionProto node count.
//   - checker::check_function is not called because checker.cc is not compiled
//     into lib_onnx_cpp.  The tests instead verify the FunctionProto
//     structure directly (node count, op names, function inputs/outputs).

#include <cassert>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "onnx_lib/common/constants.h"
#include "onnx_lib/defs/function.h"
#include "onnx_lib/defs/schema.h"
#include "onnx_alias.h"

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

// Scalar-tensor helper – identical to the one in the original test.
static TensorProto ToTensor(double value, TensorProto_DataType elem_type) {
  TensorProto t;
  t.set_data_type(elem_type);
  switch (elem_type) {
  case TensorProto_DataType_FLOAT:
    t.add_float_data(static_cast<float>(value));
    break;
  case TensorProto_DataType_DOUBLE:
    t.add_double_data(value);
    break;
  default:
    assert(false);
  }
  return t;
}

// Mirrors the original BuildFunctionProto helper: populates nodes and then
// calls BuildFunction() to set name/domain/inputs/outputs/opset_import.
static bool LocalBuildFunctionProto(FunctionProto &functionProto, const OpSchema &schema,
                                    const std::vector<FunctionBodyHelper::NodeDef> &node_defs) {
  FunctionBodyHelper::BuildNodes(functionProto, node_defs);
  schema.BuildFunction(functionProto);
  return true;
}

// ---------------------------------------------------------------------------
// Context-dependent function builders
// ---------------------------------------------------------------------------

// Monomorphic float builder – same logic as original BuildFloatFunctionBody.
static bool BuildFloatFunctionBody(const FunctionBodyBuildContext & /*ctx*/, const OpSchema &schema,
                                   FunctionProto &functionProto) {
  auto two_as_tensor = ToTensor(2.0, TensorProto_DataType_FLOAT);
  std::vector<FunctionBodyHelper::NodeDef> body{
      {{"Two"}, "Constant", {}, {{"value", two_as_tensor}}}, {{"Y"}, "Mul", {"X", "Two"}}};
  return LocalBuildFunctionProto(functionProto, schema, body);
}

// Polymorphic builder – same logic as original BuildFunctionBody.
static bool BuildPolymorphicFunctionBody(const FunctionBodyBuildContext &ctx,
                                         const OpSchema &schema, FunctionProto &functionProto) {
  const auto *const tp = ctx.getInputType(0);
  if ((tp == nullptr) || (!tp->has_tensor_type()))
    return false;
  auto elem_type = tp->ref_tensor_type().ref_elem_type();
  auto two_as_tensor = ToTensor(2.0, elem_type);
  std::vector<FunctionBodyHelper::NodeDef> body{
      {{"Two"}, "Constant", {}, {{"value", two_as_tensor}}}, {{"Y"}, "Mul", {"X", "Two"}}};
  return LocalBuildFunctionProto(functionProto, schema, body);
}

// ---------------------------------------------------------------------------
// Schema registration helpers
// ---------------------------------------------------------------------------

void RegisterCustomFuncFloatSchema() {
  OpSchema schema;
  schema.SetName("CustomFuncFloat")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(12)
      .SetDoc("This operator returns an output tensor that is twice the input tensor.")
      .Input(0, "X", "Input tensor", "T", OpSchema::Single)
      .Output(0, "Y", "Output tensor", "T", OpSchema::Single)
      .TypeConstraint("T", {"tensor(float)"}, "Type of the input and output values")
      .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
      .SetContextDependentFunctionBodyBuilder(BuildFloatFunctionBody);
  RegisterSchema(std::move(schema), 0, /*fail_duplicate_schema=*/false);
}

void RegisterCustomFunctionSchema() {
  OpSchema schema;
  schema.SetName("CustomFunction")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(12)
      .SetDoc("This operator returns an output tensor that is twice the input tensor.")
      .Input(0, "X", "Input tensor", "T", OpSchema::Single)
      .Output(0, "Y", "Output tensor", "T", OpSchema::Single)
      .TypeConstraint("T", {"tensor(float)", "tensor(double)"},
                      "Type of the input and output values")
      .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
      .SetContextDependentFunctionBodyBuilder(BuildPolymorphicFunctionBody);
  RegisterSchema(std::move(schema), 0, /*fail_duplicate_schema=*/false);
}

// Build a single-node FunctionProto body for "Z = Sub(X, Y)".
// In onnx-light, FunctionBody() only accepts std::vector<NodeProto> (not
// ONNX text), so the Sub node is constructed programmatically.
static NodeProto MakeSubNode() {
  NodeProto n;
  n.set_op_type("Sub");
  *n.add_input() = "X";
  *n.add_input() = "Y";
  *n.add_output() = "Z";
  return n;
}

class GeluFunctionBodyBuildContext final : public FunctionBodyBuildContext {
public:
  GeluFunctionBodyBuildContext(const NodeProto &node_proto,
                               const AttributeProto *approximate_attribute)
      : node_proto_(node_proto), approximate_attribute_(approximate_attribute) {}

  const AttributeProto *getAttribute(const std::string &name) const override {
    if (name == "approximate") {
      return approximate_attribute_;
    }
    return nullptr;
  }

  bool hasInput(int inputIndex) const override {
    if (inputIndex >= node_proto_.input_size()) {
      return false;
    }
    return !node_proto_.input(inputIndex).empty();
  }

  bool hasOutput(int outputIndex) const override {
    if (outputIndex >= node_proto_.output_size()) {
      return false;
    }
    return !node_proto_.output(outputIndex).empty();
  }

  const TypeProto *getInputType(int inputIndex) const override {
    // Gelu's builder only branches on the "approximate" attribute in this test.
    (void)inputIndex;
    return nullptr;
  }

private:
  const NodeProto &node_proto_;
  const AttributeProto *approximate_attribute_;
};

void RegisterMySubSchemas() {
  // MySub at sinceVersion 2: one function body at opset 2.
  OpSchema schema_ver2;
  schema_ver2.SetName("MySub")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(2)
      .SetDoc("Z = Sub (X, Y)")
      .Input(0, "X", "Input tensor X", "T", OpSchema::Single)
      .Input(1, "Y", "Input tensor Y", "T", OpSchema::Single)
      .Output(0, "Z", "Output tensor Z", "T", OpSchema::Single)
      .TypeConstraint("T", {"tensor(float)", "tensor(double)"},
                      "Type of the input and output values")
      .FunctionBody({MakeSubNode()}, 2);
  RegisterSchema(std::move(schema_ver2), 0, /*fail_duplicate_schema=*/false);

  // MySub at sinceVersion 9: function bodies at opsets 9 and 16.
  OpSchema schema_ver9;
  schema_ver9.SetName("MySub")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(9)
      .SetDoc("Z = Sub (X, Y)")
      .Input(0, "X", "Input tensor X", "T", OpSchema::Single)
      .Input(1, "Y", "Input tensor Y", "T", OpSchema::Single)
      .Output(0, "Z", "Output tensor Z", "T", OpSchema::Single)
      .TypeConstraint("T", {"tensor(float)", "tensor(double)"},
                      "Type of the input and output values")
      .FunctionBody({MakeSubNode()}, 9)
      .FunctionBody({MakeSubNode()}, 16);
  RegisterSchema(std::move(schema_ver9), 0, /*fail_duplicate_schema=*/false);
}

} // namespace

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
class FunctionContextTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    RegisterCustomFuncFloatSchema();
    RegisterCustomFunctionSchema();
    RegisterMySubSchemas();
  }

  static void TearDownTestSuite() {
    DeregisterSchema("CustomFuncFloat", 12, ONNX_DOMAIN);
    DeregisterSchema("CustomFunction", 12, ONNX_DOMAIN);
    DeregisterSchema("MySub", 2, ONNX_DOMAIN);
    DeregisterSchema("MySub", 9, ONNX_DOMAIN);
  }
};

// ---------------------------------------------------------------------------
// Test: ContextDependentFunctionTest
//
// Original: verifies a monomorphic (float-only) context-dependent function.
// Adapted:  BuildContextDependentFunction() returns void in onnx-light, so
//           success is inferred from the resulting FunctionProto node count.
//           checker::check_function is not called because checker.cc is not
//           compiled into lib_onnx_cpp; the FunctionProto structure is
//           verified directly instead.
// ---------------------------------------------------------------------------
TEST_F(FunctionContextTest, ContextDependentFunctionTest) {
  const auto *const schema = OpSchemaRegistry::Schema("CustomFuncFloat", 12, ONNX_DOMAIN);
  EXPECT_TRUE(schema);
  EXPECT_FALSE(schema->HasFunction());
  EXPECT_TRUE(schema->HasContextDependentFunction());

  NodeProto nodeProto;
  nodeProto.set_op_type("CustomFuncFloat");
  *nodeProto.add_input() = "X";
  *nodeProto.add_output() = "Y";

  FunctionBodyBuildContextImpl ctx(nodeProto);
  FunctionProto fnProto;
  // In onnx-light, BuildContextDependentFunction returns void.
  schema->BuildContextDependentFunction(ctx, fnProto);
  EXPECT_EQ(static_cast<int>(fnProto.ref_node().size()), 2);
  // Verify the function proto has correct metadata (set by BuildFunction).
  EXPECT_EQ(fnProto.ref_name().as_string(), "CustomFuncFloat");
  EXPECT_EQ(static_cast<int>(fnProto.ref_input().size()), 1);
  EXPECT_EQ(fnProto.ref_input()[0], "X");
  EXPECT_EQ(static_cast<int>(fnProto.ref_output().size()), 1);
  EXPECT_EQ(fnProto.ref_output()[0], "Y");
  // Verify that the nodes are the Constant and Mul ops.
  EXPECT_EQ(fnProto.ref_node()[0].ref_op_type().as_string(), "Constant");
  EXPECT_EQ(fnProto.ref_node()[1].ref_op_type().as_string(), "Mul");
}

// ---------------------------------------------------------------------------
// Test: VersionedFunctionBodyTest
//
// Original: illustrates versioning issues with function-op bodies.  It also
//           validates function bodies against current op schemas using
//           GetFunction(opset, validate=true), which can return nullptr or
//           throw when the body references an outdated op version.
//
// Adapted:  onnx-light ignores the `validate` parameter of GetFunction, so
//           the validation branch is not tested.  The test instead verifies
//           the version-lookup semantics: GetFunction(v) returns the function
//           body registered at the largest opset ≤ v.
// ---------------------------------------------------------------------------
TEST_F(FunctionContextTest, VersionedFunctionBodyTest) {
  // schema2: MySub sinceVersion 2, one body at opset 2.
  const auto *const schema2 = OpSchemaRegistry::Schema("MySub", 2, ONNX_DOMAIN);
  EXPECT_TRUE(schema2);

  // Requesting version 1 is before sinceVersion – no function expected.
  EXPECT_EQ(schema2->GetFunction(1), nullptr);

  // Versions 2–8 should all resolve to the v2 function body.
  for (int v = 2; v < 9; ++v) {
    const FunctionProto *function = schema2->GetFunction(v);
    ASSERT_TRUE(function) << "Expected function body for MySub v2 at opset " << v;
    EXPECT_EQ(static_cast<int>(function->ref_node().size()), 1);
    EXPECT_EQ(function->ref_node()[0].ref_op_type().as_string(), "Sub");
  }

  // schema9: MySub sinceVersion 9, bodies at opsets 9 and 16.
  const auto *const schema9 = OpSchemaRegistry::Schema("MySub", 9, ONNX_DOMAIN);
  EXPECT_TRUE(schema9);

  // Versions 9–15 resolve to the opset-9 body.
  for (int v = 9; v < 16; ++v) {
    const FunctionProto *function = schema9->GetFunction(v);
    ASSERT_TRUE(function) << "Expected function body for MySub v9 at opset " << v;
    EXPECT_EQ(static_cast<int>(function->ref_node().size()), 1);
    EXPECT_EQ(function->ref_node()[0].ref_op_type().as_string(), "Sub");
  }

  // Version 16 and above resolve to the opset-16 body.
  {
    const FunctionProto *function = schema9->GetFunction(16);
    ASSERT_TRUE(function);
    EXPECT_EQ(static_cast<int>(function->ref_node().size()), 1);
  }
}

// ---------------------------------------------------------------------------
// Test: TypeContextTest
//
// Original: verifies a polymorphic context-dependent function that
//           specialises based on the input element type.
// Adapted:  same as ContextDependentFunctionTest – BuildContextDependentFunction
//           returns void in onnx-light; FunctionProto structure is verified
//           directly instead of calling checker::check_function.
// ---------------------------------------------------------------------------
TEST_F(FunctionContextTest, TypeContextTest) {
  const auto *const schema = OpSchemaRegistry::Schema("CustomFunction", 12, ONNX_DOMAIN);
  EXPECT_TRUE(schema);
  EXPECT_FALSE(schema->HasFunction());
  EXPECT_TRUE(schema->HasContextDependentFunction());

  NodeProto nodeProto;
  nodeProto.set_op_type("CustomFunction");
  *nodeProto.add_input() = "X";
  *nodeProto.add_output() = "Y";

  TypeProto floatTypeProto;
  floatTypeProto.ref_tensor_type().set_elem_type(TensorProto::DataType::FLOAT);

  FunctionBodyBuildContextImpl ctx(nodeProto, {floatTypeProto});
  FunctionProto fnProto;
  // In onnx-light, BuildContextDependentFunction returns void.
  schema->BuildContextDependentFunction(ctx, fnProto);
  EXPECT_EQ(static_cast<int>(fnProto.ref_node().size()), 2);
  // Verify the function proto has correct metadata (set by BuildFunction).
  EXPECT_EQ(fnProto.ref_name().as_string(), "CustomFunction");
  EXPECT_EQ(static_cast<int>(fnProto.ref_input().size()), 1);
  EXPECT_EQ(fnProto.ref_input()[0], "X");
  EXPECT_EQ(static_cast<int>(fnProto.ref_output().size()), 1);
  EXPECT_EQ(fnProto.ref_output()[0], "Y");
  // The Constant node should hold a float tensor (type of X).
  const auto &constant_node = fnProto.ref_node()[0];
  EXPECT_EQ(constant_node.ref_op_type().as_string(), "Constant");
  ASSERT_EQ(static_cast<int>(constant_node.ref_attribute().size()), 1);
  EXPECT_EQ(constant_node.ref_attribute()[0].ref_t().ref_data_type(), TensorProto::DataType::FLOAT);
}

TEST_F(FunctionContextTest, BuildContextDependentFunctionBodyGeluTest) {
  auto has_node_with_op_type = [](const FunctionProto &function_proto, const std::string &op_type) {
    for (const auto &node : function_proto.ref_node()) {
      if (node.ref_op_type().as_string() == op_type) {
        return true;
      }
    }
    return false;
  };

  const auto *const schema = OpSchemaRegistry::Schema("Gelu", 20, ONNX_DOMAIN);
  ASSERT_TRUE(schema);
  EXPECT_TRUE(schema->HasContextDependentFunction());

  NodeProto node_proto_default;
  node_proto_default.set_op_type("Gelu");
  *node_proto_default.add_input() = "X";
  *node_proto_default.add_output() = "Y";

  GeluFunctionBodyBuildContext default_ctx(node_proto_default, nullptr);
  FunctionProto default_fn_proto;
  schema->BuildContextDependentFunction(default_ctx, default_fn_proto);
  EXPECT_EQ(default_fn_proto.ref_name().as_string(), "Gelu");
  EXPECT_TRUE(has_node_with_op_type(default_fn_proto, "Erf"));
  EXPECT_FALSE(has_node_with_op_type(default_fn_proto, "Tanh"));

  NodeProto node_proto_tanh;
  node_proto_tanh.set_op_type("Gelu");
  *node_proto_tanh.add_input() = "X";
  *node_proto_tanh.add_output() = "Y";
  AttributeProto approximate_attr;
  approximate_attr.set_name("approximate");
  approximate_attr.set_type(AttributeProto::AttributeType::STRING);
  approximate_attr.set_s("tanh");

  GeluFunctionBodyBuildContext tanh_ctx(node_proto_tanh, &approximate_attr);
  FunctionProto tanh_fn_proto;
  schema->BuildContextDependentFunction(tanh_ctx, tanh_fn_proto);
  EXPECT_EQ(tanh_fn_proto.ref_name().as_string(), "Gelu");
  EXPECT_TRUE(has_node_with_op_type(tanh_fn_proto, "Tanh"));
  EXPECT_FALSE(has_node_with_op_type(tanh_fn_proto, "Erf"));
}

} // namespace Test
