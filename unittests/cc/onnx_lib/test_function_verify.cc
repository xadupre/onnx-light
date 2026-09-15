// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Translated from file onnx/test/cpp/function_verify_test.cc
// and adapted to work with onnx-light.

#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "onnx_lib/checker.h"
#include "onnx_lib/common/constants.h"
#include "onnx_lib/defs/function.h"
#include "onnx_lib/defs/schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace Test {
namespace {

constexpr const char *kFunctionVerifyDomain = "onnx.light.test.function_verify";
constexpr const char *kFunctionVerifyOp = "DefaultAttrFunction";
constexpr int kFunctionVerifyOpset = 1;

void RegisterDefaultAttrFunctionSchema() {
  OpSchema schema;
  schema.SetName(kFunctionVerifyOp)
      .SetDomain(kFunctionVerifyDomain)
      .SinceVersion(kFunctionVerifyOpset)
      .SetDoc("Function used to verify FunctionExpandHelper default attributes.")
      .Input(0, "X", "Input tensor", "T")
      .Output(0, "Y", "Output tensor", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain to float tensor.")
      .Attr("axes", "Default axes propagated to function-body node.", AttributeProto::INTS,
            std::vector<int64_t>{1, 2})
      .FunctionBody(
          FunctionBodyHelper::BuildNodes({{{"Y"},
                                           "InnerReduceMean",
                                           {"X"},
                                           {MakeRefAttribute("axes", AttributeProto::INTS)},
                                           "onnx.light.test.inner"}}),
          kFunctionVerifyOpset);
  RegisterSchema(std::move(schema), 0, /*fail_duplicate_schema=*/false);
}

void DeregisterDefaultAttrFunctionSchema() {
  DeregisterSchema(kFunctionVerifyOp, kFunctionVerifyOpset, kFunctionVerifyDomain);
}

} // namespace

TEST(FunctionVerification, VerifyFunctionExpandHelper) {
  RegisterDefaultAttrFunctionSchema();

  const auto *const schema =
      OpSchemaRegistry::Schema(kFunctionVerifyOp, kFunctionVerifyOpset, kFunctionVerifyDomain);
  ASSERT_NE(nullptr, schema);
  ASSERT_TRUE(schema->HasFunction());

  const FunctionProto *func = schema->GetFunction(kFunctionVerifyOpset);
  ASSERT_NE(nullptr, func);

  GraphProto graph;
  NodeProto function_node;
  function_node.set_domain(kFunctionVerifyDomain);
  function_node.set_op_type(kFunctionVerifyOp);
  *function_node.add_input() = "x";
  *function_node.add_output() = "y";

  FunctionExpandHelper(function_node, *func, graph);
  ASSERT_EQ(graph.ref_node().size(), 1U);
  const auto &expanded_node = graph.ref_node()[0];
  ASSERT_EQ(expanded_node.ref_attribute().size(), 1U);
  EXPECT_EQ(expanded_node.ref_attribute()[0].ref_name(), "axes");
  ASSERT_EQ(expanded_node.ref_attribute()[0].ref_ints().size(), 2U);
  EXPECT_EQ(expanded_node.ref_attribute()[0].ref_ints()[0], 1);
  EXPECT_EQ(expanded_node.ref_attribute()[0].ref_ints()[1], 2);

  DeregisterDefaultAttrFunctionSchema();
}

TEST(FunctionVerification, VerifyFunctionExpandHelperMissingSchema) {
  RegisterDefaultAttrFunctionSchema();

  const auto *const schema =
      OpSchemaRegistry::Schema(kFunctionVerifyOp, kFunctionVerifyOpset, kFunctionVerifyDomain);
  ASSERT_NE(nullptr, schema);
  ASSERT_TRUE(schema->HasFunction());

  const FunctionProto *func = schema->GetFunction(kFunctionVerifyOpset);
  ASSERT_NE(nullptr, func);

  GraphProto graph;
  NodeProto function_node;
  function_node.set_name("missing_schema_node");
  function_node.set_domain(kFunctionVerifyDomain);
  function_node.set_op_type("MissingFunctionOp");
  *function_node.add_input() = "x";
  *function_node.add_output() = "y";
  const std::string expected_message =
      "No schema registered for op 'MissingFunctionOp' in domain '" +
      std::string(kFunctionVerifyDomain) +
      "' at version 1 while expanding function node missing_schema_node";

  try {
    FunctionExpandHelper(function_node, *func, graph);
    FAIL() << "Expected FunctionExpandHelper to throw for a missing schema.";
  } catch (const std::runtime_error &e) {
    EXPECT_EQ(std::string(e.what()), expected_message);
  }

  DeregisterDefaultAttrFunctionSchema();
}

TEST(FunctionVerification, VerifyFunctionBodyWithMultipleDomains) {
  OpSchema schema;
  schema.SetName("MultiDomainFunction")
      .SetDomain("onnx.light.test.main")
      .SinceVersion(1)
      .Input(0, "x", "Input tensor", "T")
      .Output(0, "y", "Output tensor", "T")
      .TypeConstraint("T", {"tensor(float)"}, "Constrain to float tensor.");

  const std::vector<FunctionBodyHelper::NodeDef> node_defs = {
      {{"z"}, "OpA", {"x"}, {}, "onnx.light.test.domain_a"},
      {{"y"}, "OpB", {"z"}, {}, "onnx.light.test.domain_b"},
  };

  FunctionBodyHelper::OperatorSetList relied_opsets;
  auto *opset_a = relied_opsets.Add();
  opset_a->set_domain("onnx.light.test.domain_a");
  opset_a->set_version(1);
  auto *opset_b = relied_opsets.Add();
  opset_b->set_domain("onnx.light.test.domain_b");
  opset_b->set_version(1);

  FunctionProto function_proto;
  ASSERT_TRUE(
      FunctionBodyHelper::BuildFunctionProto(function_proto, schema, node_defs, relied_opsets));
  ASSERT_EQ(function_proto.ref_node().size(), 2U);
  EXPECT_EQ(function_proto.ref_node()[0].ref_domain(), "onnx.light.test.domain_a");
  EXPECT_EQ(function_proto.ref_node()[1].ref_domain(), "onnx.light.test.domain_b");

  std::unordered_map<std::string, int> imported;
  for (const auto &opset : function_proto.ref_opset_import()) {
    imported[opset.ref_domain()] = static_cast<int>(opset.ref_version());
  }
  ASSERT_EQ(imported.size(), 2U);
  EXPECT_EQ(imported["onnx.light.test.domain_a"], 1);
  EXPECT_EQ(imported["onnx.light.test.domain_b"], 1);
}

TEST(FunctionVerification, VerifyParsedFunctionBodyWithMultipleDomains) {
  RegisterAllOnnxOperatorSchemas();
  FunctionProto function_body;
  FunctionBuilder(function_body).Add(R"ONNX(
    Q_Min = Constant <value = float {0.0}> ()
    Q_Max = Constant <value = float {255.0}> ()
    X_Min = ReduceMin <keepdims = 0> (x)
    X_Min_Adjusted = Min (X_Min, Q_Min)
    X_Max = ReduceMax <keepdims = 0> (x)
    X_Max_Adjusted = Max (X_Max, Q_Min)
    X_Range = Sub (X_Max_Adjusted, X_Min_Adjusted)
    Scale = Div (X_Range, Q_Max)
    Min_Scaled = Div (X_Min_Adjusted, Scale)
    Initial_ZeroPoint_FP = Sub (Q_Min, Min_Scaled)
    Clipped_ZeroPoint_FP = Clip (Initial_ZeroPoint_FP, Q_Min, Q_Max)
    Rounded_ZeroPoint_FP = Round (Clipped_ZeroPoint_FP)
    Zeropoint = Cast <to = 2> (Rounded_ZeroPoint_FP)
    y_scale = Identity (Scale)
    y_zero_point = Identity (Zeropoint)
    y = QuantizeLinear (x, Scale, Zeropoint)
  )ONNX");

  FunctionBodyHelper::OperatorSetList operator_sets;
  auto *onnx_opset = operator_sets.Add();
  onnx_opset->set_domain(ONNX_DOMAIN);
  onnx_opset->set_version(13);
  auto *test_opset = operator_sets.Add();
  test_opset->set_domain(AI_ONNX_ML_DOMAIN);
  test_opset->set_version(2);

  OpSchema function_schema;
  function_schema.SetName("DynamicQuantizeLinear_Fake")
      .SetDomain(AI_ONNX_ML_DOMAIN)
      .SinceVersion(2)
      .SetDoc("Test Op")
      .Input(0, "x", "Input tensor", "T1")
      .Output(0, "y", "Quantized output tensor", "T2")
      .Output(1, "y_scale",
              "Output scale. It's a scalar, which means a per-tensor/layer quantization.",
              "tensor(float)")
      .Output(2, "y_zero_point",
              "Output zero point. It's a scalar, which means a per-tensor/layer quantization.",
              "T2")
      .TypeConstraint("T1", {"tensor(float)"}, "Constrain 'x' to float tensor.")
      .TypeConstraint("T2", {"tensor(uint8)"},
                      "Constrain 'y_zero_point' and 'y' to 8-bit unsigned integer tensor.")
      .FunctionBody(function_body.ref_node(), operator_sets);
  RegisterSchema(std::move(function_schema), 0, /*fail_duplicate_schema=*/false);

  const auto *schema = OpSchemaRegistry::Schema("DynamicQuantizeLinear_Fake", 2, AI_ONNX_ML_DOMAIN);
  ASSERT_NE(schema, nullptr);
  EXPECT_TRUE(schema->HasFunction());
  EXPECT_FALSE(schema->HasContextDependentFunction());
  const auto *function = schema->GetFunction();
  ASSERT_NE(function, nullptr);
  ASSERT_EQ(function->node_size(), 16);

  std::unordered_map<std::string, int> imported;
  for (const auto &opset : function->ref_opset_import()) {
    imported[opset.ref_domain()] = static_cast<int>(opset.ref_version());
  }
  ASSERT_EQ(imported.size(), 2U);
  EXPECT_EQ(imported.at(ONNX_DOMAIN), 13);
  EXPECT_EQ(imported.at(AI_ONNX_ML_DOMAIN), 2);

  checker::LexicalScopeContext lexical_scope;
  checker::CheckerContext checker_context;
  checker_context.set_opset_imports(imported);
  checker_context.set_ir_version(7);
  EXPECT_NO_THROW(checker::check_function(*function, checker_context, lexical_scope));

  DeregisterSchema("DynamicQuantizeLinear_Fake", 2, AI_ONNX_ML_DOMAIN);
}

} // namespace Test
} // namespace ONNX_LIGHT_NAMESPACE
