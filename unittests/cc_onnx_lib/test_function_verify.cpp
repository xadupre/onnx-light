// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Translated from file onnx/test/cpp/function_verify_test.cc
// and adapted to work with onnx-light.

#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

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

  std::vector<OperatorSetIdProto> relied_opsets(2);
  relied_opsets[0].set_domain("onnx.light.test.domain_a");
  relied_opsets[0].set_version(1);
  relied_opsets[1].set_domain("onnx.light.test.domain_b");
  relied_opsets[1].set_version(1);

  FunctionProto function_proto;
  ASSERT_TRUE(
      FunctionBodyHelper::BuildFunctionProto(function_proto, schema, node_defs, relied_opsets));
  ASSERT_EQ(function_proto.ref_node().size(), 2U);
  EXPECT_EQ(function_proto.ref_node()[0].ref_domain(), "onnx.light.test.domain_a");
  EXPECT_EQ(function_proto.ref_node()[1].ref_domain(), "onnx.light.test.domain_b");

  std::unordered_map<std::string, int> imported;
  for (const auto &opset : function_proto.ref_opset_import()) {
    imported[opset.ref_domain().as_string()] = static_cast<int>(opset.ref_version());
  }
  ASSERT_EQ(imported.size(), 2U);
  EXPECT_EQ(imported["onnx.light.test.domain_a"], 1);
  EXPECT_EQ(imported["onnx.light.test.domain_b"], 1);
}

} // namespace Test
} // namespace ONNX_LIGHT_NAMESPACE
