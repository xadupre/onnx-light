// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "onnx_lib/checker.h"
#include "onnx_lib/defs/function.h"
#include "onnx_lib/defs/operator_sets.h"
#include "onnx_manipulations/tensor_proto_util.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace Test {
namespace {

NodeProto MakeLayoutNode(const std::string &op, std::optional<int64_t> blocksize,
                         const std::string &mode) {
  NodeProto node;
  node.set_op_type(op);
  *node.add_input() = "X";
  *node.add_output() = "Y";
  if (blocksize.has_value()) {
    auto *attribute = node.add_attribute();
    attribute->set_name("blocksize");
    attribute->set_type(AttributeProto::INT);
    attribute->set_i(*blocksize);
  }
  if (!mode.empty()) {
    auto *attribute = node.add_attribute();
    attribute->set_name("mode");
    attribute->set_type(AttributeProto::STRING);
    attribute->set_s(mode);
  }
  return node;
}

const NodeProto *FindOutput(const FunctionProto &function, const std::string &name) {
  for (const auto &node : function.node()) {
    if (node.output_size() == 1 && node.output(0) == name) {
      return &node;
    }
  }
  return nullptr;
}

const AttributeProto *FindAttribute(const NodeProto &node, const std::string &name) {
  for (const auto &attribute : node.attribute()) {
    if (attribute.name() == name) {
      return &attribute;
    }
  }
  return nullptr;
}

void ExpectConstant(const FunctionProto &function, const std::string &name, int64_t value) {
  const auto *node = FindOutput(function, name);
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(node->op_type(), "Constant");
  const auto *attribute = FindAttribute(*node, "value");
  ASSERT_NE(attribute, nullptr);
  ASSERT_TRUE(attribute->has_t());
  const auto &tensor = attribute->t();
  EXPECT_EQ(tensor.data_type(), TensorProto::INT64);
  ASSERT_EQ(tensor.dims_size(), 1);
  EXPECT_EQ(tensor.dims(0), 1);
  EXPECT_EQ(ParseData<int64_t>(&tensor), std::vector<int64_t>{value});
}

class LayoutFunctionContextTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { RegisterOnnxOperatorSetSchema(0, false); }
};

TEST_F(LayoutFunctionContextTest, BuildsChecksAndExpandsBothModes) {
  for (const std::string op : {"SpaceToDepth", "DepthToSpace"}) {
    const auto *schema = OpSchemaRegistry::Schema(op, 28, ONNX_DOMAIN);
    ASSERT_NE(schema, nullptr);
    ASSERT_EQ(schema->SinceVersion(), 28);
    ASSERT_TRUE(schema->HasContextDependentFunction());
    for (const std::string mode : {"DCR", "CRD", ""}) {
      for (int64_t blocksize : {2, 3}) {
        SCOPED_TRACE(op + " mode=" + mode + " blocksize=" + std::to_string(blocksize));
        const auto node = MakeLayoutNode(op, blocksize, mode);
        FunctionBodyBuildContextImpl context(node);
        FunctionProto function;
        ASSERT_TRUE(schema->BuildContextDependentFunction(context, function));
        EXPECT_EQ(function.name(), op);
        ASSERT_EQ(function.input_size(), 1);
        ASSERT_EQ(function.output_size(), 1);
        EXPECT_EQ(function.input(0), "input");
        EXPECT_EQ(function.output(0), "output");
        ASSERT_EQ(function.opset_import_size(), 1);
        EXPECT_EQ(function.opset_import(0).version(), 28);

        ExpectConstant(function, "blocksize", blocksize);
        ExpectConstant(function, "block_area", blocksize * blocksize);
        const auto *transpose = FindOutput(function, "tmp_transposed");
        ASSERT_NE(transpose, nullptr);
        ASSERT_EQ(transpose->op_type(), "Transpose");
        const auto *perm = FindAttribute(*transpose, "perm");
        ASSERT_NE(perm, nullptr);
        const bool dcr = mode != "CRD";
        const std::vector<int64_t> expected_perm =
            op == "SpaceToDepth" ? (dcr ? std::vector<int64_t>{0, 3, 5, 1, 2, 4}
                                        : std::vector<int64_t>{0, 1, 3, 5, 2, 4})
                                 : (dcr ? std::vector<int64_t>{0, 3, 4, 1, 5, 2}
                                        : std::vector<int64_t>{0, 1, 4, 2, 5, 3});
        EXPECT_EQ(std::vector<int64_t>(perm->ints().begin(), perm->ints().end()), expected_perm);

        const auto *shape = FindOutput(function, "tmp_shape");
        ASSERT_NE(shape, nullptr);
        ASSERT_EQ(shape->op_type(), "Concat");
        const std::vector<std::string> expected_shape =
            op == "SpaceToDepth"
                ? std::vector<std::string>{"N", "C", "H_block", "blocksize", "W_block", "blocksize"}
                : (dcr ? std::vector<std::string>{"N", "blocksize", "blocksize", "C_block", "H",
                                                  "W"}
                       : std::vector<std::string>{"N", "C_block", "blocksize", "blocksize", "H",
                                                  "W"});
        EXPECT_EQ(std::vector<std::string>(shape->input().begin(), shape->input().end()),
                  expected_shape);
        for (const std::string output : {"tmp", "output"}) {
          const auto *reshape = FindOutput(function, output);
          ASSERT_NE(reshape, nullptr);
          ASSERT_EQ(reshape->op_type(), "Reshape");
          const auto *allowzero = FindAttribute(*reshape, "allowzero");
          ASSERT_NE(allowzero, nullptr);
          EXPECT_EQ(allowzero->i(), 1);
        }

        checker::CheckerContext checker_context;
        checker_context.set_ir_version(IR_VERSION);
        checker_context.set_opset_imports({{ONNX_DOMAIN, 28}});
        checker::LexicalScopeContext lexical_scope;
        EXPECT_NO_THROW(checker::check_function(function, checker_context, lexical_scope));

        GraphProto expanded;
        ASSERT_NO_THROW(FunctionExpandHelper(node, function, expanded));
        ASSERT_EQ(expanded.node_size(), function.node_size());
        ASSERT_GT(expanded.node_size(), 0);
        EXPECT_EQ(expanded.node(expanded.node_size() - 1).output(0), "Y");
        bool reads_input = false;
        for (const auto &expanded_node : expanded.node()) {
          EXPECT_NE(expanded_node.op_type(), op);
          for (const auto &input : expanded_node.input()) {
            reads_input = reads_input || input == "X";
          }
        }
        EXPECT_TRUE(reads_input);
      }
    }
  }
}

TEST_F(LayoutFunctionContextTest, RejectsInvalidAttributes) {
  for (const std::string op : {"SpaceToDepth", "DepthToSpace"}) {
    const auto *schema = OpSchemaRegistry::Schema(op, 28, ONNX_DOMAIN);
    ASSERT_NE(schema, nullptr);
    for (const auto blocksize :
         {std::optional<int64_t>{}, std::optional<int64_t>{0}, std::optional<int64_t>{-1}}) {
      SCOPED_TRACE(op);
      const auto node = MakeLayoutNode(op, blocksize, "DCR");
      FunctionBodyBuildContextImpl context(node);
      FunctionProto function;
      EXPECT_FALSE(schema->BuildContextDependentFunction(context, function));
      EXPECT_EQ(function.node_size(), 0);
    }
    const auto node = MakeLayoutNode(op, 2, "INVALID");
    FunctionBodyBuildContextImpl context(node);
    FunctionProto function;
    EXPECT_FALSE(schema->BuildContextDependentFunction(context, function));
    EXPECT_EQ(function.node_size(), 0);
  }
}

TEST_F(LayoutFunctionContextTest, HistoricalSchemasHaveNoContextDependentFunction) {
  for (const std::string op : {"SpaceToDepth", "DepthToSpace"}) {
    const auto *schema = OpSchemaRegistry::Schema(op, 13, ONNX_DOMAIN);
    ASSERT_NE(schema, nullptr);
    EXPECT_EQ(schema->SinceVersion(), 13);
    EXPECT_FALSE(schema->HasContextDependentFunction());
  }
}

} // namespace
} // namespace Test
} // namespace ONNX_LIGHT_NAMESPACE
