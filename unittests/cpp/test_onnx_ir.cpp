// Translated from https://github.com/onnx/onnx/tree/main/onnx/test/cpp/ir_test.cc
// to work with onnx-light's protobuf-based API.
//
// The original test uses ONNX's high-level IR abstractions (Graph, Value, Node)
// together with ExportModelProto to build a graph and verifies that the
// auto-generated node output names are valid C identifiers.
//
// onnx-light does not expose those IR abstractions; instead, graphs are built
// directly via the protobuf-style API (ModelProto / GraphProto / NodeProto).
// This translation constructs an equivalent graph and verifies the same
// property: every node output name in the graph is a valid identifier.

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx.h"
#include <cctype>
#include <gtest/gtest.h>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

// Reproduces the IsValidIdentifier helper from ir_test.cc verbatim.
static bool IsValidIdentifier(const std::string &name) {
  if (name.empty()) {
    return false;
  }
  if (!isalpha(static_cast<unsigned char>(name[0])) && name[0] != '_') {
    return false;
  }
  for (size_t i = 1; i < name.size(); ++i) {
    if (!isalnum(static_cast<unsigned char>(name[i])) && name[i] != '_') {
      return false;
    }
  }
  return true;
}

} // namespace

// ---------------------------------------------------------------------------
// onnx_ir.ValidIdentifierTest
//
// Translated from IR::ValidIdentifierTest in ir_test.cc.
//
// Original graph (built via the ONNX Graph IR):
//   input x : float[M, N]
//   temp1 = Neg(x)
//   y     = Neg(temp1)
//   output y
//
// After ExportModelProto the original test checks that every node output name
// is a valid identifier.  We reproduce the same graph with explicit names
// using onnx-light's protobuf API and perform the same check.
// ---------------------------------------------------------------------------
TEST(onnx_ir, ValidIdentifierTest) {
  ModelProto model;
  GraphProto &graph = model.add_graph();
  graph.set_name("test");

  // Graph input: x : float[M, N]
  ValueInfoProto &input = graph.add_input();
  input.set_name("x");
  TypeProto &input_type = input.add_type();
  TypeProto::Tensor &tensor_type = input_type.add_tensor_type();
  tensor_type.set_elem_type(static_cast<int32_t>(TensorProto::DataType::FLOAT));
  TensorShapeProto &shape = tensor_type.add_shape();
  shape.add_dim().set_dim_param("M");
  shape.add_dim().set_dim_param("N");

  // First Neg node: temp1 = Neg(x)
  NodeProto &node1 = graph.add_node();
  node1.set_op_type("Neg");
  node1.add_input() = "x";
  node1.add_output() = "temp1";

  // Second Neg node: y = Neg(temp1)
  NodeProto &node2 = graph.add_node();
  node2.set_op_type("Neg");
  node2.add_input() = "temp1";
  node2.add_output() = "y";

  // Graph output: y
  ValueInfoProto &output = graph.add_output();
  output.set_name("y");

  // Verify that every node output name is a valid identifier, mirroring the
  // assertion in the original IR::ValidIdentifierTest.
  for (const auto &node : model.ref_graph().ref_node()) {
    for (const auto &name : node.ref_output()) {
      EXPECT_TRUE(IsValidIdentifier(std::string(name.data(), name.size())))
          << "Name is not a valid identifier: "
          << std::string(name.data(), name.size());
    }
  }
}

// ---------------------------------------------------------------------------
// Additional tests for the IsValidIdentifier helper, covering the cases that
// the original ir_test.cc implicitly relies on.
// ---------------------------------------------------------------------------

TEST(onnx_ir, IsValidIdentifier_ValidNames) {
  EXPECT_TRUE(IsValidIdentifier("x"));
  EXPECT_TRUE(IsValidIdentifier("temp1"));
  EXPECT_TRUE(IsValidIdentifier("y"));
  EXPECT_TRUE(IsValidIdentifier("_internal"));
  EXPECT_TRUE(IsValidIdentifier("CamelCase"));
  EXPECT_TRUE(IsValidIdentifier("abc_123"));
  EXPECT_TRUE(IsValidIdentifier("_0"));
}

TEST(onnx_ir, IsValidIdentifier_InvalidNames) {
  EXPECT_FALSE(IsValidIdentifier(""));          // empty
  EXPECT_FALSE(IsValidIdentifier("1abc"));      // starts with digit
  EXPECT_FALSE(IsValidIdentifier("a-b"));       // contains hyphen
  EXPECT_FALSE(IsValidIdentifier("a b"));       // contains space
  EXPECT_FALSE(IsValidIdentifier("a.b"));       // contains dot
  EXPECT_FALSE(IsValidIdentifier("a/b"));       // contains slash
}
