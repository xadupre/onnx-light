// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// C++ unit tests for onnx_gradient::GradientOfNodes and
// onnx_gradient::GradientOfFunction.

#include "onnx_gradient/gradient.h"

#include "onnx_proto/onnx.h"
#include "onnx_proto/onnx_helper.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_gradient::GradientOfFunction;
using onnx_gradient::GradientOfNodes;

// ─────────────────────────── helpers ────────────────────────────────────────

// Returns a NodeProto built from the given arguments.
static NodeProto MakeTestNode(const std::string &op_type, const std::vector<std::string> &inputs,
                              const std::vector<std::string> &outputs) {
  return MakeNode(op_type.c_str(), inputs, outputs, nullptr, nullptr);
}

// Collects all op_type strings from a FunctionProto's node list.
static std::vector<std::string> NodeTypes(const FunctionProto &func) {
  std::vector<std::string> types;
  types.reserve(static_cast<size_t>(func.node_size()));
  for (int i = 0; i < func.node_size(); ++i) {
    types.push_back(func.node(i).op_type());
  }
  return types;
}

// ═══════════════════════════════════════════════════════════════════════════
// GradientOfNodes – basic API contract
// ═══════════════════════════════════════════════════════════════════════════

// Gradient of a single MatMul node: y = X @ W
// xs = {"W"}, zs = {"X"}
// Expected backward: Transpose(W) -> W_T; MatMul(dy, W_T) -> dW; ...
TEST(GradientOfNodes, MatMulGrad_W) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("MatMul", {"X", "W"}, {"y"}));
  std::vector<std::string> inputs = {"X", "W"};
  std::vector<TensorProto> initializers;

  FunctionProto grad = GradientOfNodes(nodes, inputs, initializers, std::vector<std::string>{"W"},
                                       "y", std::vector<std::string>{"X"});

  // The function should have inputs: W, X, dy
  ASSERT_EQ(grad.input_size(), 3);
  EXPECT_EQ(grad.input()[0], "W");
  EXPECT_EQ(grad.input()[1], "X");
  EXPECT_EQ(grad.input()[2], "dy");

  // The function should have one output: grad_W
  ASSERT_EQ(grad.output_size(), 1);
  EXPECT_EQ(grad.output()[0], "grad_W");

  // The backward graph must contain at least: Transpose, MatMul
  auto types = NodeTypes(grad);
  EXPECT_TRUE(std::find(types.begin(), types.end(), "Transpose") != types.end())
      << "Expected a Transpose node in the backward graph";
  EXPECT_TRUE(std::find(types.begin(), types.end(), "MatMul") != types.end())
      << "Expected a MatMul node in the backward graph";
}

// Gradient w.r.t. both inputs of MatMul: y = X @ W
// xs = {"X", "W"}, zs = {}
TEST(GradientOfNodes, MatMulGrad_XW) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("MatMul", {"X", "W"}, {"y"}));

  FunctionProto grad = GradientOfNodes(nodes, std::vector<std::string>{"X", "W"}, {},
                                       std::vector<std::string>{"X", "W"}, "y", {});

  ASSERT_EQ(grad.output_size(), 2);
  EXPECT_EQ(grad.output()[0], "grad_X");
  EXPECT_EQ(grad.output()[1], "grad_W");
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear regression: y = X @ W (no bias)
// Gradient of y w.r.t. W
// ═══════════════════════════════════════════════════════════════════════════
TEST(GradientOfNodes, LinearRegression_NoBias) {
  // Forward: y = X @ W
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("MatMul", {"X", "W"}, {"y"}));

  FunctionProto grad =
      GradientOfNodes(nodes, std::vector<std::string>{"X", "W"}, {}, std::vector<std::string>{"W"},
                      "y", std::vector<std::string>{"X"});

  // Inputs: W, X, dy
  ASSERT_GE(grad.input_size(), 2);
  EXPECT_EQ(grad.input()[0], "W");

  // Outputs: grad_W
  ASSERT_EQ(grad.output_size(), 1);
  EXPECT_EQ(grad.output()[0], "grad_W");

  // We expect at least two nodes: Transpose(X) and MatMul(X^T, dy)
  EXPECT_GE(grad.node_size(), 2);
  auto types = NodeTypes(grad);
  EXPECT_TRUE(std::find(types.begin(), types.end(), "Transpose") != types.end());
  EXPECT_TRUE(std::find(types.begin(), types.end(), "MatMul") != types.end());
}

// ═══════════════════════════════════════════════════════════════════════════
// Two-node graph: y = MatMul(X, W) + b (Add)
// Gradient of y w.r.t. W and b
// ═══════════════════════════════════════════════════════════════════════════
TEST(GradientOfNodes, LinearRegressionWithBias) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("MatMul", {"X", "W"}, {"mm"}));
  nodes.push_back(MakeTestNode("Add", {"mm", "b"}, {"y"}));

  FunctionProto grad =
      GradientOfNodes(nodes, std::vector<std::string>{"X", "W", "b"}, {},
                      std::vector<std::string>{"W", "b"}, "y", std::vector<std::string>{"X"});

  // Outputs: grad_W, grad_b
  ASSERT_EQ(grad.output_size(), 2);
  EXPECT_EQ(grad.output()[0], "grad_W");
  EXPECT_EQ(grad.output()[1], "grad_b");

  // The backward graph must visit both Add and MatMul backward rules.
  auto types = NodeTypes(grad);
  EXPECT_TRUE(std::find(types.begin(), types.end(), "Transpose") != types.end());
  EXPECT_TRUE(std::find(types.begin(), types.end(), "MatMul") != types.end());
}

// ═══════════════════════════════════════════════════════════════════════════
// Sub backward
// ═══════════════════════════════════════════════════════════════════════════
TEST(GradientOfNodes, SubGrad) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("Sub", {"A", "B"}, {"C"}));

  FunctionProto grad = GradientOfNodes(nodes, std::vector<std::string>{"A", "B"}, {},
                                       std::vector<std::string>{"A", "B"}, "C", {});

  ASSERT_EQ(grad.output_size(), 2);
  auto types = NodeTypes(grad);
  EXPECT_TRUE(std::find(types.begin(), types.end(), "Neg") != types.end())
      << "Expected a Neg node for Sub backward";
}

// ═══════════════════════════════════════════════════════════════════════════
// Mul backward
// ═══════════════════════════════════════════════════════════════════════════
TEST(GradientOfNodes, MulGrad) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("Mul", {"A", "B"}, {"C"}));

  FunctionProto grad = GradientOfNodes(nodes, std::vector<std::string>{"A", "B"}, {},
                                       std::vector<std::string>{"A", "B"}, "C", {});

  ASSERT_EQ(grad.output_size(), 2);
  auto types = NodeTypes(grad);
  auto mul_count = std::count(types.begin(), types.end(), "Mul");
  // At least two Mul nodes: one for dA = dC * B, one for dB = dC * A
  EXPECT_GE(mul_count, 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// Error handling: y not produced by any node
// ═══════════════════════════════════════════════════════════════════════════
TEST(GradientOfNodes, ErrorYNotProduced) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("MatMul", {"X", "W"}, {"y"}));
  EXPECT_THROW(GradientOfNodes(nodes, std::vector<std::string>{"X", "W"}, {},
                               std::vector<std::string>{"W"}, "z", std::vector<std::string>{"X"}),
               std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// Error handling: empty xs
// ═══════════════════════════════════════════════════════════════════════════
TEST(GradientOfNodes, ErrorEmptyXs) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("MatMul", {"X", "W"}, {"y"}));
  EXPECT_THROW(GradientOfNodes(nodes, std::vector<std::string>{"X", "W"}, {}, {}, "y",
                               std::vector<std::string>{"X"}),
               std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════════════════
// GradientOfFunction
// ═══════════════════════════════════════════════════════════════════════════
TEST(GradientOfFunction, BasicLinearRegression) {
  // Build a FunctionProto representing y = X @ W
  FunctionProto func;
  func.set_name("linear");
  func.set_domain("");
  func.add_input("X");
  func.add_input("W");
  func.add_output("y");
  func.add_node("MatMul", {"X", "W"}, {"y"});
  func.add_opset("", 21);

  FunctionProto grad =
      GradientOfFunction(func, std::vector<std::string>{"W"}, "y", std::vector<std::string>{"X"});

  // Name should be derived from the original function.
  EXPECT_EQ(grad.name(), "linear_grad");

  // Outputs: grad_W
  ASSERT_EQ(grad.output_size(), 1);
  EXPECT_EQ(grad.output()[0], "grad_W");

  // Backward nodes must include Transpose and MatMul.
  auto types = NodeTypes(grad);
  EXPECT_TRUE(std::find(types.begin(), types.end(), "Transpose") != types.end());
  EXPECT_TRUE(std::find(types.begin(), types.end(), "MatMul") != types.end());
}

// ═══════════════════════════════════════════════════════════════════════════
// Conv gradient
// ═══════════════════════════════════════════════════════════════════════════

// Gradient of y = Conv(X, W) w.r.t. X (input).
// Expected: ConvTranspose is used in the backward graph.
TEST(GradientOfNodes, ConvGrad_dX) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("Conv", {"X", "W"}, {"y"}));

  FunctionProto grad =
      GradientOfNodes(nodes, std::vector<std::string>{"X", "W"}, {}, std::vector<std::string>{"X"},
                      "y", std::vector<std::string>{"W"});

  // Output: grad_X
  ASSERT_EQ(grad.output_size(), 1);
  EXPECT_EQ(grad.output()[0], "grad_X");

  // Backward graph must use ConvTranspose for dX.
  auto types = NodeTypes(grad);
  EXPECT_TRUE(std::find(types.begin(), types.end(), "ConvTranspose") != types.end())
      << "Expected a ConvTranspose node for the input gradient";
}

// Gradient of y = Conv(X, W) w.r.t. W (weights).
// Expected: Transpose and Conv nodes are used in the backward graph.
TEST(GradientOfNodes, ConvGrad_dW) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("Conv", {"X", "W"}, {"y"}));

  FunctionProto grad =
      GradientOfNodes(nodes, std::vector<std::string>{"X", "W"}, {}, std::vector<std::string>{"W"},
                      "y", std::vector<std::string>{"X"});

  // Output: grad_W
  ASSERT_EQ(grad.output_size(), 1);
  EXPECT_EQ(grad.output()[0], "grad_W");

  // Weight gradient uses Transpose + Conv.
  auto types = NodeTypes(grad);
  EXPECT_TRUE(std::find(types.begin(), types.end(), "Transpose") != types.end())
      << "Expected Transpose nodes for the weight gradient";
  EXPECT_TRUE(std::find(types.begin(), types.end(), "Conv") != types.end())
      << "Expected a Conv node for the weight gradient";
}

// Gradient of y = Conv(X, W) w.r.t. both X and W.
TEST(GradientOfNodes, ConvGrad_dX_dW) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("Conv", {"X", "W"}, {"y"}));

  FunctionProto grad = GradientOfNodes(nodes, std::vector<std::string>{"X", "W"}, {},
                                       std::vector<std::string>{"X", "W"}, "y", {});

  ASSERT_EQ(grad.output_size(), 2);
  EXPECT_EQ(grad.output()[0], "grad_X");
  EXPECT_EQ(grad.output()[1], "grad_W");

  auto types = NodeTypes(grad);
  EXPECT_TRUE(std::find(types.begin(), types.end(), "ConvTranspose") != types.end());
  EXPECT_TRUE(std::find(types.begin(), types.end(), "Conv") != types.end());
}

// Gradient of y = Conv(X, W, B) w.r.t. B (bias).
// Expected: ReduceSum is used in the backward graph.
TEST(GradientOfNodes, ConvGrad_dB) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("Conv", {"X", "W", "B"}, {"y"}));

  FunctionProto grad =
      GradientOfNodes(nodes, std::vector<std::string>{"X", "W", "B"}, {},
                      std::vector<std::string>{"B"}, "y", std::vector<std::string>{"X", "W"});

  // Output: grad_B
  ASSERT_EQ(grad.output_size(), 1);
  EXPECT_EQ(grad.output()[0], "grad_B");

  auto types = NodeTypes(grad);
  EXPECT_TRUE(std::find(types.begin(), types.end(), "ReduceSum") != types.end())
      << "Expected ReduceSum for bias gradient";
}

// Gradient of y = Conv(X, W, B) w.r.t. all three inputs.
TEST(GradientOfNodes, ConvGrad_dX_dW_dB) {
  std::vector<NodeProto> nodes;
  nodes.push_back(MakeTestNode("Conv", {"X", "W", "B"}, {"y"}));

  FunctionProto grad = GradientOfNodes(nodes, std::vector<std::string>{"X", "W", "B"}, {},
                                       std::vector<std::string>{"X", "W", "B"}, "y", {});

  ASSERT_EQ(grad.output_size(), 3);
  EXPECT_EQ(grad.output()[0], "grad_X");
  EXPECT_EQ(grad.output()[1], "grad_W");
  EXPECT_EQ(grad.output()[2], "grad_B");

  auto types = NodeTypes(grad);
  EXPECT_TRUE(std::find(types.begin(), types.end(), "ConvTranspose") != types.end());
  EXPECT_TRUE(std::find(types.begin(), types.end(), "Conv") != types.end());
  EXPECT_TRUE(std::find(types.begin(), types.end(), "ReduceSum") != types.end());
}
