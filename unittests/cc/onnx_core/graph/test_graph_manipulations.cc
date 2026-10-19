#include "onnx_core/graph/graph_manipulations.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::graph;

TEST(onnx_helper, CollectExternalInputs) {
  std::vector<NodeProto> nodes(3);
  nodes[0].set_op_type("Mul");
  nodes[0].add_input("x");
  nodes[0].add_input("y");
  nodes[0].add_output("t");

  nodes[1].set_op_type("Sub");
  nodes[1].add_input("t");
  nodes[1].add_input("z");
  nodes[1].add_output("out");

  nodes[2].set_op_type("Add");
  nodes[2].add_input("out");
  nodes[2].add_input("x");
  nodes[2].add_output("final");

  auto inputs = CollectExternalInputs(nodes);
  EXPECT_EQ(inputs, std::vector<std::string>({"x", "y", "z"}));
}

TEST(onnx_helper, CollectRemainingInputs) {
  std::vector<NodeProto> nodes(3);
  nodes[0].set_op_type("Mul");
  nodes[0].add_input("x");
  nodes[0].add_input("y");
  nodes[0].add_output("t");

  nodes[1].set_op_type("Sub");
  nodes[1].add_input("t");
  nodes[1].add_input("z");
  nodes[1].add_output("out");

  nodes[2].set_op_type("Add");
  nodes[2].add_input("out");
  nodes[2].add_input("x");
  nodes[2].add_output("final");

  auto remaining = CollectRemainingInputs(nodes, {"final"});
  ASSERT_EQ(remaining.size(), 3u);
  // Before node 0 every external ancestor of ``final`` must be available.
  EXPECT_EQ(remaining[0], std::vector<std::string>({"x", "y", "z"}));
  // Before node 1: ``t`` is produced by node 0 (outside the suffix) and still
  // read by node 1, ``z`` is read by node 1 and ``x`` by node 2; ``y`` is no
  // longer needed.
  EXPECT_EQ(remaining[1], std::vector<std::string>({"t", "z", "x"}));
  // Before node 2: it reads ``out`` (produced by node 1) and ``x``.
  EXPECT_EQ(remaining[2], std::vector<std::string>({"out", "x"}));
}

TEST(onnx_helper, CollectRemainingInputsPrunesDeadBranches) {
  std::vector<NodeProto> nodes(4);
  nodes[0].set_op_type("Mul");
  nodes[0].add_input("x");
  nodes[0].add_input("y");
  nodes[0].add_output("t");

  nodes[1].set_op_type("Sub");
  nodes[1].add_input("t");
  nodes[1].add_input("z");
  nodes[1].add_output("out");

  // Dead branch: ``dead`` is never an ancestor of the requested output.
  nodes[2].set_op_type("Neg");
  nodes[2].add_input("w");
  nodes[2].add_output("dead");

  nodes[3].set_op_type("Add");
  nodes[3].add_input("out");
  nodes[3].add_input("x");
  nodes[3].add_output("final");

  auto remaining = CollectRemainingInputs(nodes, {"final"});
  ASSERT_EQ(remaining.size(), 4u);
  // ``w`` never appears because the ``Neg`` node does not contribute to ``final``.
  EXPECT_EQ(remaining[0], std::vector<std::string>({"x", "y", "z"}));
  EXPECT_EQ(remaining[1], std::vector<std::string>({"t", "z", "x"}));
  // Before the dead ``Neg`` node, the inputs still required are those of the
  // remaining relevant nodes (the final ``Add``), not of the dead node itself.
  EXPECT_EQ(remaining[2], std::vector<std::string>({"out", "x"}));
  EXPECT_EQ(remaining[3], std::vector<std::string>({"out", "x"}));
}

TEST(onnx_helper, CollectRemainingInputsMultipleOutputs) {
  std::vector<NodeProto> nodes(3);
  nodes[0].set_op_type("Mul");
  nodes[0].add_input("x");
  nodes[0].add_input("y");
  nodes[0].add_output("t");

  nodes[1].set_op_type("Neg");
  nodes[1].add_input("w");
  nodes[1].add_output("n");

  nodes[2].set_op_type("Add");
  nodes[2].add_input("t");
  nodes[2].add_input("x");
  nodes[2].add_output("final");

  // Requesting both ``final`` and ``n`` keeps the ``Neg`` branch alive.
  auto remaining = CollectRemainingInputs(nodes, {"final", "n"});
  ASSERT_EQ(remaining.size(), 3u);
  EXPECT_EQ(remaining[0], std::vector<std::string>({"x", "y", "w"}));
  EXPECT_EQ(remaining[1], std::vector<std::string>({"w", "t", "x"}));
  EXPECT_EQ(remaining[2], std::vector<std::string>({"t", "x"}));
}

TEST(onnx_helper, CollectRemainingInputsEmpty) {
  std::vector<NodeProto> nodes;
  EXPECT_TRUE(CollectRemainingInputs(nodes, {"final"}).empty());
}
