// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <gtest/gtest.h>

#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

using core::symbolic::Device;
using core::symbolic::SymDim;
using core::symbolic::SymShape;

NodeProto LinearAttentionNode(const char *rule = "gated_delta", bool state_output = true) {
  NodeProto node;
  node.set_op_type("LinearAttention");
  for (const auto *name : {"q", "k", "v"})
    node.add_input(name);
  node.add_output("y");
  if (state_output)
    node.add_output("state");
  for (const auto *name : {"q_num_heads", "kv_num_heads"}) {
    auto *attr = node.add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto::INT);
    attr->set_i(1);
  }
  auto *attr = node.add_attribute();
  attr->set_name("update_rule");
  attr->set_type(AttributeProto::STRING);
  attr->set_s(rule);
  return node;
}

core::shapes::PeakMemoryEstimate Estimate(const NodeProto &node,
                                          const std::vector<SymShape> &inputs) {
  return onnx_shapes::shapes::nn::ComputePeakMemoryLinearAttention(Device::kCPU, node, inputs);
}

} // namespace

TEST(ComputePeakMemoryLinearAttention, BoundsPromotionsResultsAndOneAuxiliaryBuffer) {
  const std::vector<SymShape> inputs(3, SymShape({1, 1, 1}));
  const auto estimate = Estimate(LinearAttentionNode(), inputs);
  // Three FLOAT32 input promotions; FLOAT32 and half staging for output/state;
  // one FLOAT32 auxiliary element, not a second decay buffer.
  EXPECT_EQ(estimate.worst_case.AsInt(), 28);
  EXPECT_FALSE(estimate.average_case);
  EXPECT_EQ(Estimate(LinearAttentionNode("linear"), inputs).worst_case.AsInt(), 24);
  EXPECT_EQ(Estimate(LinearAttentionNode("gated"), inputs).worst_case.AsInt(), 24);
  EXPECT_EQ(Estimate(LinearAttentionNode("delta", false), inputs).worst_case.AsInt(), 32);
}

TEST(ComputePeakMemoryLinearAttention, OmittedStateIncludesItsTemporaryFinalCopy) {
  auto node = LinearAttentionNode("delta");
  node.add_input("past_state");
  const std::vector<SymShape> inputs = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1, 1}};
  EXPECT_EQ(Estimate(node, inputs).worst_case.AsInt(), 32);
  *node.mutable_output(1) = "";
  EXPECT_EQ(Estimate(node, inputs).worst_case.AsInt(), 36);
}

TEST(ComputePeakMemoryLinearAttention, UsesHeadAttributesWithoutSquaringKvHeads) {
  auto node = LinearAttentionNode("delta", false);
  node.mutable_attribute(0)->set_i(4);
  node.mutable_attribute(1)->set_i(2);
  const std::vector<SymShape> inputs = {{1, 1, 4}, {1, 1, 2}, {1, 1, 2}};
  // Eight promoted input elements, output of four, state of two, plus scratch of two.
  EXPECT_EQ(Estimate(node, inputs).worst_case.AsInt(), 84);
}

TEST(ComputePeakMemoryLinearAttention, SymbolicDimensionsAndOptionalPromotions) {
  auto node = LinearAttentionNode("delta", false);
  node.add_input("");
  node.add_input("");
  node.add_input("beta");
  const std::vector<SymShape> inputs = {{SymDim("B"), SymDim(2), SymDim(1)},
                                        {SymDim("B"), SymDim(2), SymDim(1)},
                                        {SymDim("B"), SymDim(2), SymDim(1)},
                                        {},
                                        {},
                                        {SymDim("B"), SymDim(2), SymDim(1)}};
  const auto estimate = Estimate(node, inputs);
  ASSERT_TRUE(estimate.worst_case.IsExpr());
  EXPECT_EQ(core::expressions::evaluate_expression(estimate.worst_case.AsExpr(), {{"B", 3}}), 174);
  EXPECT_FALSE(estimate.average_case);
}

TEST(ComputePeakMemoryLinearAttention, RejectsNegativeDimensionsAndOverflow) {
  for (size_t slot = 0; slot < 3; ++slot) {
    for (size_t axis = 0; axis < 3; ++axis) {
      std::vector<SymShape> inputs(3, SymShape({int64_t{0}, 1, 1}));
      inputs[slot][axis] = SymDim(-1);
      EXPECT_THROW(Estimate(LinearAttentionNode(), inputs), std::invalid_argument);
    }
  }
  const int64_t large = int64_t{1} << 31;
  EXPECT_THROW(Estimate(LinearAttentionNode(), std::vector<SymShape>(3, SymShape({1, 1, large}))),
               std::invalid_argument);
  EXPECT_EQ(Estimate(LinearAttentionNode(),
                     std::vector<SymShape>(3, SymShape({int64_t{0}, large, large}))),
            core::shapes::PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
}

TEST(ComputePeakMemoryLinearAttention, MissingShapesAndInvalidAttributes) {
  EXPECT_EQ(Estimate(LinearAttentionNode(), {{1, 1, 1}}),
            core::shapes::UnknownPeakMemory("", "LinearAttention"));
  std::vector<SymShape> inputs(3, SymShape({1, 1, 1}));
  inputs[2][2] = SymDim("?");
  EXPECT_EQ(Estimate(LinearAttentionNode(), inputs),
            core::shapes::UnknownPeakMemory("", "LinearAttention"));
  inputs[2][2] = SymDim(1);
  auto node = LinearAttentionNode();
  node.mutable_attribute(1)->set_i(0);
  EXPECT_THROW(Estimate(node, inputs), std::invalid_argument);
  node = LinearAttentionNode("invalid");
  EXPECT_THROW(Estimate(node, inputs), std::invalid_argument);
}

TEST(ComputePeakMemoryLinearAttention, RegistrationPassesNodeMetadata) {
  onnx_shapes::RegisterPeakMemoryFunctions();
  const auto node = LinearAttentionNode("linear", false);
  const std::vector<SymShape> inputs(3, SymShape({1, 1, 1}));
  EXPECT_EQ(core::shapes::ComputePeakMemory(Device::kCPU, node, inputs).worst_case.AsInt(), 28);
  const auto shapes_only =
      onnx_shapes::shapes::nn::ComputePeakMemoryLinearAttention(Device::kCPU, inputs);
  EXPECT_GE(shapes_only.worst_case.AsInt(), Estimate(node, inputs).worst_case.AsInt());
  EXPECT_FALSE(shapes_only.average_case);
}

} // namespace Test
