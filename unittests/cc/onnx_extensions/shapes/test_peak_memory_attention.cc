// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/compute_context.h"
#include "onnx_core/expressions/expressions.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

using core::symbolic::Device;
using core::symbolic::SymDim;
using core::symbolic::SymShape;

NodeProto AttentionNode(bool packed = false) {
  NodeProto node;
  node.set_op_type("Attention");
  for (const auto *name : {"q", "k", "v"})
    node.add_input(name);
  node.add_output("y");
  if (packed) {
    for (const auto *name : {"q_num_heads", "kv_num_heads"}) {
      auto *attr = node.add_attribute();
      attr->set_name(name);
      attr->set_type(AttributeProto::INT);
      attr->set_i(1);
    }
  }
  return node;
}

core::shapes::PeakMemoryEstimate Estimate(const NodeProto &node,
                                          const std::vector<SymShape> &inputs) {
  return onnx_shapes::shapes::nn::ComputePeakMemoryAttention(Device::kCPU, node, inputs);
}

} // namespace

TEST(ComputePeakMemoryAttention, BoundsPromotionsAndOneFloatScoreRow) {
  const std::vector<SymShape> inputs(3, SymShape({1, 1, 1, 1}));
  const auto estimate = Estimate(AttentionNode(), inputs);
  // Three promoted inputs, intermediate Y and one scratch score, all FLOAT32.
  EXPECT_EQ(estimate.worst_case.AsInt(), 20);
  EXPECT_FALSE(estimate.average_case);
}

TEST(ComputePeakMemoryAttention, DeclaredQkCountsOnlyItsFloatIntermediate) {
  auto node = AttentionNode();
  node.add_output("");
  node.add_output("");
  const std::vector<SymShape> inputs(3, SymShape({1, 1, 1, 1}));
  EXPECT_EQ(Estimate(node, inputs).worst_case.AsInt(), 20);
  node.add_output("qk");
  // The final declared qk tensor is excluded; its half-path FLOAT32 staging remains.
  EXPECT_EQ(Estimate(node, inputs).worst_case.AsInt(), 24);
}

TEST(ComputePeakMemoryAttention, PackedLayoutIncludesReorderingAndIntermediateY) {
  const auto estimate =
      Estimate(AttentionNode(true), std::vector<SymShape>(3, SymShape({1, 1, 1})));
  EXPECT_EQ(estimate.worst_case.AsInt(), 36);
  EXPECT_FALSE(estimate.average_case);
  EXPECT_EQ(onnx_shapes::shapes::nn::ComputePeakMemoryAttention(
                Device::kCPU, std::vector<SymShape>(3, SymShape({1, 1, 1}))),
            core::shapes::UnknownPeakMemory("", "Attention"));
}

TEST(ComputePeakMemoryAttention, IncludesTemporaryCachesAndOptionalMaskPromotion) {
  auto node = AttentionNode();
  node.add_input("mask");
  node.add_input("past_key");
  node.add_input("past_value");
  std::vector<SymShape> inputs = {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1},
                                  {1, 1, 1, 3}, {1, 1, 2, 1}, {1, 1, 2, 1}};
  // Seven promoted Q/K/V/cache elements, three mask elements, Y, two
  // concatenated caches of length three and one score row of length three.
  EXPECT_EQ(Estimate(node, inputs).worst_case.AsInt(), 80);
  *node.mutable_input(3) = "";
  EXPECT_EQ(Estimate(node, inputs).worst_case.AsInt(), 68);
  inputs[5][2] = SymDim(3);
  EXPECT_THROW(Estimate(node, inputs), std::invalid_argument);
}

TEST(ComputePeakMemoryAttention, PreservesSymbolicBatchSequenceAndHiddenDimensions) {
  const std::vector<SymShape> inputs = {{SymDim("B"), SymDim(1), SymDim(2), SymDim(1)},
                                        {SymDim("B"), SymDim(1), SymDim(3), SymDim(1)},
                                        {SymDim("B"), SymDim(1), SymDim(3), SymDim(1)}};
  const auto estimate = Estimate(AttentionNode(), inputs);
  ASSERT_TRUE(estimate.worst_case.IsExpr());
  EXPECT_EQ(core::expressions::evaluate_expression(estimate.worst_case.AsExpr(), {{"B", 3}}), 156);
  EXPECT_FALSE(estimate.average_case);

  const std::vector<SymShape> packed = {{SymDim(1), SymDim("T"), SymDim("D")},
                                        {SymDim(1), SymDim(3), SymDim("D")},
                                        {SymDim(1), SymDim(3), SymDim("D")}};
  const auto packed_estimate = Estimate(AttentionNode(true), packed);
  ASSERT_TRUE(packed_estimate.worst_case.IsExpr());
  EXPECT_EQ(core::expressions::evaluate_expression(packed_estimate.worst_case.AsExpr(),
                                                   {{"T", 2}, {"D", 1}}),
            92);
}

TEST(ComputePeakMemoryAttention, SymbolicCachesUseTheirMaximumWithoutClaimingAnAverage) {
  auto node = AttentionNode();
  node.add_input("");
  node.add_input("past_key");
  node.add_input("past_value");
  const std::vector<SymShape> inputs = {{1, 1, 1, 1},
                                        {1, 1, 1, 1},
                                        {1, 1, 1, 1},
                                        {},
                                        {SymDim(1), SymDim(1), SymDim("P"), SymDim(1)},
                                        {SymDim(1), SymDim(1), SymDim("R"), SymDim(1)}};
  const auto estimate = Estimate(node, inputs);
  ASSERT_TRUE(estimate.worst_case.IsExpr());
  EXPECT_EQ(
      core::expressions::evaluate_expression(estimate.worst_case.AsExpr(), {{"P", 2}, {"R", 3}}),
      84);
  EXPECT_FALSE(estimate.average_case);
}

TEST(ComputePeakMemoryAttention, RejectsNegativeDimensionsBeforeZeroShortcuts) {
  for (size_t slot = 0; slot < 3; ++slot) {
    for (size_t axis = 0; axis < 4; ++axis) {
      std::vector<SymShape> inputs(3, SymShape({int64_t{0}, 1, 1, 1}));
      inputs[slot][axis] = SymDim(-1);
      EXPECT_THROW(Estimate(AttentionNode(), inputs), std::invalid_argument);
    }
  }
}

TEST(ComputePeakMemoryAttention, RejectsOverflowAndPreservesEmptyBatch) {
  const int64_t large = int64_t{1} << 31;
  const std::vector<SymShape> large_inputs(3, SymShape({large, large, 1, 1}));
  EXPECT_THROW(Estimate(AttentionNode(), large_inputs), std::invalid_argument);
  EXPECT_EQ(
      Estimate(AttentionNode(), std::vector<SymShape>(3, SymShape({int64_t{0}, 1, large, large}))),
      core::shapes::PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
  // An empty query does not eliminate K/V promotions.
  EXPECT_EQ(Estimate(AttentionNode(), {{1, 1, int64_t{0}, 1}, {1, 1, 2, 1}, {1, 1, 2, 1}})
                .worst_case.AsInt(),
            16);
}

TEST(ComputePeakMemoryAttention, MissingShapesAndInvalidPackedHeads) {
  EXPECT_EQ(Estimate(AttentionNode(), {{1, 1, 1, 1}}),
            core::shapes::UnknownPeakMemory("", "Attention"));
  std::vector<SymShape> inputs(3, SymShape({1, 1, 1, 1}));
  inputs[2][3] = SymDim("?");
  EXPECT_EQ(Estimate(AttentionNode(), inputs), core::shapes::UnknownPeakMemory("", "Attention"));
  auto node = AttentionNode(true);
  node.mutable_attribute(0)->set_i(0);
  EXPECT_THROW(Estimate(node, std::vector<SymShape>(3, SymShape({1, 1, 1}))),
               std::invalid_argument);
}

TEST(ComputePeakMemoryAttention, ShapeAnalysisAccountsForFinalOutputsSeparately) {
  onnx_shapes::RegisterShapeFunctions();
  onnx_shapes::RegisterPeakMemoryFunctions();
  ModelProto model;
  model.add_opset("", 24);
  auto &graph = *model.mutable_graph();
  for (const auto *name : {"q", "k", "v"}) {
    auto *input = graph.add_input();
    input->set_name(name);
    auto *type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(TensorProto::FLOAT16);
    for (int64_t dim : {1, 1, 1, 1})
      type->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  auto node = AttentionNode();
  for (const auto *name : {"", "", "qk"})
    node.add_output(name);
  *graph.add_node() = node;
  graph.add_output()->set_name("y");
  graph.add_output()->set_name("qk");
  core::compute::ComputeContext ctx;
  ctx.ComputeShapes(model);
  ctx.ComputePeakMemory(graph);
  EXPECT_EQ(ctx.NodePeakMemory(0).worst_case.AsInt(), 24);
  EXPECT_FALSE(ctx.NodePeakMemory(0).average_case);
  EXPECT_EQ(ctx.Shapes().Get("qk").Shape(), SymShape({1, 1, 1, 1}));
}

} // namespace Test
