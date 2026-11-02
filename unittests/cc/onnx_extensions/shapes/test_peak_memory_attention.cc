// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/compute_context.h"
#include "onnx_core/compute/peak_memory.h"
#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

core::symbolic::SymShape Rank4(int64_t batch, int64_t heads, int64_t seq, int64_t head_size) {
  return core::symbolic::SymShape{core::symbolic::SymDim(batch), core::symbolic::SymDim(heads),
                                  core::symbolic::SymDim(seq), core::symbolic::SymDim(head_size)};
}

core::symbolic::SymShape Packed(int64_t batch, int64_t seq, int64_t hidden) {
  return core::symbolic::SymShape{core::symbolic::SymDim(batch), core::symbolic::SymDim(seq),
                                  core::symbolic::SymDim(hidden)};
}

NodeProto AttentionNode(bool packed, int64_t q_heads = 2, int64_t kv_heads = 2) {
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
      attr->set_i(std::string(name) == "q_num_heads" ? q_heads : kv_heads);
    }
  }
  return node;
}

} // namespace

// The estimate is the QK^T score buffer:
// batch * q_num_heads * q_seq * kv_seq * 4 bytes.
TEST(ComputePeakMemoryAttention, StaticRank4ReturnsScoreBufferBytes) {
  const std::vector<core::symbolic::SymShape> inputs = {
      Rank4(2, 4, 8, 16),  // Q: (batch=2, q_num_heads=4, q_seq=8, head_size=16)
      Rank4(2, 4, 32, 16), // K: kv_seq=32
      Rank4(2, 4, 32, 16), // V
  };
  const int64_t expected = 2 * 4 * 8 * 32 * 4;
  EXPECT_EQ(
      onnx_shapes::shapes::nn::ComputePeakMemoryAttention(core::symbolic::Device::kCPU, inputs),
      core::shapes::PeakMemoryEstimate::Exact(core::symbolic::SymDim(expected)));
}

TEST(ComputePeakMemoryAttention, RejectsOverflowAndPreservesZero) {
  onnx_shapes::RegisterPeakMemoryFunctions();
  const int64_t large = int64_t{1} << 31;
  std::vector<core::symbolic::SymShape> inputs = {Rank4(1, 1, large, 1), Rank4(1, 1, large, 1),
                                                  Rank4(1, 1, large, 1)};
  EXPECT_THROW(
      core::shapes::ComputePeakMemory("", "Attention", core::symbolic::Device::kCPU, inputs),
      std::invalid_argument);
  inputs[0] = Rank4(large, large, large, 1);
  inputs[1] = Rank4(large, large, 0, 1);
  EXPECT_EQ(core::shapes::ComputePeakMemory("", "Attention", core::symbolic::Device::kCPU, inputs),
            core::shapes::PeakMemoryEstimate::Exact(core::symbolic::SymDim(int64_t{0})));
}

TEST(ComputePeakMemoryAttention, NamesNegativeDimensionsEvenWithZeroFactors) {
  const char *names[] = {"batch", "q_num_heads", "q_seq_len", "kv_seq_len"};
  for (int i = 0; i < 4; ++i) {
    SCOPED_TRACE(names[i]);
    const std::vector<core::symbolic::SymShape> inputs = {
        Rank4(i == 0 ? -7 : 0, i == 1 ? -7 : 1, i == 2 ? -7 : 1, 1),
        Rank4(1, 1, i == 3 ? -7 : 0, 1),
    };
    try {
      onnx_shapes::shapes::nn::ComputePeakMemoryAttention(core::symbolic::Device::kCPU, inputs);
      FAIL() << "A zero factor hid a negative Attention scratch dimension.";
    } catch (const std::invalid_argument &error) {
      EXPECT_NE(std::string(error.what()).find(std::string("'") + names[i] + "': -7."),
                std::string::npos);
    }
  }
}

TEST(ComputePeakMemoryAttention, PreservesSymbolicDimensionsAndExactAverage) {
  std::vector<core::symbolic::SymShape> inputs = {
      core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4),
                               core::symbolic::SymDim(8), core::symbolic::SymDim(16)},
      Rank4(2, 4, 32, 16),
      Rank4(2, 4, 32, 16),
  };
  const auto estimate =
      onnx_shapes::shapes::nn::ComputePeakMemoryAttention(core::symbolic::Device::kCPU, inputs);
  ASSERT_TRUE(estimate.worst_case.IsExpr());
  EXPECT_EQ(core::expressions::evaluate_expression(estimate.worst_case.AsExpr(), {{"N", 3}}),
            3 * 4 * 8 * 32 * 4);
  ASSERT_TRUE(estimate.average_case);
  EXPECT_EQ(*estimate.average_case, estimate.worst_case);
  inputs[0][1] = core::symbolic::SymDim(int64_t{-1});
  EXPECT_THROW(
      onnx_shapes::shapes::nn::ComputePeakMemoryAttention(core::symbolic::Device::kCPU, inputs),
      std::invalid_argument);
}

// The shape-only API cannot infer head counts from packed hidden dimensions.
TEST(ComputePeakMemoryAttention, PackedWithoutAttributesReturnsUnresolvedBound) {
  const std::vector<core::symbolic::SymShape> inputs = {
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(8),
                               core::symbolic::SymDim(64)},
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(32),
                               core::symbolic::SymDim(64)},
  };
  const auto estimate =
      onnx_shapes::shapes::nn::ComputePeakMemoryAttention(core::symbolic::Device::kCPU, inputs);
  EXPECT_TRUE(estimate.worst_case.IsExpr());
  EXPECT_FALSE(estimate.average_case);
}

TEST(ComputePeakMemoryAttention, EquivalentLayoutsIncludeCachesAndGroupedQueryHeads) {
  for (int64_t kv_heads : {1, 2}) {
    for (int64_t past : {-1, 0, 7}) {
      SCOPED_TRACE(::testing::Message() << "kv_heads=" << kv_heads << " past=" << past);
      const auto expected = core::shapes::PeakMemoryEstimate::Exact(
          core::symbolic::SymDim(int64_t{2 * 2 * 8 * (32 + (past < 0 ? 0 : past)) * 4}));
      for (bool packed : {false, true}) {
        auto node = AttentionNode(packed, 2, kv_heads);
        std::vector<core::symbolic::SymShape> inputs =
            packed ? std::vector<core::symbolic::SymShape>{Packed(2, 8, 32),
                                                           Packed(2, 32, kv_heads * 16),
                                                           Packed(2, 32, kv_heads * 16)}
                   : std::vector<core::symbolic::SymShape>{Rank4(2, 2, 8, 16),
                                                           Rank4(2, kv_heads, 32, 16),
                                                           Rank4(2, kv_heads, 32, 16)};
        if (past >= 0) {
          node.add_input("");
          node.add_input("past_key");
          node.add_input("past_value");
          inputs.emplace_back();
          inputs.push_back(Rank4(2, kv_heads, past, 16));
          inputs.push_back(Rank4(2, kv_heads, past, 16));
        }
        EXPECT_EQ(core::shapes::ComputePeakMemory(core::symbolic::Device::kCPU, node, inputs),
                  expected);
        ModelProto model;
        model.add_opset("", 24);
        auto &graph = *model.mutable_graph();
        *graph.add_node() = node;
        graph.add_output()->set_name("y");
        for (std::size_t i = 0; i < inputs.size(); ++i) {
          if (node.input(i).empty())
            continue;
          auto *input = graph.add_input();
          input->set_name(node.input(i));
          auto *type = input->mutable_type()->mutable_tensor_type();
          type->set_elem_type(TensorProto::FLOAT16);
          for (std::size_t axis = 0; axis < inputs[i].Rank(); ++axis)
            type->mutable_shape()->add_dim()->set_dim_value(inputs[i][axis].AsInt());
        }
        core::compute::ComputeContext ctx;
        ctx.ComputeShapes(model);
        ctx.ComputePeakMemory(graph);
        EXPECT_EQ(ctx.NodePeakMemory(0), expected);
        if (!packed) {
          EXPECT_EQ(core::shapes::ComputePeakMemory("", "Attention", core::symbolic::Device::kCPU,
                                                    inputs),
                    expected);
        }
      }
    }
  }
}

TEST(ComputePeakMemoryAttention, PackedRejectsInvalidHeadsAndOverflow) {
  const auto device = core::symbolic::Device::kCPU;
  const std::vector<core::symbolic::SymShape> inputs = {Packed(1, 0, 64), Packed(1, 32, 64)};
  for (const auto &heads :
       std::vector<std::pair<int64_t, int64_t>>{{0, 2}, {-1, 2}, {2, 0}, {2, -1}, {3, 2}}) {
    const auto node = AttentionNode(true, heads.first, heads.second);
    EXPECT_THROW(core::shapes::ComputePeakMemory(device, node, inputs), std::invalid_argument);
  }
  const auto node = AttentionNode(true);
  for (std::size_t axis : {0, 1}) {
    auto negative = inputs;
    negative[0][axis] = core::symbolic::SymDim(-1);
    EXPECT_THROW(core::shapes::ComputePeakMemory(device, node, negative), std::invalid_argument);
  }
  auto negative_key = inputs;
  negative_key[1][1] = core::symbolic::SymDim(-1);
  EXPECT_THROW(core::shapes::ComputePeakMemory(device, node, negative_key), std::invalid_argument);
  const int64_t large = int64_t{1} << 31;
  EXPECT_THROW(
      core::shapes::ComputePeakMemory(device, node, {Packed(1, large, 64), Packed(1, large, 64)}),
      std::invalid_argument);
  EXPECT_EQ(core::shapes::ComputePeakMemory(device, node,
                                            {Packed(large, 0, 64), Packed(large, large, 64)}),
            core::shapes::PeakMemoryEstimate::Exact(core::symbolic::SymDim(int64_t{0})));
}

TEST(ComputePeakMemoryAttention, CacheUnknownNegativeAndOverflow) {
  const auto device = core::symbolic::Device::kCPU;
  for (bool packed : {false, true}) {
    auto node = AttentionNode(packed);
    node.add_input("");
    node.add_input("past_key");
    std::vector<core::symbolic::SymShape> inputs = {packed ? Packed(1, 0, 64) : Rank4(1, 2, 0, 32),
                                                    packed ? Packed(1, 32, 64)
                                                           : Rank4(1, 2, 32, 32),
                                                    {},
                                                    {},
                                                    Rank4(1, 2, -1, 32)};
    EXPECT_THROW(core::shapes::ComputePeakMemory(device, node, inputs), std::invalid_argument);
    inputs[4] = Rank4(1, 2, std::numeric_limits<int64_t>::max(), 32);
    EXPECT_THROW(core::shapes::ComputePeakMemory(device, node, inputs), std::invalid_argument);
    inputs[4] =
        core::symbolic::SymShape{core::symbolic::SymDim(1), core::symbolic::SymDim(2),
                                 core::symbolic::SymDim("past"), core::symbolic::SymDim(32)};
    EXPECT_EQ(core::shapes::ComputePeakMemory(device, node, inputs),
              core::shapes::PeakMemoryEstimate::Exact(core::symbolic::SymDim(int64_t{0})));
    inputs[0][packed ? 1 : 2] = core::symbolic::SymDim(8);
    const auto estimate = core::shapes::ComputePeakMemory(device, node, inputs);
    ASSERT_TRUE(estimate.worst_case.IsExpr());
    EXPECT_EQ(core::expressions::evaluate_expression(estimate.worst_case.AsExpr(), {{"past", 7}}),
              1 * 2 * 8 * 39 * 4);
    ASSERT_TRUE(estimate.average_case);
    EXPECT_EQ(*estimate.average_case, estimate.worst_case);
    inputs.resize(4);
    const auto missing = core::shapes::ComputePeakMemory(device, node, inputs);
    EXPECT_EQ(missing, core::shapes::UnknownPeakMemory("", "Attention"));
    inputs.push_back(Rank4(1, 2, 7, 32));
    inputs.push_back(Rank4(1, 2, 8, 32));
    node.add_input("past_value");
    EXPECT_THROW(core::shapes::ComputePeakMemory(device, node, inputs), std::invalid_argument);
    inputs[4][2] = core::symbolic::SymDim("past_key");
    inputs[5][2] = core::symbolic::SymDim("past_value");
    const auto symbolic = core::shapes::ComputePeakMemory(device, node, inputs);
    ASSERT_TRUE(symbolic.worst_case.IsExpr());
    for (int64_t past_key : {2, 13}) {
      EXPECT_EQ(core::expressions::evaluate_expression(symbolic.worst_case.AsExpr(),
                                                       {{"past_key", past_key}, {"past_value", 7}}),
                1 * 2 * 8 * (32 + std::max(past_key, int64_t{7})) * 4);
    }
    EXPECT_FALSE(symbolic.average_case);
    inputs[4][2] = core::symbolic::SymDim("?");
    EXPECT_EQ(core::shapes::ComputePeakMemory(device, node, inputs),
              core::shapes::UnknownPeakMemory("", "Attention"));
  }
}

TEST(ComputePeakMemoryAttention, LocalFunctionBodyReceivesHeadAttributes) {
  ModelProto model;
  model.add_opset("", 24);
  model.add_opset("memory_test", 1);
  auto &graph = *model.mutable_graph();
  for (const auto *name : {"q", "k", "v"}) {
    auto *input = graph.add_input();
    input->set_name(name);
    auto *type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(TensorProto::FLOAT16);
    for (int64_t dim : {1, 8, 64})
      type->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  auto *call = graph.add_node();
  call->set_domain("memory_test");
  call->set_op_type("PackedAttention");
  for (const auto *name : {"q", "k", "v"})
    call->add_input(name);
  call->add_output("y");
  graph.add_output()->set_name("y");
  auto *function = model.add_functions();
  function->set_domain("memory_test");
  function->set_name("PackedAttention");
  function->add_opset("", 24);
  for (const auto *name : {"q", "k", "v"})
    function->add_input(name);
  function->add_output("y");
  *function->add_node() = AttentionNode(true);
  core::compute::ComputeContext ctx;
  ctx.ComputeShapes(model);
  ctx.ComputePeakMemory(graph);
  EXPECT_EQ(ctx.NodePeakMemory(0), core::shapes::PeakMemoryEstimate::Exact(
                                       core::symbolic::SymDim(int64_t{1 * 2 * 8 * 8 * 4})));
}

TEST(ComputePeakMemoryAttention, PackedSymbolicDimensionsAndOptionalCacheSlots) {
  const auto device = core::symbolic::Device::kCPU;
  auto node = AttentionNode(true);
  node.add_input("");
  node.add_input("");
  node.add_input("past_value");
  std::vector<core::symbolic::SymShape> inputs = {Packed(1, 8, 64),     Packed(1, 32, 64), {}, {},
                                                  Rank4(1, 2, 100, 32), Rank4(1, 2, 7, 32)};
  const auto expected =
      core::shapes::PeakMemoryEstimate::Exact(core::symbolic::SymDim(int64_t{1 * 2 * 8 * 39 * 4}));
  EXPECT_EQ(core::shapes::ComputePeakMemory(device, node, inputs), expected);
  for (std::size_t axis : {0, 1}) {
    auto symbolic = inputs;
    symbolic[0][axis] = core::symbolic::SymDim("unknown");
    const auto estimate = core::shapes::ComputePeakMemory(device, node, symbolic);
    ASSERT_TRUE(estimate.worst_case.IsExpr());
    EXPECT_EQ(
        core::expressions::evaluate_expression(estimate.worst_case.AsExpr(), {{"unknown", 3}}),
        (axis == 0 ? 3 * 2 * 8 : 1 * 2 * 3) * 39 * 4);
    ASSERT_TRUE(estimate.average_case);
    EXPECT_EQ(*estimate.average_case, estimate.worst_case);
    for (const auto *placeholder : {"", "?"}) {
      symbolic[0][axis] = core::symbolic::SymDim(placeholder);
      EXPECT_EQ(core::shapes::ComputePeakMemory(device, node, symbolic),
                core::shapes::UnknownPeakMemory("", "Attention"));
    }
  }
  inputs[0][2] = core::symbolic::SymDim("hidden");
  EXPECT_EQ(core::shapes::ComputePeakMemory(device, node, inputs), expected);
  inputs[0][0] = core::symbolic::SymDim("batch");
  inputs[5][2] = core::symbolic::SymDim(-1);
  EXPECT_THROW(core::shapes::ComputePeakMemory(device, node, inputs), std::invalid_argument);
}

TEST(ComputePeakMemoryAttention, ScratchRemainsSeparateFromStructuralReuse) {
  for (bool packed : {false, true}) {
    ModelProto model;
    model.add_opset("", 24);
    auto &graph = *model.mutable_graph();
    auto *input = graph.add_input();
    input->set_name("x");
    auto *type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(TensorProto::FLOAT16);
    for (int64_t dim :
         packed ? std::vector<int64_t>{1, 128, 64} : std::vector<int64_t>{1, 2, 128, 32})
      type->mutable_shape()->add_dim()->set_dim_value(dim);
    for (const auto *name : {"q", "k", "v"}) {
      auto *relu = graph.add_node();
      relu->set_op_type("Relu");
      relu->add_input("x");
      relu->add_output(name);
    }
    auto node = AttentionNode(packed);
    node.add_input("");
    for (const auto *name : {"past_key", "past_value"}) {
      auto *cache = graph.add_initializer();
      cache->set_name(name);
      cache->set_data_type(TensorProto::FLOAT16);
      for (int64_t dim : {1, 2, 0, 32})
        cache->add_dims(dim);
      node.add_input(name);
    }
    node.add_output("present_key");
    node.add_output("present_value");
    *graph.add_node() = node;
    for (const auto *name : {"y", "present_key", "present_value"})
      graph.add_output()->set_name(name);

    core::compute::ComputeContext ctx;
    ctx.set_events_enabled(true);
    ctx.ComputeShapes(model);
    ctx.ComputeValueAndNodeTags(graph);
    ctx.ComputeInPlaceReuseGraph(graph, ctx.Shapes(), false, ctx.ValueTags());
    ASSERT_EQ(ctx.NodeReuse(3).size(), 3u);
    std::size_t reuse_events = 0;
    for (const auto &event : ctx.Events()) {
      if (event.node_index == 3 && event.action == core::compute::ComputeEventAction::kInPlace) {
        EXPECT_EQ(event.input_index, event.output_index);
        ++reuse_events;
      }
    }
    EXPECT_EQ(reuse_events, 3u);
    const auto bytes = [&](const char *key) {
      return std::get<int64_t>(std::get<core::expressions::DimType>(ctx.NodeMemory(3).at(key)));
    };
    EXPECT_EQ(bytes("output_allocation_bytes"), 0);
    ctx.ComputePeakMemory(graph, core::symbolic::Device::kCPU);
    EXPECT_EQ(ctx.NodePeakMemory(3),
              core::shapes::PeakMemoryEstimate::Exact(core::symbolic::SymDim(int64_t{131072})));
    EXPECT_EQ(bytes("before_bytes"), 65536);
    EXPECT_EQ(bytes("during_bytes"), 196608);
    EXPECT_EQ(bytes("after_bytes"), 65536);
    EXPECT_EQ(bytes("released_bytes"), 131072);
    EXPECT_EQ(bytes("output_allocation_bytes"), 0);
    ctx.AppendNodePeakMemory(graph.node(3), 3);
    EXPECT_EQ(bytes("during_bytes"), 196608);

    core::compute::WritePeakMemoryToMetadata(graph, ctx.Shapes());
    const auto &metadata = graph.node(3).metadata_props();
    ASSERT_EQ(metadata.size(), 3u);
    for (const auto *key : {core::compute::kNodePeakMemoryMaxMetadataKey,
                            core::compute::kNodePeakMemoryAverageMetadataKey,
                            core::compute::kNodePeakMemoryMetadataKey}) {
      EXPECT_TRUE(std::any_of(metadata.begin(), metadata.end(), [&](const auto &entry) {
        return entry.key() == key && entry.value() == "131072";
      }));
    }
  }
}

// Fewer than two inputs yields no concrete estimate.
TEST(ComputePeakMemoryAttention, TooFewInputsReturnsUnresolvedBound) {
  const std::vector<core::symbolic::SymShape> inputs = {Rank4(2, 4, 8, 16)};
  EXPECT_TRUE(
      onnx_shapes::shapes::nn::ComputePeakMemoryAttention(core::symbolic::Device::kCPU, inputs)
          .worst_case.IsExpr());
}

// RegisterPeakMemoryFunctions wires Attention into the core dispatch table so
// ComputePeakMemory resolves it; unregistered operators retain an unresolved bound.
TEST(ComputePeakMemoryAttention, RegisteredInCoreDispatchTable) {
  onnx_shapes::RegisterPeakMemoryFunctions();
  const auto &table = core::shapes::PeakMemoryDispatchTable();
  EXPECT_NE(table.find("ai.onnx:Attention"), table.end());

  const std::vector<core::symbolic::SymShape> inputs = {
      Rank4(1, 2, 4, 8),
      Rank4(1, 2, 16, 8),
      Rank4(1, 2, 16, 8),
  };
  EXPECT_EQ(
      core::shapes::ComputePeakMemory("ai.onnx", "Attention", core::symbolic::Device::kCPU, inputs),
      core::shapes::PeakMemoryEstimate::Exact(core::symbolic::SymDim(int64_t{1 * 2 * 4 * 16 * 4})));
  EXPECT_TRUE(core::shapes::ComputePeakMemory("ai.onnx", "OpWithoutPeakMemory",
                                              core::symbolic::Device::kCPU, inputs)
                  .worst_case.IsExpr());
}

} // namespace Test
