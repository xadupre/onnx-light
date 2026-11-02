// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

core::symbolic::SymShape Rank4(int64_t batch, int64_t heads, int64_t seq, int64_t head_size) {
  return core::symbolic::SymShape{core::symbolic::SymDim(batch), core::symbolic::SymDim(heads),
                                  core::symbolic::SymDim(seq), core::symbolic::SymDim(head_size)};
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

// Non-rank-4 inputs (e.g. the packed rank-3 form) yield no concrete estimate.
TEST(ComputePeakMemoryAttention, NonRank4ReturnsUnresolvedBound) {
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
