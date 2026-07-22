// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/shapes/nn/shape_nn.h"

#include <gtest/gtest.h>

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
      expected);
}

// A symbolic contributing dimension yields no concrete estimate.
TEST(ComputePeakMemoryAttention, SymbolicDimensionReturnsZero) {
  std::vector<core::symbolic::SymShape> inputs = {
      core::symbolic::SymShape{core::symbolic::SymDim("N"), core::symbolic::SymDim(4),
                               core::symbolic::SymDim(8), core::symbolic::SymDim(16)},
      Rank4(2, 4, 32, 16),
      Rank4(2, 4, 32, 16),
  };
  EXPECT_EQ(
      onnx_shapes::shapes::nn::ComputePeakMemoryAttention(core::symbolic::Device::kCPU, inputs), 0);
}

// Non-rank-4 inputs (e.g. the packed rank-3 form) yield no concrete estimate.
TEST(ComputePeakMemoryAttention, NonRank4ReturnsZero) {
  const std::vector<core::symbolic::SymShape> inputs = {
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(8),
                               core::symbolic::SymDim(64)},
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(32),
                               core::symbolic::SymDim(64)},
  };
  EXPECT_EQ(
      onnx_shapes::shapes::nn::ComputePeakMemoryAttention(core::symbolic::Device::kCPU, inputs), 0);
}

// Fewer than two inputs yields no concrete estimate.
TEST(ComputePeakMemoryAttention, TooFewInputsReturnsZero) {
  const std::vector<core::symbolic::SymShape> inputs = {Rank4(2, 4, 8, 16)};
  EXPECT_EQ(
      onnx_shapes::shapes::nn::ComputePeakMemoryAttention(core::symbolic::Device::kCPU, inputs), 0);
}

// RegisterPeakMemoryFunctions wires Attention into the core dispatch table so
// ComputePeakMemory resolves it; unregistered operators still return 0.
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
      1 * 2 * 4 * 16 * 4);
  EXPECT_EQ(core::shapes::ComputePeakMemory("ai.onnx", "OpWithoutPeakMemory",
                                            core::symbolic::Device::kCPU, inputs),
            0);
}

} // namespace Test
