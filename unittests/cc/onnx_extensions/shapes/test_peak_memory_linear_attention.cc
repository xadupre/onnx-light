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

core::symbolic::SymShape Rank3(int64_t batch, int64_t sequence, int64_t hidden) {
  return core::symbolic::SymShape{core::symbolic::SymDim(batch), core::symbolic::SymDim(sequence),
                                  core::symbolic::SymDim(hidden)};
}

} // namespace

TEST(ComputePeakMemoryLinearAttention, StaticPackedInputsReturnConservativeStateBytes) {
  const std::vector<core::symbolic::SymShape> inputs = {
      Rank3(2, 8, 64),
      Rank3(2, 8, 64),
      Rank3(2, 8, 32),
  };
  EXPECT_EQ(onnx_shapes::shapes::nn::ComputePeakMemoryLinearAttention(core::symbolic::Device::kCPU,
                                                                      inputs),
            2 * 64 * 32 * 4);
}

TEST(ComputePeakMemoryLinearAttention, PastStateReturnsExactStateBytes) {
  const std::vector<core::symbolic::SymShape> inputs = {
      Rank3(2, 8, 64),
      Rank3(2, 8, 64),
      Rank3(2, 8, 32),
      core::symbolic::SymShape{core::symbolic::SymDim(2), core::symbolic::SymDim(8),
                               core::symbolic::SymDim(8), core::symbolic::SymDim(4)},
  };
  EXPECT_EQ(onnx_shapes::shapes::nn::ComputePeakMemoryLinearAttention(core::symbolic::Device::kCPU,
                                                                      inputs),
            2 * 8 * 8 * 4 * 4);
}

TEST(ComputePeakMemoryLinearAttention, IncompleteOrSymbolicInputsReturnZero) {
  const std::vector<core::symbolic::SymShape> too_few = {Rank3(2, 8, 64), Rank3(2, 8, 64)};
  EXPECT_EQ(onnx_shapes::shapes::nn::ComputePeakMemoryLinearAttention(core::symbolic::Device::kCPU,
                                                                      too_few),
            0);

  const std::vector<core::symbolic::SymShape> symbolic = {
      core::symbolic::SymShape{core::symbolic::SymDim("batch"), core::symbolic::SymDim(8),
                               core::symbolic::SymDim(64)},
      Rank3(2, 8, 64),
      Rank3(2, 8, 32),
  };
  EXPECT_EQ(onnx_shapes::shapes::nn::ComputePeakMemoryLinearAttention(core::symbolic::Device::kCPU,
                                                                      symbolic),
            0);
}

TEST(ComputePeakMemoryLinearAttention, RegisteredInCoreDispatchTable) {
  onnx_shapes::RegisterPeakMemoryFunctions();
  const auto &table = core::shapes::PeakMemoryDispatchTable();
  EXPECT_NE(table.find("ai.onnx:LinearAttention"), table.end());

  const std::vector<core::symbolic::SymShape> inputs = {
      Rank3(1, 4, 32),
      Rank3(1, 4, 32),
      Rank3(1, 4, 16),
  };
  EXPECT_EQ(core::shapes::ComputePeakMemory("ai.onnx", "LinearAttention",
                                            core::symbolic::Device::kCPU, inputs),
            32 * 16 * 4);
}

} // namespace Test
