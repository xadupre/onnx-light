// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/compute_context.h"
#include "onnx_core/compute/peak_memory.h"
#include "onnx_core/expressions/expressions.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_extensions/shapes/dispatch_table.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {
namespace {

using core::shapes::PeakMemoryEstimate;
using core::symbolic::Device;
using core::symbolic::SymDim;
using core::symbolic::SymShape;

PeakMemoryEstimate DecoderScratch(const std::string &op, const std::vector<SymShape> &inputs) {
  onnx_shapes::RegisterPeakMemoryFunctions();
  return core::shapes::ComputePeakMemory("", op, Device::kCPU, inputs);
}

NodeProto DecoderNode(const std::string &op, const std::string &attribute, int64_t value) {
  NodeProto node;
  node.set_op_type(op);
  auto *attr = node.add_attribute();
  attr->set_name(attribute);
  attr->set_type(AttributeProto::INT);
  attr->set_i(value);
  return node;
}

} // namespace

TEST(ComputePeakMemoryDecoder, MatMulIncludesPromotedInputsAndBroadcastOutput) {
  EXPECT_EQ(DecoderScratch("MatMul", {{2, 3}, {3, 5}}).worst_case.AsInt(),
            4 * (2 * 3 + 3 * 5 + 2 * 5));
  EXPECT_EQ(DecoderScratch("MatMul", {{2, 1, 3, 4}, {1, 7, 4, 5}}).worst_case.AsInt(),
            4 * (2 * 3 * 4 + 7 * 4 * 5 + 2 * 7 * 3 * 5));
  EXPECT_EQ(DecoderScratch("MatMul", {{3}, {3}}).worst_case.AsInt(), 4 * (3 + 3 + 1));
  EXPECT_EQ(DecoderScratch("MatMul", {{3}, {3, 5}}).worst_case.AsInt(), 4 * (3 + 15 + 5));
  EXPECT_EQ(DecoderScratch("MatMul", {{2, 3}, {3}}).worst_case.AsInt(), 4 * (6 + 3 + 2));
  EXPECT_EQ(DecoderScratch("MatMul", {{2, SymDim(int64_t{0})}, {SymDim(int64_t{0}), 5}})
                .worst_case.AsInt(),
            4 * 10);
  EXPECT_FALSE(DecoderScratch("MatMul", {{2, 3}, {3, 5}}).average_case);
}

TEST(ComputePeakMemoryDecoder, IndexingAndNormalizationHaveNonzeroBounds) {
  const auto gather = DecoderScratch("Gather", {{8, 16}, {2, 3}});
  EXPECT_EQ(gather, PeakMemoryEstimate::Exact(SymDim(int64_t{48})));
  EXPECT_EQ(DecoderScratch("Gather", {{8, 16}, {}}).worst_case.AsInt(), 8);
  EXPECT_EQ(DecoderScratch("Gather", {{8, 16}, {SymDim(int64_t{0})}}).worst_case.AsInt(), 0);
  EXPECT_EQ(DecoderScratch("Slice", {{8, 16}, {2}, {2}}).worst_case.AsInt(), 64);
  EXPECT_EQ(DecoderScratch("Slice", {{8, 16}, {2}, {2}, {2}, {2}}).worst_case.AsInt(), 64);
  EXPECT_EQ(DecoderScratch("Slice", {{8, 16}, {SymDim(int64_t{0})}, {SymDim(int64_t{0})}})
                .worst_case.AsInt(),
            0);
  EXPECT_EQ(DecoderScratch("RMSNormalization", {{2, 3, 4}, {4}}).worst_case.AsInt(),
            8 * (4 + 1 + 1) + 4 * (24 + 4));
  const auto node = DecoderNode("RMSNormalization", "axis", 1);
  EXPECT_EQ(
      core::shapes::ComputePeakMemory(Device::kCPU, node, {{2, 3, 4}, {3, 4}}).worst_case.AsInt(),
      8 * (12 + 2 + 2) + 4 * (24 + 12));
  // An empty normalization block still acquires a one-element index buffer.
  EXPECT_GE(DecoderScratch("RMSNormalization", {{2, SymDim(int64_t{0})}, {SymDim(int64_t{0})}})
                .worst_case.AsInt(),
            8);
}

TEST(ComputePeakMemoryDecoder, AuditedStreamingKernelsNeedNoTensorWorkspace) {
  for (const auto *op : {"Swish", "Shape", "Transpose", "Unsqueeze"})
    EXPECT_EQ(DecoderScratch(op, {{2, 3}}), PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
  EXPECT_EQ(DecoderScratch("CumSum", {{2, 3}, {}}), PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
  EXPECT_EQ(DecoderScratch("Where", {{2, 3}, {2, 3}, {}}),
            PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
  const auto cast = DecoderNode("Cast", "to", TensorProto::FLOAT);
  EXPECT_EQ(core::shapes::ComputePeakMemory(Device::kCPU, cast, {{2, 3}}),
            PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
}

TEST(ComputePeakMemoryDecoder, BinaryKernelsIncludeHeapBackedStridesOnFastPaths) {
  for (const auto *op : {"Mul", "Sub", "GreaterOrEqual", "And"}) {
    EXPECT_EQ(DecoderScratch(op, {{2, 3}, {}}), PeakMemoryEstimate::Exact(SymDim(int64_t{32})));
    EXPECT_EQ(DecoderScratch(op, {{2, 3}, {2, 3}}).worst_case.AsInt(), 32);
    EXPECT_EQ(DecoderScratch(op, {{2, 1, 3}, {7, 3}}).worst_case.AsInt(), 48);
    EXPECT_EQ(DecoderScratch(op, {{SymDim(int64_t{0}), 3}, {3}}).worst_case.AsInt(), 32);
    EXPECT_EQ(DecoderScratch(op, {{}, {}}).worst_case.AsInt(), 0);
    EXPECT_EQ(DecoderScratch(op, {{SymDim("N"), 3}, {3}}).worst_case.AsInt(), 32);
  }
}

TEST(ComputePeakMemoryDecoder, SymbolicBoundsResolveWithDimensionBindings) {
  const auto matmul = DecoderScratch("MatMul", {{SymDim("N"), 3}, {3, 5}});
  ASSERT_TRUE(matmul.worst_case.IsExpr());
  EXPECT_EQ(core::expressions::evaluate_expression(matmul.worst_case.AsExpr(), {{"N", 2}}),
            4 * (6 + 15 + 10));
  const auto gather = DecoderScratch("Gather", {{8, 16}, {SymDim("N")}});
  EXPECT_EQ(core::expressions::evaluate_expression(gather.worst_case.AsExpr(), {{"N", 7}}), 56);
  const auto slice = DecoderScratch("Slice", {{8, 16}, {SymDim("N")}, {SymDim("N")}});
  EXPECT_EQ(core::expressions::evaluate_expression(slice.worst_case.AsExpr(), {{"N", 2}}), 64);
  const auto rms = DecoderScratch("RMSNormalization", {{2, SymDim("N")}, {SymDim("N")}});
  EXPECT_EQ(core::expressions::evaluate_expression(rms.worst_case.AsExpr(), {{"N", 4}}),
            8 * (4 + 1 + 1) + 4 * (8 + 4));
  EXPECT_GE(core::expressions::evaluate_expression(rms.worst_case.AsExpr(), {{"N", 0}}), 8);
}

TEST(ComputePeakMemoryDecoder, UnsupportedQueriesRetainUnresolvedBounds) {
  for (const auto *op : {"MatMul", "Gather", "Slice", "RMSNormalization", "Swish", "Cast",
                         "OpWithoutDecoderEstimator"}) {
    EXPECT_EQ(DecoderScratch(op, {}), core::shapes::UnknownPeakMemory("", op));
  }
  EXPECT_EQ(DecoderScratch("MatMul", {{}, {3, 5}}), core::shapes::UnknownPeakMemory("", "MatMul"));
  EXPECT_EQ(DecoderScratch("Gather", {{8}, {SymDim("?")}}),
            core::shapes::UnknownPeakMemory("", "Gather"));
  EXPECT_EQ(DecoderScratch("Slice", {{8}, {}, {}}), core::shapes::UnknownPeakMemory("", "Slice"));
  EXPECT_EQ(DecoderScratch("Slice", {{8}, {1}, {2}}), core::shapes::UnknownPeakMemory("", "Slice"));
  EXPECT_EQ(DecoderScratch("Cast", {{2, 3}}), core::shapes::UnknownPeakMemory("", "Cast"));
  const auto cast = DecoderNode("Cast", "to", TensorProto::STRING);
  EXPECT_EQ(core::shapes::ComputePeakMemory(Device::kCPU, cast, {{2, 3}}),
            core::shapes::UnknownPeakMemory("", "Cast"));
  EXPECT_EQ(core::shapes::ComputePeakMemory("custom", "MatMul", Device::kCPU, {{2, 3}, {3, 5}}),
            core::shapes::UnknownPeakMemory("custom", "MatMul"));
  EXPECT_EQ(core::shapes::ComputePeakMemory("", "MatMul", core::symbolic::MakeGPUDevice(0),
                                            {{2, 3}, {3, 5}}),
            core::shapes::UnknownPeakMemory("", "MatMul"));
}

TEST(ComputePeakMemoryDecoder, RejectsInvalidDimensionsAndOverflow) {
  const int64_t large = std::numeric_limits<int64_t>::max();
  EXPECT_THROW(DecoderScratch("MatMul", {{2, 3}, {4, 5}}), std::invalid_argument);
  EXPECT_THROW(DecoderScratch("MatMul", {{2, 3, 4}, {7, 4, 5}}), std::invalid_argument);
  EXPECT_THROW(DecoderScratch("MatMul", {{large, 2}, {2, 1}}), std::invalid_argument);
  EXPECT_THROW(DecoderScratch("Gather", {{8}, {large}}), std::invalid_argument);
  EXPECT_THROW(DecoderScratch("Slice", {{8}, {large}, {large}}), std::invalid_argument);
  EXPECT_THROW(DecoderScratch("RMSNormalization", {{large}, {large}}), std::invalid_argument);
  // No zero factor or zero-workspace fast path may hide negative dimensions.
  EXPECT_THROW(DecoderScratch("MatMul", {{SymDim(int64_t{0}), -1}, {-1, 5}}),
               std::invalid_argument);
  EXPECT_THROW(DecoderScratch("Gather", {{8}, {SymDim(int64_t{0}), -1}}), std::invalid_argument);
  EXPECT_THROW(DecoderScratch("Swish", {{SymDim(int64_t{0}), -1}}), std::invalid_argument);
  EXPECT_THROW(core::shapes::ComputePeakMemory(
                   Device::kCPU, DecoderNode("RMSNormalization", "axis", large), {{2, 3}, {3}}),
               std::invalid_argument);
}

TEST(ComputePeakMemoryDecoder, PublicAnalysisIncludesScratchAndPreservesUnknownPhases) {
  onnx_shapes::RegisterShapeFunctions();
  onnx_shapes::RegisterPeakMemoryFunctions();
  GraphProto graph;
  for (const auto *name : {"X", "W"}) {
    auto *input = graph.add_input();
    input->set_name(name);
    auto *tensor = input->add_type()->add_tensor_type();
    tensor->set_elem_type(TensorProto::FLOAT16);
    auto *shape = tensor->add_shape();
    for (int64_t dim : {2, 2})
      shape->add_dim()->set_dim_value(dim);
  }
  auto *node = graph.add_node();
  node->set_op_type("MatMul");
  node->add_input("X");
  node->add_input("W");
  node->add_output("Y");
  graph.add_output()->set_name("Y");
  core::compute::ComputeContext ctx;
  ctx.Compute(graph);
  EXPECT_EQ(ctx.NodePeakMemory(0).worst_case.AsInt(), 48);
  const auto &memory = ctx.NodeMemory(0);
  const auto total = std::get<int64_t>(
      std::get<core::expressions::DimType>(memory.at(core::compute::kNodeMemoryTotalBytesKey)));
  EXPECT_EQ(std::get<int64_t>(std::get<core::expressions::DimType>(
                memory.at(core::compute::kNodeMemoryDuringBytesKey))),
            total + 48);
  // A concrete predecessor must not turn a later unregistered phase numeric.
  node = graph.add_node();
  node->set_op_type("Abs");
  node->add_input("Y");
  node->add_output("Z");
  graph.mutable_output(0)->set_name("Z");
  ctx.Compute(graph);
  EXPECT_TRUE(ctx.NodePeakMemory(1).worst_case.IsExpr());
  EXPECT_TRUE(std::holds_alternative<std::string>(std::get<core::expressions::DimType>(
      ctx.NodeMemory(1).at(core::compute::kNodeMemoryDuringBytesKey))));
}

TEST(ComputePeakMemoryDecoder, RuntimeReplacementSurvivesRepeatedBuiltinRegistration) {
  onnx_shapes::RegisterPeakMemoryFunctions();
  const auto portable = core::shapes::PeakMemoryDispatchTable().at("ai.onnx:MatMul");
  core::shapes::RegisterComputePeakMemoryFn(
      "", "MatMul", Device::kCPU, [](Device, const NodeProto &, const std::vector<SymShape> &) {
        return PeakMemoryEstimate(int64_t{123});
      });
  onnx_shapes::RegisterPeakMemoryFunctions();
  EXPECT_EQ(DecoderScratch("MatMul", {{2, 3}, {3, 5}}).worst_case.AsInt(), 123);
  EXPECT_EQ(
      core::shapes::ComputePeakMemory("ai.onnx", "MatMul", Device::kUndefined, {{2, 3}, {3, 5}})
          .worst_case.AsInt(),
      123);
  core::shapes::RegisterComputePeakMemoryFn("", "MatMul", Device::kCPU, portable);
  EXPECT_EQ(DecoderScratch("MatMul", {{2, 3}, {3, 5}}).worst_case.AsInt(), 4 * (6 + 15 + 10));
}

} // namespace Test
