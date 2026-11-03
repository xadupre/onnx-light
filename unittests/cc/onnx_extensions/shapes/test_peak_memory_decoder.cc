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

PeakMemoryEstimate DecoderPeakMemory(const std::string &op, const std::vector<SymShape> &inputs) {
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

TEST(ComputePeakMemoryDecoder, MatMulIncludesPromotionsAndPreallocatedOutputTemporary) {
  EXPECT_EQ(DecoderPeakMemory("MatMul", {{2, 3}, {3, 2}}).worst_case.AsInt(), 72);
  EXPECT_EQ(DecoderPeakMemory("MatMul", {{2, 3}, {3, 5}}).worst_case.AsInt(),
            4 * (2 * 3 + 3 * 5) + 6 * (2 * 5));
  EXPECT_EQ(DecoderPeakMemory("MatMul", {{2, 1, 3, 4}, {1, 7, 4, 5}}).worst_case.AsInt(),
            4 * (2 * 3 * 4 + 7 * 4 * 5) + 6 * (2 * 7 * 3 * 5));
  EXPECT_EQ(DecoderPeakMemory("MatMul", {{3}, {3}}).worst_case.AsInt(), 4 * (3 + 3) + 6);
  EXPECT_EQ(DecoderPeakMemory("MatMul", {{3}, {3, 5}}).worst_case.AsInt(), 4 * (3 + 15) + 6 * 5);
  EXPECT_EQ(DecoderPeakMemory("MatMul", {{2, 3}, {3}}).worst_case.AsInt(), 4 * (6 + 3) + 6 * 2);
  EXPECT_EQ(DecoderPeakMemory("MatMul", {{2, SymDim(int64_t{0})}, {SymDim(int64_t{0}), 5}})
                .worst_case.AsInt(),
            6 * 10);
  EXPECT_FALSE(DecoderPeakMemory("MatMul", {{2, 3}, {3, 5}}).average_case);
}

TEST(ComputePeakMemoryDecoder, IndexingAndNormalizationHaveNonzeroBounds) {
  const auto gather = DecoderPeakMemory("Gather", {{8, 16}, {2, 3}});
  EXPECT_EQ(gather, PeakMemoryEstimate::Exact(SymDim(int64_t{48})));
  EXPECT_EQ(DecoderPeakMemory("Gather", {{8, 16}, {}}).worst_case.AsInt(), 8);
  EXPECT_EQ(DecoderPeakMemory("Gather", {{8, 16}, {SymDim(int64_t{0})}}).worst_case.AsInt(), 0);
  EXPECT_EQ(DecoderPeakMemory("Slice", {{8, 16}, {2}, {2}}).worst_case.AsInt(), 64);
  EXPECT_EQ(DecoderPeakMemory("Slice", {{8, 16}, {2}, {2}, {2}, {2}}).worst_case.AsInt(), 64);
  EXPECT_EQ(DecoderPeakMemory("Slice", {{8, 16}, {SymDim(int64_t{0})}, {SymDim(int64_t{0})}})
                .worst_case.AsInt(),
            0);
  EXPECT_EQ(DecoderPeakMemory("RMSNormalization", {{2, 3, 4}, {4}}).worst_case.AsInt(),
            8 * (4 + 1 + 1) + 4 * (24 + 4));
  const auto node = DecoderNode("RMSNormalization", "axis", 1);
  EXPECT_EQ(
      core::shapes::ComputePeakMemory(Device::kCPU, node, {{2, 3, 4}, {3, 4}}).worst_case.AsInt(),
      8 * (12 + 2 + 2) + 4 * (24 + 12));
  // An empty normalization block still acquires a one-element index buffer.
  EXPECT_GE(DecoderPeakMemory("RMSNormalization", {{2, SymDim(int64_t{0})}, {SymDim(int64_t{0})}})
                .worst_case.AsInt(),
            8);
}

TEST(ComputePeakMemoryDecoder, CommonOperatorsDefaultToZeroPeakMemory) {
  for (const auto *op : {"Swish", "Shape", "Transpose", "Unsqueeze"})
    EXPECT_EQ(DecoderPeakMemory(op, {{2, 3}}), PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
  EXPECT_EQ(DecoderPeakMemory("CumSum", {{2, 3}, {}}),
            PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
  EXPECT_EQ(DecoderPeakMemory("Where", {{2, 3}, {2, 3}, {}}),
            PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
  const auto cast = DecoderNode("Cast", "to", TensorProto::FLOAT);
  EXPECT_EQ(core::shapes::ComputePeakMemory(Device::kCPU, cast, {{2, 3}}),
            PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
}

TEST(ComputePeakMemoryDecoder, BinaryOperatorsDefaultToZeroPeakMemory) {
  for (const auto *op : {"Mul", "Sub", "GreaterOrEqual", "And"}) {
    EXPECT_EQ(DecoderPeakMemory(op, {{2, 3}, {}}), PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
    EXPECT_EQ(DecoderPeakMemory(op, {{2, 3}, {2, 3}}).worst_case.AsInt(), 0);
    EXPECT_EQ(DecoderPeakMemory(op, {{2, 1, 3}, {7, 3}}).worst_case.AsInt(), 0);
    EXPECT_EQ(DecoderPeakMemory(op, {{SymDim(int64_t{0}), 3}, {3}}).worst_case.AsInt(), 0);
    EXPECT_EQ(DecoderPeakMemory(op, {{}, {}}).worst_case.AsInt(), 0);
    EXPECT_EQ(DecoderPeakMemory(op, {{SymDim("N"), 3}, {3}}).worst_case.AsInt(), 0);
  }
}

TEST(ComputePeakMemoryDecoder, SymbolicBoundsResolveWithDimensionBindings) {
  const auto matmul = DecoderPeakMemory("MatMul", {{SymDim("N"), 3}, {3, 5}});
  ASSERT_TRUE(matmul.worst_case.IsExpr());
  EXPECT_EQ(core::expressions::evaluate_expression(matmul.worst_case.AsExpr(), {{"N", 2}}),
            4 * (6 + 15) + 6 * 10);
  const auto gather = DecoderPeakMemory("Gather", {{8, 16}, {SymDim("N")}});
  EXPECT_EQ(core::expressions::evaluate_expression(gather.worst_case.AsExpr(), {{"N", 7}}), 56);
  const auto slice = DecoderPeakMemory("Slice", {{8, 16}, {SymDim("N")}, {SymDim("N")}});
  EXPECT_EQ(core::expressions::evaluate_expression(slice.worst_case.AsExpr(), {{"N", 2}}), 64);
  const auto rms = DecoderPeakMemory("RMSNormalization", {{2, SymDim("N")}, {SymDim("N")}});
  EXPECT_EQ(core::expressions::evaluate_expression(rms.worst_case.AsExpr(), {{"N", 4}}),
            8 * (4 + 1 + 1) + 4 * (8 + 4));
  EXPECT_GE(core::expressions::evaluate_expression(rms.worst_case.AsExpr(), {{"N", 0}}), 8);
}

TEST(ComputePeakMemoryDecoder, UnsupportedQueriesRetainUnresolvedBounds) {
  for (const auto *op : {"MatMul", "Gather", "Slice", "RMSNormalization"}) {
    EXPECT_EQ(DecoderPeakMemory(op, {}), core::shapes::UnknownPeakMemory("", op));
  }
  EXPECT_EQ(DecoderPeakMemory("MatMul", {{}, {3, 5}}),
            core::shapes::UnknownPeakMemory("", "MatMul"));
  EXPECT_EQ(DecoderPeakMemory("Gather", {{8}, {SymDim("?")}}),
            core::shapes::UnknownPeakMemory("", "Gather"));
  EXPECT_EQ(DecoderPeakMemory("Slice", {{8}, {}, {}}),
            core::shapes::UnknownPeakMemory("", "Slice"));
  EXPECT_EQ(DecoderPeakMemory("Slice", {{8}, {1}, {2}}),
            core::shapes::UnknownPeakMemory("", "Slice"));
  EXPECT_EQ(DecoderPeakMemory("Cast", {{2, 3}}).worst_case.AsInt(), 0);
  const auto cast = DecoderNode("Cast", "to", TensorProto::STRING);
  EXPECT_EQ(core::shapes::ComputePeakMemory(Device::kCPU, cast, {{2, 3}}),
            PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
  EXPECT_EQ(core::shapes::ComputePeakMemory("custom", "MatMul", Device::kCPU, {{2, 3}, {3, 5}}),
            PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
  EXPECT_EQ(core::shapes::ComputePeakMemory("", "MatMul", core::symbolic::MakeGPUDevice(0),
                                            {{2, 3}, {3, 5}}),
            PeakMemoryEstimate::Exact(SymDim(int64_t{0})));
}

TEST(ComputePeakMemoryDecoder, PartialContextsDoNotTurnMissingInputsIntoScalars) {
  onnx_shapes::RegisterPeakMemoryFunctions();
  for (const auto *op : {"Gather", "RMSNormalization", "UnregisteredDecoderOp"}) {
    SCOPED_TRACE(op);
    GraphProto graph;
    auto &node = *graph.add_node();
    node.set_op_type(op);
    node.add_input("X");
    node.add_input("W");
    node.add_output("Y");
    core::compute::ComputeContext ctx;
    ctx.Shapes().Set(
        "X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {2, 3, 4}));
    const bool registered = std::string(op) != "UnregisteredDecoderOp";
    const auto missing = registered ? core::shapes::UnknownPeakMemory("", op)
                                    : PeakMemoryEstimate::Exact(SymDim(int64_t{0}));
    const auto check_metadata = [&](const PeakMemoryEstimate &expected) {
      core::compute::WritePeakMemoryToMetadata(graph, ctx.Shapes(), Device::kCPU);
      bool found = false;
      for (const auto &entry : node.metadata_props()) {
        if (entry.key() == core::compute::kNodePeakMemoryMaxMetadataKey) {
          found = true;
          EXPECT_EQ(entry.value(), expected.worst_case.ToString());
        }
      }
      EXPECT_TRUE(found);
    };
    ctx.ComputePeakMemory(graph);
    EXPECT_EQ(ctx.NodePeakMemory(0), missing);
    ctx.AppendNodePeakMemory(node, 0);
    EXPECT_EQ(ctx.NodePeakMemory(0), missing);
    check_metadata(missing);

    ctx.Shapes().Set("W",
                     core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {}));
    const auto scalar = DecoderPeakMemory(op, {{2, 3, 4}, {}});
    ASSERT_TRUE(scalar.worst_case.IsInt());
    ctx.ComputePeakMemory(graph);
    EXPECT_EQ(ctx.NodePeakMemory(0), scalar);
    ctx.AppendNodePeakMemory(node, 0);
    EXPECT_EQ(ctx.NodePeakMemory(0), scalar);
    check_metadata(scalar);
  }
}

TEST(ComputePeakMemoryDecoder, ContextQueriesPreserveOmittedOptionalInputSlots) {
  onnx_shapes::RegisterPeakMemoryFunctions();
  GraphProto graph;
  auto &node = *graph.add_node();
  node.set_op_type("Slice");
  for (const auto *input : {"X", "starts", "ends", "", "steps"})
    node.add_input(input);
  node.add_output("Y");
  core::compute::ComputeContext ctx;
  ctx.Shapes().Set("X",
                   core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat, {8, 16}));
  for (const auto *input : {"starts", "ends", "steps"})
    ctx.Shapes().Set(input,
                     core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kInt64, {2}));
  ctx.ComputePeakMemory(graph);
  EXPECT_EQ(ctx.NodePeakMemory(0).worst_case.AsInt(), 64);
}

TEST(ComputePeakMemoryDecoder, RejectsInvalidDimensionsAndOverflow) {
  const int64_t large = std::numeric_limits<int64_t>::max();
  EXPECT_THROW(DecoderPeakMemory("MatMul", {{2, 3}, {4, 5}}), std::invalid_argument);
  EXPECT_THROW(DecoderPeakMemory("MatMul", {{2, 3, 4}, {7, 4, 5}}), std::invalid_argument);
  EXPECT_THROW(DecoderPeakMemory("MatMul", {{large, 2}, {2, 1}}), std::invalid_argument);
  EXPECT_THROW(
      DecoderPeakMemory("MatMul", {{large / 6 + 1, SymDim(int64_t{0})}, {SymDim(int64_t{0}), 1}}),
      std::invalid_argument);
  EXPECT_THROW(DecoderPeakMemory("Gather", {{8}, {large}}), std::invalid_argument);
  EXPECT_THROW(DecoderPeakMemory("Slice", {{8}, {large}, {large}}), std::invalid_argument);
  EXPECT_THROW(DecoderPeakMemory("RMSNormalization", {{large}, {large}}), std::invalid_argument);
  // No zero factor or zero-workspace fast path may hide negative dimensions.
  EXPECT_THROW(DecoderPeakMemory("MatMul", {{SymDim(int64_t{0}), -1}, {-1, 5}}),
               std::invalid_argument);
  EXPECT_THROW(DecoderPeakMemory("Gather", {{8}, {SymDim(int64_t{0}), -1}}), std::invalid_argument);
  EXPECT_THROW(core::shapes::ComputePeakMemory(
                   Device::kCPU, DecoderNode("RMSNormalization", "axis", large), {{2, 3}, {3}}),
               std::invalid_argument);
}

TEST(ComputePeakMemoryDecoder, PublicAnalysisIncludesPeakMemoryAndDefaultsMissingEstimatesToZero) {
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
  EXPECT_EQ(ctx.NodePeakMemory(0).worst_case.AsInt(), 56);
  const auto &memory = ctx.NodeMemory(0);
  const auto total = std::get<int64_t>(
      std::get<core::expressions::DimType>(memory.at(core::compute::kNodeMemoryTotalBytesKey)));
  EXPECT_EQ(std::get<int64_t>(std::get<core::expressions::DimType>(
                memory.at(core::compute::kNodeMemoryDuringBytesKey))),
            total + 56);
  // An unregistered operator adds zero to its during-node total.
  node = graph.add_node();
  node->set_op_type("Abs");
  node->add_input("Y");
  node->add_output("Z");
  graph.mutable_output(0)->set_name("Z");
  ctx.Compute(graph);
  EXPECT_EQ(ctx.NodePeakMemory(1).worst_case.AsInt(), 0);
  EXPECT_EQ(std::get<core::expressions::DimType>(
                ctx.NodeMemory(1).at(core::compute::kNodeMemoryDuringBytesKey)),
            std::get<core::expressions::DimType>(
                ctx.NodeMemory(1).at(core::compute::kNodeMemoryTotalBytesKey)));
  graph.mutable_node(1)->add_metadata(core::compute::kNodePeakMemoryMetadataKey, "123");
  core::compute::WritePeakMemoryToMetadata(graph, ctx.Shapes(), Device::kCPU);
  const auto &metadata = graph.node(1).metadata_props();
  ASSERT_EQ(metadata.size(), 2u);
  for (const auto &entry : metadata) {
    EXPECT_TRUE(entry.key() == core::compute::kNodePeakMemoryMaxMetadataKey ||
                entry.key() == core::compute::kNodePeakMemoryAverageMetadataKey);
    EXPECT_EQ(entry.value(), "0");
  }
}

TEST(ComputePeakMemoryDecoder, RuntimeReplacementSurvivesRepeatedBuiltinRegistration) {
  onnx_shapes::RegisterPeakMemoryFunctions();
  const auto portable = core::shapes::PeakMemoryDispatchTable().at("ai.onnx:MatMul");
  core::shapes::RegisterComputePeakMemoryFn(
      "", "MatMul", Device::kCPU, [](Device, const NodeProto &, const std::vector<SymShape> &) {
        return PeakMemoryEstimate(int64_t{123});
      });
  onnx_shapes::RegisterPeakMemoryFunctions();
  EXPECT_EQ(DecoderPeakMemory("MatMul", {{2, 3}, {3, 5}}).worst_case.AsInt(), 123);
  EXPECT_EQ(
      core::shapes::ComputePeakMemory("ai.onnx", "MatMul", Device::kUndefined, {{2, 3}, {3, 5}})
          .worst_case.AsInt(),
      123);
  core::shapes::RegisterComputePeakMemoryFn("", "MatMul", Device::kCPU, portable);
  EXPECT_EQ(DecoderPeakMemory("MatMul", {{2, 3}, {3, 5}}).worst_case.AsInt(),
            4 * (6 + 15) + 6 * 10);
}

} // namespace Test
