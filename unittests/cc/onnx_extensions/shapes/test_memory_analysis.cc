// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/compute_context.h"
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

void MemoryInput(GraphProto &graph, const std::string &name, const std::vector<int64_t> &shape,
                 TensorProto::DataType dtype = TensorProto::FLOAT) {
  auto *input = graph.add_input();
  input->set_name(name);
  auto *tensor = input->add_type()->add_tensor_type();
  tensor->set_elem_type(dtype);
  auto *dims = tensor->add_shape();
  for (int64_t dim : shape)
    dims->add_dim()->set_dim_value(dim);
}

template <typename GraphOrFunction>
NodeProto &MemoryNode(GraphOrFunction &graph, const std::string &op,
                      const std::vector<std::string> &inputs, const std::string &output) {
  auto &node = *graph.add_node();
  node.set_op_type(op);
  for (const std::string &input : inputs)
    node.add_input(input);
  node.add_output(output);
  return node;
}

void MemoryInitializer(GraphProto &graph, const std::string &name,
                       const std::vector<int64_t> &shape) {
  auto *tensor = graph.add_initializer();
  tensor->set_name(name);
  tensor->set_data_type(TensorProto::FLOAT);
  for (int64_t dim : shape)
    tensor->add_dims(dim);
}

int64_t MemoryTotal(const core::compute::ComputeContext &ctx, std::size_t node = 0) {
  return std::get<int64_t>(std::get<core::expressions::DimType>(
      ctx.NodeMemory(node).at(core::compute::kNodeMemoryTotalBytesKey)));
}

constexpr int64_t kLarge = int64_t{1} << 31;

} // namespace

TEST(MemoryAnalysis, RejectsInputAndInitializerByteOverflowWithoutPayloads) {
  GraphProto input;
  MemoryInput(input, "X", {kLarge, kLarge});
  MemoryNode(input, "Identity", {"X"}, "Y");
  core::compute::ComputeContext ctx;
  ASSERT_NO_THROW(ctx.ComputeShapes(input));
  EXPECT_THROW(ctx.Compute(input), std::invalid_argument);

  GraphProto initializer;
  MemoryInitializer(initializer, "W", {kLarge, kLarge});
  ASSERT_NO_THROW(ctx.ComputeShapes(initializer));
  EXPECT_THROW(ctx.Compute(initializer), std::invalid_argument);

  ModelProto model;
  *model.mutable_graph() = input;
  EXPECT_THROW(ctx.Compute(model), std::invalid_argument);
}

TEST(MemoryAnalysis, RejectsInferredIntermediateAndOutputByteOverflow) {
  for (bool intermediate : {false, true}) {
    GraphProto graph;
    MemoryInput(graph, "X", {1});
    auto *target = graph.add_initializer();
    target->set_name("shape");
    target->set_data_type(TensorProto::INT64);
    target->add_dims(2);
    target->add_int64_data(kLarge);
    target->add_int64_data(kLarge);
    MemoryNode(graph, "Expand", {"X", "shape"}, "expanded");
    if (intermediate)
      MemoryNode(graph, "Identity", {"expanded"}, "Y");
    graph.add_output()->set_name(intermediate ? "Y" : "expanded");
    core::compute::ComputeContext ctx;
    ASSERT_NO_THROW(ctx.ComputeShapes(graph));
    ASSERT_TRUE(ctx.Shapes().Has("expanded"));
    EXPECT_EQ(ctx.Shapes().Get("expanded").Shape()[0].AsInt(), kLarge);
    EXPECT_EQ(ctx.Shapes().Get("expanded").Shape()[1].AsInt(), kLarge);
    EXPECT_THROW(ctx.Compute(graph), std::invalid_argument);
  }
}

TEST(MemoryAnalysis, RejectsAggregateInitializersEvenWithoutNodes) {
  GraphProto graph;
  for (int i = 0; i < 2048; ++i)
    MemoryInitializer(graph, "W" + std::to_string(i), {int64_t{1} << 50});
  core::compute::ComputeContext ctx;
  EXPECT_THROW(ctx.Compute(graph), std::invalid_argument);
}

TEST(MemoryAnalysis, RejectsLiveAllocationAndOutputSums) {
  for (bool output_overflow : {false, true}) {
    GraphProto graph;
    MemoryInput(graph, "X", {int64_t{1} << (output_overflow ? 60 : 59)});
    if (!output_overflow)
      MemoryInput(graph, "Z", {int64_t{1} << 59});
    MemoryNode(graph, "Neg", {"X"}, "A");
    MemoryNode(graph, "Neg", {output_overflow ? "X" : "Z"}, "B");
    MemoryNode(graph, "Add", {"A", "B"}, "Y");
    core::compute::ComputeContext ctx;
    EXPECT_THROW(ctx.Compute(graph), std::invalid_argument);
  }
}

TEST(MemoryAnalysis, RejectsScratchPlusTensorOverflow) {
  const auto device = core::symbolic::MakeGPUDevice(123);
  core::shapes::RegisterComputePeakMemoryFn(
      "", "Identity", device,
      [](core::symbolic::Device, const std::vector<core::symbolic::SymShape> &) {
        return std::numeric_limits<int64_t>::max() - 7;
      });
  GraphProto graph;
  MemoryInput(graph, "X", {1});
  MemoryNode(graph, "Identity", {"X"}, "Y");
  core::compute::ComputeContext ctx;
  EXPECT_THROW(ctx.Compute(graph, device), std::invalid_argument);
  core::shapes::RegisterComputePeakMemoryFn(
      "", "Identity", device,
      [](core::symbolic::Device, const std::vector<core::symbolic::SymShape> &) {
        return std::numeric_limits<int64_t>::max() - 8;
      });
  ASSERT_NO_THROW(ctx.Compute(graph, device));
  EXPECT_EQ(ctx.NodePeakMemory(0), std::numeric_limits<int64_t>::max() - 8);
}

TEST(MemoryAnalysis, RejectsNegativeScratchWithoutTensorProfiles) {
  const auto device = core::symbolic::MakeGPUDevice(124);
  core::shapes::RegisterComputePeakMemoryFn(
      "", "Identity", device,
      [](core::symbolic::Device, const std::vector<core::symbolic::SymShape> &) -> int64_t {
        return -1;
      });
  GraphProto graph;
  MemoryInput(graph, "X", {1});
  MemoryNode(graph, "Identity", {"X"}, "Y");
  core::compute::ComputeContext ctx;
  ctx.ComputeShapes(graph);
  EXPECT_THROW(ctx.ComputePeakMemory(graph, device), std::invalid_argument);
}

TEST(MemoryAnalysis, StandaloneScratchIgnoresUnrelatedTensorProfiles) {
  GraphProto first;
  MemoryInput(first, "X", {int64_t{1} << 59});
  MemoryNode(first, "Identity", {"X"}, "Y");
  core::compute::ComputeContext ctx;
  ctx.Compute(first);
  const auto original_profile = ctx.NodeMemory(0);

  onnx_shapes::RegisterPeakMemoryFunctions();
  GraphProto second;
  for (const auto &name : {"Q", "K", "V"})
    MemoryInput(second, name, {1, 1, int64_t{1} << 30, 1});
  MemoryNode(second, "Attention", {"Q", "K", "V"}, "Y");
  ctx.ComputeShapes(second);
  ASSERT_NO_THROW(ctx.ComputePeakMemory(second));
  EXPECT_EQ(ctx.NodePeakMemory(0), int64_t{1} << 62);
  EXPECT_EQ(ctx.NodeMemory(0), original_profile);
  ModelProto model;
  *model.mutable_graph() = second;
  ctx.Compute(first);
  ctx.ComputeShapes(model);
  EXPECT_NO_THROW(ctx.ComputePeakMemory(model.graph()));
  EXPECT_EQ(ctx.NodeMemory(0), original_profile);
  EXPECT_NO_THROW(ctx.Compute(second));
}

TEST(MemoryAnalysis, RejectsUnsupportedAllocations) {
  for (auto dtype : {TensorProto::STRING, TensorProto::UNDEFINED, TensorProto::FLOAT6E2M3}) {
    GraphProto graph;
    MemoryInput(graph, "X", {0}, dtype);
    MemoryNode(graph, "Identity", {"X"}, "Y");
    core::compute::ComputeContext ctx;
    EXPECT_THROW(ctx.Compute(graph), std::invalid_argument);
  }
  GraphProto sparse;
  sparse.add_sparse_initializer()->mutable_values()->set_name("W");
  GraphProto extended;
  extended.add_extended_initializer()->mutable_paged_cache()->set_name("W");
  GraphProto strings;
  auto *string_tensor = strings.add_initializer();
  string_tensor->set_name("W");
  string_tensor->set_data_type(TensorProto::STRING);
  core::compute::ComputeContext ctx;
  EXPECT_THROW(ctx.Compute(sparse), std::invalid_argument);
  EXPECT_THROW(ctx.Compute(extended), std::invalid_argument);
  EXPECT_THROW(ctx.Compute(strings), std::invalid_argument);
}

TEST(MemoryAnalysis, RejectsUnresolvedOutputAllocation) {
  GraphProto graph;
  MemoryInput(graph, "X", {2});
  MemoryNode(graph, "Identity", {"X"}, "Y");
  graph.add_output()->set_name("missing");
  core::compute::ComputeContext ctx;
  EXPECT_THROW(ctx.Compute(graph), std::invalid_argument);

  GraphProto nonzero;
  MemoryInput(nonzero, "X", {2});
  MemoryNode(nonzero, "NonZero", {"X"}, "Y");
  EXPECT_THROW(ctx.Compute(nonzero), std::invalid_argument);

  GraphProto missing_shape;
  MemoryInput(missing_shape, "X", {2});
  missing_shape.mutable_input(0)->mutable_type()->mutable_tensor_type()->clear_shape();
  MemoryNode(missing_shape, "Identity", {"X"}, "Y");
  EXPECT_THROW(ctx.Compute(missing_shape), std::invalid_argument);
}

TEST(MemoryAnalysis, AllowsPartialLowLevelAnalysis) {
  GraphProto unknown_dimension;
  MemoryInput(unknown_dimension, "X", {2});
  MemoryNode(unknown_dimension, "Identity", {"X"}, "Y");
  core::shapes::ShapesContext shapes;
  shapes.Set("X", core::symbolic::SymTensor(nullptr, core::symbolic::TensorType::kFloat,
                                            core::symbolic::SymShape{core::symbolic::SymDim(2)}));
  shapes.Set("Y", core::symbolic::SymTensor(
                      nullptr, core::symbolic::TensorType::kFloat,
                      core::symbolic::SymShape{core::symbolic::SymDim(std::string{})}));
  core::compute::ComputeContext ctx;
  EXPECT_NO_THROW(ctx.ComputeInPlaceReuseGraph(unknown_dimension, shapes));
}

TEST(MemoryAnalysis, RejectsUnaccountedSubgraphAllocations) {
  for (const std::string allocation : {"valid", "overflow", "unsupported"}) {
    SCOPED_TRACE(allocation);
    GraphProto graph;
    MemoryInput(graph, "condition", {}, TensorProto::BOOL);
    MemoryInput(graph, "X", {1});
    auto &node = MemoryNode(graph, "If", {"condition"}, "Y");
    for (const char *name : {"then_branch", "else_branch"}) {
      auto *attribute = node.add_attribute();
      attribute->set_name(name);
      attribute->set_type(AttributeProto::GRAPH);
      auto &branch = *attribute->mutable_g();
      MemoryInitializer(branch, "W",
                        allocation == "overflow" ? std::vector<int64_t>{kLarge, kLarge}
                                                 : std::vector<int64_t>{1});
      if (allocation == "unsupported")
        branch.mutable_initializer(0)->set_data_type(TensorProto::STRING);
      MemoryNode(branch, "Identity", {"X"}, "result");
      branch.add_output()->set_name("result");
    }
    graph.add_output()->set_name("Y");
    core::compute::ComputeContext ctx;
    ASSERT_NO_THROW(ctx.ComputeShapes(graph));
    EXPECT_NO_THROW(ctx.ComputeInPlaceReuseGraph(graph, ctx.Shapes()));
    EXPECT_THROW(ctx.Compute(graph), std::invalid_argument);
    ModelProto model;
    *model.mutable_graph() = graph;
    EXPECT_THROW(ctx.Compute(model), std::invalid_argument);
  }
}

TEST(MemoryAnalysis, ValidatesLocalFunctionBodyAllocationsRecursively) {
  for (bool nested : {false, true}) {
    for (auto dtype : {TensorProto::STRING, TensorProto::FLOAT}) {
      SCOPED_TRACE(::testing::Message() << "nested=" << nested << " dtype=" << dtype);
      ModelProto model;
      model.add_opset_import()->set_version(23);
      auto *local_opset = model.add_opset_import();
      local_opset->set_domain("memory_test");
      local_opset->set_version(1);
      MemoryInput(*model.mutable_graph(), "X", {2});
      auto &call = MemoryNode(*model.mutable_graph(), nested ? "Outer" : "Body", {"X", ""}, "Y");
      call.set_domain("memory_test");
      model.mutable_graph()->add_output()->set_name("Y");
      auto *body = model.add_functions();
      body->set_domain("memory_test");
      body->set_name("Body");
      body->add_input("x");
      body->add_input("omitted");
      body->add_output("y");
      body->add_opset_import()->set_version(23);
      auto *to = MemoryNode(*body, "Cast", {"x"}, "cast").add_attribute();
      to->set_name("to");
      to->set_type(AttributeProto::INT);
      to->set_i(dtype);
      MemoryNode(*body, "Shape", {"cast"}, "y");
      if (nested) {
        auto *outer = model.add_functions();
        outer->set_domain("memory_test");
        outer->set_name("Outer");
        outer->add_input("x");
        outer->add_input("omitted");
        outer->add_output("y");
        auto *opset = outer->add_opset_import();
        opset->set_domain("memory_test");
        opset->set_version(1);
        MemoryNode(*outer, "Body", {"x", ""}, "y").set_domain("memory_test");
      }
      core::compute::ComputeContext ctx;
      ASSERT_NO_THROW(ctx.ComputeShapes(model));
      EXPECT_NO_THROW(ctx.ComputeInPlaceReuseGraph(model.graph(), ctx.Shapes()));
      if (dtype == TensorProto::STRING)
        EXPECT_THROW(ctx.Compute(model), std::invalid_argument);
      else
        EXPECT_NO_THROW(ctx.Compute(model));
    }
  }
}

TEST(MemoryAnalysis, RejectsSymbolicAllocationCoefficientOverflow) {
  for (bool empty : {false, true}) {
    GraphProto graph;
    MemoryInput(graph, "X", {1, kLarge, kLarge, empty ? 0 : 1});
    auto *dim =
        graph.mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(
            0);
    dim->clear_dim_value();
    dim->set_dim_param("batch");
    MemoryNode(graph, "Identity", {"X"}, "Y");
    core::compute::ComputeContext ctx;
    ASSERT_NO_THROW(ctx.ComputeShapes(graph));
    if (empty) {
      ASSERT_NO_THROW(ctx.Compute(graph));
      EXPECT_EQ(MemoryTotal(ctx), 0);
    } else {
      EXPECT_THROW(ctx.Compute(graph), std::invalid_argument);
    }
  }
}

TEST(MemoryAnalysis, DoesNotDoubleCountAliasedLocalFunctionInputs) {
  ModelProto model;
  model.add_opset_import()->set_version(23);
  auto *local_opset = model.add_opset_import();
  local_opset->set_domain("memory_test");
  local_opset->set_version(1);
  MemoryInput(*model.mutable_graph(), "X", {int64_t{1} << 60});
  MemoryNode(*model.mutable_graph(), "Body", {"X", "X"}, "Y").set_domain("memory_test");
  model.mutable_graph()->add_output()->set_name("Y");
  auto *body = model.add_functions();
  body->set_domain("memory_test");
  body->set_name("Body");
  body->add_input("a");
  body->add_input("b");
  body->add_output("y");
  body->add_opset_import()->set_version(23);
  MemoryNode(*body, "Shape", {"a"}, "y");
  core::compute::ComputeContext ctx;
  ASSERT_NO_THROW(ctx.Compute(model));
  EXPECT_EQ(MemoryTotal(ctx), (int64_t{1} << 62) + 8);
}

TEST(MemoryAnalysis, PreservesEmptyShapeInferenceBounds) {
  for (const std::string op : {"NonZero", "Compress"}) {
    SCOPED_TRACE(op);
    GraphProto graph;
    MemoryInput(graph, "X", {kLarge, kLarge, kLarge, 0});
    std::vector<std::string> inputs{"X"};
    if (op == "Compress") {
      MemoryInput(graph, "condition", {0}, TensorProto::BOOL);
      inputs.push_back("condition");
    }
    MemoryNode(graph, op, inputs, "Y");
    core::compute::ComputeContext ctx;
    EXPECT_NO_THROW(ctx.ComputeShapes(graph));
    EXPECT_TRUE(ctx.Shapes().Has("Y"));
  }
}

TEST(MemoryAnalysis, PreservesZeroAndLargeRepresentableAllocations) {
  for (const auto &shape :
       {std::vector<int64_t>{kLarge, kLarge, 0}, std::vector<int64_t>{int64_t{1} << 52}}) {
    GraphProto graph;
    MemoryInput(graph, "X", shape);
    MemoryNode(graph, "Identity", {"X"}, "Y");
    core::compute::ComputeContext ctx;
    ASSERT_NO_THROW(ctx.Compute(graph));
    EXPECT_EQ(MemoryTotal(ctx), shape.size() == 3 ? 0 : int64_t{1} << 55);
  }
}

TEST(MemoryAnalysis, PreservesInputDimensionExpressions) {
  GraphProto graph;
  MemoryInput(graph, "X", {1});
  auto *dim =
      graph.mutable_input(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(
          0);
  dim->clear_dim_value();
  dim->set_dim_param("batch");
  MemoryNode(graph, "Identity", {"X"}, "Y");
  core::compute::ComputeContext ctx;
  ASSERT_NO_THROW(ctx.Compute(graph));
  EXPECT_TRUE(std::holds_alternative<std::string>(std::get<core::expressions::DimType>(
      ctx.NodeMemory(0).at(core::compute::kNodeMemoryTotalBytesKey))));
}

TEST(MemoryAnalysis, InitializerResolvesDefaultInputAllocation) {
  GraphProto graph;
  MemoryInitializer(graph, "X", {2});
  MemoryInput(graph, "X", {2});
  graph.mutable_input(0)->mutable_type()->mutable_tensor_type()->clear_shape();
  MemoryNode(graph, "Identity", {"X"}, "Y");
  core::compute::ComputeContext ctx;
  ASSERT_NO_THROW(ctx.Compute(graph));
  EXPECT_EQ(MemoryTotal(ctx), 16);
}

TEST(MemoryAnalysis, PreservesPackedRoundingAtInt64Boundary) {
  GraphProto graph;
  MemoryInput(graph, "X", {std::numeric_limits<int64_t>::max()}, TensorProto::UINT2);
  MemoryNode(graph, "Identity", {"X"}, "Y");
  core::compute::ComputeContext ctx;
  ASSERT_NO_THROW(ctx.Compute(graph));
  EXPECT_EQ(MemoryTotal(ctx), int64_t{1} << 62);

  GraphProto initializer;
  MemoryInitializer(initializer, "W", {std::numeric_limits<int64_t>::max()});
  initializer.mutable_initializer(0)->set_data_type(TensorProto::UINT8);
  EXPECT_NO_THROW(ctx.Compute(initializer));
}

TEST(MemoryAnalysis, RejectsAttentionScratchOverflowThroughPublicAnalysis) {
  onnx_shapes::RegisterPeakMemoryFunctions();
  GraphProto graph;
  for (const auto &name : {"Q", "K", "V"})
    MemoryInput(graph, name, {1, 1, kLarge, 1});
  MemoryNode(graph, "Attention", {"Q", "K", "V"}, "Y");
  core::compute::ComputeContext ctx;
  ASSERT_NO_THROW(ctx.ComputeShapes(graph));
  EXPECT_THROW(ctx.Compute(graph), std::invalid_argument);
}

TEST(MemoryAnalysis, RejectsLinearAttentionScratchWithInferredAndExplicitState) {
  onnx_shapes::RegisterPeakMemoryFunctions();
  for (bool explicit_state : {false, true}) {
    GraphProto graph;
    for (const auto &name : {"Q", "K", "V"})
      MemoryInput(graph, name, {1, 1, kLarge});
    std::vector<std::string> inputs{"Q", "K", "V"};
    if (explicit_state) {
      MemoryInput(graph, "state", {1, 1, kLarge, kLarge});
      inputs.push_back("state");
    }
    auto &node = MemoryNode(graph, "LinearAttention", inputs, "Y");
    for (const auto &name : {"q_num_heads", "kv_num_heads"}) {
      auto *attr = node.add_attribute();
      attr->set_name(name);
      attr->set_type(AttributeProto::INT);
      attr->set_i(1);
    }
    core::compute::ComputeContext ctx;
    ASSERT_NO_THROW(ctx.ComputeShapes(graph));
    EXPECT_THROW(ctx.Compute(graph), std::invalid_argument);
  }
}

} // namespace Test
