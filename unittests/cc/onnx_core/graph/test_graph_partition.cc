#include "onnx_core/graph/graph_partition.h"
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <tuple>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::graph;

namespace {

void Describe(ValueInfoProto *value, const std::string &name,
              const std::vector<int64_t> &shape = {}, int32_t dtype = TensorProto::FLOAT) {
  value->set_name(name);
  auto *tensor = value->mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(dtype);
  tensor->clear_shape();
  auto *dimensions = tensor->mutable_shape();
  for (int64_t dim : shape)
    dimensions->add_dim()->set_dim_value(dim);
}

GraphProto Chain(size_t count = 2) {
  GraphProto graph;
  Describe(graph.add_input(), "x");
  Describe(graph.add_output(), "y");
  for (size_t i = 0; i < count; ++i) {
    auto *node = graph.add_node();
    node->set_op_type("Identity");
    node->add_input(i == 0 ? "x" : "v" + std::to_string(i - 1));
    const std::string output = i + 1 == count ? "y" : "v" + std::to_string(i);
    node->add_output(output);
    if (i + 1 != count)
      Describe(graph.add_value_info(), output);
  }
  return graph;
}

} // namespace

TEST(GraphPartition, TinyBoundaryUnknownInternalAndOrderedPorts) {
  auto graph = Chain();
  graph.mutable_value_info(0)->mutable_type()->mutable_tensor_type()->clear_shape();
  graph.mutable_node(0)->add_input("");
  graph.mutable_node(0)->add_input("x");
  graph.mutable_node(0)->add_output("side");
  Describe(graph.add_output(), "side");
  const auto groups = ProposeOperatorGroups(graph, {12, 2, 128});
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].node_indices, std::vector<size_t>({0, 1}));
  EXPECT_EQ(groups[0].inputs, std::vector<std::string>({"x"}));
  EXPECT_EQ(groups[0].outputs, std::vector<std::string>({"side", "y"}));
  EXPECT_EQ(groups[0].transfer_bytes, 12u);
  EXPECT_TRUE(ProposeOperatorGroups(graph, {11, 2, 128}).empty());
}

TEST(GraphPartition, DynamicProgrammingBeatsEarlySmallGroup) {
  auto graph = Chain(4);
  // The cheap first interval loses to grouping all four operators.
  Describe(graph.mutable_value_info(0), "v0", {1000});
  Describe(graph.mutable_value_info(1), "v1", {});
  Describe(graph.mutable_value_info(2), "v2", {1000});
  auto groups = ProposeOperatorGroups(graph, {8, 2, 4});
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].node_indices, std::vector<size_t>({0, 1, 2, 3}));
  groups = ProposeOperatorGroups(graph, {8, 2, 2});
  ASSERT_EQ(groups.size(), 2u);
  EXPECT_EQ(groups[0].node_indices, std::vector<size_t>({0, 1}));
  EXPECT_EQ(groups[1].node_indices, std::vector<size_t>({2, 3}));
}

TEST(GraphPartition, EqualCostPrefersMoreGroupedNodesThenFirstPartition) {
  auto graph = Chain(5);
  std::unordered_map<std::string, uint64_t> sizes;
  for (const auto *name : {"x", "v0", "v1", "v2", "v3", "y"})
    sizes[name] = 0;
  const auto groups = ProposeOperatorGroups(graph, {0, 2, 3}, sizes);
  ASSERT_EQ(groups.size(), 2u);
  EXPECT_EQ(groups[0].node_indices, std::vector<size_t>({0, 1}));
  EXPECT_EQ(groups[1].node_indices, std::vector<size_t>({2, 3, 4}));
}

TEST(GraphPartition, DenseSparseAndStandardConstantsAreFreeButExplicit) {
  auto graph = Chain(3);
  graph.mutable_node(0)->set_op_type("Constant");
  graph.mutable_node(0)->set_domain("onnx.ai");
  graph.mutable_node(0)->clear_input();
  graph.add_initializer()->set_name("dense");
  graph.add_sparse_initializer()->mutable_values()->set_name("sparse");
  graph.mutable_node(1)->add_input("dense");
  graph.mutable_node(1)->add_input("sparse");
  const auto groups = ProposeOperatorGroups(graph, {4, 2, 2});
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].node_indices, std::vector<size_t>({1, 2}));
  EXPECT_EQ(groups[0].inputs, std::vector<std::string>({"v0", "dense", "sparse"}));
  EXPECT_EQ(groups[0].transfer_bytes, 4u);
  Describe(graph.add_input(), "sparse", {1000});
  EXPECT_TRUE(ProposeOperatorGroups(graph, {4, 2, 2}).empty());
  graph.mutable_input(1)->set_name("dense");
  EXPECT_TRUE(ProposeOperatorGroups(graph, {4, 2, 2}).empty());
}

TEST(GraphPartition, CustomAndOverloadedConstantsAreChargeable) {
  for (bool overloaded : {false, true}) {
    auto graph = Chain(3);
    auto *node = graph.mutable_node(0);
    node->set_op_type("Constant");
    node->clear_input();
    if (overloaded)
      node->set_overload("custom");
    else
      node->set_domain("custom");
    Describe(graph.mutable_value_info(0), "v0", {1000});
    Describe(graph.add_output(), "v0", {1000});
    EXPECT_TRUE(ProposeOperatorGroups(graph, {4, 2, 2}).empty());
  }
}

TEST(GraphPartition, CapturesRemainInputsAndLiveOutputs) {
  auto graph = Chain(3);
  Describe(graph.mutable_value_info(0), "v0", {1000});
  auto *body = graph.mutable_node(2)->add_attribute();
  body->set_name("body");
  body->set_type(AttributeProto::GRAPH);
  auto *capture = body->mutable_g()->add_node();
  capture->set_op_type("Identity");
  capture->add_input("v0");
  capture->add_output("nested");
  body->mutable_g()->add_output()->set_name("nested");
  EXPECT_TRUE(ProposeOperatorGroups(graph, {8, 2, 2}).empty());
  Describe(graph.add_input(), "external", {1000});
  *graph.mutable_node(1)->add_attribute() = *body;
  graph.mutable_node(2)->clear_attribute();
  capture = graph.mutable_node(1)->mutable_attribute(0)->mutable_g()->mutable_node(0);
  capture->ref_input()[0] = "external";
  EXPECT_TRUE(ProposeOperatorGroups(graph, {8, 2, 128}).empty());
  capture->ref_input()[0] = "undefined";
  EXPECT_THROW(ProposeOperatorGroups(graph), std::invalid_argument);
}

TEST(GraphPartition, UnknownBoundariesAndExplicitEstimates) {
  auto graph = Chain();
  graph.mutable_input(0)->mutable_type()->mutable_tensor_type()->clear_shape();
  EXPECT_TRUE(ProposeOperatorGroups(graph).empty());
  EXPECT_EQ(ProposeOperatorGroups(graph, {}, {{"x", 4}}).size(), 1u);
  Describe(graph.mutable_input(0), "x", {}, TensorProto::STRING);
  EXPECT_TRUE(ProposeOperatorGroups(graph).empty());
}

TEST(GraphPartition, PackedAndZeroSizes) {
  for (const auto &[dtype, count, bytes] :
       std::vector<std::tuple<int32_t, int64_t, uint64_t>>{{TensorProto::UINT2, 5, 4},
                                                           {TensorProto::INT4, 3, 4},
                                                           {TensorProto::FLOAT6E2M3, 3, 6},
                                                           {TensorProto::FLOAT6E3M2, 5, 8},
                                                           {TensorProto::BOOL, 3, 6},
                                                           {TensorProto::FLOAT, 0, 0}}) {
    auto graph = Chain();
    Describe(graph.mutable_input(0), "x", {count}, dtype);
    Describe(graph.mutable_output(0), "y", {count}, dtype);
    const auto groups = ProposeOperatorGroups(graph, {bytes, 2, 128});
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].transfer_bytes, bytes);
  }
}

TEST(GraphPartition, StaticSizeOverflowIsUnknownAndZeroDimensionsDoNotOverflow) {
  auto graph = Chain();
  const auto huge = std::numeric_limits<int64_t>::max();
  Describe(graph.mutable_input(0), "x", {huge, huge});
  EXPECT_TRUE(ProposeOperatorGroups(graph).empty());
  Describe(graph.mutable_input(0), "x", {huge, huge, 0});
  EXPECT_EQ(ProposeOperatorGroups(graph).size(), 1u);
  Describe(graph.mutable_input(0), "x", {huge}, TensorProto::COMPLEX128);
  EXPECT_TRUE(ProposeOperatorGroups(graph).empty());
}

TEST(GraphPartition, RepresentableByteSizesDoNotOverflowBitIntermediates) {
  auto graph = Chain();
  Describe(graph.mutable_input(0), "x", {std::numeric_limits<int64_t>::max(), 2},
           TensorProto::UINT8);
  const auto groups = ProposeOperatorGroups(graph, {UINT64_MAX, 2, 128}, {{"v0", 0}, {"y", 0}});
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].transfer_bytes, UINT64_MAX - 1);
  Describe(graph.mutable_input(0), "x", {int64_t{1} << 62, 4}, TensorProto::UINT2);
  const auto packed = ProposeOperatorGroups(graph, {UINT64_MAX, 2, 128}, {{"v0", 0}, {"y", 0}});
  ASSERT_EQ(packed.size(), 1u);
  EXPECT_EQ(packed[0].transfer_bytes, uint64_t{1} << 62);
}

TEST(GraphPartition, CostOverflowAndInvalidLimitsAreRejected) {
  const auto graph = Chain();
  const uint64_t half = uint64_t{1} << 63;
  EXPECT_THROW(ProposeOperatorGroups(graph, {UINT64_MAX, 2, 128}, {{"x", half}, {"y", half}}),
               std::invalid_argument);
  EXPECT_THROW(ProposeOperatorGroups(graph, {1024, 1, 128}), std::invalid_argument);
  EXPECT_THROW(ProposeOperatorGroups(graph, {1024, 3, 2}), std::invalid_argument);
}

TEST(GraphPartition, ValidatesAllDataflowEvenForIneligibleGraphs) {
  GraphProto empty;
  EXPECT_TRUE(ProposeOperatorGroups(empty).empty());
  empty.add_output()->set_name("missing");
  EXPECT_THROW(ProposeOperatorGroups(empty), std::invalid_argument);
  auto graph = Chain();
  graph.mutable_node(0)->ref_input()[0] = "y";
  EXPECT_THROW(ProposeOperatorGroups(graph), std::invalid_argument);
  graph.mutable_node(0)->ref_input()[0] = "x";
  graph.mutable_node(1)->ref_output()[0] = "x";
  EXPECT_THROW(ProposeOperatorGroups(graph), std::invalid_argument);
}
