#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_core/graph/graph_partition.h"
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

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

// Rescans each interval's complete boundary, independently of incremental costs.
std::vector<OperatorGroup> ScanningProposal(const GraphProto &graph,
                                            const OperatorGroupOptions &options,
                                            const std::unordered_map<std::string, uint64_t> &sizes,
                                            const std::set<size_t> &excluded) {
  std::unordered_set<std::string> constants, graph_inputs;
  for (const auto &value : graph.input())
    graph_inputs.insert(value.name());
  for (const auto &value : graph.initializer())
    if (!graph_inputs.count(value.name()))
      constants.insert(value.name());
  std::vector<size_t> indices;
  std::vector<std::vector<std::string>> inputs;
  std::unordered_map<std::string, size_t> producer, last_use;
  for (size_t i = 0; i < graph.node().size(); ++i) {
    const auto &node = graph.node()[i];
    if (node.op_type() == "Constant") {
      for (const auto &name : node.output())
        constants.insert(name);
      continue;
    }
    inputs.push_back(CollectNodeInputs(node));
    for (const auto &name : inputs.back())
      last_use[name] = indices.size();
    for (const auto &name : node.output())
      if (!name.empty())
        producer[name] = indices.size();
    indices.push_back(i);
  }
  for (const auto &value : graph.output())
    last_use[value.name()] = indices.size();
  struct State {
    uint64_t unknown = 0, bytes = 0;
    size_t grouped = 0;
    std::vector<OperatorGroup> groups;
  };
  std::vector<std::optional<State>> best(indices.size() + 1);
  best[0] = State{};
  auto update = [&](size_t position, const State &state) {
    const auto &old = best[position];
    if (!old || std::tie(state.unknown, state.bytes) < std::tie(old->unknown, old->bytes) ||
        (std::tie(state.unknown, state.bytes) == std::tie(old->unknown, old->bytes) &&
         state.grouped > old->grouped))
      best[position] = state;
  };
  for (size_t start = 0; start < indices.size(); ++start) {
    for (size_t end = start; end < indices.size() && end - start < options.max_group_nodes; ++end) {
      OperatorGroup group;
      std::unordered_set<std::string> seen;
      size_t constant_inputs = 0;
      bool blocked = false;
      for (size_t i = start; i <= end; ++i) {
        group.node_indices.push_back(indices[i]);
        blocked = blocked || excluded.count(indices[i]);
        for (const auto &name : inputs[i])
          if ((!producer.count(name) || producer.at(name) < start) && seen.insert(name).second) {
            group.inputs.push_back(name);
            constant_inputs += constants.count(name);
          }
        for (const auto &name : graph.node()[indices[i]].output())
          if (last_use.count(name) && last_use.at(name) > end)
            group.outputs.push_back(name);
      }
      State state = *best[start];
      uint64_t unknown = 0, bytes = 0;
      for (const auto *names : {&group.inputs, &group.outputs})
        for (const auto &name : *names) {
          if (constants.count(name))
            continue;
          if (sizes.count(name))
            bytes += sizes.at(name);
          else
            ++unknown;
        }
      state.unknown += unknown;
      state.bytes += bytes;
      if (end == start)
        update(end + 1, state);
      if (blocked ||
          (options.max_constant_inputs && constant_inputs > *options.max_constant_inputs) ||
          end - start + 1 < options.min_group_nodes || group.outputs.empty() ||
          (options.max_boundary_bytes && (unknown || bytes > *options.max_boundary_bytes)))
        continue;
      state.grouped += end - start + 1;
      if (!unknown)
        group.transfer_bytes = bytes;
      state.groups.push_back(std::move(group));
      update(end + 1, state);
    }
  }
  return best.back()->groups;
}

} // namespace

TEST(GraphPartition, IncrementalCostsMatchScanningReference) {
  std::mt19937 random(5922);
  for (size_t trial = 0; trial < 80; ++trial) {
    GraphProto graph;
    std::unordered_map<std::string, uint64_t> sizes;
    graph.add_input()->set_name("x");
    graph.add_input()->set_name("overridable");
    graph.add_initializer()->set_name("overridable");
    std::vector<std::string> available{"x", "overridable"};
    for (size_t i = 0; i < 9; ++i) {
      const auto weight = "w" + std::to_string(i);
      graph.add_initializer()->set_name(weight);
      if (i == 3) {
        auto *constant = graph.add_node();
        constant->set_op_type("Constant");
        constant->add_output("constant");
        available.push_back("constant");
      }
      auto *node = graph.add_node();
      node->set_op_type("Add");
      node->add_input(available[random() % available.size()]);
      node->add_input(available[random() % available.size()]);
      node->add_input(weight);
      node->add_input(weight);
      node->add_input("");
      for (size_t port = 0; port < 2; ++port) {
        const auto output = "v" + std::to_string(i) + "_" + std::to_string(port);
        node->add_output(output);
        available.push_back(output);
        if (random() % 3 == 0)
          graph.add_output()->set_name(output);
      }
      if (i == 6) {
        auto *body = node->add_attribute();
        body->set_type(AttributeProto::GRAPH);
        auto *capture = body->mutable_g()->add_node();
        capture->add_input("v0_0");
        capture->add_output("nested");
        body->mutable_g()->add_output()->set_name("nested");
      }
    }
    graph.add_output()->set_name(available.back());
    for (const auto &name : available)
      if (random() % 4)
        sizes[name] = random() % 5 == 0 ? 0 : random() % 100;
    for (const auto cap :
         {std::optional<uint64_t>{}, std::optional<uint64_t>{0}, std::optional<uint64_t>{200}}) {
      OperatorGroupOptions options;
      options.max_boundary_bytes = cap;
      options.min_group_nodes = 2 + trial % 2;
      options.max_group_nodes = 3 + trial % 7;
      if (trial % 3)
        options.max_constant_inputs = trial % 5;
      const std::set<size_t> excluded = trial % 2 ? std::set<size_t>{4, 7} : std::set<size_t>{};
      SCOPED_TRACE(trial);
      const auto expected = ScanningProposal(graph, options, sizes, excluded);
      const auto actual = ProposeOperatorGroups(graph, options, sizes, excluded).groups;
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual[i].node_indices, expected[i].node_indices);
        EXPECT_EQ(actual[i].inputs, expected[i].inputs);
        EXPECT_EQ(actual[i].outputs, expected[i].outputs);
        EXPECT_EQ(actual[i].transfer_bytes, expected[i].transfer_bytes);
      }
    }
  }
}

TEST(GraphPartition, ConstantInputLimitCountsDistinctWeightsBeforeInternalization) {
  auto graph = Chain(4);
  graph.add_initializer()->set_name("a");
  graph.add_initializer()->set_name("b");
  for (size_t i = 0; i < 4; ++i) {
    graph.mutable_node(i)->add_input(i < 2 ? "a" : "b");
    graph.mutable_node(i)->add_input(i < 2 ? "a" : "b");
  }
  ASSERT_EQ(ProposeOperatorGroups(graph).groups.size(), 1u);
  OperatorGroupOptions options;
  options.max_constant_inputs = 1;
  const auto groups = ProposeOperatorGroups(graph, options).groups;
  ASSERT_EQ(groups.size(), 2u);
  EXPECT_EQ(groups[0].node_indices, std::vector<size_t>({0, 1}));
  EXPECT_EQ(groups[1].node_indices, std::vector<size_t>({2, 3}));
  options.max_constant_inputs = 0;
  EXPECT_TRUE(ProposeOperatorGroups(graph, options).groups.empty());
}

TEST(GraphPartition, ConstantInputLimitIncludesExtendedAndExcludesOverridableInitializers) {
  auto graph = Chain();
  graph.add_extended_initializer()->mutable_encoded_value()->set_name("weight");
  graph.mutable_node(0)->add_input("weight");
  OperatorGroupOptions options;
  options.max_constant_inputs = 0;
  EXPECT_TRUE(ProposeOperatorGroups(graph, options).groups.empty());
  options.max_constant_inputs = 1;
  EXPECT_EQ(ProposeOperatorGroups(graph, options).groups.size(), 1u);
  Describe(graph.add_input(), "weight");
  options.max_constant_inputs = 0;
  EXPECT_EQ(ProposeOperatorGroups(graph, options).groups.size(), 1u);
}

TEST(GraphPartition, TinyBoundaryUnknownInternalAndOrderedPorts) {
  auto graph = Chain();
  graph.mutable_value_info(0)->mutable_type()->mutable_tensor_type()->clear_shape();
  graph.mutable_node(0)->add_input("");
  graph.mutable_node(0)->add_input("x");
  graph.mutable_node(0)->add_output("side");
  Describe(graph.add_output(), "side");
  const auto groups = ProposeOperatorGroups(graph, {12, 2, 128}).groups;
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].node_indices, std::vector<size_t>({0, 1}));
  EXPECT_EQ(groups[0].inputs, std::vector<std::string>({"x"}));
  EXPECT_EQ(groups[0].outputs, std::vector<std::string>({"side", "y"}));
  EXPECT_EQ(groups[0].transfer_bytes, 12u);
  EXPECT_TRUE(ProposeOperatorGroups(graph, {11, 2, 128}).groups.empty());
}

TEST(GraphPartition, DynamicProgrammingBeatsEarlySmallGroup) {
  auto graph = Chain(4);
  // The cheap first interval loses to grouping all four operators.
  Describe(graph.mutable_value_info(0), "v0", {1000});
  Describe(graph.mutable_value_info(1), "v1", {});
  Describe(graph.mutable_value_info(2), "v2", {1000});
  auto groups = ProposeOperatorGroups(graph, {8, 2, 4}).groups;
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].node_indices, std::vector<size_t>({0, 1, 2, 3}));
  groups = ProposeOperatorGroups(graph, {8, 2, 2}).groups;
  ASSERT_EQ(groups.size(), 2u);
  EXPECT_EQ(groups[0].node_indices, std::vector<size_t>({0, 1}));
  EXPECT_EQ(groups[1].node_indices, std::vector<size_t>({2, 3}));
}

TEST(GraphPartition, EqualCostPrefersMoreGroupedNodesThenFirstPartition) {
  auto graph = Chain(5);
  std::unordered_map<std::string, uint64_t> sizes;
  for (const auto *name : {"x", "v0", "v1", "v2", "v3", "y"})
    sizes[name] = 0;
  const auto groups = ProposeOperatorGroups(graph, {0, 2, 3}, sizes).groups;
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
  const auto groups = ProposeOperatorGroups(graph, {4, 2, 2}).groups;
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].node_indices, std::vector<size_t>({1, 2}));
  EXPECT_EQ(groups[0].inputs, std::vector<std::string>({"v0", "dense", "sparse"}));
  EXPECT_EQ(groups[0].transfer_bytes, 4u);
  Describe(graph.add_input(), "sparse", {1000});
  EXPECT_TRUE(ProposeOperatorGroups(graph, {4, 2, 2}).groups.empty());
  graph.mutable_input(1)->set_name("dense");
  EXPECT_TRUE(ProposeOperatorGroups(graph, {4, 2, 2}).groups.empty());
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
    EXPECT_TRUE(ProposeOperatorGroups(graph, {4, 2, 2}).groups.empty());
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
  EXPECT_TRUE(ProposeOperatorGroups(graph, {8, 2, 2}).groups.empty());
  Describe(graph.add_input(), "external", {1000});
  *graph.mutable_node(1)->add_attribute() = *body;
  graph.mutable_node(2)->clear_attribute();
  capture = graph.mutable_node(1)->mutable_attribute(0)->mutable_g()->mutable_node(0);
  capture->ref_input()[0] = "external";
  EXPECT_TRUE(ProposeOperatorGroups(graph, {8, 2, 128}).groups.empty());
  capture->ref_input()[0] = "undefined";
  EXPECT_THROW(ProposeOperatorGroups(graph), std::invalid_argument);
}

TEST(GraphPartition, UnknownBoundariesAndExplicitEstimates) {
  auto graph = Chain();
  graph.mutable_input(0)->mutable_type()->mutable_tensor_type()->clear_shape();
  const auto groups = ProposeOperatorGroups(graph).groups;
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_FALSE(groups[0].transfer_bytes.has_value());
  EXPECT_TRUE(ProposeOperatorGroups(graph, {1024, 2, 128}).groups.empty());
  EXPECT_EQ(ProposeOperatorGroups(graph, {}, {{"x", 4}}).groups.size(), 1u);
  Describe(graph.mutable_input(0), "x", {}, TensorProto::STRING);
  const auto strings = ProposeOperatorGroups(graph).groups;
  ASSERT_EQ(strings.size(), 1u);
  EXPECT_FALSE(strings[0].transfer_bytes.has_value());
}

TEST(GraphPartition, DefaultPartitionSupportsLargeAndSymbolicBoundariesWithoutInventingSizes) {
  auto large = Chain();
  Describe(large.mutable_input(0), "x", {4096});
  Describe(large.mutable_output(0), "y", {4096});
  const auto static_groups = ProposeOperatorGroups(large).groups;
  ASSERT_EQ(static_groups.size(), 1u);
  EXPECT_EQ(static_groups[0].transfer_bytes, 32768u);
  EXPECT_TRUE(ProposeOperatorGroups(large, {1024, 2, 128}).groups.empty());

  auto symbolic = Chain(257);
  for (auto *values : {&symbolic.ref_input(), &symbolic.ref_output(), &symbolic.ref_value_info()})
    for (auto &value : *values) {
      auto *shape = value.mutable_type()->mutable_tensor_type()->mutable_shape();
      shape->clear_dim();
      shape->add_dim()->set_dim_param("batch");
    }
  const auto groups = ProposeOperatorGroups(symbolic).groups;
  ASSERT_EQ(groups.size(), 3u);
  std::size_t next = 0;
  for (const auto &group : groups) {
    EXPECT_FALSE(group.transfer_bytes.has_value());
    EXPECT_GE(group.node_indices.size(), 2u);
    EXPECT_LE(group.node_indices.size(), 128u);
    for (const auto index : group.node_indices)
      EXPECT_EQ(index, next++);
  }
  EXPECT_EQ(next, symbolic.node().size());
  EXPECT_TRUE(ProposeOperatorGroups(symbolic, {UINT64_MAX, 2, 128}).groups.empty());
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
    const auto groups = ProposeOperatorGroups(graph, {bytes, 2, 128}).groups;
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].transfer_bytes, bytes);
  }
}

TEST(GraphPartition, StaticSizeOverflowIsUnknownAndZeroDimensionsDoNotOverflow) {
  auto graph = Chain();
  const auto huge = std::numeric_limits<int64_t>::max();
  Describe(graph.mutable_input(0), "x", {huge, huge});
  EXPECT_TRUE(ProposeOperatorGroups(graph, {UINT64_MAX, 2, 128}).groups.empty());
  Describe(graph.mutable_input(0), "x", {huge, huge, 0});
  EXPECT_EQ(ProposeOperatorGroups(graph).groups.size(), 1u);
  Describe(graph.mutable_input(0), "x", {huge}, TensorProto::COMPLEX128);
  EXPECT_TRUE(ProposeOperatorGroups(graph, {UINT64_MAX, 2, 128}).groups.empty());
}

TEST(GraphPartition, EmptySymbolicAndUnspecifiedDimensionsHaveKnownZeroBytes) {
  for (const bool zero_first : {false, true})
    for (const bool symbolic : {false, true}) {
      auto graph = Chain();
      for (auto *value : {graph.mutable_input(0), graph.mutable_output(0)}) {
        auto *shape = value->mutable_type()->mutable_tensor_type()->mutable_shape();
        shape->clear_dim();
        for (const bool zero : {zero_first, !zero_first}) {
          auto *dimension = shape->add_dim();
          if (zero)
            dimension->set_dim_value(0);
          else if (symbolic)
            dimension->set_dim_param("N");
        }
      }
      for (const auto &options : {OperatorGroupOptions{}, OperatorGroupOptions{0, 2, 128}}) {
        const auto groups = ProposeOperatorGroups(graph, options).groups;
        ASSERT_EQ(groups.size(), 1u);
        EXPECT_EQ(groups[0].transfer_bytes, 0u);
      }
    }
}

TEST(GraphPartition, OverflowingAlternativesDoNotRejectRepresentableOptimum) {
  auto graph = Chain();
  for (const uint64_t internal : {uint64_t{1} << 63, UINT64_MAX}) {
    const auto groups = ProposeOperatorGroups(graph, {8, 2, 128}, {{"v0", internal}}).groups;
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].node_indices, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(groups[0].transfer_bytes, 8u);
  }
  graph = Chain(3);
  const auto groups =
      ProposeOperatorGroups(graph, {}, {{"v0", UINT64_MAX}, {"v1", UINT64_MAX}}).groups;
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].node_indices, (std::vector<size_t>{0, 1, 2}));
  EXPECT_EQ(groups[0].transfer_bytes, 8u);
}

TEST(GraphPartition, ExcludedNodesSplitIntervalsWithoutDiscardingEligibleNeighbors) {
  const auto graph = Chain(5);
  const auto groups = ProposeOperatorGroups(graph, {}, {}, {2}).groups;
  ASSERT_EQ(groups.size(), 2u);
  EXPECT_EQ(groups[0].node_indices, (std::vector<size_t>{0, 1}));
  EXPECT_EQ(groups[1].node_indices, (std::vector<size_t>{3, 4}));
  EXPECT_THROW(ProposeOperatorGroups(graph, {}, {}, {5}), std::invalid_argument);

  auto constants = Chain(3);
  auto &constant = *constants.mutable_node(1);
  constant.set_op_type("Constant");
  constant.clear_input();
  EXPECT_TRUE(ProposeOperatorGroups(constants, {}, {}, {1}).groups.empty());
}

TEST(GraphPartition, ExpiringOutputsRestoreKnownRepresentableBoundaryCosts) {
  auto graph = Chain(4);
  graph.mutable_node(0)->add_output("side");
  graph.mutable_node(2)->add_input("side");
  const auto groups =
      ProposeOperatorGroups(graph, {8, 2, 3},
                            {{"v0", UINT64_MAX}, {"v1", UINT64_MAX}, {"side", UINT64_MAX}})
          .groups;
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].node_indices, (std::vector<size_t>{0, 1, 2}));
  EXPECT_EQ(groups[0].outputs, (std::vector<std::string>{"v2"}));
  EXPECT_EQ(groups[0].transfer_bytes, 8u);
  const auto unknown = ProposeOperatorGroups(graph, {8, 2, 3}).groups;
  ASSERT_EQ(unknown.size(), 1u);
  EXPECT_EQ(unknown[0].node_indices, groups[0].node_indices);
  EXPECT_EQ(unknown[0].transfer_bytes, 8u);
}

TEST(GraphPartition, RepresentableByteSizesDoNotOverflowBitIntermediates) {
  auto graph = Chain();
  Describe(graph.mutable_input(0), "x", {std::numeric_limits<int64_t>::max(), 2},
           TensorProto::UINT8);
  const auto groups =
      ProposeOperatorGroups(graph, {UINT64_MAX, 2, 128}, {{"v0", 0}, {"y", 0}}).groups;
  ASSERT_EQ(groups.size(), 1u);
  EXPECT_EQ(groups[0].transfer_bytes, UINT64_MAX - 1);
  Describe(graph.mutable_input(0), "x", {int64_t{1} << 62, 4}, TensorProto::UINT2);
  const auto packed =
      ProposeOperatorGroups(graph, {UINT64_MAX, 2, 128}, {{"v0", 0}, {"y", 0}}).groups;
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
  EXPECT_TRUE(ProposeOperatorGroups(empty).groups.empty());
  empty.add_output()->set_name("missing");
  EXPECT_THROW(ProposeOperatorGroups(empty), std::invalid_argument);
  auto graph = Chain();
  graph.mutable_node(0)->ref_input()[0] = "y";
  EXPECT_THROW(ProposeOperatorGroups(graph), std::invalid_argument);
  graph.mutable_node(0)->ref_input()[0] = "x";
  graph.mutable_node(1)->ref_output()[0] = "x";
  EXPECT_THROW(ProposeOperatorGroups(graph), std::invalid_argument);
}
