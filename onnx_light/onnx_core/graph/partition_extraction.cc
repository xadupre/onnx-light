// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/graph/graph_partition.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/shapes/shape_inference.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::graph {
namespace {

using core::builder::GraphGraph;
using ValueInfos = std::unordered_map<std::string, const ValueInfoProto *>;

bool IsGroup(const NodeProto &node) {
  return node.domain() == "ai.rt" &&
         (node.op_type() == "PartitionGroup" || node.op_type().find("PartitionGroup_") == 0);
}

ValueInfos Descriptors(const GraphProto &graph) {
  ValueInfos result;
  for (const auto *values : {&graph.input(), &graph.output(), &graph.value_info()})
    for (const auto &value : *values)
      result.emplace(value.name().value(), &value);
  return result;
}

bool Extractable(const NodeProto &node, const core::shapes::ShapesContext &shapes,
                 const ValueInfos &descriptors) {
  if (IsGroup(node) || !node.device_configurations().empty())
    return false;
  if (!core::shapes::HasShapeInference(shapes, node) &&
      !core::shapes::CanSeedAnnotatedOutputsForUnknownCustomOp(shapes, node, descriptors))
    return false;
  const auto extractable_graph = [&](const GraphProto &body) {
    const auto annotations = Descriptors(body);
    return std::all_of(body.node().begin(), body.node().end(), [&](const NodeProto &inner) {
      return Extractable(inner, shapes, annotations);
    });
  };
  for (const auto &attribute : node.attribute()) {
    if (!attribute.ref_attr_name().empty())
      return false;
    if (attribute.has_g() && !extractable_graph(attribute.g()))
      return false;
    for (const auto &body : attribute.graphs())
      if (!extractable_graph(body))
        return false;
  }
  return true;
}

GraphProto AnalysisGraph(GraphGraph &graph) {
  auto &builder = graph.Builder();
  GraphProto snapshot = builder.BuildGraph();
  auto describe = [&](ValueInfoProto &value) {
    const std::string name = value.name().value();
    if (builder.Shapes().HasType(name))
      *value.mutable_type() = builder.Shapes().GetType(name);
    if (graph.HasShape(name))
      core::symbolic::SymTensorToValueInfo(graph.GetShape(name), value);
    const auto &tags = builder.Compute().ValueTags();
    const auto tag = tags.find(name);
    if (tag != tags.end()) {
      StringStringEntryProto *metadata = nullptr;
      for (auto &entry : value.ref_metadata_props())
        if (entry.key() == core::compute::kValueTagMetadataKey) {
          metadata = &entry;
          break;
        }
      if (metadata == nullptr) {
        metadata = value.add_metadata_props();
        metadata->set_key(core::compute::kValueTagMetadataKey);
      }
      metadata->set_value(tag->second);
    }
  };
  for (auto &value : snapshot.ref_input())
    describe(value);
  for (auto &value : snapshot.ref_output())
    describe(value);
  std::unordered_set<std::string> described;
  for (auto &value : snapshot.ref_value_info()) {
    describe(value);
    described.insert(value.name().value());
  }
  for (const auto &node : snapshot.node())
    for (const auto *names : {&node.input(), &node.output()})
      for (const auto &name : *names)
        if (!name.empty() && described.insert(name.value()).second) {
          auto &value = *snapshot.add_value_info();
          value.set_name(name);
          describe(value);
        }
  std::unordered_set<std::string> captures;
  builder.CollectImplicitInputs(captures);
  std::vector<std::string> ordered(captures.begin(), captures.end());
  std::sort(ordered.begin(), ordered.end());
  for (const auto &name : ordered) {
    // Analysis-only declarations preserve constant provenance without loading weights.
    const bool constant = graph.IsConstant(name);
    if (constant)
      snapshot.add_initializer()->set_name(name);
    auto &value = *(constant ? snapshot.add_value_info() : snapshot.add_input());
    value.set_name(name);
    describe(value);
  }
  return snapshot;
}

OperatorGroupProposal Groups(GraphGraph &graph, const GraphProto &snapshot,
                             const OperatorGroupOptions &options) {
  const auto descriptors = Descriptors(snapshot);
  std::set<size_t> excluded;
  for (size_t index = 0; index < snapshot.node().size(); ++index)
    if (!Extractable(snapshot.node()[index], graph.Builder().Shapes(), descriptors))
      excluded.insert(index);
  return ProposeOperatorGroups(snapshot, options, {}, excluded);
}

NodeProto MakeGroup(GraphGraph &graph, const GraphProto &snapshot, const OperatorGroup &group,
                    const ValueInfos &descriptors, size_t &suffix) {
  auto &source = graph.Builder();
  source.SetOpsetVersion("ai.rt", 1);
  GraphProto body;
  std::unordered_set<std::string> annotated;
  auto annotate = [&](const std::string &name) {
    const auto found = descriptors.find(name);
    if (found != descriptors.end() && annotated.insert(name).second)
      *body.add_value_info() = *found->second;
  };
  for (const auto &input : group.inputs) {
    auto &value = *body.add_input();
    value.set_name(input);
    const auto found = descriptors.find(input);
    if (found != descriptors.end())
      value = *found->second;
    annotated.insert(input);
  }
  for (const auto &output : group.outputs) {
    auto &value = *body.add_output();
    value.set_name(output);
    const auto found = descriptors.find(output);
    if (found != descriptors.end())
      value = *found->second;
    annotated.insert(output);
  }
  for (size_t index : group.node_indices) {
    const auto &node = snapshot.node()[index];
    *body.add_node() = utils::CopyWithSharedTensorStorage(node);
    for (const auto &output : node.output())
      if (!output.empty())
        annotate(output.value());
  }
  const auto producers = FindCommonOperatorGroupInputs(graph, {&group, 1});
  const auto inputs = InlineOperatorGroupInputs(body, group, producers);
  const std::string name =
      AddOperatorGroupFunction(source, std::move(body), "PartitionGroup", suffix);
  return MakeNode(name.c_str(), inputs, group.outputs, "ai.rt",
                  source.UniqueName("PartitionGroup").c_str());
}

OperatorGroupStatus Eligibility(const GraphGraph &graph) {
  if (graph.Builder().IsInsideLocalFunction())
    return OperatorGroupStatus::kInsideFunction;
  const int version = graph.Builder().OpsetVersion("ai.rt");
  if (version > 0 && version != 1)
    return OperatorGroupStatus::kIncompatibleOpset;
  return OperatorGroupStatus::kFound;
}

} // namespace

OperatorGroupProposal ProposePartitionGroups(GraphGraph &graph,
                                             const OperatorGroupOptions &options) {
  if (options.min_group_nodes < 2 || options.max_group_nodes < options.min_group_nodes)
    throw std::invalid_argument(
        "Operator grouping requires 2 <= min_group_nodes <= max_group_nodes.");
  if (graph.Builder().Nodes().empty())
    return {};
  const auto status = Eligibility(graph);
  if (status != OperatorGroupStatus::kFound) {
    OperatorGroupProposal proposal;
    proposal.status = status;
    return proposal;
  }
  return Groups(graph, AnalysisGraph(graph), options);
}

utils::RepeatedProtoField<NodeProto>
MakePartitionGroupReplacements(GraphGraph &graph, const std::vector<const NodeProto *> &nodes,
                               const OperatorGroupOptions &options) {
  if (nodes.empty() || nodes[0] == nullptr || graph.Builder().Nodes().empty())
    throw core::builder::BuilderError("PartitionGroup::Apply expects a nonempty match.");
  graph.Position(*nodes[0]);
  if (Eligibility(graph) != OperatorGroupStatus::kFound)
    throw core::builder::BuilderError("PartitionGroup::Apply received an inconsistent match.");
  const GraphProto snapshot = AnalysisGraph(graph);
  const auto proposal = Groups(graph, snapshot, options);
  if (proposal.status != OperatorGroupStatus::kFound ||
      proposal.MatchedNodes(graph.Builder().Nodes()) != nodes)
    throw core::builder::BuilderError("PartitionGroup::Apply received an inconsistent match.");
  const auto descriptors = Descriptors(snapshot);
  size_t suffix = 0;
  return MakeOperatorGroupReplacements(graph, proposal, [&](std::size_t group) {
    return MakeGroup(graph, snapshot, proposal.groups[group], descriptors, suffix);
  });
}

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
