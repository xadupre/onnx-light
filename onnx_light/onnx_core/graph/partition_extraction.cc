// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/graph/graph_partition.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::graph {
namespace {

using core::builder::GraphGraph;
using ValueInfos = std::unordered_map<std::string, const ValueInfoProto *>;

bool IsGroup(const NodeProto &node) {
  return node.domain() == "ai.rt" &&
         (node.op_type() == "PartitionGroup" || node.op_type().find("PartitionGroup_") == 0);
}

bool Extractable(const NodeProto &node) {
  if (IsGroup(node) || !node.device_configurations().empty())
    return false;
  for (const auto &attribute : node.attribute()) {
    if (!attribute.ref_attr_name().empty())
      return false;
    if (attribute.has_g())
      for (const auto &inner : attribute.g().node())
        if (!Extractable(inner))
          return false;
    for (const auto &body : attribute.graphs())
      for (const auto &inner : body.node())
        if (!Extractable(inner))
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
    auto &value = *snapshot.add_input();
    value.set_name(name);
    describe(value);
  }
  // The core analysis understands dense/sparse initializers; mark the builder's
  // extended initializers as ordinary constants without reading their payloads.
  for (const auto &value : snapshot.extended_initializer())
    snapshot.add_initializer()->set_name(value.name());
  return snapshot;
}

OperatorGroupProposal Groups(const GraphProto &snapshot, const OperatorGroupOptions &options) {
  auto proposal = ProposeOperatorGroups(snapshot, options);
  const auto removed = std::erase_if(proposal.groups, [&](const OperatorGroup &group) {
    for (size_t index = group.node_indices.front(); index <= group.node_indices.back(); ++index)
      if (!Extractable(snapshot.node()[index]))
        return true;
    return false;
  });
  if (removed)
    return MakeIntervalGroupProposal(std::move(proposal.groups), snapshot.node().size());
  return proposal;
}

NodeProto MakeGroup(GraphGraph &graph, const GraphProto &snapshot, const OperatorGroup &group,
                    const ValueInfos &descriptors, size_t &suffix) {
  auto &source = graph.Builder();
  auto &model = source.ModelBuilder();
  source.SetOpsetVersion("ai.rt", 1);
  std::string name;
  do {
    name = suffix == 0 ? "PartitionGroup" : "PartitionGroup_" + std::to_string(suffix);
    ++suffix;
  } while (model.HasName(name) ||
           source.Shapes().HasLocalFunction(
               core::shapes::ShapesContext::LocalFunctionKey("ai.rt", name, "")) ||
           source.IsRegisteredOperator(MakeNode(name.c_str(), {}, {}, "ai.rt")));
  FunctionProto function;
  function.set_name(name);
  function.set_domain("ai.rt");
  for (const auto &[domain, version] : source.OpsetVersions()) {
    auto &opset = *function.add_opset_import();
    opset.set_domain(domain);
    opset.set_version(version);
  }
  std::unordered_set<std::string> annotated;
  auto annotate = [&](const std::string &name) {
    const auto found = descriptors.find(name);
    if (found != descriptors.end() && annotated.insert(name).second)
      *function.add_value_info() = *found->second;
  };
  for (const auto &input : group.inputs) {
    function.add_input(input);
    annotate(input);
  }
  for (const auto &output : group.outputs)
    function.add_output(output);
  for (size_t index : group.node_indices) {
    const auto &node = snapshot.node()[index];
    *function.add_node() = utils::CopyWithSharedTensorStorage(node);
    for (const auto &output : node.output())
      if (!output.empty())
        annotate(output.value());
  }
  model.MakeLocalFunction(function);
  return MakeNode(name.c_str(), group.inputs, group.outputs, "ai.rt",
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
  return Groups(AnalysisGraph(graph), options);
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
  const auto proposal = Groups(snapshot, options);
  if (proposal.status != OperatorGroupStatus::kFound ||
      proposal.MatchedNodes(graph.Builder().Nodes()) != nodes)
    throw core::builder::BuilderError("PartitionGroup::Apply received an inconsistent match.");
  ValueInfos descriptors;
  for (const auto *values : {&snapshot.input(), &snapshot.output(), &snapshot.value_info()})
    for (const auto &value : *values)
      descriptors.emplace(value.name().value(), &value);
  size_t suffix = 0;
  return MakeOperatorGroupReplacements(graph, proposal, [&](std::size_t group) {
    return MakeGroup(graph, snapshot, proposal.groups[group], descriptors, suffix);
  });
}

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
