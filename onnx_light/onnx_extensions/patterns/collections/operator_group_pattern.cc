// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/collections/operator_group_pattern.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {
namespace {

using core::builder::GraphGraph;
using core::graph::OperatorGroup;
using ValueInfos = std::unordered_map<std::string, const ValueInfoProto *>;

bool IsGroup(const NodeProto &node) {
  return node.domain() == "ai.rt" &&
         (node.op_type() == "OperatorGroup" || node.op_type().find("OperatorGroup_") == 0);
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

std::vector<OperatorGroup> Groups(const GraphProto &snapshot,
                                  const core::graph::OperatorGroupOptions &options) {
  auto groups = core::graph::ProposeOperatorGroups(snapshot, options);
  std::erase_if(groups, [&](const OperatorGroup &group) {
    for (size_t index = group.node_indices.front(); index <= group.node_indices.back(); ++index)
      if (!Extractable(snapshot.node()[index]))
        return true;
    return false;
  });
  return groups;
}

NodeProto MakeGroup(GraphGraph &graph, const GraphProto &snapshot, const OperatorGroup &group,
                    const ValueInfos &descriptors, size_t &suffix) {
  auto &source = graph.Builder();
  auto &model = source.ModelBuilder();
  source.SetOpsetVersion("ai.rt", 1);
  std::string name;
  do {
    name = suffix == 0 ? "OperatorGroup" : "OperatorGroup_" + std::to_string(suffix);
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
                  source.UniqueName("OperatorGroup").c_str());
}

} // namespace

OperatorGroupPattern::OperatorGroupPattern(int priority, uint64_t max_boundary_bytes,
                                           uint64_t min_group_nodes, uint64_t max_group_nodes)
    : PatternOptimization(priority, "OperatorGroup"),
      options_{max_boundary_bytes, min_group_nodes, max_group_nodes} {
  if (min_group_nodes < 2 || max_group_nodes < min_group_nodes)
    throw std::invalid_argument("OperatorGroup requires 2 <= min_group_nodes <= max_group_nodes.");
}

std::set<std::string> OperatorGroupPattern::FastOpType() const { return {}; }

core::builder::MatchResult OperatorGroupPattern::Match(GraphGraph &graph,
                                                       const NodeProto &candidate) const {
  const auto &nodes = graph.Builder().Nodes();
  if (nodes.empty() || &candidate != &nodes[0])
    return NoMatch(candidate, "operator grouping runs only at the first graph node");
  if (graph.Builder().IsInsideLocalFunction())
    return NoMatch(candidate, "local-function bodies must not introduce recursive groups");
  const int version = graph.Builder().OpsetVersion("ai.rt");
  if (version > 0 && version != 1)
    return NoMatch(candidate, "OperatorGroup requires ai.rt opset 1");
  const auto groups = Groups(AnalysisGraph(graph), options_);
  if (groups.empty())
    return NoMatch(candidate, "no extractable low-transfer operator interval");
  const size_t first = groups.front().node_indices.front();
  const size_t last = groups.back().node_indices.back();
  std::vector<const NodeProto *> matched;
  for (size_t position = first; position <= last; ++position)
    matched.push_back(&nodes[position]);
  return {this, std::move(matched), &nodes[first]};
}

utils::RepeatedProtoField<NodeProto>
OperatorGroupPattern::Apply(GraphGraph &graph, const std::vector<const NodeProto *> &nodes) const {
  if (nodes.empty() || nodes[0] == nullptr || graph.Builder().Nodes().empty())
    throw core::builder::BuilderError("OperatorGroup::Apply expects a nonempty match.");
  const auto verified = Match(graph, graph.Builder().Nodes()[0]);
  if (verified.pattern != this || verified.nodes != nodes)
    throw core::builder::BuilderError("OperatorGroup::Apply received an inconsistent match.");
  const GraphProto snapshot = AnalysisGraph(graph);
  const auto groups = Groups(snapshot, options_);
  ValueInfos descriptors;
  for (const auto *values : {&snapshot.input(), &snapshot.output(), &snapshot.value_info()})
    for (const auto &value : *values)
      descriptors.emplace(value.name().value(), &value);
  utils::RepeatedProtoField<NodeProto> replacement;
  size_t suffix = 0;
  size_t position = groups.front().node_indices.front();
  for (const auto &group : groups) {
    for (size_t index : group.node_indices) {
      while (position < index)
        replacement.push_back(
            utils::CopyWithSharedTensorStorage(graph.Builder().Nodes()[position++]));
      ++position;
    }
    replacement.push_back(MakeGroup(graph, snapshot, group, descriptors, suffix));
  }
  return replacement;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
