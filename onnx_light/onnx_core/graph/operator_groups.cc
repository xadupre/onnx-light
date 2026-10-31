// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/graph/operator_groups.h"

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#include "onnx_proto/onnx_helper.h"

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_map>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::graph {
namespace {

constexpr std::size_t kMaxAnchorDepth = 16;
constexpr std::size_t kMaxAnchorWork = 4096;
constexpr std::size_t kSmallConstantElementLimit = 32;

std::optional<TensorProto> SmallConstant(const builder::GraphGraph &graph,
                                         const std::string &name) {
  const auto *scope = graph.GetConstantScope(name);
  const TensorProto *tensor = scope == nullptr ? nullptr : scope->GetComputedConstant(name);
  TensorProto literal;
  const NodeProto *node = (scope == nullptr ? graph : *scope).NodeBefore(name);
  if (node != nullptr &&
      (!node->device_configurations().empty() || !node->overload().empty() ||
       std::any_of(node->attribute().begin(), node->attribute().end(),
                   [](const auto &attribute) { return !attribute.ref_attr_name().empty(); })))
    return {};
  if (tensor == nullptr) {
    if (node == nullptr || NormaliseDomain(node->domain()) != "ai.onnx" ||
        node->op_type() != "Constant" || !graph.IsConstant(name))
      return {};
    for (const auto &attribute : node->attribute()) {
      if (attribute.name() == "value_float" && attribute.has_f()) {
        literal.set_data_type(TensorProto::FLOAT);
        literal.add_float_data(attribute.f());
      } else if (attribute.name() == "value_int" && attribute.has_i()) {
        literal.set_data_type(TensorProto::INT64);
        literal.add_int64_data(attribute.i());
      } else if (attribute.name() == "value_string" && attribute.has_s()) {
        literal.set_data_type(TensorProto::STRING);
        literal.add_string_data(attribute.s());
      } else if (attribute.name() == "value_floats" &&
                 attribute.floats().size() < kSmallConstantElementLimit) {
        literal.set_data_type(TensorProto::FLOAT);
        literal.add_dims(static_cast<int64_t>(attribute.floats().size()));
        literal.ref_float_data() = attribute.floats();
      } else if (attribute.name() == "value_ints" &&
                 attribute.ints().size() < kSmallConstantElementLimit) {
        literal.set_data_type(TensorProto::INT64);
        literal.add_dims(static_cast<int64_t>(attribute.ints().size()));
        literal.ref_int64_data() = attribute.ints();
      } else if (attribute.name() == "value_strings" &&
                 attribute.strings().size() < kSmallConstantElementLimit) {
        literal.set_data_type(TensorProto::STRING);
        literal.add_dims(static_cast<int64_t>(attribute.strings().size()));
        literal.ref_string_data() = attribute.strings();
      }
    }
    if (literal.data_type() == TensorProto::UNDEFINED)
      return {};
    tensor = &literal;
  }
  if (tensor->has_segment())
    return {};
  bool empty = false;
  for (const auto dim : tensor->dims()) {
    if (dim < 0)
      throw builder::BuilderError("Operator-group constant has a negative dimension.");
    empty |= dim == 0;
  }
  std::size_t count = empty ? 0 : 1;
  if (!empty)
    for (const auto dim : tensor->dims()) {
      if (static_cast<uint64_t>(dim) > (kSmallConstantElementLimit - 1) / count)
        return {};
      count *= static_cast<std::size_t>(dim);
    }
  if (count != 0 && tensor->raw_data().empty() && tensor->float_data().empty() &&
      tensor->int32_data().empty() && tensor->int64_data().empty() &&
      tensor->double_data().empty() && tensor->uint64_data().empty() &&
      tensor->string_data().empty())
    return {};
  for (const auto &entry : tensor->metadata_props())
    if (entry.key() == "onnx_light.encryption")
      return {};
  TensorProto canonical;
  canonical.set_data_type(tensor->data_type());
  canonical.ref_dims() = tensor->dims();
  canonical.ref_metadata_props() = tensor->metadata_props();
  if (tensor->data_type() == TensorProto::STRING) {
    if (tensor->string_data().size() != count)
      throw builder::BuilderError("Operator-group string constant has an invalid payload size.");
    canonical.ref_string_data() = tensor->string_data();
  } else {
    const auto value = core::runtime::TensorFromProto(*tensor);
    canonical.set_raw_data(value.bytes(), value.size_bytes());
  }
  return canonical;
}

struct InputProducer {
  std::optional<std::string> source;
  NodeProto node;
};

std::optional<InputProducer> FindInputProducer(const builder::GraphGraph &graph,
                                               const std::string &name) {
  if (const auto value = SmallConstant(graph, name)) {
    NodeProto producer = MakeNode("Constant", {}, {""});
    auto &attribute = *producer.add_attribute();
    attribute.set_name("value");
    attribute.set_type(AttributeProto::TENSOR);
    *attribute.mutable_t() = utils::CopyWithSharedTensorStorage(*value);
    return InputProducer{{}, std::move(producer)};
  }
  const auto *node = graph.NodeBefore(name);
  if (node != nullptr && NormaliseDomain(node->domain()) == "ai.onnx" &&
      node->op_type() == "Shape" && node->overload().empty() &&
      node->device_configurations().empty() && node->input().size() == 1 &&
      node->output().size() == 1 &&
      !graph.Builder().Shapes().HasLocalFunction(
          core::shapes::ShapesContext::LocalFunctionKey(node->domain().value(), "Shape", ""))) {
    int64_t start = 0, end = std::numeric_limits<int64_t>::max();
    bool has_start = false, has_end = false;
    for (const auto &attribute : node->attribute()) {
      if (!attribute.ref_attr_name().empty() || attribute.type() != AttributeProto::INT ||
          !attribute.has_i())
        return {};
      if (attribute.name() == "start" && !has_start) {
        start = attribute.i();
        has_start = true;
      } else if (attribute.name() == "end" && !has_end) {
        end = attribute.i();
        has_end = true;
      } else {
        return {};
      }
    }
    NodeProto producer = MakeNode("Shape", {""}, {""});
    producer.ref_metadata_props() = node->metadata_props();
    const auto add_attribute = [&](const char *key, int64_t value) {
      auto &attribute = *producer.add_attribute();
      attribute.set_name(key);
      attribute.set_type(AttributeProto::INT);
      attribute.set_i(value);
    };
    if (start != 0)
      add_attribute("start", start);
    if (end != std::numeric_limits<int64_t>::max())
      add_attribute("end", end);
    return InputProducer{node->input(0).value(), std::move(producer)};
  }
  return {};
}

bool IsAnchor(const NodeProto &node, const builder::GraphBuilder &builder, std::size_t depth,
              std::size_t &work, std::set<std::string> &visiting, bool in_function = false) {
  const std::string &domain = node.domain().value();
  const std::string &op = node.op_type().value();
  if (domain == "ai.rt" && (op == "Layer" || op.rfind("Layer_", 0) == 0))
    return false;
  const std::string key =
      core::shapes::ShapesContext::LocalFunctionKey(domain, op, node.overload().value());
  const FunctionProto *function = builder.Shapes().GetLocalFunction(key);
  if (function == nullptr && node.overload().value().empty() &&
      (((domain.empty() || domain == "ai.onnx") && (op == "MatMul" || op == "Attention")) ||
       (domain == "com.microsoft" && op == "GroupQueryAttention") ||
       (domain == "ai.rt" && op == "PagedAttention")))
    return true;
  if (depth >= kMaxAnchorDepth || work >= kMaxAnchorWork)
    return false;
  bool found = false;
  if (function != nullptr && visiting.insert(key).second) {
    for (const NodeProto &inner : function->node()) {
      if (++work > kMaxAnchorWork)
        break;
      if (IsAnchor(inner, builder, depth + 1, work, visiting, true)) {
        found = true;
        break;
      }
    }
    visiting.erase(key);
  }
  if (!found && in_function) {
    auto search = [&](const GraphProto &graph) {
      for (const NodeProto &inner : graph.node()) {
        if (++work > kMaxAnchorWork)
          return false;
        if (IsAnchor(inner, builder, depth + 1, work, visiting, true))
          return true;
      }
      return false;
    };
    for (const auto &attribute : node.attribute()) {
      if (attribute.has_g() && search(attribute.g()))
        return true;
      for (const GraphProto &nested : attribute.graphs()) {
        if (search(nested))
          return true;
      }
    }
  }
  return found;
}

} // namespace

bool IsOperatorGroupAnchor(const NodeProto &node, const builder::GraphBuilder &builder,
                           std::size_t &work) {
  std::set<std::string> visiting;
  return IsAnchor(node, builder, 0, work, visiting);
}

std::string MakeOperatorGroupFunctionName(builder::GraphBuilder &source,
                                          const utils::RepeatedProtoField<NodeProto> &nodes,
                                          const std::string &prefix, std::size_t &suffix) {
  if (prefix.empty() || nodes.empty() || nodes[0].op_type().empty())
    throw builder::BuilderError("Operator-group naming requires a prefix and a nonempty body.");
  std::string main_op;
  std::size_t work = 0;
  for (const auto &node : nodes) {
    if (main_op.empty() &&
        !(NormaliseDomain(node.domain()) == "ai.onnx" && node.op_type() == "Constant"))
      main_op = node.op_type().value();
    if (IsOperatorGroupAnchor(node, source, work)) {
      main_op = node.op_type().value();
      break;
    }
  }
  if (main_op.empty())
    main_op = nodes[0].op_type().value();
  auto &model = source.ModelBuilder();
  const std::string base = prefix + "_" + main_op + "_";
  for (;;) {
    const auto current = suffix;
    if (suffix < std::numeric_limits<std::size_t>::max())
      ++suffix;
    const std::string name = base + std::to_string(current);
    if (!model.HasName(name) &&
        !source.Shapes().HasLocalFunction(
            core::shapes::ShapesContext::LocalFunctionKey("ai.rt", name, "")) &&
        !source.IsRegisteredOperator(MakeNode(name.c_str(), {}, {}, "ai.rt")))
      return name;
    if (current == std::numeric_limits<std::size_t>::max())
      throw builder::BuilderError("Operator-group function name indices are exhausted.");
  }
}

std::vector<OperatorGroupInput>
FindCommonOperatorGroupInputs(const builder::GraphGraph &graph,
                              std::span<const OperatorGroup> instances) {
  std::vector<OperatorGroupInput> producers;
  if (instances.empty())
    return producers;
  std::unordered_map<std::string, std::optional<InputProducer>> values;
  std::vector<std::unordered_map<std::string, std::size_t>> ports(instances.size());
  for (std::size_t i = 0; i < instances.size(); ++i)
    for (std::size_t port = 0; port < instances[i].inputs.size(); ++port)
      ports[i].emplace(instances[i].inputs[port], port);
  const auto get =
      [&](std::size_t instance,
          std::size_t port) -> std::pair<const NodeProto *, std::optional<std::size_t>> {
    if (port >= instances[instance].inputs.size())
      return {nullptr, {}};
    const auto &name = instances[instance].inputs[port];
    auto [found, inserted] = values.try_emplace(name);
    if (inserted)
      found->second = FindInputProducer(graph, name);
    if (!found->second)
      return {nullptr, {}};
    const auto &producer = *found->second;
    if (producer.source) {
      const auto source = ports[instance].find(*producer.source);
      if (source == ports[instance].end())
        return {nullptr, {}};
      return {&producer.node, source->second};
    }
    return {&producer.node, {}};
  };
  for (std::size_t port = 0; port < instances[0].inputs.size(); ++port) {
    const auto [value, source_input] = get(0, port);
    if (!value)
      continue;
    bool common = true;
    for (std::size_t i = 1; i < instances.size() && common; ++i) {
      const auto [other, other_source] = get(i, port);
      common = other && source_input == other_source && value->Equals(*other);
    }
    if (common)
      producers.push_back({port, source_input, utils::CopyWithSharedTensorStorage(*value)});
  }
  return producers;
}

std::vector<std::string>
InlineOperatorGroupInputs(GraphProto &body, const OperatorGroup &group,
                          const std::vector<OperatorGroupInput> &producers) {
  if (body.input().size() != group.inputs.size())
    throw builder::BuilderError("Operator-group producer ports do not match the body interface.");
  for (std::size_t i = 0; i < producers.size(); ++i) {
    const auto &producer = producers[i];
    if (producer.input_index >= group.inputs.size() ||
        (i && producers[i - 1].input_index >= producer.input_index) ||
        (producer.source_input && *producer.source_input >= group.inputs.size()) ||
        producer.node.output().size() != 1 ||
        producer.node.input().size() != (producer.source_input ? 1u : 0u))
      throw builder::BuilderError(
          "Operator-group producer ports must be valid, ordered and unique.");
  }
  std::vector<std::string> arguments;
  if (producers.empty())
    return group.inputs;
  utils::RepeatedProtoField<NodeProto> nodes;
  utils::RepeatedProtoField<ValueInfoProto> inputs;
  std::vector<std::size_t> owners(group.inputs.size(), producers.size());
  for (std::size_t i = 0; i < producers.size(); ++i)
    owners[producers[i].input_index] = i;
  std::vector<std::vector<std::size_t>> consumers(producers.size());
  std::vector<std::size_t> ready;
  ready.reserve(producers.size());
  for (std::size_t i = 0; i < producers.size(); ++i) {
    const auto source = producers[i].source_input;
    if (source && owners[*source] != producers.size())
      consumers[owners[*source]].push_back(i);
    else
      ready.push_back(i);
  }
  for (std::size_t i = 0; i < ready.size(); ++i) {
    const auto &producer = producers[ready[i]];
    NodeProto node = utils::CopyWithSharedTensorStorage(producer.node);
    *node.mutable_output(0) = body.input(producer.input_index).name();
    if (producer.source_input)
      *node.mutable_input(0) = body.input(*producer.source_input).name();
    nodes.push_back(std::move(node));
    for (const auto consumer : consumers[ready[i]])
      ready.push_back(consumer);
  }
  if (ready.size() != producers.size())
    throw builder::BuilderError("Operator-group input producers contain a dependency cycle.");
  std::size_t next = 0;
  for (std::size_t port = 0; port < group.inputs.size(); ++port) {
    if (next < producers.size() && producers[next].input_index == port) {
      ++next;
      *body.add_value_info() = body.input(port);
    } else {
      arguments.push_back(group.inputs[port]);
      inputs.push_back(body.input(port));
    }
  }
  for (auto &node : body.ref_node())
    nodes.push_back(std::move(node));
  body.ref_node() = std::move(nodes);
  body.ref_input() = std::move(inputs);
  return arguments;
}

std::string AddOperatorGroupFunction(builder::GraphBuilder &source, GraphProto body,
                                     const std::string &prefix, std::size_t &suffix) {
  if (const std::string existing = FindOperatorGroupFunction(source, body, prefix);
      !existing.empty())
    return existing;
  const std::string name = MakeOperatorGroupFunctionName(source, body.node(), prefix, suffix);
  FunctionProto definition;
  definition.set_name(name);
  definition.set_domain("ai.rt");
  for (const auto &[domain, version] : source.OpsetVersions()) {
    auto &opset = *definition.add_opset_import();
    opset.set_domain(domain);
    opset.set_version(version);
  }
  for (auto &input : body.ref_input()) {
    definition.add_input(input.name());
    *definition.add_value_info() = std::move(input);
  }
  for (auto &output : body.ref_output()) {
    definition.add_output(output.name());
    *definition.add_value_info() = std::move(output);
  }
  for (auto &info : body.ref_value_info())
    *definition.add_value_info() = std::move(info);
  definition.ref_node() = std::move(body.ref_node());
  source.ModelBuilder().MakeLocalFunction(definition);
  return name;
}

void OperatorGroupProposal::Validate(std::size_t node_count) const {
  const auto fail = []() {
    throw builder::BuilderError("Invalid operator-group proposal or replacement schedule.");
  };
  switch (status) {
  case OperatorGroupStatus::kFound:
    break;
  case OperatorGroupStatus::kNoMatch:
  case OperatorGroupStatus::kInsideFunction:
  case OperatorGroupStatus::kIncompatibleOpset:
    if (!groups.empty() || !schedule.empty())
      fail();
    return;
  default:
    fail();
  }
  if (groups.empty() || first > last || last >= node_count)
    fail();
  const std::size_t unowned = groups.size();
  std::vector<std::size_t> owners(last - first + 1, unowned);
  for (std::size_t group = 0; group < groups.size(); ++group) {
    const auto &indices = groups[group].node_indices;
    if (indices.empty())
      fail();
    for (std::size_t i = 0; i < indices.size(); ++i) {
      const auto position = indices[i];
      if (position < first || position > last || (i && position <= indices[i - 1]))
        fail();
      auto &owner = owners[position - first];
      if (owner != unowned)
        fail();
      owner = group;
    }
  }
  std::vector<bool> emitted_groups(groups.size(), false), emitted_nodes(owners.size(), false);
  for (const auto &[is_group, index] : schedule) {
    if (is_group) {
      if (index >= groups.size() || emitted_groups[index])
        fail();
      emitted_groups[index] = true;
      for (const auto position : groups[index].node_indices)
        emitted_nodes[position - first] = true;
    } else {
      if (index < first || index > last || owners[index - first] != unowned ||
          emitted_nodes[index - first])
        fail();
      emitted_nodes[index - first] = true;
    }
  }
  if (std::find(emitted_nodes.begin(), emitted_nodes.end(), false) != emitted_nodes.end() ||
      std::find(emitted_groups.begin(), emitted_groups.end(), false) != emitted_groups.end())
    fail();
}

std::vector<const NodeProto *>
OperatorGroupProposal::MatchedNodes(const utils::RepeatedProtoField<NodeProto> &nodes) const {
  Validate(nodes.size());
  if (status != OperatorGroupStatus::kFound)
    return {};
  std::vector<const NodeProto *> matched;
  matched.reserve(last - first + 1);
  for (std::size_t i = first; i <= last; ++i)
    matched.push_back(&nodes[i]);
  return matched;
}

OperatorGroupProposal MakeIntervalGroupProposal(std::vector<OperatorGroup> groups,
                                                std::size_t node_count) {
  OperatorGroupProposal proposal;
  if (groups.empty())
    return proposal;
  for (std::size_t i = 0; i < groups.size(); ++i) {
    const auto &indices = groups[i].node_indices;
    if (indices.empty() || !std::is_sorted(indices.begin(), indices.end()) ||
        std::adjacent_find(indices.begin(), indices.end()) != indices.end() ||
        (i && indices.front() <= groups[i - 1].node_indices.back()) || indices.back() >= node_count)
      throw builder::BuilderError("Operator-group intervals must be ordered and disjoint.");
  }
  proposal.status = OperatorGroupStatus::kFound;
  proposal.first = groups.front().node_indices.front();
  proposal.last = groups.back().node_indices.back();
  std::size_t position = proposal.first;
  for (std::size_t group = 0; group < groups.size(); ++group) {
    for (const auto index : groups[group].node_indices) {
      while (position < index)
        proposal.schedule.push_back({false, position++});
      ++position;
    }
    proposal.schedule.push_back({true, group});
  }
  proposal.groups = std::move(groups);
  return proposal;
}

utils::RepeatedProtoField<NodeProto>
MakeOperatorGroupReplacements(builder::GraphGraph &graph, const OperatorGroupProposal &proposal,
                              const std::function<NodeProto(std::size_t)> &make_group) {
  proposal.Validate(graph.Builder().Nodes().size());
  if (proposal.status != OperatorGroupStatus::kFound || !make_group)
    throw builder::BuilderError("Operator-group replacement requires a successful proposal.");
  utils::RepeatedProtoField<NodeProto> replacement;
  for (const auto &[is_group, index] : proposal.schedule) {
    if (is_group)
      replacement.push_back(make_group(index));
    else
      replacement.push_back(utils::CopyWithSharedTensorStorage(graph.Builder().Nodes()[index]));
  }
  return replacement;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
