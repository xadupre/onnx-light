// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/function_pattern.h"

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"
#include "onnx_proto/onnx_verify.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {
namespace {

using core::builder::BuilderError;
using core::builder::GraphGraph;
using AttributeType = AttributeProto::AttributeType;
using References = std::unordered_map<std::string, AttributeType>;

References Validate(const FunctionProto &function, bool is_template) {
  VerifyFunction(function);
  std::unordered_set<std::string> inputs, used;
  References references;
  for (const auto &input : function.input())
    inputs.insert(input.value());
  for (const auto &node : function.node()) {
    for (const auto &input : node.input())
      if (!input.empty())
        used.insert(input.value());
    for (const auto &attribute : node.attribute()) {
      if (attribute.type() == AttributeType::GRAPH || attribute.type() == AttributeType::GRAPHS ||
          attribute.has_g() || attribute.graphs_size() != 0)
        throw BuilderError("FunctionPattern: graph-valued attributes are not supported.");
      if (!attribute.ref_attr_name().empty()) {
        references.emplace(attribute.ref_attr_name().value(), attribute.type());
      }
    }
  }
  if (function.output_size() == 0 || (is_template && function.node_size() == 0))
    throw BuilderError("FunctionPattern: a template needs nodes and functions need outputs.");
  if (is_template) {
    std::unordered_set<std::string> outputs;
    for (const auto &output : function.output())
      if (inputs.contains(output.value()) || !outputs.insert(output.value()).second)
        throw BuilderError("FunctionPattern: template outputs must be distinct node results.");
    for (const auto &input : inputs)
      if (!used.contains(input))
        throw BuilderError("FunctionPattern: template inputs must be used.");
  }
  return references;
}

bool SameAttribute(const AttributeProto &left, const AttributeProto &right) {
  AttributeProto first = utils::CopyWithSharedTensorStorage(left);
  AttributeProto second = utils::CopyWithSharedTensorStorage(right);
  first.clear_name();
  second.clear_name();
  first.clear_doc_string();
  second.clear_doc_string();
  return first.Equals(second);
}

struct Bindings {
  std::unordered_map<std::string, std::string> values;
  std::unordered_map<std::string, AttributeProto> attributes;
  std::vector<const NodeProto *> nodes;
};

bool BindValue(Bindings &bindings, const std::string &formal, const std::string &actual) {
  if (formal.empty() || actual.empty())
    return formal.empty() && actual.empty();
  const auto [it, inserted] = bindings.values.emplace(formal, actual);
  return inserted || it->second == actual;
}

bool BindNode(const NodeProto &formal, const NodeProto &actual, Bindings &bindings) {
  if (!actual.device_configurations().empty() || formal.op_type() != actual.op_type() ||
      NormaliseDomain(formal.domain()) != NormaliseDomain(actual.domain()) ||
      formal.overload().sv() != actual.overload().sv() ||
      formal.input_size() != actual.input_size() || formal.output_size() != actual.output_size() ||
      formal.attribute_size() != actual.attribute_size())
    return false;
  for (int i = 0; i < formal.input_size(); ++i)
    if (!BindValue(bindings, formal.input(i).value(), actual.input(i).value()))
      return false;
  for (int i = 0; i < formal.output_size(); ++i)
    if (!BindValue(bindings, formal.output(i).value(), actual.output(i).value()))
      return false;
  for (const auto &expected : formal.attribute()) {
    const AttributeProto *attribute = nullptr;
    for (const auto &value : actual.attribute()) {
      if (value.name() == expected.name()) {
        if (attribute != nullptr)
          return false;
        attribute = &value;
      }
    }
    if (attribute == nullptr || attribute->type() != expected.type())
      return false;
    if (expected.ref_attr_name().empty()) {
      if (!SameAttribute(expected, *attribute))
        return false;
    } else {
      const auto [it, inserted] = bindings.attributes.emplace(
          expected.ref_attr_name().value(), utils::CopyWithSharedTensorStorage(*attribute));
      if (!inserted && !SameAttribute(it->second, *attribute))
        return false;
    }
  }
  return true;
}

const NodeProto *LastNode(GraphGraph &graph, const std::vector<const NodeProto *> &nodes) {
  return *std::max_element(nodes.begin(), nodes.end(), [&](const auto *left, const auto *right) {
    return graph.Position(*left) < graph.Position(*right);
  });
}

bool SafeMatch(GraphGraph &graph, const FunctionProto &pattern, const Bindings &bindings) {
  const NodeProto &anchor = *LastNode(graph, bindings.nodes);
  const std::unordered_set<const NodeProto *> nodes(bindings.nodes.begin(), bindings.nodes.end());
  std::unordered_set<std::string> outputs;
  for (const auto &output : pattern.output())
    outputs.insert(bindings.values.at(output.value()));
  for (const auto &input : pattern.input()) {
    const NodeProto *producer = graph.NodeBefore(bindings.values.at(input.value()));
    if (producer != nullptr &&
        (nodes.contains(producer) || graph.Position(*producer) >= graph.Position(anchor)))
      return false;
  }
  for (const auto *node : bindings.nodes) {
    for (const auto &output : node->output()) {
      if (output.empty())
        continue;
      const bool exported = outputs.contains(output.value());
      if (graph.IsUsedBySubgraph(output) || (!exported && graph.IsOutput(output)))
        return false;
      for (const auto *consumer : graph.NextNodes(output)) {
        if (!nodes.contains(consumer) &&
            (!exported || graph.Position(*consumer) < graph.Position(anchor)))
          return false;
      }
    }
  }
  return true;
}

bool CompatibleOpsets(GraphGraph &graph, const FunctionProto &function) {
  for (const auto &node : function.node()) {
    for (const auto &opset : function.opset_import()) {
      if (NormaliseDomain(node.domain()) == NormaliseDomain(opset.domain()) &&
          graph.Builder().OpsetVersion(node.domain().value()) != opset.version())
        return false;
    }
  }
  return true;
}

bool NeedsOutputIdentity(const FunctionProto &function) {
  std::unordered_set<std::string> values;
  for (const auto &input : function.input())
    values.insert(input.value());
  for (const auto &output : function.output())
    if (!values.insert(output.value()).second)
      return true;
  return false;
}

} // namespace

FunctionPattern::FunctionPattern(const FunctionProto &pattern, const FunctionProto &replacement,
                                 int priority, std::string name, core::symbolic::Device device,
                                 int anchor)
    : PatternOptimization(priority, std::move(name), device),
      pattern_(utils::CopyWithSharedTensorStorage(pattern)),
      replacement_(utils::CopyWithSharedTensorStorage(replacement)) {
  const References captured = Validate(pattern_, true);
  const References required = Validate(replacement_, false);
  if (anchor < -1 || anchor >= pattern_.node_size())
    throw BuilderError("FunctionPattern: anchor must identify a template node.");
  anchor_index_ = anchor == -1 ? pattern_.node().size() - 1 : static_cast<std::size_t>(anchor);
  if (pattern_.input_size() != replacement_.input_size() ||
      pattern_.output_size() != replacement_.output_size())
    throw BuilderError("FunctionPattern: template and replacement arities must agree.");
  for (const auto &[reference, type] : required) {
    const auto it = captured.find(reference);
    if (it == captured.end() || it->second != type)
      throw BuilderError(
          "FunctionPattern: replacement references must be captured by the template.");
  }
  std::unordered_map<std::string, std::vector<std::size_t>> adjacent;
  for (std::size_t index = 0; index < pattern_.node().size(); ++index) {
    for (const auto &input : pattern_.node()[index].input())
      if (!input.empty())
        adjacent[input.value()].push_back(index);
    for (const auto &output : pattern_.node()[index].output())
      if (!output.empty())
        adjacent[output.value()].push_back(index);
  }
  std::vector<bool> visited(pattern_.node().size(), false);
  match_order_.push_back(anchor_index_);
  visited[match_order_.front()] = true;
  const auto visit = [&](const auto &value) {
    const auto found = adjacent.find(value.value());
    if (found == adjacent.end())
      return;
    for (const auto index : found->second)
      if (!visited[index]) {
        visited[index] = true;
        match_order_.push_back(index);
      }
    adjacent.erase(found);
  };
  for (std::size_t position = 0; position < match_order_.size(); ++position) {
    const auto &node = pattern_.node()[match_order_[position]];
    for (const auto &input : node.input())
      visit(input);
    for (const auto &output : node.output())
      visit(output);
  }
  if (match_order_.size() != pattern_.node().size())
    throw BuilderError("FunctionPattern: template nodes must be connected through values.");
}

std::set<std::string> FunctionPattern::FastOpType() const {
  return {pattern_.node()[anchor_index_].op_type().value()};
}

core::builder::MatchResult FunctionPattern::Match(GraphGraph &graph,
                                                  const NodeProto &candidate) const {
  if (!CompatibleOpsets(graph, pattern_) || !CompatibleOpsets(graph, replacement_))
    return NoMatch(candidate, "function opsets differ from the target graph");
  if (NeedsOutputIdentity(replacement_) && graph.Builder().OpsetVersion("") <= 0)
    return NoMatch(candidate, "output aliases require an imported default ONNX opset");
  Bindings initial;
  initial.nodes.resize(pattern_.node_size());
  if (!BindNode(pattern_.node()[anchor_index_], candidate, initial))
    return NoMatch(candidate, "candidate differs from the template anchor");
  initial.nodes[anchor_index_] = &candidate;
  std::function<bool(std::size_t, Bindings &)> search = [&](std::size_t position,
                                                            Bindings &bindings) {
    if (position == match_order_.size())
      return SafeMatch(graph, pattern_, bindings);
    const std::size_t index = match_order_[position];
    const NodeProto &formal = pattern_.node()[index];
    std::vector<const NodeProto *> producer_choice;
    const std::vector<const NodeProto *> *choices = nullptr;
    for (const auto &output : formal.output()) {
      if (output.empty())
        continue;
      const auto it = bindings.values.find(output.value());
      if (it != bindings.values.end()) {
        const NodeProto *producer = graph.NodeBefore(it->second);
        if (producer == nullptr)
          return false;
        producer_choice.push_back(producer);
        choices = &producer_choice;
        break;
      }
    }
    if (choices == nullptr) {
      for (const auto &input : formal.input()) {
        if (input.empty())
          continue;
        const auto it = bindings.values.find(input.value());
        if (it != bindings.values.end()) {
          const auto &consumers = graph.NextNodes(it->second);
          if (choices == nullptr || consumers.size() < choices->size())
            choices = &consumers;
        }
      }
    }
    if (choices == nullptr)
      throw BuilderError("FunctionPattern: local matching order has no bound value.");
    for (const auto *node : *choices) {
      if (std::find(bindings.nodes.begin(), bindings.nodes.end(), node) != bindings.nodes.end())
        continue;
      Bindings next;
      next.values = bindings.values;
      next.nodes = bindings.nodes;
      for (const auto &[name, value] : bindings.attributes)
        next.attributes.emplace(name, utils::CopyWithSharedTensorStorage(value));
      if (!BindNode(formal, *node, next))
        continue;
      next.nodes[index] = node;
      if (search(position + 1, next)) {
        bindings = std::move(next);
        return true;
      }
    }
    return false;
  };
  if (!search(1, initial))
    return NoMatch(candidate, "template structure, attributes or use constraints differ");
  const NodeProto *insert_at = LastNode(graph, initial.nodes);
  return {this, std::move(initial.nodes), insert_at};
}

utils::RepeatedProtoField<NodeProto>
FunctionPattern::Apply(GraphGraph &graph, const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != static_cast<std::size_t>(pattern_.node_size()) ||
      nodes[anchor_index_] == nullptr)
    throw BuilderError("FunctionPattern::Apply: invalid matched nodes.");
  const auto verified = Match(graph, *nodes[anchor_index_]);
  if (verified.pattern != this || verified.nodes != nodes)
    throw BuilderError("FunctionPattern::Apply: unsafe or inconsistent match.");
  Bindings bindings;
  for (int i = 0; i < pattern_.node_size(); ++i)
    if (!BindNode(pattern_.node(i), *nodes[i], bindings))
      throw BuilderError("FunctionPattern::Apply: inconsistent attribute bindings.");
  std::unordered_map<std::string, std::string> values{{"", ""}};
  for (int i = 0; i < replacement_.input_size(); ++i)
    values.emplace(replacement_.input(i).value(), bindings.values.at(pattern_.input(i).value()));
  std::unordered_map<std::string, std::string> output_names;
  for (int i = 0; i < replacement_.output_size(); ++i)
    output_names.emplace(replacement_.output(i).value(),
                         bindings.values.at(pattern_.output(i).value()));
  utils::RepeatedProtoField<NodeProto> result;
  for (const auto &node : replacement_.node()) {
    NodeProto copy = utils::CopyWithSharedTensorStorage(node);
    for (int i = 0; i < node.input_size(); ++i)
      copy.ref_input()[i] = values.at(node.input(i).value());
    for (int i = 0; i < node.output_size(); ++i) {
      const std::string formal = node.output(i).value();
      if (!formal.empty()) {
        const auto output = output_names.find(formal);
        values.emplace(formal, output == output_names.end()
                                   ? graph.Builder().UniqueName(Name() + "_" + formal)
                                   : output->second);
      }
      copy.ref_output()[i] = values.at(formal);
    }
    for (auto &configuration : copy.ref_device_configurations())
      for (auto &spec : configuration.ref_sharding_spec())
        spec.set_tensor_name(values.at(spec.tensor_name().value()));
    for (auto &attribute : copy.ref_attribute()) {
      if (!attribute.ref_attr_name().empty()) {
        const std::string name = attribute.name().value();
        attribute = utils::CopyWithSharedTensorStorage(
            bindings.attributes.at(attribute.ref_attr_name().value()));
        attribute.set_name(name);
      }
    }
    result.push_back(std::move(copy));
  }
  for (int i = 0; i < replacement_.output_size(); ++i) {
    const std::string &source = values.at(replacement_.output(i).value());
    const std::string &destination = bindings.values.at(pattern_.output(i).value());
    if (source != destination)
      result.push_back(MakeNode("Identity", {source}, {destination}));
  }
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
