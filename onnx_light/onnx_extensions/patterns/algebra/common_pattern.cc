// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/algebra/common_pattern.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::IsDefaultOp;
using core::builder::BuilderError;

bool SameAttributes(const NodeProto &first, const NodeProto &second) {
  if (first.attribute_size() != second.attribute_size()) {
    return false;
  }
  for (int index = 0; index < first.attribute_size(); ++index) {
    const AttributeProto &left = first.attribute(index);
    const AttributeProto &right = second.attribute(index);
    if (left.name() != right.name() || left.type() != right.type()) {
      return false;
    }
    switch (left.type()) {
    case AttributeProto::AttributeType::INT:
      if (left.i() != right.i()) {
        return false;
      }
      break;
    case AttributeProto::AttributeType::FLOAT:
      if (left.f() != right.f()) {
        return false;
      }
      break;
    case AttributeProto::AttributeType::STRING:
      if (left.s() != right.s()) {
        return false;
      }
      break;
    default:
      if (left.SerializeAsString() != right.SerializeAsString()) {
        return false;
      }
      break;
    }
  }
  return true;
}

bool SameInputs(const NodeProto &first, const NodeProto &second) {
  if (first.input_size() != second.input_size()) {
    return false;
  }
  bool equal = true;
  for (int index = 0; index < first.input_size(); ++index) {
    if (first.input(index).value() != second.input(index).value()) {
      equal = false;
      break;
    }
  }
  if (equal) {
    return true;
  }
  const std::string &type = first.op_type().value();
  return (type == "Add" || type == "Mul") && first.input_size() == 2 &&
         first.input()[0].value() == second.input()[1].value() &&
         first.input()[1].value() == second.input()[0].value();
}

bool SameOutputs(const NodeProto &first, const NodeProto &second) {
  return first.output_size() == second.output_size();
}

bool IsUnaryLike(const std::string &op_type) {
  static const std::unordered_set<std::string> types = {"Abs",
                                                        "Acos",
                                                        "Acosh",
                                                        "Asin",
                                                        "Asinh",
                                                        "Atan",
                                                        "Atanh",
                                                        "BitShift",
                                                        "BitwiseNot",
                                                        "Cast",
                                                        "CastLike",
                                                        "Ceil",
                                                        "Celu",
                                                        "Clip",
                                                        "Cos",
                                                        "Cosh",
                                                        "CumSum",
                                                        "DequantizeLinear",
                                                        "DynamicQuantizeLinear",
                                                        "Elu",
                                                        "Erf",
                                                        "Exp",
                                                        "Floor",
                                                        "HardSigmoid",
                                                        "HardSwish",
                                                        "IsInf",
                                                        "LeakyRelu",
                                                        "Log",
                                                        "LogSoftmax",
                                                        "LpNormalization",
                                                        "LRN",
                                                        "MeanVarianceNormalization",
                                                        "Mish",
                                                        "Neg",
                                                        "Not",
                                                        "PRelu",
                                                        "QuantizeLinear",
                                                        "Reciprocal",
                                                        "Relu",
                                                        "Round",
                                                        "Selu",
                                                        "Shrink",
                                                        "Sigmoid",
                                                        "Sign",
                                                        "Sin",
                                                        "Sinh",
                                                        "Softmax",
                                                        "Softplus",
                                                        "Softsign",
                                                        "Sqrt",
                                                        "Tan",
                                                        "Tanh",
                                                        "ThresholdedRelu",
                                                        "ThresholdRelu",
                                                        "Trilu",
                                                        "Trunc"};
  return types.find(op_type) != types.end();
}

bool HasOneElementVectorShape(core::builder::GraphGraph &graph, const std::string &name) {
  if (!graph.HasShape(name)) {
    return false;
  }
  const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
  return shape.Rank() == 1 && shape[0].IsInt() && shape[0].AsInt() == 1;
}

} // namespace

bool SameChildrenPattern::SameNode(const NodeProto &first, const NodeProto &second) {
  return SameInputs(first, second) && SameOutputs(first, second) && SameAttributes(first, second);
}

bool SameChildrenPattern::SameNodeWithAliases(
    const NodeProto &first, const NodeProto &second,
    const std::unordered_map<std::string, std::unordered_set<std::string>> &aliases) {
  if (first.input_size() != second.input_size() || first.output_size() != second.output_size() ||
      first.attribute_size() != second.attribute_size()) {
    return false;
  }
  for (int index = 0; index < first.input_size(); ++index) {
    const std::string &left = first.input(index).value();
    const std::string &right = second.input(index).value();
    const auto it = aliases.find(right);
    if (left != right && (it == aliases.end() || it->second.find(left) == it->second.end())) {
      return false;
    }
  }
  return SameAttributes(first, second);
}

core::builder::MatchResult
SameChildrenPattern::MatchWithNodes(core::builder::GraphGraph &graph, const NodeProto &candidate,
                                    const std::vector<const NodeProto *> &next_nodes) const {
  (void)candidate;
  std::vector<const NodeProto *> nodes;
  if (next_nodes.size() == 2) {
    const NodeProto *first = next_nodes[0];
    const NodeProto *second = next_nodes[1];
    if (first == nullptr || second == nullptr ||
        first->op_type().value() != second->op_type().value() ||
        first->op_type().value() == "Identity" || !SameNode(*first, *second)) {
      return {};
    }
    nodes = {first, second};
  } else {
    std::vector<std::pair<std::string, std::vector<const NodeProto *>>> groups;
    for (const NodeProto *node : next_nodes) {
      if (node == nullptr || node->op_type().value() == "Identity") {
        continue;
      }
      const auto group = std::find_if(groups.begin(), groups.end(), [node](const auto &entry) {
        return entry.first == node->op_type().value();
      });
      if (group == groups.end()) {
        groups.push_back({node->op_type().value(), {node}});
      } else {
        group->second.push_back(node);
      }
    }
    for (const auto &entry : groups) {
      const std::vector<const NodeProto *> &group = entry.second;
      bool found = false;
      for (std::size_t first_index = 0; first_index + 1 < group.size() && !found; ++first_index) {
        for (std::size_t second_index = first_index + 1; second_index < group.size();
             ++second_index) {
          if (SameNode(*group[first_index], *group[second_index])) {
            nodes.push_back(group[first_index]);
            nodes.push_back(group[second_index]);
            found = true;
            break;
          }
        }
      }
    }
    if (nodes.empty()) {
      return {};
    }
  }
  for (std::size_t index = 0; index < nodes.size(); index += 2) {
    const NodeProto &first = *nodes[index];
    const NodeProto &second = *nodes[index + 1];
    if (first.output_size() == 0 ||
        (graph.HasType(first.output()[0].value()) && graph.HasType(second.output()[0].value()) &&
         graph.GetType(first.output()[0].value()) != graph.GetType(second.output()[0].value()))) {
      return {};
    }
  }
  return core::builder::MatchResult{this, std::move(nodes), nullptr};
}

core::builder::MatchResult SameChildrenPattern::Match(core::builder::GraphGraph &graph,
                                                      const NodeProto &candidate) const {
  core::builder::MatchResult match;
  for (int output_index = 0; output_index < candidate.output_size(); ++output_index) {
    const std::vector<const NodeProto *> &next_nodes =
        graph.NextNodes(candidate.output()[output_index].value());
    if (next_nodes.size() <= 1) {
      continue;
    }
    match = MatchWithNodes(graph, candidate, next_nodes);
    if (match.pattern != nullptr) {
      break;
    }
  }
  if (match.pattern != nullptr && match.nodes.size() == 2) {
    std::vector<const NodeProto *> nodes = match.nodes;
    std::unordered_map<std::string, std::unordered_set<std::string>> aliases;
    std::unordered_set<std::string> used_outputs;
    for (const auto &output : nodes[0]->output()) {
      if (!output.value().empty()) {
        used_outputs.insert(output.value());
      }
    }
    std::vector<std::pair<const NodeProto *, const NodeProto *>> stack = {{nodes[0], nodes[1]}};
    bool cannot_continue = false;
    while (!stack.empty() && !cannot_continue) {
      const auto [first, second] = stack.back();
      stack.pop_back();
      const int output_count = std::min(first->output_size(), second->output_size());
      for (int index = 0; index < output_count; ++index) {
        const std::string &left = first->output(index).value();
        const std::string &right = second->output(index).value();
        aliases[left].insert(right);
        aliases[right].insert(left);
      }
      if (first->output_size() != 1 || second->output_size() != 1) {
        break;
      }
      const std::vector<const NodeProto *> &first_next =
          graph.NextNodes(first->output()[0].value());
      const std::vector<const NodeProto *> &second_next =
          graph.NextNodes(second->output()[0].value());
      std::unordered_map<std::string, std::vector<const NodeProto *>> first_by_type;
      std::unordered_map<std::string, std::vector<const NodeProto *>> second_by_type;
      for (const NodeProto *node : first_next) {
        first_by_type[node->op_type().value()].push_back(node);
      }
      for (const NodeProto *node : second_next) {
        second_by_type[node->op_type().value()].push_back(node);
      }
      for (const auto &[type, first_nodes] : first_by_type) {
        if (type == "Identity") {
          continue;
        }
        const auto second_it = second_by_type.find(type);
        if (second_it == second_by_type.end()) {
          continue;
        }
        for (const NodeProto *left : first_nodes) {
          for (const NodeProto *right : second_it->second) {
            if (left == right || left->domain().value() != right->domain().value() ||
                !SameNodeWithAliases(*left, *right, aliases)) {
              continue;
            }
            const auto has_used_output = [&used_outputs](const NodeProto &node) {
              for (const auto &output : node.output()) {
                if (!output.value().empty() &&
                    used_outputs.find(output.value()) != used_outputs.end()) {
                  return true;
                }
              }
              return false;
            };
            if (has_used_output(*left) || has_used_output(*right)) {
              cannot_continue = true;
              break;
            }
            nodes.push_back(left);
            nodes.push_back(right);
            stack.push_back({left, right});
            for (const auto &output : left->output()) {
              if (!output.value().empty()) {
                used_outputs.insert(output.value());
              }
            }
            for (const auto &output : right->output()) {
              if (!output.value().empty()) {
                used_outputs.insert(output.value());
              }
            }
          }
          if (cannot_continue) {
            break;
          }
        }
        if (cannot_continue) {
          break;
        }
      }
    }
    match = core::builder::MatchResult{this, std::move(nodes), nullptr};
  }
  return match.pattern != nullptr ? match : NoMatch(candidate, "no equivalent sibling nodes found");
}

utils::RepeatedProtoField<NodeProto>
SameChildrenPattern::Apply(core::builder::GraphGraph &,
                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.empty() || nodes.size() % 2 != 0) {
    throw BuilderError("SameChildrenPattern::Apply expects a non-empty sequence of node pairs.");
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  std::unordered_set<const NodeProto *> already_added;
  for (std::size_t index = 0; index < nodes.size(); index += 2) {
    if (nodes[index] == nullptr || nodes[index + 1] == nullptr) {
      throw BuilderError("SameChildrenPattern::Apply does not accept null node pairs.");
    }
    const NodeProto &first = *nodes[index];
    const NodeProto &second = *nodes[index + 1];
    if (already_added.insert(&first).second) {
      replacements.add() = first;
    }
    const int output_count = std::min(first.output_size(), second.output_size());
    for (int output_index = 0; output_index < output_count; ++output_index) {
      const std::string &first_output = first.output(output_index).value();
      const std::string &second_output = second.output(output_index).value();
      if (first_output.empty() && second_output.empty()) {
        continue;
      }
      if (first_output.empty()) {
        throw BuilderError("SameChildrenPattern::Apply cannot redirect an empty first output.");
      }
      NodeProto identity = MakeNode("Identity", {first_output}, {second_output}, "",
                                    ("SameChildrenPattern--" + second.name().value()).c_str());
      if (second.has_doc_string()) {
        identity.set_doc_string(second.doc_string().value());
      }
      replacements.push_back(std::move(identity));
    }
  }
  return replacements;
}

core::builder::MatchResult SameChildrenFromInputPattern::Match(core::builder::GraphGraph &graph,
                                                               const NodeProto &candidate) const {
  if (candidate.input_size() == 0 || graph.NodeBefore(candidate.input()[0].value()) != nullptr) {
    return NoMatch(candidate, "the candidate first input is not a graph input");
  }
  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.input()[0].value());
  if (next_nodes.size() <= 1) {
    return NoMatch(candidate, "the graph input has fewer than two consumers");
  }
  core::builder::MatchResult match = MatchWithNodes(graph, candidate, next_nodes);
  return match.pattern != nullptr
             ? match
             : NoMatch(candidate, "the graph input consumers are not equivalent");
}

std::set<std::string> ShapeBasedSameChildrenPattern::FastOpType() const {
  return {"Expand", "Reshape"};
}

core::builder::MatchResult ShapeBasedSameChildrenPattern::Match(core::builder::GraphGraph &graph,
                                                                const NodeProto &candidate) const {
  const bool is_expand = IsDefaultOp(candidate, "Expand");
  const bool is_reshape = IsDefaultOp(candidate, "Reshape");
  if ((!is_expand && !is_reshape) || candidate.input_size() < 1 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Expand or Reshape");
  }
  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.input()[0].value());
  if (next_nodes.size() <= 1) {
    return NoMatch(candidate, "the common input has fewer than two consumers");
  }
  for (const char *op_type : {"Expand", "Reshape"}) {
    std::vector<const NodeProto *> selected;
    for (const NodeProto *node : next_nodes) {
      if (node != nullptr && IsDefaultOp(*node, op_type) && node->input_size() >= 1 &&
          node->output_size() >= 1 && node->input()[0].value() == candidate.input()[0].value() &&
          graph.HasShape(node->output()[0].value())) {
        selected.push_back(node);
      }
    }
    if (selected.size() < 2) {
      continue;
    }
    const core::symbolic::SymShape &shape =
        graph.GetShape(selected[0]->output()[0].value()).Shape();
    const bool same_shapes =
        std::all_of(selected.begin() + 1, selected.end(), [&graph, &shape](const NodeProto *node) {
          return graph.GetShape(node->output()[0].value()).Shape() == shape;
        });
    if (same_shapes) {
      return core::builder::MatchResult{this, std::move(selected), nullptr};
    }
  }
  return NoMatch(candidate, "equivalent Expand or Reshape output shapes were not found");
}

utils::RepeatedProtoField<NodeProto>
ShapeBasedSameChildrenPattern::Apply(core::builder::GraphGraph &,
                                     const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() < 2 || nodes[0] == nullptr) {
    throw BuilderError("ShapeBasedSameChildrenPattern::Apply expects at least two nodes.");
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.add() = *nodes[0];
  for (std::size_t index = 1; index < nodes.size(); ++index) {
    if (nodes[index] == nullptr || nodes[index]->output_size() == 0) {
      throw BuilderError("ShapeBasedSameChildrenPattern::Apply received an invalid node.");
    }
    replacements.push_back(
        MakeNode("Identity", {nodes[0]->output()[0].value()}, {nodes[index]->output()[0].value()},
                 "", ("ShapeBasedSameChildrenPattern--" + nodes[index]->name().value()).c_str()));
  }
  return replacements;
}

std::set<std::string> ShapeBasedIdentityPattern::FastOpType() const {
  return {"Slice", "Transpose"};
}

core::builder::MatchResult ShapeBasedIdentityPattern::Match(core::builder::GraphGraph &graph,
                                                            const NodeProto &candidate) const {
  if ((!IsDefaultOp(candidate, "Transpose") && !IsDefaultOp(candidate, "Slice")) ||
      candidate.input_size() < 1 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Transpose or Slice");
  }
  if (candidate.op_type().value() != "Slice") {
    return NoMatch(candidate, "the upstream pattern only proves Slice identities");
  }
  if (candidate.input_size() == 5) {
    const std::string &steps = candidate.input()[4].value();
    const TensorProto *steps_tensor =
        graph.IsConstant(steps) ? graph.GetComputedConstant(steps) : nullptr;
    std::vector<int64_t> values;
    if (steps_tensor == nullptr || !ReadIntegerValues(*steps_tensor, values) || values.empty() ||
        !std::all_of(values.begin(), values.end(), [](int64_t value) { return value == 1; })) {
      return NoMatch(candidate, "the Slice steps are not a materialised all-one constant");
    }
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.output()[0].value())) {
    return NoMatch(candidate, "the Slice input or output shape is unknown");
  }
  if (graph.GetShape(candidate.input()[0].value()).Shape() !=
      graph.GetShape(candidate.output()[0].value()).Shape()) {
    return NoMatch(candidate, "the Slice input and output shapes differ");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ShapeBasedIdentityPattern::Apply(core::builder::GraphGraph &graph,
                                 const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("ShapeBasedIdentityPattern::Apply expects one Slice node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ShapeBasedIdentityPattern::Apply received an unsafe or inconsistent match.");
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(
      MakeNode("Identity", {nodes[0]->input()[0].value()}, {nodes[0]->output()[0].value()}, "",
               ("ShapeBasedIdentityPattern--" + nodes[0]->name().value()).c_str()));
  return replacements;
}

std::set<std::string> SwapUnaryPattern::FastOpType() const {
  return {"Reshape", "Squeeze", "Transpose", "Unsqueeze"};
}

core::builder::MatchResult SwapUnaryPattern::Match(core::builder::GraphGraph &graph,
                                                   const NodeProto &candidate) const {
  const bool layout = IsDefaultOp(candidate, "Transpose") || IsDefaultOp(candidate, "Reshape") ||
                      IsDefaultOp(candidate, "Squeeze") || IsDefaultOp(candidate, "Unsqueeze");
  if (!layout || candidate.input_size() < 1 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a supported default-domain layout operation");
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      graph.GetShape(candidate.input()[0].value()).Shape().Rank() == 0) {
    return NoMatch(candidate, "the layout input rank is unknown or zero");
  }
  if (graph.IsUsedMoreThanOnce(candidate.output()[0].value()) ||
      candidate.op_type().value() == "Squeeze" || candidate.op_type().value() == "Unsqueeze") {
    return NoMatch(candidate, "the layout output is shared or the operation changes rank");
  }
  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.output()[0].value());
  if (next_nodes.size() != 1 || next_nodes[0] == nullptr) {
    return NoMatch(candidate, "the layout output does not have one consumer");
  }
  const NodeProto *next = next_nodes[0];
  const bool scalar_binary =
      (next->op_type().value() == "Mul" || next->op_type().value() == "Add" ||
       next->op_type().value() == "Div" || next->op_type().value() == "Sub") &&
      next->input_size() >= 2 && HasOneElementVectorShape(graph, next->input()[1].value());
  if (!IsUnaryLike(next->op_type().value()) && !scalar_binary) {
    return NoMatch(candidate, "the layout consumer is not unary-like or binary with shape [1]");
  }
  return core::builder::MatchResult{this, {&candidate, next}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
SwapUnaryPattern::Apply(core::builder::GraphGraph &graph,
                        const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("SwapUnaryPattern::Apply expects a layout and a following unary node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("SwapUnaryPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &layout = *nodes[0];
  const NodeProto &unary = *nodes[1];
  std::vector<std::string> unary_inputs = {layout.input()[0].value()};
  for (int index = 1; index < unary.input_size(); ++index) {
    unary_inputs.push_back(unary.input()[index].value());
  }
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string temporary =
      builder.UniqueName("SwapUnaryPattern--" + layout.output()[0].value());
  const std::string name = "SwapUnaryPattern--" + unary.name().value();
  NodeProto first =
      MakeNode(unary.op_type().value().c_str(), unary_inputs, {temporary}, "", name.c_str());
  for (const AttributeProto &attribute : unary.attribute()) {
    first.mutable_attribute()->push_back(attribute);
  }
  std::vector<std::string> layout_inputs = {temporary};
  for (int index = 1; index < layout.input_size(); ++index) {
    layout_inputs.push_back(layout.input()[index].value());
  }
  NodeProto second =
      MakeNode(layout.op_type().value().c_str(), layout_inputs, {unary.output()[0].value()}, "",
               ("SwapUnaryPattern--" + layout.name().value()).c_str());
  for (const AttributeProto &attribute : layout.attribute()) {
    second.mutable_attribute()->push_back(attribute);
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(first));
  replacements.push_back(std::move(second));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
