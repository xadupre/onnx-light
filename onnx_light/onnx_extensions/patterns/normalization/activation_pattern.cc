// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/normalization/activation_pattern.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::IsDefaultOp;
using core::builder::BuilderError;
using core::symbolic::TensorType;

bool MainOpsetAtLeast(core::builder::GraphGraph &graph, int minimum) {
  const int opset = graph.Builder().OpsetVersion("");
  return opset == core::shapes::kUnknownOpsetVersion || opset >= minimum;
}

bool IsNode(const NodeProto *node, const char *op_type, int inputs) {
  return node != nullptr && IsDefaultOp(*node, op_type) && node->input_size() == inputs &&
         node->output_size() == 1;
}

bool HasOnlyConsumers(core::builder::GraphGraph &graph, const NodeProto &node,
                      std::initializer_list<const NodeProto *> expected) {
  if (node.output_size() != 1) {
    return false;
  }
  const std::string &name = node.output()[0].value();
  if (graph.IsOutput(name) || graph.IsUsedBySubgraph(name)) {
    return false;
  }
  const std::vector<const NodeProto *> &actual = graph.NextNodes(name);
  if (actual.size() != expected.size()) {
    return false;
  }
  return std::all_of(expected.begin(), expected.end(), [&](const NodeProto *consumer) {
    return consumer != nullptr && std::find(actual.begin(), actual.end(), consumer) != actual.end();
  });
}

bool ReadScalar(core::builder::GraphGraph &graph, const std::string &name, double &value) {
  if (!graph.IsConstantScalar(name, false)) {
    return false;
  }
  const TensorProto *tensor = graph.GetComputedConstant(name);
  if (tensor == nullptr) {
    const NodeProto *constant = graph.NodeBefore(name);
    if (constant == nullptr || !IsDefaultOp(*constant, "Constant")) {
      return false;
    }
    const AttributeProto *value_int = FindAttribute(*constant, "value_int");
    if (value_int != nullptr && value_int->type() == AttributeProto::AttributeType::INT) {
      value = static_cast<double>(value_int->i());
      return true;
    }
    const AttributeProto *value_float = FindAttribute(*constant, "value_float");
    if (value_float != nullptr && value_float->type() == AttributeProto::AttributeType::FLOAT) {
      value = static_cast<double>(value_float->f());
      return true;
    }
    return false;
  }
  const auto type = static_cast<TensorProto::DataType>(tensor->data_type());
  if (type != TensorProto::DataType::FLOAT16 && type != TensorProto::DataType::BFLOAT16) {
    return ReadScalarAsDouble(*tensor, value);
  }
  std::uint16_t bits = 0;
  if (tensor->int32_data().size() > 0) {
    bits = static_cast<std::uint16_t>(tensor->int32_data()[0]);
  } else if (tensor->is_raw_data() && tensor->ref_raw_data().size() >= 2) {
    bits = static_cast<std::uint16_t>(tensor->ref_raw_data()[0]) |
           (static_cast<std::uint16_t>(tensor->ref_raw_data()[1]) << 8);
  } else {
    return false;
  }
  value = type == TensorProto::DataType::FLOAT16
              ? static_cast<double>(core::runtime::Float16BitsToFloat(bits))
              : static_cast<double>(core::runtime::Bfloat16BitsToFloat(bits));
  return true;
}

bool IsScalar(core::builder::GraphGraph &graph, const std::string &name, double expected) {
  double value = 0.0;
  return ReadScalar(graph, name, value) && value == expected;
}

bool HasIntAttribute(const NodeProto &node, const char *name, int64_t expected) {
  const AttributeProto *attribute = FindAttribute(node, name);
  return attribute != nullptr && attribute->type() == AttributeProto::AttributeType::INT &&
         attribute->i() == expected;
}

} // namespace

std::set<std::string> GeluPattern::FastOpType() const { return {"Mul"}; }

core::builder::MatchResult GeluPattern::Match(core::builder::GraphGraph &graph,
                                              const NodeProto &candidate) const {
  if (!MainOpsetAtLeast(graph, min_opset_)) {
    return NoMatch(candidate, "the default-domain opset is below the configured minimum");
  }
  if (!IsDefaultOp(candidate, "Mul") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not the final default-domain two-input Mul");
  }
  const NodeProto *x_half = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *tanh_one = graph.NodeBefore(candidate.input()[1].value());
  if (!IsNode(x_half, "Mul", 2) || !IsNode(tanh_one, "Add", 2)) {
    return NoMatch(candidate, "the final Mul inputs are not Mul(x, 0.5) and Add(tanh, 1)");
  }
  const NodeProto *tanh = graph.NodeBefore(tanh_one->input()[0].value());
  const NodeProto *scaled = tanh == nullptr ? nullptr : graph.NodeBefore(tanh->input()[0].value());
  const NodeProto *add = scaled == nullptr ? nullptr : graph.NodeBefore(scaled->input()[0].value());
  const NodeProto *cubic = add == nullptr ? nullptr : graph.NodeBefore(add->input()[1].value());
  const NodeProto *power = cubic == nullptr ? nullptr : graph.NodeBefore(cubic->input()[0].value());
  if (!IsNode(tanh, "Tanh", 1) || !IsNode(scaled, "Mul", 2) || !IsNode(add, "Add", 2) ||
      !IsNode(cubic, "Mul", 2) || !IsNode(power, "Pow", 2)) {
    return NoMatch(candidate, "the tanh-based Gelu topology is incomplete");
  }
  const std::string &x = power->input()[0].value();
  if (add->input()[0].value() != x || x_half->input()[0].value() != x) {
    return NoMatch(candidate, "the three Gelu branches do not share the same input");
  }
  if (!HasOnlyConsumers(graph, *power, {cubic}) || !HasOnlyConsumers(graph, *cubic, {add}) ||
      !HasOnlyConsumers(graph, *add, {scaled}) || !HasOnlyConsumers(graph, *scaled, {tanh}) ||
      !HasOnlyConsumers(graph, *tanh, {tanh_one}) ||
      !HasOnlyConsumers(graph, *tanh_one, {&candidate}) ||
      !HasOnlyConsumers(graph, *x_half, {&candidate})) {
    return NoMatch(candidate, "an intermediate Gelu result is externally used");
  }

  double cubic_scale = 0.0;
  if (!IsScalar(graph, power->input()[1].value(), 3.0) ||
      !ReadScalar(graph, cubic->input()[1].value(), cubic_scale) ||
      (cubic_scale != 0.044715 && cubic_scale != 0.044708251953125) ||
      !IsScalar(graph, scaled->input()[1].value(), 0.7978515625) ||
      !IsScalar(graph, tanh_one->input()[1].value(), 1.0) ||
      !IsScalar(graph, x_half->input()[1].value(), 0.5)) {
    return NoMatch(candidate, "one Gelu scalar constant has an unexpected value");
  }
  return core::builder::MatchResult{
      this, {power, cubic, add, scaled, tanh, tanh_one, x_half, &candidate}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
GeluPattern::Apply(core::builder::GraphGraph &graph,
                   const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 8 || nodes[0] == nullptr || nodes[7] == nullptr) {
    throw BuilderError("GeluPattern::Apply expects the eight-node tanh Gelu decomposition.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[7]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("GeluPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &power = *nodes[0];
  const NodeProto &root = *nodes[7];
  NodeProto replacement =
      MakeNode("Gelu", {power.input()[0].value()}, {root.output()[0].value()}, domain_.c_str(),
               ("GeluPattern--" + root.name().value()).c_str());
  AddAttribute<std::string>(replacement, "approximate", "tanh");
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> LeakyReluPattern::FastOpType() const { return {"Where"}; }

core::builder::MatchResult LeakyReluPattern::Match(core::builder::GraphGraph &graph,
                                                   const NodeProto &candidate) const {
  if (!MainOpsetAtLeast(graph, min_opset_)) {
    return NoMatch(candidate, "the default-domain opset is below the configured minimum");
  }
  if (!IsDefaultOp(candidate, "Where") || candidate.input_size() != 3 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain three-input Where");
  }
  const NodeProto *greater = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *mul = graph.NodeBefore(candidate.input()[2].value());
  if (!IsNode(greater, "Greater", 2) || !IsNode(mul, "Mul", 2)) {
    return NoMatch(candidate, "Where inputs are not produced by Greater and Mul");
  }
  const std::string &x = candidate.input()[1].value();
  if (greater->input()[0].value() != x || mul->input()[0].value() != x) {
    return NoMatch(candidate, "Greater, Mul, and Where do not share the same data input");
  }
  if (!HasOnlyConsumers(graph, *greater, {&candidate}) ||
      !HasOnlyConsumers(graph, *mul, {&candidate})) {
    return NoMatch(candidate, "a LeakyRelu intermediate result is externally used");
  }
  double slope = 0.0;
  if (!IsScalar(graph, greater->input()[1].value(), 0.0) ||
      !ReadScalar(graph, mul->input()[1].value(), slope)) {
    return NoMatch(candidate, "the threshold is not zero or the slope is not scalar");
  }
  return core::builder::MatchResult{this, {greater, mul, &candidate}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
LeakyReluPattern::Apply(core::builder::GraphGraph &graph,
                        const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError("LeakyReluPattern::Apply expects Greater, Mul, and Where.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("LeakyReluPattern::Apply received an unsafe or inconsistent match.");
  }
  double slope = 0.0;
  if (!ReadScalar(graph, nodes[1]->input()[1].value(), slope)) {
    throw BuilderError("LeakyReluPattern::Apply could not read the slope.");
  }
  const NodeProto &where = *nodes[2];
  NodeProto replacement =
      MakeNode("LeakyRelu", {where.input()[1].value()}, {where.output()[0].value()}, "",
               ("LeakyReluPattern--" + where.name().value()).c_str());
  AddAttribute<float>(replacement, "alpha", static_cast<float>(slope));
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> SoftmaxCrossEntropyLossCastPattern::FastOpType() const { return {"Div"}; }

core::builder::MatchResult
SoftmaxCrossEntropyLossCastPattern::Match(core::builder::GraphGraph &graph,
                                          const NodeProto &candidate) const {
  if (!MainOpsetAtLeast(graph, min_opset_)) {
    return NoMatch(candidate, "the default-domain opset is below the configured minimum");
  }
  if (!IsDefaultOp(candidate, "Div") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not the final default-domain two-input Div");
  }

  const NodeProto *numerator_cast = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *denominator_cast = graph.NodeBefore(candidate.input()[1].value());
  const NodeProto *numerator_reduce =
      numerator_cast == nullptr ? nullptr : graph.NodeBefore(numerator_cast->input()[0].value());
  const NodeProto *denominator_reduce =
      denominator_cast == nullptr ? nullptr
                                  : graph.NodeBefore(denominator_cast->input()[0].value());
  const NodeProto *numerator_input_cast =
      numerator_reduce == nullptr ? nullptr
                                  : graph.NodeBefore(numerator_reduce->input()[0].value());
  const NodeProto *denominator_input_cast =
      denominator_reduce == nullptr ? nullptr
                                    : graph.NodeBefore(denominator_reduce->input()[0].value());
  if (!IsNode(numerator_cast, "Cast", 1) || !IsNode(denominator_cast, "Cast", 1) ||
      !IsNode(numerator_reduce, "ReduceSum", 1) || !IsNode(denominator_reduce, "ReduceSum", 1) ||
      !IsNode(numerator_input_cast, "Cast", 1) || !IsNode(denominator_input_cast, "Cast", 1)) {
    return NoMatch(candidate, "the numerator or denominator cast-reduction chain is incomplete");
  }
  if (!HasIntAttribute(*numerator_cast, "to", TensorProto::DataType::FLOAT16) ||
      !HasIntAttribute(*denominator_cast, "to", TensorProto::DataType::FLOAT16) ||
      !HasIntAttribute(*numerator_input_cast, "to", TensorProto::DataType::FLOAT) ||
      !HasIntAttribute(*denominator_input_cast, "to", TensorProto::DataType::FLOAT) ||
      !HasIntAttribute(*numerator_reduce, "keepdims", 0) ||
      !HasIntAttribute(*denominator_reduce, "keepdims", 0)) {
    return NoMatch(candidate, "the cast targets or ReduceSum keepdims attributes differ");
  }

  const NodeProto *where_loss = graph.NodeBefore(numerator_input_cast->input()[0].value());
  const NodeProto *not_node = graph.NodeBefore(denominator_input_cast->input()[0].value());
  if (!IsNode(where_loss, "Where", 3) || !IsNode(not_node, "Not", 1) ||
      where_loss->input()[0].value() != not_node->output()[0].value()) {
    return NoMatch(candidate, "the masked loss and denominator do not share one Not condition");
  }
  const NodeProto *negative = graph.NodeBefore(where_loss->input()[1].value());
  const NodeProto *squeeze =
      negative == nullptr ? nullptr : graph.NodeBefore(negative->input()[0].value());
  const NodeProto *gather =
      squeeze == nullptr ? nullptr : graph.NodeBefore(squeeze->input()[0].value());
  const NodeProto *log_softmax =
      gather == nullptr ? nullptr : graph.NodeBefore(gather->input()[0].value());
  const NodeProto *unsqueeze =
      gather == nullptr ? nullptr : graph.NodeBefore(gather->input()[1].value());
  const NodeProto *where_indices =
      unsqueeze == nullptr ? nullptr : graph.NodeBefore(unsqueeze->input()[0].value());
  const NodeProto *equal =
      not_node == nullptr ? nullptr : graph.NodeBefore(not_node->input()[0].value());
  if (!IsNode(negative, "Neg", 1) || !IsNode(squeeze, "Squeeze", 2) ||
      !IsNode(gather, "GatherElements", 2) || !IsNode(log_softmax, "LogSoftmax", 1) ||
      !IsNode(unsqueeze, "Unsqueeze", 2) || !IsNode(where_indices, "Where", 3) ||
      !IsNode(equal, "Equal", 2)) {
    return NoMatch(candidate, "the masked GatherElements loss topology is incomplete");
  }
  if (!HasIntAttribute(*log_softmax, "axis", 1) || !HasIntAttribute(*gather, "axis", 1)) {
    return NoMatch(candidate, "LogSoftmax or GatherElements does not use axis one");
  }

  const std::string &indices = equal->input()[0].value();
  if (where_indices->input()[0].value() != not_node->output()[0].value() ||
      where_indices->input()[1].value() != indices ||
      unsqueeze->input()[1].value() != squeeze->input()[1].value()) {
    return NoMatch(candidate, "the index mask, labels, or squeeze axes are inconsistent");
  }
  if (!IsScalar(graph, equal->input()[1].value(), -100.0) ||
      !IsScalar(graph, unsqueeze->input()[1].value(), 1.0) ||
      !IsScalar(graph, squeeze->input()[1].value(), 1.0)) {
    return NoMatch(candidate, "the ignore index or squeeze axes constants differ");
  }
  if (!IsScalar(graph, where_indices->input()[2].value(), 0.0) ||
      !IsScalar(graph, where_loss->input()[2].value(), 0.0)) {
    return NoMatch(candidate, "an ignored-label Where branch is not exactly zero");
  }
  const std::string &scores = log_softmax->input()[0].value();
  const std::string &masked_losses = where_loss->output()[0].value();
  const std::string &output = candidate.output()[0].value();
  if (!graph.HasType(scores) || !graph.HasType(indices) ||
      !graph.HasType(where_indices->input()[2].value()) ||
      !graph.HasType(where_loss->input()[2].value()) || !graph.HasType(masked_losses) ||
      !graph.HasType(output) || graph.GetType(scores) != TensorType::kFloat16 ||
      graph.GetType(indices) != TensorType::kInt64 ||
      graph.GetType(where_indices->input()[2].value()) != TensorType::kInt64 ||
      graph.GetType(where_loss->input()[2].value()) != TensorType::kFloat16 ||
      graph.GetType(masked_losses) != TensorType::kFloat16 ||
      graph.GetType(output) != TensorType::kFloat16) {
    return NoMatch(candidate, "the score, label, zero-branch, loss, or output types differ");
  }

  if (!HasOnlyConsumers(graph, *equal, {not_node}) ||
      !HasOnlyConsumers(graph, *not_node, {where_indices, where_loss, denominator_input_cast}) ||
      !HasOnlyConsumers(graph, *where_indices, {unsqueeze}) ||
      !HasOnlyConsumers(graph, *unsqueeze, {gather}) ||
      !HasOnlyConsumers(graph, *log_softmax, {gather}) ||
      !HasOnlyConsumers(graph, *gather, {squeeze}) ||
      !HasOnlyConsumers(graph, *squeeze, {negative}) ||
      !HasOnlyConsumers(graph, *negative, {where_loss}) ||
      !HasOnlyConsumers(graph, *where_loss, {numerator_input_cast}) ||
      !HasOnlyConsumers(graph, *denominator_input_cast, {denominator_reduce}) ||
      !HasOnlyConsumers(graph, *denominator_reduce, {denominator_cast}) ||
      !HasOnlyConsumers(graph, *denominator_cast, {&candidate}) ||
      !HasOnlyConsumers(graph, *numerator_input_cast, {numerator_reduce}) ||
      !HasOnlyConsumers(graph, *numerator_reduce, {numerator_cast}) ||
      !HasOnlyConsumers(graph, *numerator_cast, {&candidate})) {
    return NoMatch(candidate, "an intermediate loss result is externally used");
  }

  return core::builder::MatchResult{this,
                                    {equal, not_node, where_indices, unsqueeze, log_softmax, gather,
                                     squeeze, negative, where_loss, denominator_input_cast,
                                     denominator_reduce, denominator_cast, numerator_input_cast,
                                     numerator_reduce, numerator_cast, &candidate},
                                    nullptr};
}

utils::RepeatedProtoField<NodeProto>
SoftmaxCrossEntropyLossCastPattern::Apply(core::builder::GraphGraph &graph,
                                          const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 16 || nodes[0] == nullptr || nodes[4] == nullptr || nodes[15] == nullptr) {
    throw BuilderError(
        "SoftmaxCrossEntropyLossCastPattern::Apply expects the sixteen-node loss graph.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[15]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "SoftmaxCrossEntropyLossCastPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &equal = *nodes[0];
  const NodeProto &log_softmax = *nodes[4];
  const NodeProto &div = *nodes[15];
  NodeProto replacement = MakeNode(
      "SoftmaxCrossEntropyLoss", {log_softmax.input()[0].value(), equal.input()[0].value()},
      {div.output()[0].value()}, domain_.c_str(),
      ("SoftmaxCrossEntropyLossCastPattern--" + div.name().value()).c_str());
  AddAttribute<int64_t>(replacement, "ignore_index", -100);
  AddAttribute<std::string>(replacement, "reduction", "mean");
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> MaxReluPattern::FastOpType() const { return {"Max"}; }

core::builder::MatchResult MaxReluPattern::Match(core::builder::GraphGraph &graph,
                                                 const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Max") || candidate.input_size() != 2 ||
      candidate.output_size() < 1) {
    return NoMatch(candidate, "candidate is not a two-input default-domain Max");
  }
  if (!graph.HasType(candidate.input()[0].value())) {
    return NoMatch(candidate, "the first input type is unknown");
  }
  const TensorType type = graph.GetType(candidate.input()[0].value());
  if (type != TensorType::kFloat && type != TensorType::kFloat16 && type != TensorType::kInt16 &&
      type != TensorType::kInt32) {
    return NoMatch(candidate, "the first input type is not supported by the rewrite");
  }
  if ((type == TensorType::kInt16 || type == TensorType::kInt32) &&
      graph.Builder().OpsetVersion("") < 14) {
    return NoMatch(candidate, "integer Relu requires default-domain opset 14 or newer");
  }
  int zero_inputs = 0;
  for (const auto &input : candidate.input()) {
    if (IsScalar(graph, input.value(), 0.0)) {
      ++zero_inputs;
    }
  }
  if (zero_inputs != 1) {
    return NoMatch(candidate, "Max does not have exactly one scalar zero input");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
MaxReluPattern::Apply(core::builder::GraphGraph &graph,
                      const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("MaxReluPattern::Apply expects one Max node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("MaxReluPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &maximum = *nodes[0];
  std::string x;
  for (const auto &input : maximum.input()) {
    if (!IsScalar(graph, input.value(), 0.0)) {
      x = input.value();
      break;
    }
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("Relu", {x}, {maximum.output()[0].value()}, "",
                                  ("MaxReluPattern--" + maximum.name().value()).c_str()));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
