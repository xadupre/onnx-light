// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/collections/concat_pattern.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::GetAxis;
using collections::IsDefaultOp;
using core::builder::BuilderError;

// Shape-preserving unary operators that may be pushed through a Concat(x, x).
const std::set<std::string> &UnaryLikeOpTypes() {
  static const std::set<std::string> types = {"Abs",
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
  return types;
}

// Returns the indices of ``concat`` inputs that are empty along ``axis``.
std::vector<int> EmptyConcatInputs(core::builder::GraphGraph &graph, const NodeProto &concat) {
  const int64_t axis = GetAxis(concat, 0);
  std::vector<int> empty;
  for (int i = 0; i < concat.input_size(); ++i) {
    const std::string &name = concat.input()[i].value();
    if (!graph.HasShape(name)) {
      continue;
    }
    const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
    const int64_t rank = static_cast<int64_t>(shape.Rank());
    int64_t normalized = axis < 0 ? axis + rank : axis;
    if (normalized < 0 || normalized >= rank) {
      continue;
    }
    const auto &dim = shape[static_cast<std::size_t>(normalized)];
    if (dim.IsInt() && dim.AsInt() == 0) {
      empty.push_back(i);
    }
  }
  return empty;
}

// Locates which ``concat`` input holds element ``idx`` along axis 0. On success
// returns ``true`` and fills the input name, its local index and its size.
bool ConcatInputBucket(core::builder::GraphGraph &graph, const NodeProto &concat, int64_t idx,
                       std::string &input_name, int64_t &local_index, int64_t &input_size) {
  int64_t offset = 0;
  for (const auto &input : concat.input()) {
    const std::string &name = input.value();
    if (!graph.HasShape(name)) {
      return false;
    }
    const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
    if (shape.Rank() != 1 || !shape[0].IsInt()) {
      return false;
    }
    const int64_t n = shape[0].AsInt();
    if (offset + n > idx) {
      input_name = name;
      local_index = idx - offset;
      input_size = n;
      return true;
    }
    offset += n;
  }
  return false;
}

// Returns true when ``unary`` may be pushed ahead of ``concat`` (see pattern).
bool ValidUnaryConsumer(core::builder::GraphGraph &graph, const NodeProto &concat,
                        const NodeProto &unary) {
  const std::string &op_type = unary.op_type().value();
  if (NormaliseDomain(unary.domain().value()) != kDefaultOnnxDomain) {
    return false;
  }
  if (UnaryLikeOpTypes().count(op_type) != 0) {
    return true;
  }
  if (op_type == "Unsqueeze" && unary.input_size() >= 2 &&
      graph.IsConstantScalar(unary.input()[1].value())) {
    int64_t cst = 0;
    if (!collections::ReadScalarInt(*graph.GetComputedConstant(unary.input()[1].value()), cst)) {
      return false;
    }
    const int64_t axis = GetAxis(concat, 0);
    if (axis == -1 && cst != -1 && graph.HasShape(unary.input()[0].value()) &&
        cst < static_cast<int64_t>(graph.GetShape(unary.input()[0].value()).Shape().Rank())) {
      return true;
    }
  }
  static const std::set<std::string> binary = {"Mul", "Add", "Div", "Sub"};
  if (binary.count(op_type) != 0 && unary.input_size() >= 2 &&
      graph.IsConstantScalar(unary.input()[1].value())) {
    return true;
  }
  return false;
}

} // namespace

std::set<std::string> ConcatEmptyPattern::FastOpType() const { return {"Concat"}; }

core::builder::MatchResult ConcatEmptyPattern::Match(core::builder::GraphGraph &graph,
                                                     const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Concat")) {
    return NoMatch(candidate, "candidate is not a default-domain Concat");
  }
  const std::vector<int> empty = EmptyConcatInputs(graph, candidate);
  if (empty.empty()) {
    return NoMatch(candidate, "no Concat input is empty along the axis");
  }
  if (static_cast<int>(empty.size()) >= candidate.input_size()) {
    return NoMatch(candidate, "every Concat input is empty along the axis");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ConcatEmptyPattern::Apply(core::builder::GraphGraph &graph,
                          const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("ConcatEmptyPattern::Apply expects one Concat node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ConcatEmptyPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &node = *nodes[0];
  const std::vector<int> empty = EmptyConcatInputs(graph, node);
  std::vector<bool> drop(node.input_size(), false);
  for (int i : empty) {
    drop[i] = true;
  }
  std::vector<std::string> kept;
  for (int i = 0; i < node.input_size(); ++i) {
    if (!drop[i]) {
      kept.push_back(node.input()[i].value());
    }
  }

  const std::string name = "ConcatEmptyPattern--" + node.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  if (kept.size() == 1) {
    replacements.push_back(
        MakeNode("Identity", kept, {node.output()[0].value()}, "", name.c_str()));
    return replacements;
  }
  NodeProto concat = MakeNode("Concat", kept, {node.output()[0].value()}, "", name.c_str());
  for (const auto &attribute : node.attribute()) {
    *concat.add_attribute() = attribute;
  }
  replacements.push_back(concat);
  return replacements;
}

std::set<std::string> ConcatGatherPattern::FastOpType() const { return {"Gather"}; }

core::builder::MatchResult ConcatGatherPattern::Match(core::builder::GraphGraph &graph,
                                                      const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Gather")) {
    return NoMatch(candidate, "candidate is not a default-domain Gather");
  }
  const std::string &index_name = candidate.input()[1].value();
  if (!graph.IsConstant(index_name)) {
    return NoMatch(candidate, "the Gather index is not constant");
  }
  const TensorProto *index = graph.GetComputedConstant(index_name);
  if (index == nullptr ||
      static_cast<TensorProto::DataType>(index->data_type()) != TensorProto::DataType::INT64) {
    return NoMatch(candidate, "the Gather index is not a materialised int64 tensor");
  }
  if (index->dims_size() != 1 || index->dims()[0] != 1) {
    return NoMatch(candidate, "the Gather index is not a single-element vector");
  }
  const NodeProto *concat = graph.NodeBefore(candidate.input()[0].value());
  if (concat == nullptr || !IsDefaultOp(*concat, "Concat")) {
    return NoMatch(candidate, "the gathered value is not produced by a Concat");
  }
  std::vector<int64_t> index_values;
  if (!ReadIntegerValues(*index, index_values) || index_values.empty()) {
    return NoMatch(candidate, "the Gather index could not be read");
  }
  std::string input_name;
  int64_t local_index = 0;
  int64_t input_size = 0;
  if (!ConcatInputBucket(graph, *concat, index_values[0], input_name, local_index, input_size)) {
    return NoMatch(candidate, "the index does not map to a statically-sized Concat input");
  }
  return core::builder::MatchResult{this, {concat, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ConcatGatherPattern::Apply(core::builder::GraphGraph &graph,
                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ConcatGatherPattern::Apply expects a Concat and a Gather node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ConcatGatherPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &concat = *nodes[0];
  const NodeProto &gather = *nodes[1];

  const TensorProto *index = graph.GetComputedConstant(gather.input()[1].value());
  std::vector<int64_t> index_values;
  if (index == nullptr || !ReadIntegerValues(*index, index_values) || index_values.empty()) {
    throw BuilderError("ConcatGatherPattern::Apply could not read the Gather index.");
  }
  std::string input_name;
  int64_t local_index = 0;
  int64_t input_size = 0;
  if (!ConcatInputBucket(graph, concat, index_values[0], input_name, local_index, input_size)) {
    throw BuilderError("ConcatGatherPattern::Apply could not locate the Concat input.");
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "ConcatGatherPattern--" + gather.name().value();
  const std::string &output = gather.output()[0].value();

  NodeProto replacement;
  if (input_size == 1) {
    replacement = MakeNode("Identity", {input_name}, {output}, "", name.c_str());
  } else {
    const NodeProto *producer = graph.NodeBefore(input_name);
    if (producer != nullptr && IsDefaultOp(*producer, "Shape")) {
      const int64_t start = GetAttributeOr<int64_t>(*producer, "start", 0);
      replacement = MakeNode("Shape", {producer->input()[0].value()}, {output}, "", name.c_str());
      AddAttribute<int64_t>(replacement, "start", start + local_index);
      AddAttribute<int64_t>(replacement, "end", start + local_index + 1);
    } else {
      const std::string index_init = collections::FreeInitializerName(builder, name + "_idx");
      builder.MakeInitializer(MakeInitializerShape(index_init.c_str(), {local_index}));
      replacement = MakeNode("Gather", {input_name, index_init}, {output}, "", name.c_str());
    }
  }

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(concat.output()[0].value())) {
    replacements.add() = concat;
  }
  replacements.push_back(replacement);
  return replacements;
}

std::set<std::string> ConcatTwiceUnaryPattern::FastOpType() const { return {"Concat"}; }

core::builder::MatchResult ConcatTwiceUnaryPattern::Match(core::builder::GraphGraph &graph,
                                                          const NodeProto &candidate) const {
  if (graph.Builder().OpsetVersion("") < 18) {
    return NoMatch(candidate, "the default opset is older than 18");
  }
  if (!IsDefaultOp(candidate, "Concat")) {
    return NoMatch(candidate, "candidate is not a default-domain Concat");
  }
  if (candidate.input_size() != 2 || candidate.input()[0].value() != candidate.input()[1].value()) {
    return NoMatch(candidate, "the Concat does not repeat a single input twice");
  }
  for (const NodeProto *user : graph.NextNodes(candidate.output()[0].value())) {
    if (ValidUnaryConsumer(graph, candidate, *user)) {
      return core::builder::MatchResult{this, {&candidate, user}, &candidate};
    }
  }
  return NoMatch(candidate, "the Concat is not consumed by a shape-preserving unary op");
}

utils::RepeatedProtoField<NodeProto>
ConcatTwiceUnaryPattern::Apply(core::builder::GraphGraph &graph,
                               const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ConcatTwiceUnaryPattern::Apply expects a Concat and a unary node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ConcatTwiceUnaryPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &concat = *nodes[0];
  const NodeProto &unary = *nodes[1];

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string new_name = builder.UniqueName("u" + unary.output()[0].value());
  const std::string name = "ConcatTwiceUnaryPattern--" + unary.name().value();

  std::vector<std::string> unary_inputs{concat.input()[0].value()};
  for (int i = 1; i < unary.input_size(); ++i) {
    unary_inputs.push_back(unary.input()[i].value());
  }
  NodeProto new_unary =
      MakeNode(unary.op_type().value().c_str(), unary_inputs, {new_name}, "", name.c_str());
  for (const auto &attribute : unary.attribute()) {
    *new_unary.add_attribute() = attribute;
  }
  NodeProto new_concat = MakeNode(concat.op_type().value().c_str(), {new_name, new_name},
                                  {unary.output()[0].value()}, "", name.c_str());
  for (const auto &attribute : concat.attribute()) {
    *new_concat.add_attribute() = attribute;
  }

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(concat.output()[0].value())) {
    replacements.add() = concat;
  }
  replacements.push_back(new_unary);
  replacements.push_back(new_concat);
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
