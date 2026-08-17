// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/transpose/transpose_pattern.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;

bool IsDefaultOp(const NodeProto &node, const char *op_type) {
  return node.op_type().value() == op_type &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain;
}

/// Applies ``perm`` to ``on``: ``result[i] = on[perm[i]]``.
std::vector<int64_t> ApplyPerm(const std::vector<int64_t> &perm, const std::vector<int64_t> &on) {
  std::vector<int64_t> result(perm.size(), 0);
  for (std::size_t i = 0; i < perm.size(); ++i) {
    result[i] = on[static_cast<std::size_t>(perm[i])];
  }
  return result;
}

/// Composes ``second`` after ``first`` (``second`` is applied last), so that the
/// result is equivalent to a single transpose: ``result[i] = first[second[i]]``.
std::vector<int64_t> ComposePerms(const std::vector<int64_t> &first,
                                  const std::vector<int64_t> &second) {
  return ApplyPerm(second, first);
}

/// Returns ``true`` when ``perm`` is a valid permutation of ``[0, perm.size())``.
bool IsValidPerm(const std::vector<int64_t> &perm) {
  std::vector<int64_t> sorted = perm;
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    if (sorted[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

bool IsIdentityPerm(const std::vector<int64_t> &perm) {
  for (std::size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

/// Returns the rank of ``name`` when its shape is known.
bool GetRank(core::builder::GraphGraph &graph, const std::string &name, std::size_t &rank) {
  if (!graph.HasShape(name)) {
    return false;
  }
  rank = graph.GetShape(name).Shape().Rank();
  return true;
}

} // namespace

std::set<std::string> TransposeTransposePattern::FastOpType() const { return {"Transpose"}; }

core::builder::MatchResult TransposeTransposePattern::Match(core::builder::GraphGraph &graph,
                                                            const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Transpose") || candidate.input_size() != 1 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Transpose");
  }
  const NodeProto *next_node = nullptr;
  for (const NodeProto *consumer : graph.NextNodes(candidate.output()[0].value())) {
    if (consumer != nullptr && IsDefaultOp(*consumer, "Transpose") && consumer->input_size() == 1 &&
        consumer->output_size() == 1) {
      next_node = consumer;
      break;
    }
  }
  if (next_node == nullptr) {
    return NoMatch(candidate, "the Transpose output is not consumed by another Transpose");
  }

  std::vector<int64_t> first_perm;
  std::vector<int64_t> second_perm;
  if (!GetAttributeInts(candidate, "perm", first_perm) ||
      !GetAttributeInts(*next_node, "perm", second_perm)) {
    return NoMatch(candidate, "both Transpose nodes must define a perm attribute");
  }
  if (first_perm.size() != second_perm.size() || !IsValidPerm(first_perm) ||
      !IsValidPerm(second_perm)) {
    return NoMatch(candidate, "the perms are not compatible permutations");
  }

  const std::vector<int64_t> composed = ComposePerms(first_perm, second_perm);
  if (!IsIdentityPerm(composed) && graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
    return NoMatch(candidate,
                   "the intermediate Transpose is shared and the perms do not cancel out");
  }

  return core::builder::MatchResult{this, {&candidate, next_node}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
TransposeTransposePattern::Apply(core::builder::GraphGraph &graph,
                                 const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("TransposeTransposePattern::Apply expects two consecutive Transpose nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "TransposeTransposePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &first = *nodes[0];
  const NodeProto &second = *nodes[1];

  std::vector<int64_t> first_perm;
  std::vector<int64_t> second_perm;
  GetAttributeInts(first, "perm", first_perm);
  GetAttributeInts(second, "perm", second_perm);
  const std::vector<int64_t> composed = ComposePerms(first_perm, second_perm);

  const std::string name = "TransposeTransposePattern--" + first.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(first.output()[0].value())) {
    replacements.add() = first;
  }
  if (IsIdentityPerm(composed)) {
    replacements.push_back(MakeNode("Identity", {first.input()[0].value()},
                                    {second.output()[0].value()}, "", name.c_str()));
  } else {
    NodeProto merged = MakeNode("Transpose", {first.input()[0].value()},
                                {second.output()[0].value()}, "", name.c_str());
    AddAttribute(merged, "perm", composed);
    replacements.push_back(std::move(merged));
  }
  return replacements;
}

std::set<std::string> TransposeGatherPattern::FastOpType() const { return {"Gather"}; }

core::builder::MatchResult TransposeGatherPattern::Match(core::builder::GraphGraph &graph,
                                                         const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Gather") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Gather with two inputs");
  }
  const NodeProto *transpose = graph.NodeBefore(candidate.input()[0].value());
  if (transpose == nullptr || !IsDefaultOp(*transpose, "Transpose") ||
      transpose->input_size() != 1 || transpose->output_size() != 1) {
    return NoMatch(candidate, "the Gather data is not produced by a default-domain Transpose");
  }
  std::size_t index_rank = 0;
  if (!GetRank(graph, candidate.input()[1].value(), index_rank)) {
    return NoMatch(candidate, "the Gather index rank is unknown");
  }
  if (index_rank != 0) {
    return NoMatch(candidate, "the Gather index is not a scalar");
  }
  std::vector<int64_t> perm;
  if (!GetAttributeInts(*transpose, "perm", perm) || !IsValidPerm(perm)) {
    return NoMatch(candidate, "the Transpose has no valid perm attribute");
  }
  const int64_t axis = GetAttributeOr<int64_t>(candidate, "axis", 0);
  const int64_t rank = static_cast<int64_t>(perm.size());
  const int64_t normalized_axis = axis < 0 ? axis + rank : axis;
  if (normalized_axis < 0 || normalized_axis >= rank) {
    return NoMatch(candidate, "the Gather axis is out of range for the Transpose rank");
  }
  return core::builder::MatchResult{this, {transpose, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
TransposeGatherPattern::Apply(core::builder::GraphGraph &graph,
                              const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("TransposeGatherPattern::Apply expects one Transpose and one Gather.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("TransposeGatherPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &transpose = *nodes[0];
  const NodeProto &gather = *nodes[1];

  std::vector<int64_t> perm;
  GetAttributeInts(transpose, "perm", perm);
  const int64_t rank = static_cast<int64_t>(perm.size());
  int64_t axis = GetAttributeOr<int64_t>(gather, "axis", 0);
  if (axis < 0) {
    axis += rank;
  }
  const int64_t new_axis = perm[static_cast<std::size_t>(axis)];

  std::vector<int64_t> perm_less;
  perm_less.reserve(perm.size() - 1);
  for (std::size_t i = 0; i < perm.size(); ++i) {
    if (static_cast<int64_t>(i) != axis) {
      perm_less.push_back(perm[i]);
    }
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "TransposeGatherPattern--" + gather.name().value();
  const bool keep_transpose = graph.IsUsedMoreThanOnce(transpose.output()[0].value());

  utils::RepeatedProtoField<NodeProto> replacements;
  if (keep_transpose) {
    replacements.add() = transpose;
  }

  if (std::is_sorted(perm_less.begin(), perm_less.end())) {
    NodeProto new_gather =
        MakeNode("Gather", {transpose.input()[0].value(), gather.input()[1].value()},
                 {gather.output()[0].value()}, "", name.c_str());
    AddAttribute(new_gather, "axis", static_cast<int64_t>(new_axis));
    replacements.push_back(std::move(new_gather));
  } else {
    std::vector<int64_t> new_perm;
    new_perm.reserve(perm_less.size());
    for (int64_t p : perm_less) {
      new_perm.push_back(p > new_axis ? p - 1 : p);
    }
    const std::string gather_out =
        builder.UniqueName("TransposeGatherPattern_" + gather.output()[0].value());
    NodeProto new_gather =
        MakeNode("Gather", {transpose.input()[0].value(), gather.input()[1].value()}, {gather_out},
                 "", (name + "--gather").c_str());
    AddAttribute(new_gather, "axis", static_cast<int64_t>(new_axis));
    replacements.push_back(std::move(new_gather));
    NodeProto new_transpose = MakeNode("Transpose", {gather_out}, {gather.output()[0].value()}, "",
                                       (name + "--transpose").c_str());
    AddAttribute(new_transpose, "perm", new_perm);
    replacements.push_back(std::move(new_transpose));
  }
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
