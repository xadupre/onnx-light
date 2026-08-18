// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/layout/layout_pattern.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::FreeInitializerName;
using collections::IsDefaultOp;
using core::builder::BuilderError;

bool MainOpsetAtLeast(core::builder::GraphGraph &graph, int minimum) {
  const int opset = graph.Builder().OpsetVersion("");
  return opset == core::shapes::kUnknownOpsetVersion || opset >= minimum;
}

bool ReadConstantIntegers(core::builder::GraphGraph &graph, const std::string &name,
                          std::vector<int64_t> &values) {
  if (!graph.IsConstant(name)) {
    return false;
  }
  const TensorProto *tensor = graph.GetComputedConstant(name);
  return tensor != nullptr && ReadIntegerValues(*tensor, values);
}

bool ReadConstantShape(core::builder::GraphGraph &graph, const std::string &name,
                       std::vector<int64_t> &values) {
  if (!graph.HasType(name) || graph.GetType(name) != core::symbolic::TensorType::kInt64 ||
      !graph.IsConstant(name)) {
    return false;
  }
  const TensorProto *tensor = graph.GetComputedConstant(name);
  return tensor != nullptr && tensor->dims_size() == 1 && ReadIntegerValues(*tensor, values);
}

std::vector<std::string> InputsAfterFirst(const NodeProto &node) {
  std::vector<std::string> inputs;
  inputs.reserve(static_cast<std::size_t>(node.input_size()));
  for (int i = 1; i < node.input_size(); ++i) {
    inputs.push_back(node.input()[i].value());
  }
  return inputs;
}

std::string MakeShapeInitializer(core::builder::GraphBuilder &builder, const std::string &base,
                                 const std::vector<int64_t> &values) {
  const std::string name = FreeInitializerName(builder, base);
  return builder.MakeInitializer(MakeInitializerShape(name.c_str(), values));
}

bool IsBinaryOp(const NodeProto &node) {
  const std::string &op_type = node.op_type().value();
  return op_type == "Add" || op_type == "Div" || op_type == "Mul" || op_type == "Sub";
}

bool IsValidPermutation(const std::vector<int64_t> &perm) {
  std::vector<int64_t> sorted = perm;
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    if (sorted[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

bool IsSingleDimensionShape(core::builder::GraphGraph &graph, const std::string &name) {
  if (!graph.HasShape(name)) {
    return false;
  }
  const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
  return shape.Rank() == 1 && shape[0].IsInt() && shape[0].AsInt() == 1;
}

bool ResolveSqueezeAxes(core::builder::GraphGraph &graph, const NodeProto &squeeze,
                        std::vector<int64_t> &axes) {
  if (squeeze.input_size() == 2) {
    return ReadConstantShape(graph, squeeze.input()[1].value(), axes);
  }
  return squeeze.input_size() == 1 && IsSingleDimensionShape(graph, squeeze.input()[0].value()) &&
         (axes = {0}, true);
}

struct ReshapeTransposePlan {
  std::vector<int64_t> perm;
  std::vector<int64_t> shape;
  bool after = false;
};

bool AlignShape(
    const std::vector<int64_t> &shape, const std::vector<int64_t> &new_shape,
    std::vector<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>> &mapped) {
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < shape.size() && j < new_shape.size()) {
    if (shape[i] == new_shape[j]) {
      mapped.push_back({{i}, {j}});
      ++i;
      ++j;
      continue;
    }

    std::vector<std::size_t> ii{i};
    std::vector<std::size_t> jj{j};
    int64_t first = shape[i];
    int64_t second = new_shape[j];
    while (first != second && i < shape.size() && j < new_shape.size()) {
      if (first < second) {
        if (++i == shape.size() || shape[i] == 0 ||
            (first > 0 && shape[i] > std::numeric_limits<int64_t>::max() / first)) {
          return false;
        }
        first *= shape[i];
        ii.push_back(i);
      } else {
        if (++j == new_shape.size() || new_shape[j] == 0 ||
            (second > 0 && new_shape[j] > std::numeric_limits<int64_t>::max() / second)) {
          return false;
        }
        second *= new_shape[j];
        jj.push_back(j);
      }
    }
    if (std::min(ii.size(), jj.size()) != 1) {
      return false;
    }
    mapped.push_back({std::move(ii), std::move(jj)});
    ++i;
    ++j;
  }
  return i == shape.size() && j == new_shape.size();
}

bool MakeReshapeTransposePlan(core::builder::GraphGraph &graph, const NodeProto &first,
                              const NodeProto &reshape, const NodeProto &second,
                              ReshapeTransposePlan &plan) {
  std::vector<int64_t> first_perm;
  std::vector<int64_t> second_perm;
  std::vector<int64_t> new_shape;
  if (!GetAttributeInts(first, "perm", first_perm) ||
      !GetAttributeInts(second, "perm", second_perm) || !IsValidPermutation(first_perm) ||
      !IsValidPermutation(second_perm) ||
      !ReadConstantIntegers(graph, reshape.input()[1].value(), new_shape) ||
      std::find(new_shape.begin(), new_shape.end(), -1) != new_shape.end() ||
      !graph.HasShape(reshape.input()[0].value())) {
    return false;
  }

  const core::symbolic::SymShape &inferred = graph.GetShape(reshape.input()[0].value()).Shape();
  std::vector<int64_t> shape;
  shape.reserve(inferred.Rank());
  for (std::size_t i = 0; i < inferred.Rank(); ++i) {
    if (!inferred[i].IsInt()) {
      return false;
    }
    shape.push_back(inferred[i].AsInt());
  }

  std::vector<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>> mapped;
  if (!AlignShape(shape, new_shape, mapped)) {
    return false;
  }

  if (second_perm.size() <= first_perm.size()) {
    if (mapped.size() != second_perm.size()) {
      return false;
    }
    for (int64_t p : second_perm) {
      const std::vector<std::size_t> &indices = mapped[static_cast<std::size_t>(p)].first;
      for (std::size_t index : indices) {
        plan.perm.push_back(static_cast<int64_t>(index));
      }
      plan.shape.push_back(new_shape[static_cast<std::size_t>(p)]);
    }
    plan.after = true;
    return true;
  }

  if (mapped.size() != first_perm.size()) {
    return false;
  }
  std::vector<std::size_t> reverse_first(first_perm.size());
  for (std::size_t i = 0; i < first_perm.size(); ++i) {
    reverse_first[static_cast<std::size_t>(first_perm[i])] = i;
  }
  std::vector<std::size_t> indices;
  for (std::size_t p : reverse_first) {
    const std::vector<std::size_t> &mapped_indices = mapped[p].second;
    indices.insert(indices.end(), mapped_indices.begin(), mapped_indices.end());
  }
  for (std::size_t index : indices) {
    plan.shape.push_back(new_shape[index]);
  }
  std::vector<std::size_t> reverse_indices(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    reverse_indices[indices[i]] = i;
  }
  for (std::size_t index : reverse_indices) {
    plan.perm.push_back(static_cast<int64_t>(index));
  }
  return true;
}

} // namespace

std::set<std::string> SqueezeAddPattern::FastOpType() const { return {"Add"}; }

core::builder::MatchResult SqueezeAddPattern::Match(core::builder::GraphGraph &graph,
                                                    const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Add") || !MainOpsetAtLeast(graph, 13) ||
      candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not an opset-13 default-domain Add");
  }
  const NodeProto *first = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *second = graph.NodeBefore(candidate.input()[1].value());
  if (first == nullptr || second == nullptr || first->op_type().value() != "Squeeze" ||
      second->op_type().value() != "Squeeze") {
    return NoMatch(candidate, "both Add inputs are not produced by Squeeze nodes");
  }

  std::vector<int64_t> first_axes;
  std::vector<int64_t> second_axes;
  if (!ResolveSqueezeAxes(graph, *first, first_axes) ||
      !ResolveSqueezeAxes(graph, *second, second_axes) || first_axes != second_axes) {
    return NoMatch(candidate, "the Squeeze axes are not equal");
  }
  return core::builder::MatchResult{this, {first, second, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
SqueezeAddPattern::Apply(core::builder::GraphGraph &graph,
                         const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError("SqueezeAddPattern::Apply expects two Squeeze nodes and one Add node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("SqueezeAddPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &first = *nodes[0];
  const NodeProto &second = *nodes[1];
  const NodeProto &add = *nodes[2];
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "SqueezeAddPattern--" + add.name().value();
  const std::string intermediate =
      builder.UniqueName("SqueezeAddPattern_" + add.output()[0].value());
  std::vector<std::string> squeeze_inputs{intermediate};
  const std::vector<std::string> axes = InputsAfterFirst(first);
  squeeze_inputs.insert(squeeze_inputs.end(), axes.begin(), axes.end());

  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(add.input()[0].value())) {
    replacements.add() = first;
  }
  if (graph.IsUsedMoreThanOnce(add.input()[1].value())) {
    replacements.add() = second;
  }
  replacements.push_back(MakeNode("Add", {first.input()[0].value(), second.input()[0].value()},
                                  {intermediate}, "", name.c_str()));
  replacements.push_back(MakeNode("Squeeze", squeeze_inputs, {add.output()[0].value()}, "",
                                  ("SqueezeAddPattern--" + first.name().value()).c_str()));
  return replacements;
}

std::set<std::string> MulUnsqueezeUnsqueezePattern::FastOpType() const { return {"Mul"}; }

core::builder::MatchResult MulUnsqueezeUnsqueezePattern::Match(core::builder::GraphGraph &graph,
                                                               const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Mul") || !MainOpsetAtLeast(graph, 13) ||
      candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not an opset-13 default-domain Mul");
  }
  const NodeProto *first = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *second = graph.NodeBefore(candidate.input()[1].value());
  if (first == nullptr || second == nullptr || first->op_type().value() != "Unsqueeze" ||
      second->op_type().value() != "Unsqueeze" || first->input_size() < 2 ||
      second->input_size() < 2) {
    return NoMatch(candidate, "both Mul inputs are not Unsqueeze nodes with axes");
  }
  std::vector<int64_t> first_axes;
  std::vector<int64_t> second_axes;
  if (!ReadConstantIntegers(graph, first->input()[1].value(), first_axes) ||
      !ReadConstantIntegers(graph, second->input()[1].value(), second_axes) ||
      first_axes != second_axes) {
    return NoMatch(candidate, "the Unsqueeze axes are not equal constants");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value()) ||
      graph.IsUsedMoreThanOnce(candidate.input()[1].value())) {
    return NoMatch(candidate, "an Unsqueeze output is shared");
  }
  return core::builder::MatchResult{this, {first, second, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
MulUnsqueezeUnsqueezePattern::Apply(core::builder::GraphGraph &graph,
                                    const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError(
        "MulUnsqueezeUnsqueezePattern::Apply expects two Unsqueeze nodes and one Mul node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("MulUnsqueezeUnsqueezePattern::Apply received an unsafe match.");
  }
  const NodeProto &first = *nodes[0];
  const NodeProto &second = *nodes[1];
  const NodeProto &mul = *nodes[2];
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string intermediate =
      builder.UniqueName("MulUnsqueezeUnsqueezePattern_" + mul.output()[0].value());
  std::vector<std::string> unsqueeze_inputs{intermediate};
  const std::vector<std::string> axes = InputsAfterFirst(first);
  unsqueeze_inputs.insert(unsqueeze_inputs.end(), axes.begin(), axes.end());

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("Mul", {first.input()[0].value(), second.input()[0].value()},
                                  {intermediate}, "",
                                  ("MulUnsqueezeUnsqueezePattern--" + mul.name().value()).c_str()));
  replacements.push_back(
      MakeNode("Unsqueeze", unsqueeze_inputs, {mul.output()[0].value()}, "",
               ("MulUnsqueezeUnsqueezePattern--" + first.name().value()).c_str()));
  return replacements;
}

std::set<std::string> SqueezeBinaryUnsqueezePattern::FastOpType() const { return {"Unsqueeze"}; }

core::builder::MatchResult SqueezeBinaryUnsqueezePattern::Match(core::builder::GraphGraph &graph,
                                                                const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Unsqueeze") || !MainOpsetAtLeast(graph, 13) ||
      candidate.input_size() != 2 || candidate.output_size() != 1 ||
      !graph.IsConstantScalar(candidate.input()[1].value(), 0.0, false)) {
    return NoMatch(candidate, "candidate is not an Unsqueeze on constant axis zero");
  }
  const NodeProto *binary = graph.NodeBefore(candidate.input()[0].value());
  if (binary == nullptr || !IsBinaryOp(*binary) || binary->input_size() != 2 ||
      binary->output_size() != 1 || graph.IsUsedMoreThanOnce(binary->output()[0].value()) ||
      graph.IsUsedMoreThanOnce(binary->input()[0].value()) ||
      !graph.HasShape(binary->input()[1].value()) ||
      graph.GetShape(binary->input()[1].value()).Shape().Rank() != 0) {
    return NoMatch(candidate, "the preceding binary operation does not have a scalar right input");
  }
  const NodeProto *squeeze = graph.NodeBefore(binary->input()[0].value());
  if (squeeze == nullptr || squeeze->input_size() != 1 ||
      !graph.HasShape(squeeze->input()[0].value()) ||
      graph.GetShape(squeeze->input()[0].value()).Shape().Rank() != 1) {
    return NoMatch(candidate, "the binary left input is not produced from a rank-one value");
  }
  return core::builder::MatchResult{this, {squeeze, binary, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
SqueezeBinaryUnsqueezePattern::Apply(core::builder::GraphGraph &graph,
                                     const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError(
        "SqueezeBinaryUnsqueezePattern::Apply expects a Squeeze, binary node and Unsqueeze.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("SqueezeBinaryUnsqueezePattern::Apply received an unsafe match.");
  }
  const NodeProto &squeeze = *nodes[0];
  const NodeProto &binary = *nodes[1];
  const NodeProto &unsqueeze = *nodes[2];
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string intermediate =
      builder.UniqueName("SqueezeBinaryUnsqueezePattern_" + binary.input()[1].value());
  const std::string unsqueeze_name = "SqueezeBinaryUnsqueezePattern--" + unsqueeze.name().value();
  const std::string binary_name = "SqueezeBinaryUnsqueezePattern--" + binary.name().value();

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("Unsqueeze",
                                  {binary.input()[1].value(), unsqueeze.input()[1].value()},
                                  {intermediate}, "", unsqueeze_name.c_str()));
  replacements.push_back(MakeNode(binary.op_type().value().c_str(),
                                  {squeeze.input()[0].value(), intermediate},
                                  {unsqueeze.output()[0].value()}, "", binary_name.c_str()));
  return replacements;
}

std::set<std::string> SwapUnsqueezeTransposePattern::FastOpType() const { return {"Transpose"}; }

core::builder::MatchResult SwapUnsqueezeTransposePattern::Match(core::builder::GraphGraph &graph,
                                                                const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Transpose") || candidate.input_size() != 1 ||
      candidate.output_size() != 1 || graph.IsUsedMoreThanOnce(candidate.input()[0].value())) {
    return NoMatch(candidate, "candidate is not an unshared default-domain Transpose");
  }
  const NodeProto *unsqueeze = graph.NodeBefore(candidate.input()[0].value());
  if (unsqueeze == nullptr || !IsDefaultOp(*unsqueeze, "Unsqueeze") ||
      unsqueeze->input_size() != 2 || unsqueeze->output_size() != 1 ||
      !graph.IsConstant(unsqueeze->input()[1].value()) ||
      graph.GetComputedConstant(unsqueeze->input()[1].value()) == nullptr) {
    return NoMatch(candidate,
                   "the Transpose input is not produced by an Unsqueeze with constant axes");
  }
  std::vector<int64_t> perm;
  std::vector<int64_t> axes;
  if (!GetAttributeInts(candidate, "perm", perm) || !IsValidPermutation(perm) ||
      !ReadConstantIntegers(graph, unsqueeze->input()[1].value(), axes) || axes.empty()) {
    return NoMatch(candidate, "the Transpose perm or Unsqueeze axes are invalid");
  }
  const bool has_negative =
      std::any_of(axes.begin(), axes.end(), [](int64_t axis) { return axis < 0; });
  for (int64_t axis : axes) {
    const int64_t normalized =
        has_negative
            ? ((axis + static_cast<int64_t>(perm.size())) % static_cast<int64_t>(perm.size()) +
               static_cast<int64_t>(perm.size())) %
                  static_cast<int64_t>(perm.size())
            : axis;
    if (normalized < 0 || normalized >= static_cast<int64_t>(perm.size())) {
      return NoMatch(candidate, "an Unsqueeze axis is out of range for the Transpose");
    }
  }
  return core::builder::MatchResult{this, {unsqueeze, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
SwapUnsqueezeTransposePattern::Apply(core::builder::GraphGraph &graph,
                                     const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("SwapUnsqueezeTransposePattern::Apply expects an Unsqueeze and Transpose.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("SwapUnsqueezeTransposePattern::Apply received an unsafe match.");
  }
  const NodeProto &unsqueeze = *nodes[0];
  const NodeProto &transpose = *nodes[1];
  std::vector<int64_t> axes;
  std::vector<int64_t> perm;
  ReadConstantIntegers(graph, unsqueeze.input()[1].value(), axes);
  GetAttributeInts(transpose, "perm", perm);

  if (std::any_of(axes.begin(), axes.end(), [](int64_t axis) { return axis < 0; })) {
    const int64_t rank = static_cast<int64_t>(perm.size());
    for (int64_t &axis : axes) {
      axis = ((axis + rank) % rank + rank) % rank;
    }
  }
  const std::set<int64_t> axes_set(axes.begin(), axes.end());
  std::vector<int64_t> filtered_perm;
  for (int64_t p : perm) {
    if (axes_set.find(p) == axes_set.end()) {
      filtered_perm.push_back(p);
    }
  }
  std::vector<std::pair<int64_t, int64_t>> inverse_perm;
  for (std::size_t i = 0; i < filtered_perm.size(); ++i) {
    inverse_perm.push_back({filtered_perm[i], static_cast<int64_t>(i)});
  }
  std::sort(inverse_perm.begin(), inverse_perm.end());
  std::vector<std::pair<int64_t, int64_t>> new_perm_pairs;
  for (std::size_t i = 0; i < inverse_perm.size(); ++i) {
    new_perm_pairs.push_back({inverse_perm[i].second, static_cast<int64_t>(i)});
  }
  std::sort(new_perm_pairs.begin(), new_perm_pairs.end());
  std::vector<int64_t> new_perm;
  for (const auto &[_, value] : new_perm_pairs) {
    new_perm.push_back(value);
  }

  std::vector<int64_t> new_axes;
  new_axes.reserve(axes.size());
  for (int64_t axis : axes) {
    new_axes.push_back(perm[static_cast<std::size_t>(axis)]);
  }
  std::sort(new_axes.begin(), new_axes.end());

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string intermediate =
      builder.UniqueName("SwapUnsqueezeTransposePattern_" + transpose.output()[0].value());
  const std::string axes_name =
      MakeShapeInitializer(builder, "SwapUnsqueezeTransposePattern_axes", new_axes);
  const std::string transpose_name = "SwapUnsqueezeTransposePattern--" + transpose.name().value();
  const std::string unsqueeze_name = "SwapUnsqueezeTransposePattern--" + unsqueeze.name().value();

  NodeProto new_transpose = MakeNode("Transpose", {unsqueeze.input()[0].value()}, {intermediate},
                                     "", transpose_name.c_str());
  AddAttribute(new_transpose, "perm", new_perm);
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(new_transpose));
  replacements.push_back(MakeNode("Unsqueeze", {intermediate, axes_name},
                                  {transpose.output()[0].value()}, "", unsqueeze_name.c_str()));
  return replacements;
}

std::set<std::string> TransposeEqualReshapePattern::FastOpType() const { return {"Transpose"}; }

core::builder::MatchResult TransposeEqualReshapePattern::Match(core::builder::GraphGraph &graph,
                                                               const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Transpose") || candidate.input_size() != 1 ||
      candidate.output_size() != 1 || !graph.HasShape(candidate.input()[0].value())) {
    return NoMatch(candidate, "candidate is not a Transpose with a known input shape");
  }
  std::vector<int64_t> perm;
  if (!GetAttributeInts(candidate, "perm", perm) || !IsValidPermutation(perm)) {
    return NoMatch(candidate, "the Transpose has no valid perm attribute");
  }
  const core::symbolic::SymShape &shape = graph.GetShape(candidate.input()[0].value()).Shape();
  if (shape.Rank() != perm.size()) {
    return NoMatch(candidate, "the Transpose perm does not match the input rank");
  }

  std::size_t first = 0;
  while (first < perm.size() && perm[first] == static_cast<int64_t>(first)) {
    ++first;
  }
  std::size_t last = perm.size();
  while (last > first && perm[last - 1] == static_cast<int64_t>(last - 1)) {
    --last;
  }
  int non_one = 0;
  for (std::size_t i = first; i < last; ++i) {
    if (!shape[i].IsInt() || shape[i].AsInt() != 1) {
      ++non_one;
    }
  }
  if (non_one > 1) {
    return NoMatch(candidate, "the Transpose moves more than one non-size-one dimension");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
TransposeEqualReshapePattern::Apply(core::builder::GraphGraph &graph,
                                    const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("TransposeEqualReshapePattern::Apply expects one Transpose node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("TransposeEqualReshapePattern::Apply received an unsafe match.");
  }
  const NodeProto &transpose = *nodes[0];
  std::vector<int64_t> perm;
  GetAttributeInts(transpose, "perm", perm);
  const core::symbolic::SymShape &input_shape =
      graph.GetShape(transpose.input()[0].value()).Shape();
  std::vector<int64_t> new_shape;
  new_shape.reserve(perm.size());
  for (std::size_t i = 0; i < perm.size(); ++i) {
    const int64_t p = perm[i];
    const core::symbolic::SymDim &dim = input_shape[static_cast<std::size_t>(p)];
    if (i == static_cast<std::size_t>(p)) {
      new_shape.push_back(0);
    } else if (dim.IsInt() && dim.AsInt() == 1) {
      new_shape.push_back(1);
    } else if (dim.IsInt()) {
      new_shape.push_back(dim.AsInt());
    } else {
      new_shape.push_back(-1);
    }
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string shape_name =
      MakeShapeInitializer(builder, "TransposeEqualReshapePattern_shape", new_shape);
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode(
      "Reshape", {transpose.input()[0].value(), shape_name}, {transpose.output()[0].value()}, "",
      ("TransposeEqualReshapePattern--B--" + transpose.name().value()).c_str()));
  return replacements;
}

std::set<std::string> TransposeReshapeTransposePattern::FastOpType() const { return {"Transpose"}; }

core::builder::MatchResult
TransposeReshapeTransposePattern::Match(core::builder::GraphGraph &graph,
                                        const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Transpose") || candidate.input_size() != 1 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Transpose");
  }
  const std::vector<const NodeProto *> &first_next = graph.NextNodes(candidate.output()[0].value());
  if (first_next.size() != 1 || !IsDefaultOp(*first_next[0], "Reshape") ||
      first_next[0]->input_size() != 2 || first_next[0]->output_size() != 1 ||
      !graph.IsConstant(first_next[0]->input()[1].value())) {
    return NoMatch(candidate, "the Transpose is not followed by a constant-shape Reshape");
  }
  const NodeProto *reshape = first_next[0];
  const std::vector<const NodeProto *> &second_next = graph.NextNodes(reshape->output()[0].value());
  if (second_next.size() != 1 || !IsDefaultOp(*second_next[0], "Transpose") ||
      second_next[0]->input_size() != 1 || second_next[0]->output_size() != 1) {
    return NoMatch(candidate, "the Reshape is not followed by a Transpose");
  }
  ReshapeTransposePlan plan;
  if (!MakeReshapeTransposePlan(graph, candidate, *reshape, *second_next[0], plan)) {
    return NoMatch(candidate, "the Reshape shape cannot be aligned with the Transposes");
  }
  return core::builder::MatchResult{this, {&candidate, reshape, second_next[0]}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
TransposeReshapeTransposePattern::Apply(core::builder::GraphGraph &graph,
                                        const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError(
        "TransposeReshapeTransposePattern::Apply expects Transpose, Reshape, Transpose.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("TransposeReshapeTransposePattern::Apply received an unsafe match.");
  }
  const NodeProto &first = *nodes[0];
  const NodeProto &reshape = *nodes[1];
  const NodeProto &second = *nodes[2];
  ReshapeTransposePlan plan;
  if (!MakeReshapeTransposePlan(graph, first, reshape, second, plan)) {
    throw BuilderError("TransposeReshapeTransposePattern::Apply lost its reshape plan.");
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string intermediate =
      builder.UniqueName("TransposeReshapeTransposePattern_" + first.output()[0].value());
  const std::string shape_name =
      MakeShapeInitializer(builder, "TransposeReshapeTransposePattern_shape", plan.shape);
  utils::RepeatedProtoField<NodeProto> replacements;
  if (plan.after) {
    replacements.add() = first;
    NodeProto new_transpose =
        MakeNode("Transpose", {first.output()[0].value()}, {intermediate}, "",
                 ("TransposeReshapeTransposePattern--C--" + second.name().value()).c_str());
    AddAttribute(new_transpose, "perm", plan.perm);
    replacements.push_back(std::move(new_transpose));
    replacements.push_back(
        MakeNode("Reshape", {intermediate, shape_name}, {second.output()[0].value()}, "",
                 ("TransposeReshapeTransposePattern--D--" + reshape.name().value()).c_str()));
    return replacements;
  }

  replacements.push_back(
      MakeNode("Reshape", {first.input()[0].value(), shape_name}, {intermediate}, "",
               ("TransposeReshapeTransposePattern--A--" + reshape.name().value()).c_str()));
  NodeProto new_transpose =
      MakeNode("Transpose", {intermediate}, {second.input()[0].value()}, "",
               ("TransposeReshapeTransposePattern--B--" + first.name().value()).c_str());
  AddAttribute(new_transpose, "perm", plan.perm);
  replacements.push_back(std::move(new_transpose));
  replacements.add() = second;
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
