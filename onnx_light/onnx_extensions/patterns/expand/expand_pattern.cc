// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/expand/expand_pattern.h"

#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/shapes/shape_broadcast.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;

bool IsDefaultOp(const NodeProto &node, const char *op_type) {
  return node.op_type().value() == op_type &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain;
}

// Returns ``true`` unless the builder records a default-domain opset strictly
// below ``minimum``. An unrecorded opset is treated as acceptable, matching the
// modern (axes-as-input) form these patterns already require.
bool MainOpsetAtLeast(core::builder::GraphGraph &graph, int minimum) {
  const int opset = graph.Builder().OpsetVersion("");
  return opset == core::shapes::kUnknownOpsetVersion || opset >= minimum;
}

// Returns a fresh initializer name derived from ``base`` that is not already
// used by ``builder`` (without reserving it, so ``MakeInitializer`` can).
std::string FreeInitializerName(core::builder::GraphBuilder &builder, const std::string &base) {
  if (!builder.HasName(base)) {
    return base;
  }
  for (int suffix = 0;; ++suffix) {
    const std::string candidate = base + "_" + std::to_string(suffix);
    if (!builder.HasName(candidate)) {
      return candidate;
    }
  }
}

// Inserts a ``1`` at every position of ``axes`` (normalised against the output
// rank and sorted ascending) into ``base``, producing the shape after an
// ``Unsqueeze``.
std::vector<int64_t> InsertUnsqueezeOnes(const std::vector<int64_t> &base,
                                         const std::vector<int64_t> &axes) {
  const std::size_t rank_out = base.size() + axes.size();
  std::vector<int64_t> sorted_axes;
  sorted_axes.reserve(axes.size());
  for (int64_t axis : axes) {
    int64_t normalized = axis < 0 ? axis + static_cast<int64_t>(rank_out) : axis;
    // Clamp out-of-range axes so the insertion index below is always valid.
    normalized =
        std::max<int64_t>(0, std::min<int64_t>(normalized, static_cast<int64_t>(rank_out)));
    sorted_axes.push_back(normalized);
  }
  std::sort(sorted_axes.begin(), sorted_axes.end());

  std::vector<int64_t> result = base;
  for (int64_t axis : sorted_axes) {
    const std::size_t position = std::min(static_cast<std::size_t>(axis), result.size());
    result.insert(result.begin() + static_cast<std::ptrdiff_t>(position), 1);
  }
  return result;
}

// Element-wise binary and comparison operators that broadcast their inputs.
const std::set<std::string> &BinaryOpTypes() {
  static const std::set<std::string> types = {"Add",
                                              "And",
                                              "Div",
                                              "Mul",
                                              "Mod",
                                              "Or",
                                              "Sub",
                                              "Xor",
                                              "Equal",
                                              "Greater",
                                              "GreaterOrEqual",
                                              "Less",
                                              "LessOrEqual"};
  return types;
}

// Unary-like operators that preserve the shape of their first input.
const std::set<std::string> &UnaryLikeOpTypes() {
  static const std::set<std::string> types = {"Abs",
                                              "Acos",
                                              "Acosh",
                                              "Asin",
                                              "Asinh",
                                              "Atan",
                                              "Atanh",
                                              "BitShift",
                                              "Cast",
                                              "CastLike",
                                              "Ceil",
                                              "Celu",
                                              "Clip",
                                              "Cos",
                                              "Cosh",
                                              "DequantizeLinear",
                                              "DynamicQuantizeLinear",
                                              "Elu",
                                              "Erf",
                                              "Exp",
                                              "IsInf",
                                              "Log",
                                              "LogSoftmax",
                                              "Neg",
                                              "Not",
                                              "PRelu",
                                              "QuantizeLinear",
                                              "Reciprocal",
                                              "Relu",
                                              "Round",
                                              "Selu",
                                              "Sigmoid",
                                              "Sign",
                                              "Sin",
                                              "Sinh",
                                              "Softmax",
                                              "SoftmaxCrossEntropyLoss",
                                              "Softplus",
                                              "Softsign",
                                              "Sqrt",
                                              "Tan",
                                              "Tanh",
                                              "ThresholdRelu"};
  return types;
}

// Operators handled by ExpandSwapPattern regardless of their domain.
const std::set<std::string> &SwapOtherOpTypes() {
  static const std::set<std::string> types = {"NegXplus1", "ReplaceZero", "Pow"};
  return types;
}

// Reads the fully-static shape of ``name`` into ``dims``, returning ``false``
// when the shape is unknown or contains a symbolic dimension.
bool ReadStaticShape(core::builder::GraphGraph &graph, const std::string &name,
                     std::vector<int64_t> &dims) {
  if (!graph.HasShape(name)) {
    return false;
  }
  const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
  dims.clear();
  for (std::size_t i = 0; i < shape.Rank(); ++i) {
    if (!shape[i].IsInt()) {
      return false;
    }
    dims.push_back(shape[i].AsInt());
  }
  return true;
}

// Reads the constant integer tensor ``name`` into ``values``, returning
// ``false`` when it is not a constant INT32/INT64 tensor.
bool ReadConstantShape(core::builder::GraphGraph &graph, const std::string &name,
                       std::vector<int64_t> &values) {
  if (!graph.IsConstant(name)) {
    return false;
  }
  const TensorProto *tensor = graph.GetComputedConstant(name);
  if (tensor == nullptr) {
    return false;
  }
  const auto dtype = static_cast<TensorProto::DataType>(tensor->data_type());
  if (dtype != TensorProto::DataType::INT32 && dtype != TensorProto::DataType::INT64) {
    return false;
  }
  return ReadIntegerValues(*tensor, values);
}

bool IsExpand(const NodeProto *node) {
  return node != nullptr && IsDefaultOp(*node, "Expand") && node->input_size() == 2 &&
         node->output_size() == 1;
}

const core::symbolic::SymDim &AlignedDim(const core::symbolic::SymShape &shape,
                                         std::size_t output_rank, std::size_t index,
                                         const core::symbolic::SymDim &one) {
  const std::size_t offset = output_rank - shape.Rank();
  return index < offset ? one : shape[index - offset];
}

std::optional<core::symbolic::SymDim> StrictBroadcastDim(const core::symbolic::SymDim &left,
                                                         const core::symbolic::SymDim &right) {
  if (left.IsInt() && right.IsInt()) {
    if (left.AsInt() == right.AsInt()) {
      return left;
    }
    if (left.AsInt() == 1) {
      return right;
    }
    if (right.AsInt() == 1) {
      return left;
    }
    return std::nullopt;
  }
  if (left.IsInt() && left.AsInt() == 1) {
    return right;
  }
  if (right.IsInt() && right.AsInt() == 1) {
    return left;
  }
  if (left == right) {
    return left;
  }
  if (left.IsInt()) {
    return left;
  }
  if (right.IsInt()) {
    return right;
  }
  return std::nullopt;
}

bool BroadcastsTo(const core::symbolic::SymShape &left, const core::symbolic::SymShape &right,
                  const core::symbolic::SymShape &output) {
  const std::size_t rank = std::max(left.Rank(), right.Rank());
  if (rank != output.Rank()) {
    return false;
  }
  const core::symbolic::SymDim one(1);
  for (std::size_t i = 0; i < rank; ++i) {
    const core::symbolic::SymDim &l = AlignedDim(left, rank, i, one);
    const core::symbolic::SymDim &r = AlignedDim(right, rank, i, one);
    const core::symbolic::SymDim &out = output[i];
    if (l.IsInt() && l.AsInt() == 1) {
      if (r != out) {
        return false;
      }
    } else if (r.IsInt() && r.AsInt() == 1) {
      if (l != out) {
        return false;
      }
    } else if (l != r || l != out) {
      return false;
    }
  }
  return true;
}

core::symbolic::SymShape ShapePrefix(const core::symbolic::SymShape &shape,
                                     std::size_t trailing_dimensions) {
  core::symbolic::SymShape prefix;
  for (std::size_t i = 0; i + trailing_dimensions < shape.Rank(); ++i) {
    prefix.PushBack(shape[i]);
  }
  return prefix;
}

const NodeProto *InputExpand(core::builder::GraphGraph &graph, const NodeProto &node,
                             int input_index) {
  const NodeProto *expand = graph.NodeBefore(node.input()[input_index].value());
  return IsExpand(expand) ? expand : nullptr;
}

core::builder::MatchResult
MatchShapeBasedBroadcast(const core::builder::PatternOptimization *pattern,
                         core::builder::GraphGraph &graph, const NodeProto &candidate,
                         bool matmul) {
  if (candidate.input_size() != 2 || candidate.output_size() != 1 ||
      NormaliseDomain(candidate.domain().value()) != kDefaultOnnxDomain) {
    return {};
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.input()[1].value()) ||
      !graph.HasShape(candidate.output()[0].value())) {
    return {};
  }

  const NodeProto *left_expand = InputExpand(graph, candidate, 0);
  const NodeProto *right_expand = InputExpand(graph, candidate, 1);
  if (left_expand == nullptr && right_expand == nullptr) {
    return {};
  }

  const std::string &left_name =
      left_expand == nullptr ? candidate.input()[0].value() : left_expand->input()[0].value();
  const std::string &right_name =
      right_expand == nullptr ? candidate.input()[1].value() : right_expand->input()[0].value();
  if (!graph.HasShape(left_name) || !graph.HasShape(right_name)) {
    return {};
  }

  const core::symbolic::SymShape &left = graph.GetShape(left_name).Shape();
  const core::symbolic::SymShape &right = graph.GetShape(right_name).Shape();
  const core::symbolic::SymShape &output = graph.GetShape(candidate.output()[0].value()).Shape();
  bool compatible = false;
  if (matmul) {
    compatible = left.Rank() >= 2 && right.Rank() >= 2 && output.Rank() >= 2 &&
                 BroadcastsTo(ShapePrefix(left, 2), ShapePrefix(right, 2), ShapePrefix(output, 2));
  } else {
    compatible = BroadcastsTo(left, right, output);
  }
  if (!compatible) {
    return {};
  }

  return core::builder::MatchResult{pattern, {left_expand, right_expand, &candidate}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
ApplyShapeBasedBroadcast(const char *pattern_name, core::builder::GraphGraph &graph,
                         const std::vector<const NodeProto *> &nodes) {
  if (nodes.size() != 3 || nodes[2] == nullptr) {
    throw BuilderError(std::string(pattern_name) +
                       "::Apply expects left Expand, right Expand, and binary slots.");
  }
  const NodeProto *left_expand = nodes[0];
  const NodeProto *right_expand = nodes[1];
  const NodeProto &binary = *nodes[2];
  const std::vector<std::string> inputs = {
      left_expand == nullptr ? binary.input()[0].value() : left_expand->input()[0].value(),
      right_expand == nullptr ? binary.input()[1].value() : right_expand->input()[0].value()};

  utils::RepeatedProtoField<NodeProto> replacements;
  if (left_expand != nullptr && graph.IsUsedMoreThanOnce(left_expand->output()[0].value())) {
    replacements.push_back(*left_expand);
  }
  if (right_expand != nullptr && right_expand != left_expand &&
      graph.IsUsedMoreThanOnce(right_expand->output()[0].value())) {
    replacements.push_back(*right_expand);
  }
  replacements.push_back(
      MakeNode(binary.op_type().value().c_str(), inputs, {binary.output()[0].value()}, "",
               (std::string(pattern_name) + "--" + binary.name().value()).c_str()));
  return replacements;
}

std::optional<core::symbolic::SymShape>
StrictBroadcastShape(const core::symbolic::SymShape &left, const core::symbolic::SymShape &right) {
  const std::size_t left_rank = left.Rank();
  const std::size_t right_rank = right.Rank();
  const std::size_t rank = std::max(left_rank, right_rank);
  std::vector<core::symbolic::SymDim> dims;
  dims.reserve(rank);
  for (std::size_t i = 0; i < rank; ++i) {
    const bool has_left = i + left_rank >= rank;
    const bool has_right = i + right_rank >= rank;
    if (!has_left) {
      dims.push_back(right[i - (rank - right_rank)]);
      continue;
    }
    if (!has_right) {
      dims.push_back(left[i - (rank - left_rank)]);
      continue;
    }
    const std::optional<core::symbolic::SymDim> dim =
        StrictBroadcastDim(left[i - (rank - left_rank)], right[i - (rank - right_rank)]);
    if (!dim.has_value()) {
      return std::nullopt;
    }
    dims.push_back(*dim);
  }
  return core::symbolic::SymShape(dims);
}

bool StaticExpandTarget(const core::symbolic::SymShape &input,
                        const core::symbolic::SymShape &output, std::vector<int64_t> &target) {
  if (input.Rank() != output.Rank()) {
    return false;
  }
  target.clear();
  target.reserve(output.Rank());
  for (std::size_t i = 0; i < output.Rank(); ++i) {
    if (input[i] == output[i]) {
      target.push_back(1);
    } else if (input[i].IsInt() && output[i].IsInt()) {
      target.push_back(output[i].AsInt());
    } else {
      return false;
    }
  }
  return true;
}

bool CanSwapOneExpand(const core::symbolic::SymShape &before,
                      const core::symbolic::SymShape &expanded,
                      const core::symbolic::SymShape &other,
                      const core::symbolic::SymShape &output) {
  if (before == expanded || expanded == other || output != expanded) {
    return false;
  }
  return StrictBroadcastShape(before, other).has_value();
}

bool CanSwapTwoExpands(const core::symbolic::SymShape &left,
                       const core::symbolic::SymShape &left_expanded,
                       const core::symbolic::SymShape &right,
                       const core::symbolic::SymShape &right_expanded,
                       const core::symbolic::SymShape &output) {
  if (left_expanded != right_expanded || left_expanded != output || left.Rank() != right.Rank() ||
      left == left_expanded || left_expanded == right) {
    return false;
  }
  const std::optional<core::symbolic::SymShape> broadcast = StrictBroadcastShape(left, right);
  return broadcast.has_value() && *broadcast != output;
}

bool CompatibleCastWhereShapes(const core::symbolic::SymShape &condition,
                               const core::symbolic::SymShape &other,
                               const core::symbolic::SymShape &output,
                               const core::symbolic::SymShape &before) {
  if (condition != output) {
    return false;
  }
  const std::optional<core::symbolic::SymShape> broadcast = StrictBroadcastShape(before, other);
  if (!broadcast.has_value() || broadcast->Rank() != output.Rank()) {
    return false;
  }
  const core::symbolic::SymDim one(1);
  for (std::size_t i = 0; i < output.Rank(); ++i) {
    const core::symbolic::SymDim &before_dim = AlignedDim(before, output.Rank(), i, one);
    if ((*broadcast)[i] != output[i] && (*broadcast)[i] != before_dim) {
      return false;
    }
  }
  return true;
}

} // namespace

std::set<std::string> ExpandPattern::FastOpType() const { return {"Expand"}; }

core::builder::MatchResult ExpandPattern::Match(core::builder::GraphGraph &graph,
                                                const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Expand") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Expand with two inputs");
  }
  std::vector<int64_t> shape;
  if (!ReadStaticShape(graph, candidate.input()[0].value(), shape)) {
    return NoMatch(candidate, "the Expand input has no fully-static shape");
  }
  std::vector<int64_t> new_shape;
  if (!ReadConstantShape(graph, candidate.input()[1].value(), new_shape)) {
    return NoMatch(candidate, "the Expand target shape is not a constant integer tensor");
  }
  if (shape != new_shape) {
    return NoMatch(candidate, "the Expand changes the input shape");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ExpandPattern::Apply(core::builder::GraphGraph &graph,
                     const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("ExpandPattern::Apply expects one Expand node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ExpandPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &expand = *nodes[0];
  const std::string name = "ExpandPattern--" + expand.name().value();

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("Identity", {expand.input()[0].value()},
                                  {expand.output()[0].value()}, "", name.c_str()));
  return replacements;
}

std::set<std::string> ExpandBroadcastPattern::FastOpType() const { return {"Expand"}; }

core::builder::MatchResult ExpandBroadcastPattern::Match(core::builder::GraphGraph &graph,
                                                         const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Expand") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Expand with two inputs");
  }
  std::vector<int64_t> shape;
  if (!ReadStaticShape(graph, candidate.input()[0].value(), shape)) {
    return NoMatch(candidate, "the Expand input has no fully-static shape");
  }
  std::vector<int64_t> new_shape;
  if (!ReadConstantShape(graph, candidate.input()[1].value(), new_shape)) {
    return NoMatch(candidate, "the Expand target shape is not a constant integer tensor");
  }
  if (graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
    return NoMatch(candidate, "the Expand output is shared");
  }

  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.output()[0].value());
  if (next_nodes.size() != 1 || next_nodes[0] == nullptr) {
    return NoMatch(candidate, "the Expand output is not consumed by exactly one node");
  }
  const NodeProto &next_node = *next_nodes[0];
  if (NormaliseDomain(next_node.domain().value()) != kDefaultOnnxDomain ||
      BinaryOpTypes().find(next_node.op_type().value()) == BinaryOpTypes().end() ||
      next_node.input_size() != 2) {
    return NoMatch(candidate, "the consumer is not a default-domain element-wise binary operator");
  }

  const std::string &other = next_node.input()[0].value() == candidate.output()[0].value()
                                 ? next_node.input()[1].value()
                                 : next_node.input()[0].value();
  std::vector<int64_t> other_shape;
  if (!ReadStaticShape(graph, other, other_shape)) {
    return NoMatch(candidate, "the other operand has no fully-static shape");
  }
  if (new_shape != other_shape) {
    return NoMatch(candidate, "Expand does not target the shape of the other operand");
  }
  if (shape.size() != other_shape.size()) {
    return NoMatch(candidate, "the operands have different ranks");
  }
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != other_shape[i] && shape[i] != 1 && other_shape[i] != 1) {
      return NoMatch(candidate, "the operands do not broadcast together");
    }
  }

  return core::builder::MatchResult{this, {&candidate, &next_node}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ExpandBroadcastPattern::Apply(core::builder::GraphGraph &graph,
                              const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ExpandBroadcastPattern::Apply expects one Expand and one binary node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ExpandBroadcastPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &expand = *nodes[0];
  const NodeProto &next_node = *nodes[1];

  std::vector<std::string> inputs;
  if (next_node.input()[0].value() == expand.output()[0].value()) {
    inputs = {expand.input()[0].value(), next_node.input()[1].value()};
  } else {
    inputs = {next_node.input()[0].value(), expand.input()[0].value()};
  }
  const std::string name = "ExpandBroadcastPattern--" + expand.name().value();

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode(next_node.op_type().value().c_str(), inputs,
                                  {next_node.output()[0].value()}, "", name.c_str()));
  return replacements;
}

std::set<std::string> ShapeBasedConcatExpandPattern::FastOpType() const { return {"Expand"}; }

core::builder::MatchResult ShapeBasedConcatExpandPattern::Match(core::builder::GraphGraph &graph,
                                                                const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Expand") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Expand with two inputs");
  }
  const std::string &target = candidate.input()[1].value();
  if (graph.IsConstant(target)) {
    return NoMatch(candidate, "the Expand target shape is already constant");
  }
  if (graph.IsUsedMoreThanOnce(target)) {
    return NoMatch(candidate, "the Expand target shape is shared");
  }

  const NodeProto *concat = graph.NodeBefore(target);
  if (concat == nullptr || !IsDefaultOp(*concat, "Concat") || concat->output_size() != 1 ||
      GetAttributeOr<int64_t>(*concat, "axis", 0) != 0) {
    return NoMatch(candidate, "the target shape is not produced by an axis-0 Concat");
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.output()[0].value())) {
    return NoMatch(candidate, "the Expand input or output shape is unknown");
  }

  const core::symbolic::SymShape &input_shape =
      graph.GetShape(candidate.input()[0].value()).Shape();
  const core::symbolic::SymShape &output_shape =
      graph.GetShape(candidate.output()[0].value()).Shape();
  if (input_shape.Rank() != output_shape.Rank() ||
      output_shape.Rank() != static_cast<std::size_t>(concat->input_size())) {
    return NoMatch(candidate, "the input, output, and Concat target ranks differ");
  }

  std::size_t changed_index = output_shape.Rank();
  for (std::size_t i = 0; i < output_shape.Rank(); ++i) {
    const std::string &name = concat->input()[static_cast<int>(i)].value();
    if (!graph.HasShape(name)) {
      return NoMatch(candidate, "a Concat input has no known shape");
    }
    const core::symbolic::SymShape &part_shape = graph.GetShape(name).Shape();
    if (part_shape.Rank() != 1 || !part_shape[0].IsInt() || part_shape[0].AsInt() != 1) {
      return NoMatch(candidate, "each Concat input must be a one-element vector");
    }
    if (input_shape[i] != output_shape[i]) {
      if (changed_index != output_shape.Rank()) {
        return NoMatch(candidate, "the Expand changes more than one dimension");
      }
      changed_index = i;
    }
  }
  if (changed_index == output_shape.Rank()) {
    return NoMatch(candidate, "the Expand does not change any dimension");
  }

  bool simplifies_target = false;
  for (std::size_t i = 0; i < output_shape.Rank(); ++i) {
    if (i != changed_index &&
        !graph.IsConstantScalar(concat->input()[static_cast<int>(i)].value(), 1.0, false)) {
      simplifies_target = true;
      break;
    }
  }
  if (!simplifies_target) {
    return NoMatch(candidate, "the unchanged target dimensions are already constant ones");
  }

  return core::builder::MatchResult{this, {concat, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ShapeBasedConcatExpandPattern::Apply(core::builder::GraphGraph &graph,
                                     const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError(
        "ShapeBasedConcatExpandPattern::Apply expects one Concat and one Expand node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ShapeBasedConcatExpandPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &concat = *nodes[0];
  const NodeProto &expand = *nodes[1];
  const core::symbolic::SymShape &input_shape = graph.GetShape(expand.input()[0].value()).Shape();
  const core::symbolic::SymShape &output_shape = graph.GetShape(expand.output()[0].value()).Shape();

  std::size_t changed_index = output_shape.Rank();
  for (std::size_t i = 0; i < output_shape.Rank(); ++i) {
    if (input_shape[i] != output_shape[i]) {
      changed_index = i;
      break;
    }
  }
  if (changed_index == output_shape.Rank()) {
    throw BuilderError("ShapeBasedConcatExpandPattern::Apply found no expanded dimension.");
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "ShapeBasedConcatExpandPattern--" + expand.name().value();
  const std::string one = FreeInitializerName(builder, name + "_one");
  builder.MakeInitializer(MakeInitializerShape(one.c_str(), {1}));

  std::vector<std::string> target_inputs(static_cast<std::size_t>(concat.input_size()), one);
  target_inputs[changed_index] = concat.input()[static_cast<int>(changed_index)].value();
  const std::string new_target = builder.UniqueName(name + "_shape");
  NodeProto replacement_concat =
      MakeNode("Concat", target_inputs, {new_target}, "", (name + "--concat").c_str());
  AddAttribute<int64_t>(replacement_concat, "axis", 0);

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement_concat));
  replacements.push_back(MakeNode("Expand", {expand.input()[0].value(), new_target},
                                  {expand.output()[0].value()}, "", (name + "--expand").c_str()));
  return replacements;
}

std::set<std::string> ShapeBasedExpandBroadcastPattern::FastOpType() const {
  return BinaryOpTypes();
}

core::builder::MatchResult
ShapeBasedExpandBroadcastPattern::Match(core::builder::GraphGraph &graph,
                                        const NodeProto &candidate) const {
  if (BinaryOpTypes().find(candidate.op_type().value()) == BinaryOpTypes().end() ||
      candidate.attribute_size() != 0) {
    return NoMatch(candidate, "candidate is not an attribute-free broadcasting binary operator");
  }
  core::builder::MatchResult match = MatchShapeBasedBroadcast(this, graph, candidate, false);
  if (match.pattern == nullptr) {
    return NoMatch(candidate, "the pre-expansion shapes do not broadcast to the output shape");
  }
  return match;
}

utils::RepeatedProtoField<NodeProto>
ShapeBasedExpandBroadcastPattern::Apply(core::builder::GraphGraph &graph,
                                        const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[2] == nullptr) {
    throw BuilderError(
        "ShapeBasedExpandBroadcastPattern::Apply expects two Expand slots and one binary node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ShapeBasedExpandBroadcastPattern::Apply received an unsafe or inconsistent match.");
  }
  return ApplyShapeBasedBroadcast("ShapeBasedExpandBroadcastPattern", graph, nodes);
}

std::set<std::string> ShapeBasedExpandBroadcastMatMulPattern::FastOpType() const {
  return {"MatMul"};
}

core::builder::MatchResult
ShapeBasedExpandBroadcastMatMulPattern::Match(core::builder::GraphGraph &graph,
                                              const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "MatMul") || candidate.attribute_size() != 0) {
    return NoMatch(candidate, "candidate is not an attribute-free default-domain MatMul");
  }
  core::builder::MatchResult match = MatchShapeBasedBroadcast(this, graph, candidate, true);
  if (match.pattern == nullptr) {
    return NoMatch(candidate,
                   "the pre-expansion batch shapes do not broadcast to the MatMul output");
  }
  return match;
}

utils::RepeatedProtoField<NodeProto>
ShapeBasedExpandBroadcastMatMulPattern::Apply(core::builder::GraphGraph &graph,
                                              const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[2] == nullptr) {
    throw BuilderError(
        "ShapeBasedExpandBroadcastMatMulPattern::Apply expects two Expand slots and one MatMul.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ShapeBasedExpandBroadcastMatMulPattern::Apply received an unsafe or inconsistent "
        "match.");
  }
  return ApplyShapeBasedBroadcast("ShapeBasedExpandBroadcastMatMulPattern", graph, nodes);
}

std::set<std::string> ShapeBasedStaticExpandPattern::FastOpType() const { return {"Expand"}; }

core::builder::MatchResult ShapeBasedStaticExpandPattern::Match(core::builder::GraphGraph &graph,
                                                                const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Expand") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Expand with two inputs");
  }
  if (graph.IsConstant(candidate.input()[1].value())) {
    return NoMatch(candidate, "the Expand target shape is already constant");
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.output()[0].value())) {
    return NoMatch(candidate, "the Expand input or output shape is unknown");
  }
  std::vector<int64_t> target;
  if (!StaticExpandTarget(graph.GetShape(candidate.input()[0].value()).Shape(),
                          graph.GetShape(candidate.output()[0].value()).Shape(), target)) {
    return NoMatch(candidate, "the effective Expand target is not fully static");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ShapeBasedStaticExpandPattern::Apply(core::builder::GraphGraph &graph,
                                     const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("ShapeBasedStaticExpandPattern::Apply expects one Expand node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ShapeBasedStaticExpandPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &expand = *nodes[0];
  std::vector<int64_t> target;
  if (!StaticExpandTarget(graph.GetShape(expand.input()[0].value()).Shape(),
                          graph.GetShape(expand.output()[0].value()).Shape(), target)) {
    throw BuilderError("ShapeBasedStaticExpandPattern::Apply could not derive a static target.");
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "ShapeBasedStaticExpandPattern--" + expand.name().value();
  const std::string shape_name = FreeInitializerName(builder, name + "_shape");
  builder.MakeInitializer(MakeInitializerShape(shape_name.c_str(), target));

  utils::RepeatedProtoField<NodeProto> replacements;
  NodeProto replacement = MakeNode("Expand", {expand.input()[0].value(), shape_name},
                                   {expand.output()[0].value()}, "", name.c_str());
  replacement.set_doc_string(expand.doc_string().value());
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> ShapeBasedExpandSwapPattern::FastOpType() const { return BinaryOpTypes(); }

core::builder::MatchResult ShapeBasedExpandSwapPattern::Match(core::builder::GraphGraph &graph,
                                                              const NodeProto &candidate) const {
  if (BinaryOpTypes().find(candidate.op_type().value()) == BinaryOpTypes().end() ||
      NormaliseDomain(candidate.domain().value()) != kDefaultOnnxDomain ||
      candidate.input_size() != 2 || candidate.output_size() != 1 ||
      candidate.attribute_size() != 0) {
    return NoMatch(candidate, "candidate is not an attribute-free broadcasting binary operator");
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.input()[1].value()) ||
      !graph.HasShape(candidate.output()[0].value())) {
    return NoMatch(candidate, "the binary input or output shape is unknown");
  }

  const NodeProto *left_expand = InputExpand(graph, candidate, 0);
  const NodeProto *right_expand = InputExpand(graph, candidate, 1);
  if (left_expand == nullptr && right_expand == nullptr) {
    return NoMatch(candidate, "neither binary input is produced by Expand");
  }

  const core::symbolic::SymShape &left_expanded =
      graph.GetShape(candidate.input()[0].value()).Shape();
  const core::symbolic::SymShape &right_expanded =
      graph.GetShape(candidate.input()[1].value()).Shape();
  const core::symbolic::SymShape &output = graph.GetShape(candidate.output()[0].value()).Shape();
  if (left_expand == nullptr || right_expand == nullptr) {
    const NodeProto &expand = *(left_expand == nullptr ? right_expand : left_expand);
    const std::string &before_name = expand.input()[0].value();
    if (!graph.HasShape(before_name)) {
      return NoMatch(candidate, "the pre-expansion input shape is unknown");
    }
    const core::symbolic::SymShape &before = graph.GetShape(before_name).Shape();
    const core::symbolic::SymShape &expanded =
        left_expand == nullptr ? right_expanded : left_expanded;
    const core::symbolic::SymShape &other = left_expand == nullptr ? left_expanded : right_expanded;
    if (!CanSwapOneExpand(before, expanded, other, output)) {
      return NoMatch(candidate, "the binary operation cannot safely run before Expand");
    }
  } else {
    if (left_expand->input()[1].value() != right_expand->input()[1].value()) {
      return NoMatch(candidate, "the two Expand nodes use different target values");
    }
    if (!graph.HasShape(left_expand->input()[0].value()) ||
        !graph.HasShape(right_expand->input()[0].value())) {
      return NoMatch(candidate, "a pre-expansion input shape is unknown");
    }
    if (!CanSwapTwoExpands(graph.GetShape(left_expand->input()[0].value()).Shape(), left_expanded,
                           graph.GetShape(right_expand->input()[0].value()).Shape(), right_expanded,
                           output)) {
      return NoMatch(candidate, "the two expansions cannot safely move after the binary operator");
    }
  }
  return core::builder::MatchResult{this, {left_expand, right_expand, &candidate}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
ShapeBasedExpandSwapPattern::Apply(core::builder::GraphGraph &graph,
                                   const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[2] == nullptr || (nodes[0] == nullptr && nodes[1] == nullptr)) {
    throw BuilderError(
        "ShapeBasedExpandSwapPattern::Apply expects two Expand slots and one binary node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ShapeBasedExpandSwapPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto *left_expand = nodes[0];
  const NodeProto *right_expand = nodes[1];
  const NodeProto &binary = *nodes[2];
  const NodeProto &target_expand = right_expand == nullptr ? *left_expand : *right_expand;
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "ShapeBasedExpandSwapPattern--" + binary.name().value();
  const std::string binary_output = builder.UniqueName(name + "_binary");

  utils::RepeatedProtoField<NodeProto> replacements;
  if (left_expand != nullptr && graph.IsUsedMoreThanOnce(left_expand->output()[0].value())) {
    replacements.push_back(*left_expand);
  }
  if (right_expand != nullptr && right_expand != left_expand &&
      graph.IsUsedMoreThanOnce(right_expand->output()[0].value())) {
    replacements.push_back(*right_expand);
  }
  replacements.push_back(MakeNode(
      binary.op_type().value().c_str(),
      {left_expand == nullptr ? binary.input()[0].value() : left_expand->input()[0].value(),
       right_expand == nullptr ? binary.input()[1].value() : right_expand->input()[0].value()},
      {binary_output}, "", (name + "--binary").c_str()));
  replacements.push_back(MakeNode("Expand", {binary_output, target_expand.input()[1].value()},
                                  {binary.output()[0].value()}, "", (name + "--expand").c_str()));
  return replacements;
}

std::set<std::string> ShapeBasedExpandCastWhereSwapPattern::FastOpType() const { return {"Where"}; }

core::builder::MatchResult
ShapeBasedExpandCastWhereSwapPattern::Match(core::builder::GraphGraph &graph,
                                            const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Where") || candidate.input_size() != 3 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Where with three inputs");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value())) {
    return NoMatch(candidate, "the Where condition is shared");
  }
  const NodeProto *cast = graph.NodeBefore(candidate.input()[0].value());
  if (cast == nullptr || !IsDefaultOp(*cast, "Cast") || cast->input_size() != 1 ||
      cast->output_size() != 1) {
    return NoMatch(candidate, "the Where condition is not produced by Cast");
  }
  const int branch = candidate.input()[1].value() == cast->input()[0].value()   ? 1
                     : candidate.input()[2].value() == cast->input()[0].value() ? 2
                                                                                : 0;
  if (branch == 0) {
    return NoMatch(candidate, "the Cast input is not a Where value branch");
  }
  const NodeProto *expand = graph.NodeBefore(cast->input()[0].value());
  if (!IsExpand(expand)) {
    return NoMatch(candidate, "the Cast input is not produced by Expand");
  }
  const std::vector<const NodeProto *> &consumers = graph.NextNodes(expand->output()[0].value());
  if (consumers.size() != 2) {
    return NoMatch(candidate, "the Expand output is not used only by Cast and Where");
  }

  const std::string &other = candidate.input()[3 - branch].value();
  if (!graph.HasShape(candidate.input()[0].value()) || !graph.HasShape(other) ||
      !graph.HasShape(candidate.output()[0].value()) ||
      !graph.HasShape(expand->input()[0].value())) {
    return NoMatch(candidate, "a required Cast/Where shape is unknown");
  }
  if (!CompatibleCastWhereShapes(graph.GetShape(candidate.input()[0].value()).Shape(),
                                 graph.GetShape(other).Shape(),
                                 graph.GetShape(candidate.output()[0].value()).Shape(),
                                 graph.GetShape(expand->input()[0].value()).Shape())) {
    return NoMatch(candidate, "the pre-expansion Where does not preserve the output shape");
  }
  return core::builder::MatchResult{this, {expand, cast, &candidate}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
ShapeBasedExpandCastWhereSwapPattern::Apply(core::builder::GraphGraph &graph,
                                            const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError("ShapeBasedExpandCastWhereSwapPattern::Apply expects Expand, Cast, and "
                       "Where nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ShapeBasedExpandCastWhereSwapPattern::Apply received an unsafe or "
                       "inconsistent match.");
  }
  const NodeProto &expand = *nodes[0];
  const NodeProto &cast = *nodes[1];
  const NodeProto &where = *nodes[2];
  const int branch = where.input()[1].value() == expand.output()[0].value() ? 1 : 2;
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "ShapeBasedExpandCastWhereSwapPattern--" + where.name().value();
  const std::string cast_output = builder.UniqueName(name + "_cast");
  const std::string where_output = builder.UniqueName(name + "_where");

  NodeProto replacement_cast =
      MakeNode("Cast", {expand.input()[0].value()}, {cast_output}, "", (name + "--cast").c_str());
  for (const AttributeProto &attribute : cast.attribute()) {
    replacement_cast.mutable_attribute()->push_back(attribute);
  }
  const std::vector<std::string> where_inputs =
      branch == 1 ? std::vector<std::string>{cast_output, expand.input()[0].value(),
                                             where.input()[2].value()}
                  : std::vector<std::string>{cast_output, where.input()[1].value(),
                                             expand.input()[0].value()};

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement_cast));
  replacements.push_back(
      MakeNode("Where", where_inputs, {where_output}, "", (name + "--where").c_str()));
  replacements.push_back(MakeNode("Expand", {where_output, expand.input()[1].value()},
                                  {where.output()[0].value()}, "", (name + "--expand").c_str()));
  return replacements;
}

std::set<std::string> ExpandSwapPattern::FastOpType() const { return {"Expand"}; }

core::builder::MatchResult ExpandSwapPattern::Match(core::builder::GraphGraph &graph,
                                                    const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Expand") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Expand with two inputs");
  }
  if (!graph.HasShape(candidate.input()[0].value())) {
    return NoMatch(candidate, "the Expand input has no known shape");
  }
  if (graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
    return NoMatch(candidate, "the Expand output is shared");
  }

  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.output()[0].value());
  if (next_nodes.size() != 1 || next_nodes[0] == nullptr) {
    return NoMatch(candidate, "the Expand output is not consumed by exactly one node");
  }
  const NodeProto &next_node = *next_nodes[0];
  const std::string &op_type = next_node.op_type().value();
  const bool is_other = SwapOtherOpTypes().find(op_type) != SwapOtherOpTypes().end();
  const bool is_unary = UnaryLikeOpTypes().find(op_type) != UnaryLikeOpTypes().end() &&
                        NormaliseDomain(next_node.domain().value()) == kDefaultOnnxDomain;
  if (!is_other && !is_unary) {
    return NoMatch(candidate, "the consumer is not a unary-like operator");
  }
  if (next_node.output_size() != 1) {
    return NoMatch(candidate, "the consumer does not have a single output");
  }

  return core::builder::MatchResult{this, {&candidate, &next_node}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ExpandSwapPattern::Apply(core::builder::GraphGraph &graph,
                         const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ExpandSwapPattern::Apply expects one Expand and one unary-like node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ExpandSwapPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &expand = *nodes[0];
  const NodeProto &next_node = *nodes[1];

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "ExpandSwapPattern--" + expand.name().value();
  const std::string new_name = builder.UniqueName("ExpandSwapPattern_" + expand.input()[0].value());

  std::vector<std::string> unary_inputs{expand.input()[0].value()};
  for (int i = 1; i < next_node.input_size(); ++i) {
    unary_inputs.push_back(next_node.input()[i].value());
  }

  NodeProto unary = MakeNode(next_node.op_type().value().c_str(), unary_inputs, {new_name},
                             next_node.domain().value().c_str(), (name + "--unary").c_str());
  for (const AttributeProto &attribute : next_node.attribute()) {
    unary.mutable_attribute()->push_back(attribute);
  }

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(unary));
  replacements.push_back(MakeNode("Expand", {new_name, expand.input()[1].value()},
                                  {next_node.output()[0].value()}, "",
                                  (name + "--expand").c_str()));
  return replacements;
}

std::set<std::string> SwapExpandUnsqueezePattern::FastOpType() const { return {"Expand"}; }

core::builder::MatchResult SwapExpandUnsqueezePattern::Match(core::builder::GraphGraph &graph,
                                                             const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Expand") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Expand with two inputs");
  }
  if (graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
    return NoMatch(candidate, "the Expand output is shared");
  }
  if (!MainOpsetAtLeast(graph, 13)) {
    return NoMatch(candidate, "the model opset is below 13");
  }

  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.output()[0].value());
  if (next_nodes.size() != 1 || next_nodes[0] == nullptr) {
    return NoMatch(candidate, "the Expand output is not consumed by exactly one node");
  }
  const NodeProto &unsqueeze = *next_nodes[0];
  if (!IsDefaultOp(unsqueeze, "Unsqueeze") || unsqueeze.input_size() != 2 ||
      unsqueeze.output_size() != 1) {
    return NoMatch(candidate, "the consumer is not a default-domain Unsqueeze with two inputs");
  }
  if (!graph.IsConstant(unsqueeze.input()[1].value())) {
    return NoMatch(candidate, "the Unsqueeze axes are not constant");
  }

  // The new Expand target shape is derived either from the constant Expand shape
  // or from the fully-static shape of the Expand output.
  std::vector<int64_t> base_shape;
  if (!ReadConstantShape(graph, candidate.input()[1].value(), base_shape) &&
      !ReadStaticShape(graph, candidate.output()[0].value(), base_shape)) {
    return NoMatch(candidate,
                   "neither the Expand target shape nor the Expand output shape is known");
  }

  return core::builder::MatchResult{this, {&candidate, &unsqueeze}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
SwapExpandUnsqueezePattern::Apply(core::builder::GraphGraph &graph,
                                  const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("SwapExpandUnsqueezePattern::Apply expects one Expand and one Unsqueeze.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "SwapExpandUnsqueezePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &expand = *nodes[0];
  const NodeProto &unsqueeze = *nodes[1];

  std::vector<int64_t> axes;
  if (!ReadConstantShape(graph, unsqueeze.input()[1].value(), axes)) {
    throw BuilderError("SwapExpandUnsqueezePattern::Apply could not read the Unsqueeze axes.");
  }
  std::vector<int64_t> base_shape;
  if (!ReadConstantShape(graph, expand.input()[1].value(), base_shape) &&
      !ReadStaticShape(graph, expand.output()[0].value(), base_shape)) {
    throw BuilderError("SwapExpandUnsqueezePattern::Apply could not read the Expand shape.");
  }
  const std::vector<int64_t> new_shape = InsertUnsqueezeOnes(base_shape, axes);

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "SwapExpandUnsqueezePattern--" + expand.name().value();
  const std::string shape_init = FreeInitializerName(builder, name + "_shape");
  builder.MakeInitializer(MakeInitializerShape(shape_init.c_str(), new_shape));
  const std::string new_name =
      builder.UniqueName("SwapExpandUnsqueezePattern_" + expand.input()[0].value());

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("Unsqueeze",
                                  {expand.input()[0].value(), unsqueeze.input()[1].value()},
                                  {new_name}, "", (name + "--unsqueeze").c_str()));
  replacements.push_back(MakeNode("Expand", {new_name, shape_init}, {unsqueeze.output()[0].value()},
                                  "", (name + "--expand").c_str()));
  return replacements;
}

std::set<std::string> ExpandUnsqueezeExpandPattern::FastOpType() const { return {"Expand"}; }

core::builder::MatchResult ExpandUnsqueezeExpandPattern::Match(core::builder::GraphGraph &graph,
                                                               const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Expand") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Expand with two inputs");
  }
  if (graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
    return NoMatch(candidate, "the first Expand output is shared");
  }
  if (!MainOpsetAtLeast(graph, 13)) {
    return NoMatch(candidate, "the model opset is below 13");
  }

  const std::vector<const NodeProto *> &after_expand =
      graph.NextNodes(candidate.output()[0].value());
  if (after_expand.size() != 1 || after_expand[0] == nullptr) {
    return NoMatch(candidate, "the first Expand output is not consumed by exactly one node");
  }
  const NodeProto &unsqueeze = *after_expand[0];
  if (!IsDefaultOp(unsqueeze, "Unsqueeze") || unsqueeze.input_size() != 2 ||
      unsqueeze.output_size() != 1) {
    return NoMatch(candidate, "the consumer is not a default-domain Unsqueeze with two inputs");
  }
  if (graph.IsUsedMoreThanOnce(unsqueeze.output()[0].value())) {
    return NoMatch(candidate, "the Unsqueeze output is shared");
  }
  if (!graph.IsConstant(unsqueeze.input()[1].value())) {
    return NoMatch(candidate, "the Unsqueeze axes are not constant");
  }

  const std::vector<const NodeProto *> &after_unsqueeze =
      graph.NextNodes(unsqueeze.output()[0].value());
  if (after_unsqueeze.size() != 1 || after_unsqueeze[0] == nullptr) {
    return NoMatch(candidate, "the Unsqueeze output is not consumed by exactly one node");
  }
  const NodeProto &expand2 = *after_unsqueeze[0];
  if (!IsDefaultOp(expand2, "Expand") || expand2.input_size() != 2 || expand2.output_size() != 1) {
    return NoMatch(candidate, "the second consumer is not a default-domain Expand with two inputs");
  }

  std::vector<int64_t> first_shape;
  std::vector<int64_t> axes;
  std::vector<int64_t> second_shape;
  if (!ReadConstantShape(graph, candidate.input()[1].value(), first_shape) ||
      !ReadConstantShape(graph, unsqueeze.input()[1].value(), axes) ||
      !ReadConstantShape(graph, expand2.input()[1].value(), second_shape)) {
    return NoMatch(candidate, "the Expand shapes or the Unsqueeze axes are not constant integers");
  }
  if (first_shape.size() + axes.size() != second_shape.size()) {
    return NoMatch(candidate,
                   "the unsqueezed first shape and the second shape have different ranks");
  }

  return core::builder::MatchResult{this, {&candidate, &unsqueeze, &expand2}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ExpandUnsqueezeExpandPattern::Apply(core::builder::GraphGraph &graph,
                                    const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError(
        "ExpandUnsqueezeExpandPattern::Apply expects two Expand nodes and one Unsqueeze.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ExpandUnsqueezeExpandPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &expand1 = *nodes[0];
  const NodeProto &unsqueeze = *nodes[1];
  const NodeProto &expand2 = *nodes[2];

  std::vector<int64_t> first_shape;
  std::vector<int64_t> axes;
  std::vector<int64_t> second_shape;
  if (!ReadConstantShape(graph, expand1.input()[1].value(), first_shape) ||
      !ReadConstantShape(graph, unsqueeze.input()[1].value(), axes) ||
      !ReadConstantShape(graph, expand2.input()[1].value(), second_shape)) {
    throw BuilderError("ExpandUnsqueezeExpandPattern::Apply could not read the constant shapes.");
  }
  const std::vector<int64_t> first_unsqueezed = InsertUnsqueezeOnes(first_shape, axes);
  if (first_unsqueezed.size() != second_shape.size()) {
    throw BuilderError("ExpandUnsqueezeExpandPattern::Apply produced mismatched shape ranks.");
  }
  std::vector<int64_t> combined_shape(second_shape.size());
  for (std::size_t i = 0; i < second_shape.size(); ++i) {
    combined_shape[i] = std::max(first_unsqueezed[i], second_shape[i]);
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "ExpandUnsqueezeExpandPattern--" + expand1.name().value();
  const std::string shape_init = FreeInitializerName(builder, name + "_shape");
  builder.MakeInitializer(MakeInitializerShape(shape_init.c_str(), combined_shape));
  const std::string new_name =
      builder.UniqueName("ExpandUnsqueezeExpandPattern_" + unsqueeze.output()[0].value());

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("Unsqueeze",
                                  {expand1.input()[0].value(), unsqueeze.input()[1].value()},
                                  {new_name}, "", (name + "--unsqueeze").c_str()));
  replacements.push_back(MakeNode("Expand", {new_name, shape_init}, {expand2.output()[0].value()},
                                  "", (name + "--expand").c_str()));
  return replacements;
}

std::set<std::string> SwapExpandReshapePattern::FastOpType() const { return {"Reshape"}; }

core::builder::MatchResult SwapExpandReshapePattern::Match(core::builder::GraphGraph &graph,
                                                           const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Reshape") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Reshape with two inputs");
  }
  std::vector<int64_t> reshape_target;
  if (!ReadConstantShape(graph, candidate.input()[1].value(), reshape_target) ||
      reshape_target != std::vector<int64_t>({0, 1, -1})) {
    return NoMatch(candidate, "the Reshape target is not the constant [0, 1, -1]");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value())) {
    return NoMatch(candidate, "the Expand output is shared");
  }
  const NodeProto *expand = graph.NodeBefore(candidate.input()[0].value());
  if (!IsExpand(expand)) {
    return NoMatch(candidate, "the Reshape input is not produced by Expand");
  }
  if (!graph.HasShape(expand->input()[0].value()) ||
      graph.GetShape(expand->input()[0].value()).Shape().Rank() != 3) {
    return NoMatch(candidate, "the pre-expansion input does not have rank three");
  }
  if (!graph.HasShape(expand->input()[1].value())) {
    return NoMatch(candidate, "the Expand target has no inferred shape value");
  }
  const core::symbolic::SymTensor &target = graph.GetShape(expand->input()[1].value());
  if (!target.HasValueAsShape() || target.ValueAsShape().Rank() != 3 ||
      !target.ValueAsShape()[1].IsInt() || target.ValueAsShape()[1].AsInt() != 1 ||
      !target.ValueAsShape()[2].IsInt() || target.ValueAsShape()[2].AsInt() != 1) {
    return NoMatch(candidate, "the Expand target does not end with [1, 1]");
  }
  return core::builder::MatchResult{this, {expand, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
SwapExpandReshapePattern::Apply(core::builder::GraphGraph &graph,
                                const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("SwapExpandReshapePattern::Apply expects one Expand and one Reshape.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("SwapExpandReshapePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &expand = *nodes[0];
  const NodeProto &reshape = *nodes[1];
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "SwapExpandReshapePattern--" + reshape.name().value();
  const std::string reshape_output = builder.UniqueName(name + "_reshape");

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("Reshape",
                                  {expand.input()[0].value(), reshape.input()[1].value()},
                                  {reshape_output}, "", (name + "--reshape").c_str()));
  replacements.push_back(MakeNode("Expand", {reshape_output, expand.input()[1].value()},
                                  {reshape.output()[0].value()}, "", (name + "--expand").c_str()));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
