// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/expand/expand_pattern.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
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

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
