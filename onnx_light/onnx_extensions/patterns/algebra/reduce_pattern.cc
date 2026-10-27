// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/algebra/reduce_pattern.h"

#include <string>
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

bool ReadAxes(const TensorProto &tensor, std::vector<int64_t> &axes) {
  return ReadIntegerValues(tensor, axes) && !axes.empty();
}

void CopyAttributes(const NodeProto &from, NodeProto &to) {
  for (const AttributeProto &attribute : from.attribute()) {
    to.mutable_attribute()->push_back(attribute);
  }
}

} // namespace

std::set<std::string> ReduceArgTopKPattern::FastOpType() const { return {"ArgMax", "ArgMin"}; }

core::builder::MatchResult ReduceArgTopKPattern::Match(core::builder::GraphGraph &graph,
                                                       const NodeProto &candidate) const {
  if (!MainOpsetAtLeast(graph, 18)) {
    return NoMatch(candidate, "the default-domain opset is below 18");
  }
  if ((!IsDefaultOp(candidate, "ArgMin") && !IsDefaultOp(candidate, "ArgMax")) ||
      candidate.input_size() < 1 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain ArgMin or ArgMax");
  }
  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.input()[0].value());
  if (next_nodes.size() < 2) {
    return NoMatch(candidate, "the Arg input has fewer than two consumers");
  }
  const std::string reduce_type =
      candidate.op_type().value() == "ArgMin" ? "ReduceMin" : "ReduceMax";
  const NodeProto *reduce = nullptr;
  for (const NodeProto *node : next_nodes) {
    if (node != nullptr && IsDefaultOp(*node, reduce_type.c_str())) {
      if (reduce != nullptr) {
        return NoMatch(candidate, "the Arg input has multiple matching Reduce consumers");
      }
      reduce = node;
    }
  }
  if (reduce == nullptr || reduce->input_size() < 2 || reduce->output_size() != 1 ||
      !graph.IsConstant(reduce->input()[1].value())) {
    return NoMatch(candidate, "the matching Reduce does not have constant axes");
  }
  const TensorProto *axes_tensor = graph.GetComputedConstant(reduce->input()[1].value());
  std::vector<int64_t> axes;
  if (axes_tensor == nullptr || !ReadAxes(*axes_tensor, axes) || axes.size() != 1) {
    return NoMatch(candidate, "the Reduce axes are not one readable axis");
  }
  const int64_t axis = GetAttributeOr<int64_t>(candidate, "axis", 0);
  if (axis != axes[0]) {
    return NoMatch(candidate, "the Arg axis differs from the Reduce axis");
  }
  if (GetAttributeOr<int64_t>(candidate, "keepdims", 1) !=
      GetAttributeOr<int64_t>(*reduce, "keepdims", 1)) {
    return NoMatch(candidate, "the Arg and Reduce keepdims attributes differ");
  }
  if (GetAttributeOr<int64_t>(*reduce, "noop_with_empty_axes", 0) != 0 ||
      GetAttributeOr<int64_t>(*reduce, "select_last_index", 0) == 1) {
    return NoMatch(candidate, "the Reduce attributes are not TopK-compatible");
  }
  return core::builder::MatchResult{this, {reduce, &candidate}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
ReduceArgTopKPattern::Apply(core::builder::GraphGraph &graph,
                            const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ReduceArgTopKPattern::Apply expects a Reduce and an Arg node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ReduceArgTopKPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &reduce = *nodes[0];
  const NodeProto &arg = *nodes[1];
  const int64_t keepdims = GetAttributeOr<int64_t>(arg, "keepdims", 1);
  const int64_t axis = GetAttributeOr<int64_t>(arg, "axis", 0);
  core::builder::GraphBuilder &builder = graph.Builder();

  const std::string k_name = FreeInitializerName(builder, "ReduceArgTopKPattern.K");
  builder.MakeInitializer(MakeInitializerShape(k_name.c_str(), {1}));

  std::vector<std::string> topk_outputs;
  if (keepdims != 0) {
    topk_outputs = {reduce.output()[0].value(), arg.output()[0].value()};
  } else {
    topk_outputs = {builder.UniqueName("ReduceArgTopKPattern_" + reduce.output()[0].value()),
                    builder.UniqueName("ReduceArgTopKPattern_" + arg.output()[0].value())};
  }
  NodeProto topk = MakeNode("TopK", {reduce.input()[0].value(), k_name}, topk_outputs, "",
                            ("ReduceArgTopKPattern--" + arg.name().value()).c_str());
  AddAttribute<int64_t>(topk, "axis", axis);
  AddAttribute<int64_t>(topk, "largest", arg.op_type().value() == "ArgMax" ? 1 : 0);

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(topk));
  if (keepdims == 0) {
    const std::string axes_name = FreeInitializerName(builder, "ReduceArgTopKPattern.axes");
    builder.MakeInitializer(MakeInitializerShape(axes_name.c_str(), {axis}));
    replacements.push_back(MakeNode("Squeeze", {topk_outputs[0], axes_name},
                                    {reduce.output()[0].value()}, "",
                                    ("ReduceArgTopKPattern--" + reduce.name().value()).c_str()));
    replacements.push_back(MakeNode("Squeeze", {topk_outputs[1], axes_name},
                                    {arg.output()[0].value()}, "",
                                    ("ReduceArgTopKPattern--" + arg.name().value()).c_str()));
  }
  return replacements;
}

std::set<std::string> ReduceSumNormalizePattern::FastOpType() const { return {"ReduceSum"}; }

core::builder::MatchResult ReduceSumNormalizePattern::Match(core::builder::GraphGraph &graph,
                                                            const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "ReduceSum") || candidate.input_size() < 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain ReduceSum with axes input");
  }
  const NodeProto *cast = graph.NodeBefore(candidate.input()[0].value());
  if (cast == nullptr || !IsDefaultOp(*cast, "Cast") || cast->input_size() != 1) {
    return NoMatch(candidate, "the ReduceSum input is not produced by a default-domain Cast");
  }
  const std::vector<const NodeProto *> &mul_nodes = graph.NextNodes(candidate.output()[0].value());
  if (mul_nodes.size() != 1 || !IsDefaultOp(*mul_nodes[0], "Mul") ||
      mul_nodes[0]->input_size() != 2 || mul_nodes[0]->output_size() != 1) {
    return NoMatch(candidate, "the ReduceSum is not followed by one default-domain Mul");
  }
  const NodeProto *mul = mul_nodes[0];
  const std::vector<const NodeProto *> &sub_nodes = graph.NextNodes(mul->output()[0].value());
  if (sub_nodes.size() != 1 || !IsDefaultOp(*sub_nodes[0], "Sub") ||
      sub_nodes[0]->input_size() != 2 || sub_nodes[0]->output_size() != 1) {
    return NoMatch(candidate, "the Mul is not followed by one default-domain Sub");
  }
  const NodeProto *sub = sub_nodes[0];
  const std::vector<const NodeProto *> &cast_nodes = graph.NextNodes(sub->output()[0].value());
  if (cast_nodes.size() != 1 || !IsDefaultOp(*cast_nodes[0], "Cast") ||
      cast_nodes[0]->input_size() != 1 || cast_nodes[0]->output_size() != 1) {
    return NoMatch(candidate, "the Sub is not followed by one default-domain Cast");
  }
  const NodeProto *cast2 = cast_nodes[0];
  if (sub->input()[0].value() != candidate.input()[0].value() &&
      sub->input()[1].value() != candidate.input()[0].value() &&
      sub->input()[0].value() != candidate.input()[1].value() &&
      sub->input()[1].value() != candidate.input()[1].value()) {
    return NoMatch(candidate, "the Sub does not reuse a ReduceSum input");
  }
  if (!graph.HasType(cast->input()[0].value()) || !graph.HasType(cast2->output()[0].value()) ||
      graph.GetType(cast->input()[0].value()) != graph.GetType(cast2->output()[0].value())) {
    return NoMatch(candidate, "the Cast input and final output types differ or are unknown");
  }
  const AttributeProto *to = FindAttribute(*cast2, "to");
  if (to == nullptr || to->type() != AttributeProto::AttributeType::INT) {
    return NoMatch(candidate, "the final Cast does not define an integer 'to' attribute");
  }
  return core::builder::MatchResult{this, {cast, &candidate, mul, sub, cast2}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
ReduceSumNormalizePattern::Apply(core::builder::GraphGraph &graph,
                                 const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 5 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr ||
      nodes[3] == nullptr || nodes[4] == nullptr) {
    throw BuilderError("ReduceSumNormalizePattern::Apply expects Cast, ReduceSum, Mul, Sub, Cast.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ReduceSumNormalizePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &cast = *nodes[0];
  const NodeProto &reduce = *nodes[1];
  const NodeProto &mul = *nodes[2];
  const NodeProto &sub = *nodes[3];
  const NodeProto &cast2 = *nodes[4];
  core::builder::GraphBuilder &builder = graph.Builder();

  const std::string reduced =
      builder.UniqueName("ReduceSumNormalizePattern_" + reduce.output()[0].value());
  NodeProto replacement_reduce =
      MakeNode("ReduceSum", {cast.input()[0].value(), reduce.input()[1].value()}, {reduced}, "",
               ("ReduceSumNormalizePattern--" + reduce.name().value()).c_str());
  CopyAttributes(reduce, replacement_reduce);

  const std::string other = mul.input()[0].value() == reduce.output()[0].value()
                                ? mul.input()[1].value()
                                : mul.input()[0].value();
  const std::string converted = builder.UniqueName("ReduceSumNormalizePattern_" + other);
  NodeProto replacement_cast =
      MakeNode("Cast", {other}, {converted}, "",
               ("ReduceSumNormalizePattern--" + cast.name().value()).c_str());
  AddAttribute<int64_t>(replacement_cast, "to", FindAttribute(cast2, "to")->i());

  const std::string product =
      builder.UniqueName("ReduceSumNormalizePattern_" + mul.output()[0].value());
  NodeProto replacement_mul =
      MakeNode("Mul", {reduced, converted}, {product}, "",
               ("ReduceSumNormalizePattern--" + mul.name().value()).c_str());
  const std::vector<std::string> sub_inputs =
      mul.output()[0].value() == sub.input()[0].value()
          ? std::vector<std::string>{product, cast.input()[0].value()}
          : std::vector<std::string>{cast.input()[0].value(), product};
  NodeProto replacement_sub =
      MakeNode("Sub", sub_inputs, {cast2.output()[0].value()}, "",
               ("ReduceSumNormalizePattern--" + sub.name().value()).c_str());

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement_reduce));
  replacements.push_back(std::move(replacement_cast));
  replacements.push_back(std::move(replacement_mul));
  replacements.push_back(std::move(replacement_sub));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
