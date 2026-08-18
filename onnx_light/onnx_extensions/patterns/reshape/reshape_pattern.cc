// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/reshape/reshape_pattern.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
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
using core::symbolic::SymDim;
using core::symbolic::SymShape;

bool IsReshape(const NodeProto &node) { return IsDefaultOp(node, "Reshape"); }

bool IsBinaryElementWise(const NodeProto &node) {
  static const std::set<std::string> op_types = {
      "Add", "And", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "Div", "Max", "Mean",
      "Min", "Mul", "Mod",        "Or",        "Sub",        "Sum", "Xor",
  };
  return NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain &&
         op_types.count(node.op_type().value()) != 0;
}

std::vector<std::string> Inputs(const NodeProto &node) {
  std::vector<std::string> result;
  result.reserve(node.input_size());
  for (int i = 0; i < node.input_size(); ++i) {
    result.push_back(node.input()[i].value());
  }
  return result;
}

bool ReadConstantInts(core::builder::GraphGraph &graph, const std::string &name,
                      std::vector<int64_t> &values) {
  if (!graph.IsConstant(name)) {
    return false;
  }
  const TensorProto *tensor = graph.GetComputedConstant(name);
  return tensor != nullptr && ReadIntegerValues(*tensor, values);
}

bool IsShapeOne(core::builder::GraphGraph &graph, const std::string &name) {
  if (!graph.HasShape(name)) {
    return false;
  }
  const SymShape &shape = graph.GetShape(name).Shape();
  return shape.Rank() == 1 && shape[0].IsInt() && shape[0].AsInt() == 1;
}

bool IsFullyStatic(const SymShape &shape) { return shape.IsFullyKnown(); }

bool HasRank(core::builder::GraphGraph &graph, const std::string &name) {
  return graph.HasShape(name);
}

std::size_t Rank(core::builder::GraphGraph &graph, const std::string &name) {
  return graph.GetShape(name).Shape().Rank();
}

std::optional<int64_t> Product(const SymShape &shape, std::size_t first, std::size_t count) {
  int64_t result = 1;
  for (std::size_t i = first; i < first + count; ++i) {
    if (!shape[i].IsInt()) {
      return std::nullopt;
    }
    result *= shape[i].AsInt();
  }
  return result;
}

std::string AddShapeInitializer(core::builder::GraphGraph &graph, const std::string &base,
                                const std::vector<int64_t> &values) {
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = FreeInitializerName(builder, base);
  return builder.MakeInitializer(MakeInitializerShape(name.c_str(), values));
}

NodeProto MakeReplacement(const char *op_type, const std::vector<std::string> &inputs,
                          const std::vector<std::string> &outputs, const std::string &name,
                          const NodeProto &source) {
  NodeProto replacement = MakeNode(op_type, inputs, outputs, "", name.c_str());
  replacement.set_doc_string(source.doc_string().value());
  return replacement;
}

void CopyAttributesExcept(const NodeProto &source, NodeProto &destination, const char *excluded) {
  for (int i = 0; i < source.attribute_size(); ++i) {
    const AttributeProto &attribute = source.attribute(i);
    if (attribute.name().value() != excluded) {
      *destination.add_attribute() = attribute;
    }
  }
}

bool HasInt(const std::vector<int64_t> &values, int64_t value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

bool MainOpsetAtLeast(core::builder::GraphGraph &graph, int minimum) {
  const int opset = graph.Builder().OpsetVersion("");
  return opset == core::shapes::kUnknownOpsetVersion || opset >= minimum;
}

utils::RepeatedProtoField<NodeProto> SingleIdentity(const char *pattern_name,
                                                    const NodeProto &node) {
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(
      MakeReplacement("Identity", {node.input()[0].value()}, {node.output()[0].value()},
                      std::string(pattern_name) + "--" + node.name().value(), node));
  return replacements;
}

bool ApplicableReshape(const SymShape &input_shape, const std::vector<int64_t> &shape) {
  bool inferred = false;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) {
      if (inferred || i >= input_shape.Rank()) {
        return false;
      }
    } else if (shape[i] < 0) {
      if (inferred) {
        return false;
      }
      inferred = true;
    }
  }
  if (shape.size() == input_shape.Rank()) {
    bool equal = true;
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if ((shape[i] > 0 && (!input_shape[i].IsInt() || input_shape[i].AsInt() != shape[i])) ||
          (shape[i] == 0 && !input_shape[i].IsInt())) {
        equal = false;
        break;
      }
    }
    if (equal) {
      return true;
    }
  }
  const int zeros = static_cast<int>(std::count(shape.begin(), shape.end(), 0));
  if (zeros > 1 && !shape.empty() && shape.back() != -1) {
    return false;
  }
  return true;
}

} // namespace

std::set<std::string> ConcatReshapePattern::FastOpType() const { return {"Reshape"}; }

core::builder::MatchResult ConcatReshapePattern::Match(core::builder::GraphGraph &graph,
                                                       const NodeProto &candidate) const {
  if (!IsReshape(candidate) || candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Reshape with two inputs");
  }
  const NodeProto *concat = graph.NodeBefore(candidate.input()[1].value());
  if (concat == nullptr || !IsDefaultOp(*concat, "Concat") || concat->output_size() != 1 ||
      collections::GetAxis(*concat) != 0) {
    return NoMatch(candidate, "the reshape shape is not produced by an axis-0 Concat");
  }

  std::set<std::string> dynamic_producers;
  int shape_count = 0;
  int dynamic_count = 0;
  for (int i = 0; i < concat->input_size(); ++i) {
    const std::string &input = concat->input()[i].value();
    if (graph.IsConstant(input)) {
      std::vector<int64_t> values;
      if (!ReadConstantInts(graph, input, values) || values.size() != 1 || HasInt(values, -1)) {
        return NoMatch(candidate, "a constant Concat shape input is not a one-element shape");
      }
      continue;
    }
    const NodeProto *producer = graph.NodeBefore(input);
    if (producer == nullptr || !IsShapeOne(graph, input)) {
      return NoMatch(candidate, "a dynamic Concat shape input does not produce one element");
    }
    ++dynamic_count;
    dynamic_producers.insert(producer->op_type().value());
    if (IsDefaultOp(*producer, "Shape")) {
      ++shape_count;
    }
  }

  if (dynamic_count == 0 || dynamic_producers.size() > 2 ||
      (dynamic_producers.size() == 1 && !dynamic_producers.contains("Shape")) ||
      (dynamic_producers.size() == 2 &&
       (!dynamic_producers.contains("Shape") || shape_count != dynamic_count - 1))) {
    return NoMatch(candidate, "the dynamic Concat shape inputs cannot be represented by one -1");
  }
  const NodeProto *insert_at =
      graph.IsUsedMoreThanOnce(candidate.input()[1].value()) ? nullptr : &candidate;
  return core::builder::MatchResult{this, {concat, &candidate}, insert_at};
}

utils::RepeatedProtoField<NodeProto>
ConcatReshapePattern::Apply(core::builder::GraphGraph &graph,
                            const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ConcatReshapePattern::Apply expects a Concat and a Reshape node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ConcatReshapePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &concat = *nodes[0];
  const NodeProto &reshape = *nodes[1];
  const std::string m1 = AddShapeInitializer(graph, "ConcatReshapePattern_m1", {-1});

  std::vector<std::string> inputs;
  int last_shape = -1;
  bool replaced = false;
  for (int i = 0; i < concat.input_size(); ++i) {
    const std::string &input = concat.input()[i].value();
    if (graph.IsConstant(input)) {
      inputs.push_back(input);
      continue;
    }
    const NodeProto *producer = graph.NodeBefore(input);
    if (producer != nullptr && IsDefaultOp(*producer, "Shape")) {
      last_shape = static_cast<int>(inputs.size());
      inputs.push_back(input);
    } else {
      inputs.push_back(m1);
      replaced = true;
    }
  }
  if (!replaced) {
    if (last_shape < 0) {
      throw BuilderError("ConcatReshapePattern::Apply found no Shape input to replace.");
    }
    inputs[static_cast<std::size_t>(last_shape)] = m1;
  }

  const std::string name = "ConcatReshapePattern--" + concat.name().value();
  const std::string concat_out =
      graph.Builder().UniqueName(concat.output()[0].value() + "--concat");
  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(concat.output()[0].value())) {
    replacements.add() = concat;
  }
  NodeProto new_concat = MakeReplacement("Concat", inputs, {concat_out}, name, concat);
  AddAttribute<int64_t>(new_concat, "axis", 0);
  replacements.push_back(std::move(new_concat));
  replacements.push_back(MakeReplacement(
      "Reshape", {reshape.input()[0].value(), concat_out}, {reshape.output()[0].value()},
      "ConcatReshapePattern--" + reshape.name().value(), reshape));
  return replacements;
}

std::set<std::string> StaticConcatReshapePattern::FastOpType() const { return {"Reshape"}; }

core::builder::MatchResult StaticConcatReshapePattern::Match(core::builder::GraphGraph &graph,
                                                             const NodeProto &candidate) const {
  if (!IsReshape(candidate) || candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Reshape with two inputs");
  }
  const NodeProto *concat = graph.NodeBefore(candidate.input()[1].value());
  if (concat == nullptr || !IsDefaultOp(*concat, "Concat") || concat->output_size() != 1 ||
      collections::GetAxis(*concat) != 0) {
    return NoMatch(candidate, "the reshape shape is not produced by an axis-0 Concat");
  }
  int dynamic = 0;
  for (int i = 0; i < concat->input_size(); ++i) {
    const std::string &input = concat->input()[i].value();
    if (graph.IsConstant(input)) {
      std::vector<int64_t> values;
      if (!ReadConstantInts(graph, input, values) || HasInt(values, -1)) {
        return NoMatch(candidate, "a constant Concat input is not a usable static shape");
      }
    } else if (IsShapeOne(graph, input)) {
      ++dynamic;
    } else {
      return NoMatch(candidate, "a dynamic Concat input does not have shape (1)");
    }
  }
  if (dynamic != 1) {
    return NoMatch(candidate, "the Concat does not have exactly one dynamic shape element");
  }
  const NodeProto *insert_at =
      graph.IsUsedMoreThanOnce(candidate.input()[1].value()) ? nullptr : &candidate;
  return core::builder::MatchResult{this, {concat, &candidate}, insert_at};
}

utils::RepeatedProtoField<NodeProto>
StaticConcatReshapePattern::Apply(core::builder::GraphGraph &graph,
                                  const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("StaticConcatReshapePattern::Apply expects a Concat and a Reshape node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "StaticConcatReshapePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &concat = *nodes[0];
  const NodeProto &reshape = *nodes[1];
  const std::string m1 = AddShapeInitializer(graph, "StaticConcatReshapePattern_m1", {-1});
  std::vector<std::string> inputs;
  bool replaced = false;
  for (int i = 0; i < concat.input_size(); ++i) {
    const std::string &input = concat.input()[i].value();
    if (graph.IsConstant(input)) {
      inputs.push_back(input);
    } else if (IsShapeOne(graph, input) && !replaced) {
      inputs.push_back(m1);
      replaced = true;
    } else {
      throw BuilderError("StaticConcatReshapePattern::Apply found an invalid dynamic input.");
    }
  }
  if (!replaced) {
    throw BuilderError("StaticConcatReshapePattern::Apply found no dynamic input to replace.");
  }
  const std::string concat_out =
      graph.Builder().UniqueName(concat.output()[0].value() + "--static-concat");
  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(concat.output()[0].value())) {
    replacements.add() = concat;
  }
  NodeProto new_concat =
      MakeReplacement("Concat", inputs, {concat_out},
                      "StaticConcatReshapePattern--" + concat.name().value(), concat);
  AddAttribute<int64_t>(new_concat, "axis", 0);
  replacements.push_back(std::move(new_concat));
  replacements.push_back(MakeReplacement(
      "Reshape", {reshape.input()[0].value(), concat_out}, {reshape.output()[0].value()},
      "StaticConcatReshapePattern--" + reshape.name().value(), reshape));
  return replacements;
}

std::set<std::string> ReshapePattern::FastOpType() const { return {"Reshape"}; }

core::builder::MatchResult ReshapePattern::Match(core::builder::GraphGraph &graph,
                                                 const NodeProto &candidate) const {
  if (!IsReshape(candidate) || candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Reshape with two inputs");
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !IsFullyStatic(graph.GetShape(candidate.input()[0].value()).Shape())) {
    return NoMatch(candidate, "the reshape input shape is not fully static");
  }
  std::vector<int64_t> target;
  if (!ReadConstantInts(graph, candidate.input()[1].value(), target)) {
    return NoMatch(candidate, "the reshape target shape is not a materialised constant");
  }
  const SymShape &input_shape = graph.GetShape(candidate.input()[0].value()).Shape();
  if (target.size() != input_shape.Rank()) {
    return NoMatch(candidate, "the reshape target rank differs from the input rank");
  }
  for (std::size_t i = 0; i < target.size(); ++i) {
    if (target[i] != input_shape[i].AsInt()) {
      return NoMatch(candidate, "the reshape target shape differs from the input shape");
    }
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ReshapePattern::Apply(core::builder::GraphGraph &graph,
                      const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("ReshapePattern::Apply expects one Reshape node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ReshapePattern::Apply received an unsafe or inconsistent match.");
  }
  return SingleIdentity("ReshapePattern", *nodes[0]);
}

std::set<std::string> ShapedBasedReshapePattern::FastOpType() const { return {"Reshape"}; }

core::builder::MatchResult ShapedBasedReshapePattern::Match(core::builder::GraphGraph &graph,
                                                            const NodeProto &candidate) const {
  if (!IsReshape(candidate) || candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Reshape with two inputs");
  }
  std::vector<int64_t> target;
  if (!ReadConstantInts(graph, candidate.input()[1].value(), target) || target.empty() ||
      target.back() == 0) {
    return NoMatch(candidate, "the reshape target is not a supported zero-prefix shape");
  }
  if (!std::all_of(target.begin(), target.end() - 1, [](int64_t value) { return value == 0; })) {
    return NoMatch(candidate, "the reshape target does not contain only leading zero dimensions");
  }
  if (!HasRank(graph, candidate.input()[0].value()) ||
      Rank(graph, candidate.input()[0].value()) != target.size()) {
    return NoMatch(candidate, "the reshape input rank does not equal the target rank");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ShapedBasedReshapePattern::Apply(core::builder::GraphGraph &graph,
                                 const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("ShapedBasedReshapePattern::Apply expects one Reshape node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ShapedBasedReshapePattern::Apply received an unsafe or inconsistent match.");
  }
  return SingleIdentity("ShapedBasedReshapePattern", *nodes[0]);
}

std::set<std::string> ReduceReshapePattern::FastOpType() const {
  return {"ReduceL1",   "ReduceL2",  "ReduceLogSum", "ReduceLogSumExp", "ReduceMax",
          "ReduceMean", "ReduceMin", "ReduceProd",   "ReduceSum",       "ReduceSumSquare"};
}

core::builder::MatchResult ReduceReshapePattern::Match(core::builder::GraphGraph &graph,
                                                       const NodeProto &candidate) const {
  if (candidate.op_type().value().rfind("Reduce", 0) != 0 ||
      NormaliseDomain(candidate.domain().value()) != kDefaultOnnxDomain ||
      candidate.input_size() < 1 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain reduction");
  }
  if (graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
    return NoMatch(candidate, "the reduction output is shared");
  }
  if (GetAttributeOr<int64_t>(candidate, "keepdims", 1) == 0) {
    return NoMatch(candidate, "the reduction already removes its reduced dimensions");
  }

  if (!HasRank(graph, candidate.input()[0].value())) {
    return NoMatch(candidate, "the reduction input rank is unknown");
  }
  std::vector<int64_t> axes;
  if (candidate.input_size() == 2) {
    if (!ReadConstantInts(graph, candidate.input()[1].value(), axes)) {
      return NoMatch(candidate, "the reduction axes input is not constant");
    }
  } else if (!GetAttributeInts(candidate, "axes", axes)) {
    for (std::size_t i = 0; i < Rank(graph, candidate.input()[0].value()); ++i) {
      axes.push_back(static_cast<int64_t>(i));
    }
  }
  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.output()[0].value());
  if (next_nodes.size() != 1 || next_nodes[0] == nullptr || !IsReshape(*next_nodes[0]) ||
      next_nodes[0]->input_size() != 2 || next_nodes[0]->output_size() != 1 ||
      next_nodes[0]->input()[0].value() != candidate.output()[0].value()) {
    return NoMatch(candidate, "the reduction is not followed by exactly one Reshape");
  }
  const NodeProto *reshape = next_nodes[0];
  if (!HasRank(graph, reshape->output()[0].value()) ||
      Rank(graph, candidate.input()[0].value()) !=
          Rank(graph, reshape->output()[0].value()) + axes.size()) {
    return NoMatch(candidate, "the reshape rank is inconsistent with the reduced axes");
  }
  if (Rank(graph, reshape->output()[0].value()) > 1) {
    if (!graph.HasShape(candidate.input()[0].value()) ||
        !graph.HasShape(reshape->output()[0].value())) {
      return NoMatch(candidate, "the reduction input or reshape output shape is unknown");
    }
    const SymShape &input_shape = graph.GetShape(candidate.input()[0].value()).Shape();
    const SymShape &reshape_shape = graph.GetShape(reshape->output()[0].value()).Shape();
    std::vector<SymDim> expected;
    for (std::size_t i = 0; i < input_shape.Rank(); ++i) {
      if (!HasInt(axes, static_cast<int64_t>(i))) {
        expected.push_back(input_shape[i]);
      }
    }
    if (expected != reshape_shape.Dims()) {
      return NoMatch(candidate, "the reshape output does not equal the reduced input shape");
    }
  }
  return core::builder::MatchResult{this, {&candidate, reshape}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ReduceReshapePattern::Apply(core::builder::GraphGraph &graph,
                            const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ReduceReshapePattern::Apply expects a reduction and a Reshape node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ReduceReshapePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &reduce = *nodes[0];
  const NodeProto &reshape = *nodes[1];
  NodeProto replacement = MakeReplacement(reduce.op_type().value().c_str(), Inputs(reduce),
                                          {reshape.output()[0].value()},
                                          "ReduceReshapePattern--" + reduce.name().value(), reduce);
  CopyAttributesExcept(reduce, replacement, "keepdims");
  AddAttribute<int64_t>(replacement, "keepdims", 0);
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> ReshapeReshapePattern::FastOpType() const { return {"Reshape"}; }

core::builder::MatchResult ReshapeReshapePattern::Match(core::builder::GraphGraph &graph,
                                                        const NodeProto &candidate) const {
  if (!IsReshape(candidate) || candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Reshape with two inputs");
  }
  if (graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
    return NoMatch(candidate, "the first Reshape output is shared");
  }
  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.output()[0].value());
  if (next_nodes.size() != 1 || next_nodes[0] == nullptr || !IsReshape(*next_nodes[0]) ||
      next_nodes[0]->input_size() != 2 || next_nodes[0]->output_size() != 1 ||
      next_nodes[0]->input()[0].value() != candidate.output()[0].value()) {
    return NoMatch(candidate, "the first Reshape is not followed by exactly one Reshape");
  }
  const NodeProto *next = next_nodes[0];
  std::vector<int64_t> first_shape;
  std::vector<int64_t> second_shape;
  const bool first_constant = ReadConstantInts(graph, candidate.input()[1].value(), first_shape);
  const bool second_constant = ReadConstantInts(graph, next->input()[1].value(), second_shape);

  if (second_constant && !HasInt(second_shape, 0)) {
    return core::builder::MatchResult{this, {&candidate, next}, next};
  }
  if (first_constant && second_constant) {
    if (std::all_of(second_shape.begin(), second_shape.end(),
                    [](int64_t value) { return value > 0; })) {
      return core::builder::MatchResult{this, {&candidate, next}, next};
    }
    if (!first_shape.empty() && !second_shape.empty() && first_shape[0] == second_shape[0]) {
      std::size_t common = 0;
      while (common < first_shape.size() && common < second_shape.size() &&
             first_shape[common] == second_shape[common] && first_shape[common] >= 0) {
        ++common;
      }
      std::vector<int64_t> expected(common, 0);
      expected.push_back(-1);
      if (expected == second_shape && second_shape.size() <= first_shape.size()) {
        return core::builder::MatchResult{this, {&candidate, next}, next};
      }
      if (common < second_shape.size() &&
          std::all_of(
              second_shape.begin() + static_cast<std::ptrdiff_t>(common), second_shape.end(),
              [](int64_t value) { return value > 0; })) {
        return core::builder::MatchResult{this, {&candidate, next}, next};
      }
    }
    if (std::all_of(first_shape.begin(), first_shape.end(),
                    [](int64_t value) { return value > 0; }) &&
        HasInt(second_shape, 0) && second_shape.size() <= first_shape.size()) {
      return core::builder::MatchResult{this, {&candidate, next}, next};
    }
  }
  if (first_constant && HasInt(first_shape, -1) &&
      (!second_constant || std::any_of(second_shape.begin(), second_shape.end(),
                                       [](int64_t value) { return value <= 0; }))) {
    return NoMatch(candidate, "an inferred first shape requires a positive constant second shape");
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.output()[0].value()) ||
      !graph.HasShape(next->output()[0].value())) {
    return NoMatch(candidate, "a shape required to compose the Reshapes is unknown");
  }
  if (!graph.HasShape(candidate.input()[1].value()) || !graph.HasShape(next->input()[1].value())) {
    return NoMatch(candidate, "a symbolic Reshape target shape is unknown");
  }

  const core::symbolic::SymTensor &first_value = graph.GetShape(candidate.input()[1].value());
  const core::symbolic::SymTensor &second_value = graph.GetShape(next->input()[1].value());
  const SymShape *first_value_shape =
      first_value.HasValueAsShape() ? &first_value.ValueAsShape() : nullptr;
  const SymShape *second_value_shape =
      second_value.HasValueAsShape() ? &second_value.ValueAsShape() : nullptr;
  const bool first_has_inferred =
      first_value_shape != nullptr &&
      std::any_of(first_value_shape->Dims().begin(), first_value_shape->Dims().end(),
                  [](const SymDim &value) { return value.IsInt() && value.AsInt() == -1; });
  const bool second_only_inferred =
      second_value_shape == nullptr ||
      (std::any_of(second_value_shape->Dims().begin(), second_value_shape->Dims().end(),
                   [](const SymDim &value) { return value.IsInt() && value.AsInt() == -1; }) &&
       !std::any_of(second_value_shape->Dims().begin(), second_value_shape->Dims().end(),
                    [](const SymDim &value) { return value.IsInt() && value.AsInt() == 0; }));
  if (first_has_inferred && second_only_inferred) {
    return NoMatch(candidate, "both shape values require inferred dimensions");
  }
  if (second_constant &&
      !ApplicableReshape(graph.GetShape(candidate.input()[0].value()).Shape(), second_shape) &&
      (first_constant ||
       Rank(graph, candidate.output()[0].value()) != Rank(graph, next->output()[0].value()))) {
    return NoMatch(candidate, "the second target cannot be safely applied to the original input");
  }
  return core::builder::MatchResult{this, {&candidate, next}, next};
}

utils::RepeatedProtoField<NodeProto>
ReshapeReshapePattern::Apply(core::builder::GraphGraph &graph,
                             const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ReshapeReshapePattern::Apply expects two Reshape nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ReshapeReshapePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &first = *nodes[0];
  const NodeProto &second = *nodes[1];
  std::vector<int64_t> first_shape;
  std::vector<int64_t> second_shape;
  const bool first_constant = ReadConstantInts(graph, first.input()[1].value(), first_shape);
  const bool second_constant = ReadConstantInts(graph, second.input()[1].value(), second_shape);
  std::string target = second.input()[1].value();

  if (second_constant) {
    bool valid = true;
    if (HasInt(second_shape, 0)) {
      if (!first_constant || first_shape.size() < second_shape.size()) {
        valid = false;
      } else if (std::all_of(first_shape.begin(), first_shape.end(),
                             [](int64_t value) { return value > 0; })) {
        std::vector<int64_t> substituted = second_shape;
        for (std::size_t i = 0; i < substituted.size(); ++i) {
          if (substituted[i] == 0) {
            substituted[i] = first_shape[i];
          }
        }
        target = AddShapeInitializer(graph, "ReshapeReshapePattern_zero_sub", substituted);
      }
    }
    if (valid && !HasInt(second_shape, 0)) {
      utils::RepeatedProtoField<NodeProto> replacements;
      replacements.push_back(MakeReplacement(
          "Reshape", {first.input()[0].value(), target}, {second.output()[0].value()},
          "ReshapeReshapePattern--" + second.name().value(), second));
      return replacements;
    }
    if (valid && first_constant &&
        std::all_of(first_shape.begin(), first_shape.end(),
                    [](int64_t value) { return value > 0; })) {
      utils::RepeatedProtoField<NodeProto> replacements;
      replacements.push_back(MakeReplacement(
          "Reshape", {first.input()[0].value(), target}, {second.output()[0].value()},
          "ReshapeReshapePattern--" + second.name().value(), second));
      return replacements;
    }
  }

  utils::RepeatedProtoField<NodeProto> replacements;
  if (second_constant &&
      Rank(graph, first.input()[0].value()) != Rank(graph, second.output()[0].value()) &&
      Rank(graph, first.output()[0].value()) == Rank(graph, second.output()[0].value()) &&
      HasInt(second_shape, 0)) {
    if (first_constant) {
      std::vector<int64_t> substituted = second_shape;
      for (std::size_t i = 0; i < substituted.size(); ++i) {
        if (substituted[i] == 0) {
          substituted[i] = first_shape[i];
        }
      }
      target = AddShapeInitializer(graph, "ReshapeReshapePattern_shape", substituted);
    } else {
      std::vector<std::string> concat_inputs;
      for (std::size_t i = 0; i < second_shape.size(); ++i) {
        if (second_shape[i] == 0) {
          const std::string axis = AddShapeInitializer(
              graph, "ReshapeReshapePattern_axis_" + std::to_string(i), {static_cast<int64_t>(i)});
          const std::string gather_out =
              graph.Builder().UniqueName(second.input()[0].value() + "--dim" + std::to_string(i));
          NodeProto gather =
              MakeReplacement("Gather", {first.input()[1].value(), axis}, {gather_out},
                              "ReshapeReshapePattern--" + second.name().value(), second);
          AddAttribute<int64_t>(gather, "axis", 0);
          replacements.push_back(std::move(gather));
          concat_inputs.push_back(gather_out);
        } else {
          concat_inputs.push_back(AddShapeInitializer(
              graph, "ReshapeReshapePattern_dim_" + std::to_string(i), {second_shape[i]}));
        }
      }
      target = graph.Builder().UniqueName(second.input()[0].value() + "--concat");
      NodeProto concat = MakeReplacement("Concat", concat_inputs, {target},
                                         "ReshapeReshapePattern--" + second.name().value(), second);
      AddAttribute<int64_t>(concat, "axis", 0);
      replacements.push_back(std::move(concat));
    }
  } else if (second_constant && !std::all_of(second_shape.begin(), second_shape.end(),
                                             [](int64_t value) { return value > 0; })) {
    std::vector<int64_t> applicable = second_shape;
    if (HasInt(applicable, 0)) {
      for (int64_t &value : applicable) {
        if (value == 0) {
          value = -1;
        }
      }
      if (std::count(applicable.begin(), applicable.end(), -1) > 1) {
        throw BuilderError(
            "ReshapeReshapePattern::Apply would create more than one inferred dimension.");
      }
    }
    if (applicable != second_shape) {
      target = AddShapeInitializer(graph, "ReshapeReshapePattern_applicable", applicable);
    }
  }
  replacements.push_back(MakeReplacement("Reshape", {first.input()[0].value(), target},
                                         {second.output()[0].value()},
                                         "ReshapeReshapePattern--" + first.name().value(), second));
  return replacements;
}

std::set<std::string> ReshapeReshapeBinaryPattern::FastOpType() const {
  return {"Add", "And", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "Div", "Max", "Mean",
          "Min", "Mul", "Mod",        "Or",        "Sub",        "Sum", "Xor"};
}

core::builder::MatchResult ReshapeReshapeBinaryPattern::Match(core::builder::GraphGraph &graph,
                                                              const NodeProto &candidate) const {
  if (!IsBinaryElementWise(candidate) || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain element-wise binary operator");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value()) ||
      graph.IsUsedMoreThanOnce(candidate.input()[1].value())) {
    return NoMatch(candidate, "a binary input reshape output is shared");
  }
  const NodeProto *left = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *right = graph.NodeBefore(candidate.input()[1].value());
  if (left == nullptr || right == nullptr || !IsReshape(*left) || !IsReshape(*right) ||
      left->input_size() != 2 || right->input_size() != 2 || left->output_size() != 1 ||
      right->output_size() != 1) {
    return NoMatch(candidate, "both binary inputs are not produced by Reshape nodes");
  }
  std::vector<int64_t> left_target;
  std::vector<int64_t> right_target;
  if (!ReadConstantInts(graph, left->input()[1].value(), left_target) ||
      !ReadConstantInts(graph, right->input()[1].value(), right_target) ||
      left_target != right_target) {
    return NoMatch(candidate, "the two Reshape targets are not equal constants");
  }
  if (!graph.HasShape(left->input()[0].value()) || !graph.HasShape(right->input()[0].value()) ||
      graph.GetShape(left->input()[0].value()).Shape() !=
          graph.GetShape(right->input()[0].value()).Shape()) {
    return NoMatch(candidate, "the binary inputs do not have equal source shapes");
  }
  return core::builder::MatchResult{this, {left, right, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ReshapeReshapeBinaryPattern::Apply(core::builder::GraphGraph &graph,
                                   const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError(
        "ReshapeReshapeBinaryPattern::Apply expects two Reshapes and a binary node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ReshapeReshapeBinaryPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &left = *nodes[0];
  const NodeProto &right = *nodes[1];
  const NodeProto &binary = *nodes[2];
  const std::string name = "ReshapeReshapeBinaryPattern--" + binary.name().value();
  const std::string binary_output = graph.Builder().UniqueName(name + "_binary");
  NodeProto new_binary = MakeReplacement(binary.op_type().value().c_str(),
                                         {left.input()[0].value(), right.input()[0].value()},
                                         {binary_output}, name, binary);
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(new_binary));
  replacements.push_back(MakeReplacement("Reshape", {binary_output, left.input()[1].value()},
                                         {binary.output()[0].value()}, name, binary));
  return replacements;
}

std::set<std::string> Reshape2Of3Pattern::FastOpType() const {
  return {"Add", "And", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "Div", "Max", "Mean",
          "Min", "Mul", "Mod",        "Or",        "Sub",        "Sum", "Xor"};
}

core::builder::MatchResult Reshape2Of3Pattern::Match(core::builder::GraphGraph &graph,
                                                     const NodeProto &candidate) const {
  if (!IsBinaryElementWise(candidate) || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain element-wise binary operator");
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.input()[1].value()) ||
      !graph.HasShape(candidate.output()[0].value())) {
    return NoMatch(candidate, "the binary input or output shape is unknown");
  }
  const SymShape &output_shape = graph.GetShape(candidate.output()[0].value()).Shape();
  if (output_shape != graph.GetShape(candidate.input()[0].value()).Shape() ||
      output_shape != graph.GetShape(candidate.input()[1].value()).Shape()) {
    return NoMatch(candidate, "the binary operation involves broadcasting");
  }
  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.output()[0].value());
  if (next_nodes.size() > 1 ||
      (next_nodes.empty() && !graph.IsOutput(candidate.output()[0].value()))) {
    return NoMatch(candidate, "the binary output is shared or unused");
  }
  const NodeProto *next = next_nodes.empty() ? nullptr : next_nodes[0];
  const NodeProto *left = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *right = graph.NodeBefore(candidate.input()[1].value());
  const bool left_reshape =
      left != nullptr && IsReshape(*left) && left->input_size() == 2 && left->output_size() == 1;
  const bool right_reshape = right != nullptr && IsReshape(*right) && right->input_size() == 2 &&
                             right->output_size() == 1;
  const bool next_reshape = next != nullptr && IsReshape(*next) && next->input_size() == 2 &&
                            next->output_size() == 1 &&
                            next->input()[0].value() == candidate.output()[0].value();
  const int reshape_count = static_cast<int>(left_reshape) + static_cast<int>(right_reshape) +
                            static_cast<int>(next_reshape);
  if (reshape_count < 2) {
    return NoMatch(candidate, "fewer than two of the surrounding nodes are Reshapes");
  }
  left = left_reshape ? left : nullptr;
  right = right_reshape ? right : nullptr;
  next = next_reshape ? next : nullptr;

  std::vector<const SymShape *> surrounding_shapes;
  if (left != nullptr) {
    if (!graph.HasShape(left->input()[0].value())) {
      return NoMatch(candidate, "the left Reshape source shape is unknown");
    }
    surrounding_shapes.push_back(&graph.GetShape(left->input()[0].value()).Shape());
  }
  if (right != nullptr) {
    if (!graph.HasShape(right->input()[0].value())) {
      return NoMatch(candidate, "the right Reshape source shape is unknown");
    }
    surrounding_shapes.push_back(&graph.GetShape(right->input()[0].value()).Shape());
  }
  if (next != nullptr) {
    if (!graph.HasShape(next->output()[0].value())) {
      return NoMatch(candidate, "the output Reshape target shape is unknown");
    }
    surrounding_shapes.push_back(&graph.GetShape(next->output()[0].value()).Shape());
  }
  if (surrounding_shapes.size() < 2 ||
      !std::all_of(surrounding_shapes.begin() + 1, surrounding_shapes.end(),
                   [&](const SymShape *shape) { return *shape == *surrounding_shapes.front(); })) {
    return NoMatch(candidate, "the surrounding Reshapes do not have equal outer shapes");
  }
  return core::builder::MatchResult{this, {left, right, next, &candidate}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
Reshape2Of3Pattern::Apply(core::builder::GraphGraph &graph,
                          const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 4 || nodes[3] == nullptr) {
    throw BuilderError(
        "Reshape2Of3Pattern::Apply expects optional left/right/output Reshapes and a binary node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[3]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("Reshape2Of3Pattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto *left_reshape = nodes[0];
  const NodeProto *right_reshape = nodes[1];
  const NodeProto *output_reshape = nodes[2];
  const NodeProto &binary = *nodes[3];
  const std::string compute_shape = left_reshape != nullptr ? left_reshape->input()[1].value()
                                                            : right_reshape->input()[1].value();
  const std::string final_shape =
      output_reshape == nullptr ? compute_shape : output_reshape->input()[1].value();
  const std::string name = "Reshape2Of3Pattern--" + binary.name().value();

  utils::RepeatedProtoField<NodeProto> replacements;
  std::string left_name;
  if (left_reshape == nullptr) {
    left_name = graph.Builder().UniqueName("Reshape2Of3PatternL_" + binary.input()[0].value());
    replacements.push_back(MakeReplacement("Reshape", {binary.input()[0].value(), final_shape},
                                           {left_name}, name, binary));
  } else if (graph.IsUsedMoreThanOnce(left_reshape->output()[0].value())) {
    replacements.add() = *left_reshape;
    left_name = left_reshape->input()[0].value();
  } else {
    left_name = left_reshape->input()[0].value();
  }

  std::string right_name;
  if (right_reshape == nullptr) {
    right_name = graph.Builder().UniqueName("Reshape2Of3PatternR_" + binary.input()[1].value());
    replacements.push_back(MakeReplacement("Reshape", {binary.input()[1].value(), final_shape},
                                           {right_name}, name, binary));
  } else if (graph.IsUsedMoreThanOnce(right_reshape->output()[0].value())) {
    replacements.add() = *right_reshape;
    right_name = right_reshape->input()[0].value();
  } else {
    right_name = right_reshape->input()[0].value();
  }

  if (output_reshape == nullptr) {
    const std::string binary_out =
        graph.Builder().UniqueName("Reshape2Of3PatternO_" + binary.output()[0].value());
    replacements.push_back(MakeReplacement(binary.op_type().value().c_str(),
                                           {left_name, right_name}, {binary_out}, name, binary));
    replacements.push_back(MakeReplacement("Reshape", {binary_out, final_shape},
                                           {binary.output()[0].value()}, name, binary));
  } else {
    NodeProto moved = MakeReplacement(binary.op_type().value().c_str(), {left_name, right_name},
                                      {output_reshape->output()[0].value()}, name, binary);
    replacements.push_back(std::move(moved));
    if (graph.IsUsedMoreThanOnce(binary.output()[0].value())) {
      replacements.push_back(MakeReplacement("Reshape",
                                             {output_reshape->output()[0].value(), compute_shape},
                                             {binary.output()[0].value()}, name, binary));
    }
  }
  return replacements;
}

namespace {

struct EditStep {
  std::size_t di = 0;
  std::size_t dj = 0;
  std::size_t pi = 0;
  std::size_t pj = 0;
  bool valid = false;
};

bool AllInts(const SymShape &shape, std::size_t first, std::size_t count) {
  for (std::size_t i = first; i < first + count; ++i) {
    if (!shape[i].IsInt()) {
      return false;
    }
  }
  return true;
}

std::optional<std::vector<int64_t>> AlignShapes(const SymShape &first, const SymShape &second) {
  const std::size_t rows = first.Rank();
  const std::size_t cols = second.Rank();
  const double infinity = static_cast<double>(std::max(rows, cols) + 10);
  std::vector<std::vector<double>> distance(rows + 1, std::vector<double>(cols + 1, infinity));
  std::vector<std::vector<EditStep>> predecessor(rows + 1, std::vector<EditStep>(cols + 1));
  distance[0][0] = 0.0;

  for (std::size_t i = 1; i <= rows; ++i) {
    for (std::size_t j = 1; j <= cols; ++j) {
      const double compare_cost = first[i - 1] == second[j - 1]
                                      ? 0.0
                                      : (first[i - 1].IsInt() && second[j - 1].IsInt() ? 1.0 : 0.5);
      double best = distance[i - 1][j - 1] + compare_cost;
      EditStep best_step{1, 1, i - 1, j - 1, true};
      for (std::size_t di = 1; di <= 4 && di <= i; ++di) {
        for (std::size_t dj = 1; dj <= 4 && dj <= j; ++dj) {
          if (di == 1 && dj == 1) {
            continue;
          }
          if ((i - di == 0 && j - dj != 0) || (i - di != 0 && j - dj == 0)) {
            continue;
          }
          const std::optional<int64_t> lhs = Product(first, i - di, di);
          const std::optional<int64_t> rhs = Product(second, j - dj, dj);
          double cost = infinity;
          if (lhs.has_value() && rhs.has_value()) {
            if (*lhs == *rhs) {
              cost = distance[i - di][j - dj];
            }
          } else {
            int symbols_lhs = 0;
            int symbols_rhs = 0;
            for (std::size_t k = i - di; k < i; ++k) {
              symbols_lhs += !first[k].IsInt();
            }
            for (std::size_t k = j - dj; k < j; ++k) {
              symbols_rhs += !second[k].IsInt();
            }
            if (symbols_lhs <= 1 && symbols_rhs <= 1) {
              cost = distance[i - di][j - dj] + 0.5;
            }
          }
          if (cost < best) {
            best = cost;
            best_step = EditStep{di, dj, i - di, j - dj, true};
          }
        }
      }
      distance[i][j] = best;
      predecessor[i][j] = best_step;
    }
  }
  if (distance[rows][cols] >= 1.0 || !predecessor[rows][cols].valid) {
    return std::nullopt;
  }

  std::vector<EditStep> path;
  std::size_t i = rows;
  std::size_t j = cols;
  while (i != 0 || j != 0) {
    const EditStep step = predecessor[i][j];
    if (!step.valid) {
      return std::nullopt;
    }
    path.push_back(step);
    i = step.pi;
    j = step.pj;
  }
  std::reverse(path.begin(), path.end());

  std::vector<int64_t> result;
  int inferred = 0;
  for (const EditStep &step : path) {
    if (AllInts(second, step.pj, step.dj)) {
      for (std::size_t k = step.pj; k < step.pj + step.dj; ++k) {
        result.push_back(second[k].AsInt());
      }
    } else if (AllInts(first, step.pi, step.di)) {
      if (step.dj != 1) {
        return std::nullopt;
      }
      const std::optional<int64_t> product = Product(first, step.pi, step.di);
      if (!product.has_value()) {
        return std::nullopt;
      }
      result.push_back(*product);
    } else if (step.di == 1 && step.dj == 1) {
      result.push_back(step.pi == step.pj && first[step.pi] == second[step.pj] ? 0 : -1);
      inferred += result.back() == -1;
    } else {
      for (std::size_t k = step.pj; k < step.pj + step.dj; ++k) {
        if (second[k].IsInt()) {
          result.push_back(second[k].AsInt());
        } else {
          result.push_back(-1);
          ++inferred;
        }
      }
    }
  }
  return inferred > 1 ? std::nullopt : std::optional<std::vector<int64_t>>(std::move(result));
}

std::vector<SymDim> RemoveUnitDimensions(const SymShape &shape) {
  std::vector<SymDim> result;
  for (const SymDim &dimension : shape.Dims()) {
    if (!dimension.IsInt() || dimension.AsInt() != 1) {
      result.push_back(dimension);
    }
  }
  return result;
}

std::optional<std::pair<std::string, std::vector<int64_t>>>
SqueezeOrUnsqueezeAxes(const SymShape &input, const SymShape &output) {
  if (input == output || RemoveUnitDimensions(input) != RemoveUnitDimensions(output)) {
    return std::nullopt;
  }
  const SymShape &with_units = input.Rank() < output.Rank() ? output : input;
  const SymShape &without_units = input.Rank() < output.Rank() ? input : output;
  if (RemoveUnitDimensions(with_units) != without_units.Dims()) {
    return std::nullopt;
  }
  std::vector<int64_t> axes;
  for (std::size_t i = 0; i < with_units.Rank(); ++i) {
    if (with_units[i].IsInt() && with_units[i].AsInt() == 1) {
      axes.push_back(static_cast<int64_t>(i));
    }
  }
  return std::make_pair(input.Rank() < output.Rank() ? "Unsqueeze" : "Squeeze", std::move(axes));
}

} // namespace

std::set<std::string> ShapeBasedEditDistanceReshapePattern::FastOpType() const {
  return {"Reshape"};
}

core::builder::MatchResult
ShapeBasedEditDistanceReshapePattern::Match(core::builder::GraphGraph &graph,
                                            const NodeProto &candidate) const {
  if (!IsReshape(candidate) || candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Reshape with two inputs");
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.output()[0].value())) {
    return NoMatch(candidate, "the reshape input or output shape is unknown");
  }
  const std::optional<std::vector<int64_t>> aligned =
      AlignShapes(graph.GetShape(candidate.input()[0].value()).Shape(),
                  graph.GetShape(candidate.output()[0].value()).Shape());
  if (!aligned.has_value() || aligned->size() != Rank(graph, candidate.output()[0].value())) {
    return NoMatch(candidate, "the input and output shapes cannot be aligned into a static target");
  }
  const NodeProto *producer = graph.NodeBefore(candidate.input()[1].value());
  if (producer == nullptr || !IsDefaultOp(*producer, "Concat")) {
    return NoMatch(candidate, "the reshape target is not produced by a Concat");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ShapeBasedEditDistanceReshapePattern::Apply(core::builder::GraphGraph &graph,
                                            const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("ShapeBasedEditDistanceReshapePattern::Apply expects one Reshape node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ShapeBasedEditDistanceReshapePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &reshape = *nodes[0];
  const std::optional<std::vector<int64_t>> aligned =
      AlignShapes(graph.GetShape(reshape.input()[0].value()).Shape(),
                  graph.GetShape(reshape.output()[0].value()).Shape());
  if (!aligned.has_value()) {
    throw BuilderError(
        "ShapeBasedEditDistanceReshapePattern::Apply could not align matched shapes.");
  }
  const std::string target =
      AddShapeInitializer(graph, "ShapeBasedEditDistanceReshapePattern_shape", *aligned);
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeReplacement(
      "Reshape", {reshape.input()[0].value(), target}, {reshape.output()[0].value()},
      "ShapeBasedEditDistanceReshapePattern--" + reshape.name().value(), reshape));
  return replacements;
}

std::set<std::string> ShapeBasedReshapeIsSqueezePattern::FastOpType() const {
  return {"Expand", "Reshape"};
}

core::builder::MatchResult
ShapeBasedReshapeIsSqueezePattern::Match(core::builder::GraphGraph &graph,
                                         const NodeProto &candidate) const {
  if (!MainOpsetAtLeast(graph, 18)) {
    return NoMatch(candidate, "the model opset is below 18");
  }
  if ((candidate.op_type().value() != "Reshape" && candidate.op_type().value() != "Expand") ||
      NormaliseDomain(candidate.domain().value()) != kDefaultOnnxDomain ||
      candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate,
                   "candidate is not a default-domain Reshape or Expand with two inputs");
  }
  if (candidate.op_type().value() == "Expand") {
    std::vector<int64_t> target;
    if (!ReadConstantInts(graph, candidate.input()[1].value(), target) || target.empty() ||
        !std::all_of(target.begin(), target.end(), [](int64_t value) { return value == 1; })) {
      return NoMatch(candidate, "the Expand target is not an all-one constant shape");
    }
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.output()[0].value()) ||
      !SqueezeOrUnsqueezeAxes(graph.GetShape(candidate.input()[0].value()).Shape(),
                              graph.GetShape(candidate.output()[0].value()).Shape())
           .has_value()) {
    return NoMatch(candidate, "the input and output shapes do not differ only by unit dimensions");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ShapeBasedReshapeIsSqueezePattern::Apply(core::builder::GraphGraph &graph,
                                         const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("ShapeBasedReshapeIsSqueezePattern::Apply expects one node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ShapeBasedReshapeIsSqueezePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &reshape = *nodes[0];
  const auto rewrite = SqueezeOrUnsqueezeAxes(graph.GetShape(reshape.input()[0].value()).Shape(),
                                              graph.GetShape(reshape.output()[0].value()).Shape());
  if (!rewrite.has_value()) {
    throw BuilderError("ShapeBasedReshapeIsSqueezePattern::Apply could not recover matched axes.");
  }
  const std::string axes =
      AddShapeInitializer(graph, "ShapeBasedReshapeIsSqueezePattern_axes", rewrite->second);
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeReplacement(
      rewrite->first.c_str(), {reshape.input()[0].value(), axes}, {reshape.output()[0].value()},
      "ShapeBasedReshapeIsSqueezePattern--" + reshape.name().value(), reshape));
  return replacements;
}

std::set<std::string> UnsqueezeReshapePattern::FastOpType() const { return {"Reshape"}; }

core::builder::MatchResult UnsqueezeReshapePattern::Match(core::builder::GraphGraph &graph,
                                                          const NodeProto &candidate) const {
  if (!IsReshape(candidate) || candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Reshape with two inputs");
  }
  std::vector<int64_t> target;
  if (!ReadConstantInts(graph, candidate.input()[1].value(), target)) {
    return NoMatch(candidate, "the Reshape target is not constant");
  }
  const NodeProto *unsqueeze = graph.NodeBefore(candidate.input()[0].value());
  if (unsqueeze == nullptr || !IsDefaultOp(*unsqueeze, "Unsqueeze") ||
      unsqueeze->input_size() != 2 || unsqueeze->output_size() != 1 ||
      graph.IsUsedMoreThanOnce(candidate.input()[0].value())) {
    return NoMatch(candidate, "the Reshape input is not an unshared two-input Unsqueeze");
  }
  if (!HasRank(graph, unsqueeze->input()[0].value()) ||
      Rank(graph, unsqueeze->input()[0].value()) != target.size() - 1) {
    return NoMatch(candidate, "the Unsqueeze source rank is inconsistent with the Reshape target");
  }
  std::vector<int64_t> axes;
  if (!ReadConstantInts(graph, unsqueeze->input()[1].value(), axes) ||
      axes != std::vector<int64_t>{2} || target != std::vector<int64_t>({0, 1, -1, 0})) {
    return NoMatch(candidate,
                   "the Unsqueeze axes and Reshape target are not the supported special case");
  }
  return core::builder::MatchResult{this, {unsqueeze, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
UnsqueezeReshapePattern::Apply(core::builder::GraphGraph &graph,
                               const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("UnsqueezeReshapePattern::Apply expects an Unsqueeze and a Reshape node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("UnsqueezeReshapePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &unsqueeze = *nodes[0];
  const NodeProto &reshape = *nodes[1];
  const std::string axes = AddShapeInitializer(graph, "UnsqueezeReshapePattern_axes", {1});
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeReplacement(
      "Unsqueeze", {unsqueeze.input()[0].value(), axes}, {reshape.output()[0].value()},
      "UnsqueezeReshapePattern--" + unsqueeze.name().value(), unsqueeze));
  return replacements;
}

std::set<std::string> UnsqueezeOrSqueezeReshapePattern::FastOpType() const { return {"Reshape"}; }

core::builder::MatchResult
UnsqueezeOrSqueezeReshapePattern::Match(core::builder::GraphGraph &graph,
                                        const NodeProto &candidate) const {
  if (!IsReshape(candidate) || candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Reshape with two inputs");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value())) {
    return NoMatch(candidate, "the Reshape input is shared");
  }
  const NodeProto *before = graph.NodeBefore(candidate.input()[0].value());
  if (before == nullptr ||
      (before->op_type().value() != "Squeeze" && before->op_type().value() != "Unsqueeze") ||
      NormaliseDomain(before->domain().value()) != kDefaultOnnxDomain || before->input_size() < 1 ||
      before->output_size() != 1) {
    return NoMatch(candidate, "the Reshape input is not a default-domain Squeeze or Unsqueeze");
  }
  std::vector<int64_t> target;
  if (!ReadConstantInts(graph, candidate.input()[1].value(), target)) {
    return NoMatch(candidate, "the Reshape target is not constant");
  }
  if (HasInt(target, 0)) {
    std::vector<int64_t> axes;
    if (before->input_size() != 2 || !ReadConstantInts(graph, before->input()[1].value(), axes) ||
        axes.empty()) {
      return NoMatch(candidate,
                     "zero-copy target dimensions require constant Squeeze/Unsqueeze axes");
    }
    const int64_t last_zero = static_cast<int64_t>(
        std::find(target.rbegin(), target.rend(), 0).base() - target.begin() - 1);
    int64_t min_axis = *std::min_element(axes.begin(), axes.end());
    if (min_axis < 0) {
      if (!HasRank(graph, before->input()[0].value())) {
        return NoMatch(candidate, "a source rank is required to normalise a negative axis");
      }
      min_axis += static_cast<int64_t>(Rank(graph, before->input()[0].value()));
    }
    if (last_zero >= min_axis) {
      return NoMatch(candidate,
                     "a zero-copy target dimension follows a squeezed or unsqueezed axis");
    }
  }
  return core::builder::MatchResult{this, {before, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
UnsqueezeOrSqueezeReshapePattern::Apply(core::builder::GraphGraph &graph,
                                        const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError(
        "UnsqueezeOrSqueezeReshapePattern::Apply expects a Squeeze/Unsqueeze and a Reshape node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "UnsqueezeOrSqueezeReshapePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &before = *nodes[0];
  const NodeProto &reshape = *nodes[1];
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(
      MakeReplacement("Reshape", {before.input()[0].value(), reshape.input()[1].value()},
                      {reshape.output()[0].value()},
                      "UnsqueezeOrSqueezeReshapePattern--" + reshape.name().value(), reshape));
  return replacements;
}

std::set<std::string> ReshapeSqueezePattern::FastOpType() const { return {"Squeeze"}; }

core::builder::MatchResult ReshapeSqueezePattern::Match(core::builder::GraphGraph &graph,
                                                        const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Squeeze") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain Squeeze with explicit axes");
  }
  std::vector<int64_t> axes;
  if (!ReadConstantInts(graph, candidate.input()[1].value(), axes) || axes.empty()) {
    return NoMatch(candidate, "the Squeeze axes are not a non-empty constant");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value())) {
    return NoMatch(candidate, "the Squeeze input is shared");
  }
  const NodeProto *reshape = graph.NodeBefore(candidate.input()[0].value());
  if (reshape == nullptr || !IsReshape(*reshape) || reshape->input_size() != 2 ||
      reshape->output_size() != 1) {
    return NoMatch(candidate, "the Squeeze input is not produced by a Reshape");
  }
  std::vector<int64_t> shape;
  if (!ReadConstantInts(graph, reshape->input()[1].value(), shape) || shape.empty()) {
    return NoMatch(candidate, "the Reshape target is not a non-empty constant");
  }
  const int64_t rank = static_cast<int64_t>(shape.size());
  std::set<int64_t> normalised;
  for (int64_t axis : axes) {
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      return NoMatch(candidate, "a Squeeze axis is out of range for the Reshape target");
    }
    if (shape[static_cast<std::size_t>(axis)] != 1) {
      return NoMatch(candidate,
                     "the Reshape does not introduce a unit dimension at every Squeeze axis");
    }
    normalised.insert(axis);
  }
  const int64_t first_axis = *normalised.begin();
  for (int64_t i = first_axis + 1; i < rank; ++i) {
    if (!normalised.contains(i) && shape[static_cast<std::size_t>(i)] == 0) {
      return NoMatch(candidate, "a copied Reshape dimension follows a removed axis");
    }
  }
  return core::builder::MatchResult{this, {reshape, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ReshapeSqueezePattern::Apply(core::builder::GraphGraph &graph,
                             const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("ReshapeSqueezePattern::Apply expects a Reshape and a Squeeze node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("ReshapeSqueezePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &reshape = *nodes[0];
  const NodeProto &squeeze = *nodes[1];
  std::vector<int64_t> shape;
  std::vector<int64_t> axes;
  if (!ReadConstantInts(graph, reshape.input()[1].value(), shape) ||
      !ReadConstantInts(graph, squeeze.input()[1].value(), axes)) {
    throw BuilderError("ReshapeSqueezePattern::Apply could not read matched constants.");
  }
  const int64_t rank = static_cast<int64_t>(shape.size());
  std::set<int64_t> normalised;
  for (int64_t axis : axes) {
    normalised.insert(axis < 0 ? axis + rank : axis);
  }
  std::vector<int64_t> target;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (!normalised.contains(static_cast<int64_t>(i))) {
      target.push_back(shape[i]);
    }
  }
  const std::string target_name = AddShapeInitializer(graph, "ReshapeSqueezePattern_shape", target);
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeReplacement(
      "Reshape", {reshape.input()[0].value(), target_name}, {squeeze.output()[0].value()},
      "ReshapeSqueezePattern--" + reshape.name().value(), reshape));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
