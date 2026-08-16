// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/canonicalization/identity_pattern.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;
using core::symbolic::TensorType;

const std::set<std::string> &BinaryOps() {
  static const std::set<std::string> ops = {"Add", "Mul", "Div", "Sub", "And", "Or"};
  return ops;
}

bool IsDefaultDomain(const NodeProto &node) {
  return NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain;
}

// Reads all values of the constant ``name`` as doubles. Returns ``false`` when
// the value is not a materialised numeric constant.
bool ReadConstDoubles(core::builder::GraphGraph &graph, const std::string &name,
                      std::vector<double> &out) {
  const TensorProto *tensor = graph.GetComputedConstant(name);
  if (tensor == nullptr) {
    return false;
  }
  if (ReadFloatingValues(*tensor, out)) {
    return true;
  }
  std::vector<int64_t> integers;
  if (ReadIntegerValues(*tensor, integers)) {
    out.assign(integers.begin(), integers.end());
    return true;
  }
  return false;
}

bool ReadConstInts(core::builder::GraphGraph &graph, const std::string &name,
                   std::vector<int64_t> &out) {
  const TensorProto *tensor = graph.GetComputedConstant(name);
  if (tensor == nullptr) {
    return false;
  }
  return ReadIntegerValues(*tensor, out);
}

bool HasRank(core::builder::GraphGraph &graph, const std::string &name, std::size_t &rank) {
  if (!graph.HasShape(name)) {
    return false;
  }
  rank = graph.GetShape(name).Shape().Rank();
  return true;
}

bool AllEqual(const std::vector<double> &values, double target) {
  if (values.empty()) {
    return false;
  }
  for (double value : values) {
    if (value != target) {
      return false;
    }
  }
  return true;
}

bool AllUnique(const std::vector<double> &values, double &unique) {
  if (values.empty()) {
    return false;
  }
  unique = values[0];
  for (double value : values) {
    if (value != unique) {
      return false;
    }
  }
  return true;
}

// Returns ``true`` when the known shape of ``name`` is a scalar, i.e. rank 0 or
// rank 1 with a single element.
bool IsScalarShape(core::builder::GraphGraph &graph, const std::string &name) {
  if (!graph.HasShape(name)) {
    return false;
  }
  const core::symbolic::SymShape &shape = graph.GetShape(name).Shape();
  if (shape.Rank() == 0) {
    return true;
  }
  return shape.Rank() == 1 && shape[0].IsInt() && shape[0].AsInt() == 1;
}

} // namespace

std::set<std::string> IdentityPattern::FastOpType() const {
  return {"Add",
          "Mul",
          "Div",
          "Sub",
          "And",
          "Or",
          "Transpose",
          "Slice",
          "Expand",
          "Reshape",
          "BatchNormalization"};
}

core::builder::MatchResult IdentityPattern::Match(core::builder::GraphGraph &graph,
                                                  const NodeProto &candidate) const {
  const std::string &op_type = candidate.op_type().value();
  if (!IsDefaultDomain(candidate) || FastOpType().count(op_type) == 0) {
    return NoMatch(candidate, "candidate is not a supported default-domain operation");
  }

  if (op_type == "Slice") {
    if (candidate.input_size() < 3) {
      return NoMatch(candidate, "the Slice node has no starts/ends inputs");
    }
    if (candidate.input_size() == 5) {
      std::vector<double> steps;
      if (!graph.IsConstant(candidate.input()[4].value()) ||
          !ReadConstDoubles(graph, candidate.input()[4].value(), steps) || !AllEqual(steps, 1.0)) {
        return NoMatch(candidate, "the Slice steps are not all ones");
      }
    }
    std::vector<double> starts;
    if (!graph.IsConstant(candidate.input()[1].value()) ||
        !ReadConstDoubles(graph, candidate.input()[1].value(), starts) || !AllEqual(starts, 0.0)) {
      return NoMatch(candidate, "the Slice starts are not all zeros");
    }
    std::vector<int64_t> ends;
    if (!graph.IsConstant(candidate.input()[2].value()) ||
        !ReadConstInts(graph, candidate.input()[2].value(), ends) ||
        !AllEqual(std::vector<double>(ends.begin(), ends.end()),
                  static_cast<double>(std::numeric_limits<int64_t>::max()))) {
      return NoMatch(candidate, "the Slice ends do not select the full range");
    }
    return core::builder::MatchResult{this, {&candidate}, &candidate};
  }

  if (op_type == "Reshape") {
    if (candidate.input_size() < 2 || !graph.IsConstant(candidate.input()[1].value())) {
      return NoMatch(candidate, "the Reshape shape is not constant");
    }
    std::vector<double> shape;
    if (!ReadConstDoubles(graph, candidate.input()[1].value(), shape) || !AllEqual(shape, 0.0)) {
      return NoMatch(candidate, "the Reshape shape is not all zeros");
    }
    std::size_t rank = 0;
    if (!HasRank(graph, candidate.input()[0].value(), rank) || rank != shape.size()) {
      return NoMatch(candidate, "the Reshape shape rank differs from the input rank");
    }
    return core::builder::MatchResult{this, {&candidate}, &candidate};
  }

  if (op_type == "Transpose") {
    std::vector<int64_t> perm;
    if (!GetAttributeInts(candidate, "perm", perm)) {
      return NoMatch(candidate, "the Transpose node has no perm attribute");
    }
    for (std::size_t i = 0; i < perm.size(); ++i) {
      if (perm[i] != static_cast<int64_t>(i)) {
        return NoMatch(candidate, "the Transpose perm is not the identity permutation");
      }
    }
    return core::builder::MatchResult{this, {&candidate}, &candidate};
  }

  if (op_type == "Expand") {
    if (candidate.input_size() < 2 || !graph.IsConstant(candidate.input()[1].value())) {
      return NoMatch(candidate, "the Expand shape is not constant");
    }
    std::vector<double> shape;
    if (!ReadConstDoubles(graph, candidate.input()[1].value(), shape)) {
      return NoMatch(candidate, "the Expand shape cannot be read");
    }
    std::size_t rank = 0;
    if (!HasRank(graph, candidate.input()[0].value(), rank) || rank != shape.size()) {
      return NoMatch(candidate, "the Expand shape rank differs from the input rank");
    }
    if (!AllEqual(shape, 1.0)) {
      return NoMatch(candidate, "the Expand shape is not all ones");
    }
    return core::builder::MatchResult{this, {&candidate}, &candidate};
  }

  if (op_type == "BatchNormalization") {
    if (candidate.input_size() < 5) {
      return NoMatch(candidate, "the BatchNormalization node has too few inputs");
    }
    if (!graph.HasType(candidate.input()[0].value())) {
      return NoMatch(candidate, "the BatchNormalization input type is unknown");
    }
    const TensorType type = graph.GetType(candidate.input()[0].value());
    if (type != TensorType::kFloat16 && type != TensorType::kBfloat16) {
      return NoMatch(candidate, "the BatchNormalization input is not half precision");
    }
    if (GetAttributeOr<int64_t>(candidate, "training_mode", 0) != 0 ||
        GetAttributeOr<float>(candidate, "epsilon", 1e-5f) > 1e-5f) {
      return NoMatch(candidate, "the BatchNormalization node is not in inference mode");
    }
    const double expected[4] = {1.0, 0.0, 0.0, 1.0};
    for (int i = 1; i < 5; ++i) {
      if (!graph.IsConstant(candidate.input()[i].value())) {
        return NoMatch(candidate, "a BatchNormalization parameter is not constant");
      }
      std::vector<double> values;
      double unique = 0.0;
      if (!ReadConstDoubles(graph, candidate.input()[i].value(), values) ||
          !AllUnique(values, unique) || unique != expected[i - 1]) {
        return NoMatch(candidate, "the BatchNormalization parameters are not identity");
      }
    }
    return core::builder::MatchResult{this, {&candidate}, &candidate};
  }

  // Binary arithmetic operations Add / Mul / Div / Sub.
  if (candidate.input_size() != 2) {
    return NoMatch(candidate, "the binary operation does not have two inputs");
  }
  const std::string &input0 = candidate.input()[0].value();
  const std::string &input1 = candidate.input()[1].value();

  if (graph.IsConstant(input1)) {
    std::size_t rank = 0;
    if (HasRank(graph, input1, rank) && rank > 1) {
      return NoMatch(candidate, "the constant operand has rank greater than one");
    }
    if (IsScalarShape(graph, input1)) {
      if (graph.IsConstantScalar(input1, 0.0, false) &&
          (op_type == "Add" || op_type == "Sub" || op_type == "Or")) {
        return core::builder::MatchResult{this, {&candidate}, &candidate};
      }
      if (graph.IsConstantScalar(input1, 1.0, false) &&
          (op_type == "Mul" || op_type == "Div" || op_type == "And")) {
        return core::builder::MatchResult{this, {&candidate}, &candidate};
      }
      return NoMatch(candidate, "the scalar constant operand is not a neutral element");
    }
    if (HasRank(graph, input1, rank) && rank == 1 &&
        (op_type == "Add" || op_type == "Mul" || op_type == "Sub" || op_type == "Div")) {
      std::vector<double> values;
      double unique = 0.0;
      if (!ReadConstDoubles(graph, input1, values) || !AllUnique(values, unique)) {
        return NoMatch(candidate, "the constant operand is not uniform");
      }
      if (!graph.HasShape(input0)) {
        return NoMatch(candidate, "the other operand shape is unknown");
      }
      const core::symbolic::SymShape &shape = graph.GetShape(input0).Shape();
      if (shape.Rank() == 0 || !shape[shape.Rank() - 1].IsInt() ||
          shape[shape.Rank() - 1].AsInt() != static_cast<int64_t>(values.size())) {
        return NoMatch(candidate, "the constant does not broadcast on the last dimension");
      }
      if ((op_type == "Add" || op_type == "Sub") && unique != 0.0) {
        return NoMatch(candidate, "the additive constant is not zero");
      }
      if ((op_type == "Mul" || op_type == "Div") && unique != 1.0) {
        return NoMatch(candidate, "the multiplicative constant is not one");
      }
      return core::builder::MatchResult{this, {&candidate}, &candidate};
    }
    return NoMatch(candidate, "the constant operand is not a neutral element");
  }

  if (graph.IsConstant(input0)) {
    std::size_t rank = 0;
    if (HasRank(graph, input0, rank) && rank > 1) {
      return NoMatch(candidate, "the constant operand has rank greater than one");
    }
    if (IsScalarShape(graph, input0)) {
      if (graph.IsConstantScalar(input0, 0.0, false) && (op_type == "Add" || op_type == "Or")) {
        return core::builder::MatchResult{this, {&candidate}, &candidate};
      }
      if (graph.IsConstantScalar(input0, 1.0, false) && (op_type == "Mul" || op_type == "And")) {
        return core::builder::MatchResult{this, {&candidate}, &candidate};
      }
    }
    return NoMatch(candidate, "the scalar constant operand is not a neutral element");
  }

  return NoMatch(candidate, "no operand is a neutral constant");
}

utils::RepeatedProtoField<NodeProto>
IdentityPattern::Apply(core::builder::GraphGraph &graph,
                       const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("IdentityPattern::Apply expects one node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("IdentityPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &node = *nodes[0];
  const std::string &op_type = node.op_type().value();

  std::size_t index = 0;
  if (node.input_size() == 2 && BinaryOps().count(op_type) != 0 &&
      graph.IsConstant(node.input()[0].value()) && IsScalarShape(graph, node.input()[0].value())) {
    index = 1;
  }

  const std::string name = "IdentityPattern--" + node.name().value() + "." + op_type;
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode("Identity", {node.input()[index].value()},
                                  {node.output()[0].value()}, "", name.c_str()));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
