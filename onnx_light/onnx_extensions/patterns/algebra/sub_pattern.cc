// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/algebra/sub_pattern.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::IsDefaultOp;
using core::builder::BuilderError;

bool IsAllOne(const TensorProto &tensor) {
  const auto dtype = static_cast<TensorProto::DataType>(tensor.data_type());
  if (dtype == TensorProto::DataType::FLOAT || dtype == TensorProto::DataType::DOUBLE) {
    std::vector<double> values;
    if (!ReadFloatingValues(tensor, values) || values.empty()) {
      return false;
    }
    for (double value : values) {
      if (value != 1.0) {
        return false;
      }
    }
    return true;
  }
  if (dtype == TensorProto::DataType::FLOAT16 || dtype == TensorProto::DataType::BFLOAT16) {
    if (tensor.int32_data().size() == 0) {
      return false;
    }
    for (std::size_t index = 0; index < tensor.int32_data().size(); ++index) {
      const std::uint16_t bits = static_cast<std::uint16_t>(tensor.int32_data()[index]);
      const float value = dtype == TensorProto::DataType::FLOAT16
                              ? core::runtime::Float16BitsToFloat(bits)
                              : core::runtime::Bfloat16BitsToFloat(bits);
      if (value != 1.0F) {
        return false;
      }
    }
    return true;
  }
  std::vector<int64_t> values;
  if (!ReadIntegerValues(tensor, values) || values.empty()) {
    return false;
  }
  for (int64_t value : values) {
    if (value != 1) {
      return false;
    }
  }
  return true;
}

bool IsAllOneConstantOfShape(const NodeProto &node) {
  if (!IsDefaultOp(node, "ConstantOfShape") || node.input_size() != 1 || node.output_size() != 1) {
    return false;
  }
  const AttributeProto *value = FindAttribute(node, "value");
  return value != nullptr && value->type() == AttributeProto::AttributeType::TENSOR &&
         IsAllOne(value->ref_t());
}

const core::symbolic::SymDim &AlignedDim(const core::symbolic::SymShape &shape,
                                         std::size_t output_rank, std::size_t index,
                                         const core::symbolic::SymDim &one) {
  const std::size_t offset = output_rank - shape.Rank();
  return index < offset ? one : shape[index - offset];
}

bool BroadcastsTo(const core::symbolic::SymShape &left, const core::symbolic::SymShape &right,
                  const core::symbolic::SymShape &output) {
  const std::size_t rank = left.Rank() > right.Rank() ? left.Rank() : right.Rank();
  if (rank != output.Rank()) {
    return false;
  }
  const core::symbolic::SymDim one(1);
  for (std::size_t index = 0; index < rank; ++index) {
    const core::symbolic::SymDim &left_dim = AlignedDim(left, rank, index, one);
    const core::symbolic::SymDim &right_dim = AlignedDim(right, rank, index, one);
    const core::symbolic::SymDim &output_dim = output[index];
    if (left_dim.IsInt() && left_dim.AsInt() == 1) {
      if (right_dim != output_dim) {
        return false;
      }
    } else if (right_dim.IsInt() && right_dim.AsInt() == 1) {
      if (left_dim != output_dim) {
        return false;
      }
    } else if (left_dim != right_dim || left_dim != output_dim) {
      return false;
    }
  }
  return true;
}

const NodeProto *OneMinusNode(core::builder::GraphGraph &graph, const NodeProto *node) {
  if (node == nullptr || !IsDefaultOp(*node, "Sub") || node->input_size() != 2) {
    return nullptr;
  }
  const std::string &one = node->input()[0].value();
  if (graph.IsConstant(one)) {
    const TensorProto *constant = graph.GetComputedConstant(one);
    if (constant != nullptr && IsAllOne(*constant)) {
      return node;
    }
  }
  const NodeProto *constant_of_shape = graph.NodeBefore(one);
  return constant_of_shape != nullptr && IsAllOneConstantOfShape(*constant_of_shape) ? node
                                                                                     : nullptr;
}

bool ConstantOfShapeRewritePreservesShape(core::builder::GraphGraph &graph,
                                          const NodeProto &one_minus, const std::string &other,
                                          const NodeProto &mul) {
  const NodeProto *constant_of_shape = graph.NodeBefore(one_minus.input()[0].value());
  if (constant_of_shape == nullptr || !IsAllOneConstantOfShape(*constant_of_shape)) {
    return true;
  }
  const std::string &subtracted = one_minus.input()[1].value();
  if (!graph.HasShape(subtracted) || !graph.HasShape(other) ||
      !graph.HasShape(mul.output()[0].value())) {
    return false;
  }
  return BroadcastsTo(graph.GetShape(subtracted).Shape(), graph.GetShape(other).Shape(),
                      graph.GetShape(mul.output()[0].value()).Shape());
}

} // namespace

std::set<std::string> Sub1MulPattern::FastOpType() const { return {"Mul"}; }

core::builder::MatchResult Sub1MulPattern::Match(core::builder::GraphGraph &graph,
                                                 const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Mul") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain binary Mul");
  }
  const NodeProto *left = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *right = graph.NodeBefore(candidate.input()[1].value());
  const bool left_is_sub = left != nullptr && IsDefaultOp(*left, "Sub");
  const bool right_is_sub = right != nullptr && IsDefaultOp(*right, "Sub");
  if (!left_is_sub && !right_is_sub) {
    return NoMatch(candidate, "neither Mul input is produced by a Sub");
  }
  if ((left_is_sub && graph.IsUsedMoreThanOnce(candidate.input()[0].value())) ||
      (right_is_sub && graph.IsUsedMoreThanOnce(candidate.input()[1].value()))) {
    return NoMatch(candidate, "a Sub output has another use");
  }
  const NodeProto *left_one_minus = OneMinusNode(graph, left);
  const NodeProto *right_one_minus = OneMinusNode(graph, right);
  if (left_one_minus == nullptr && right_one_minus == nullptr) {
    return NoMatch(candidate, "neither Sub has a constant first input equal to one");
  }
  const NodeProto *one_minus = left_one_minus != nullptr ? left_one_minus : right_one_minus;
  const std::string &other =
      left_one_minus != nullptr ? candidate.input()[1].value() : candidate.input()[0].value();
  if (!ConstantOfShapeRewritePreservesShape(graph, *one_minus, other, candidate)) {
    return NoMatch(candidate,
                   "removing ConstantOfShape would change or cannot prove the output shape");
  }
  return core::builder::MatchResult{this, {&candidate, left, right}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
Sub1MulPattern::Apply(core::builder::GraphGraph &graph,
                      const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr) {
    throw BuilderError("Sub1MulPattern::Apply expects a Mul and its two predecessors.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("Sub1MulPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &mul = *nodes[0];
  const NodeProto *left = nodes[1];
  const NodeProto *right = nodes[2];
  const NodeProto *left_one_minus = OneMinusNode(graph, left);

  std::string multiplied_input;
  std::string other_input;
  const NodeProto *keep = nullptr;
  if (left_one_minus != nullptr) {
    multiplied_input = left->input()[1].value();
    other_input = mul.input()[1].value();
    keep = right;
  } else {
    multiplied_input = right->input()[1].value();
    other_input = mul.input()[0].value();
    keep = left;
  }

  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string intermediate = builder.UniqueName("Sub1MulPattern--" + mul.output()[0].value());
  const std::string name = "Sub1MulPattern--" + mul.name().value();

  utils::RepeatedProtoField<NodeProto> replacements;
  if (keep != nullptr) {
    replacements.add() = *keep;
  }
  replacements.push_back(
      MakeNode("Mul", {multiplied_input, other_input}, {intermediate}, "", name.c_str()));
  NodeProto replacement =
      MakeNode("Sub", {other_input, intermediate}, {mul.output()[0].value()}, "", name.c_str());
  if (mul.has_doc_string()) {
    replacement.set_doc_string(mul.doc_string().value());
  }
  replacements.push_back(std::move(replacement));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
