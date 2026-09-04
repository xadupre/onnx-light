// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/algebra/mul_pattern.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/patterns/collections/collections_utils.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using collections::FreeInitializerName;
using collections::IsDefaultOp;
using core::builder::BuilderError;
using core::symbolic::SymDim;
using core::symbolic::SymShape;

bool IsScalarOrOneElementVector(const TensorProto &tensor) {
  return tensor.dims_size() == 0 ||
         (tensor.dims_size() == 1 && tensor.dims(0) == static_cast<int64_t>(1));
}

bool IsFloatingForDiv(const TensorProto &tensor) {
  const auto type = static_cast<TensorProto::DataType>(tensor.data_type());
  return type == TensorProto::DataType::FLOAT16 || type == TensorProto::DataType::FLOAT ||
         type == TensorProto::DataType::DOUBLE;
}

bool Read16BitFloat(const TensorProto &tensor, bool bfloat, float &value) {
  std::uint16_t bits = 0;
  if (tensor.int32_data().size() > 0) {
    bits = static_cast<std::uint16_t>(tensor.int32_data()[0]);
  } else {
    if (!tensor.is_raw_data() || tensor.ref_raw_data().size() < 2) {
      return false;
    }
    bits = static_cast<std::uint16_t>(tensor.ref_raw_data()[0]) |
           (static_cast<std::uint16_t>(tensor.ref_raw_data()[1]) << 8);
  }
  value =
      bfloat ? core::runtime::Bfloat16BitsToFloat(bits) : core::runtime::Float16BitsToFloat(bits);
  return true;
}

bool ReadScalar(const TensorProto &tensor, double &value) {
  const auto type = static_cast<TensorProto::DataType>(tensor.data_type());
  if (type == TensorProto::DataType::FLOAT16) {
    float half = 0.0F;
    if (!Read16BitFloat(tensor, false, half)) {
      return false;
    }
    value = half;
    return true;
  }
  if (type == TensorProto::DataType::BFLOAT16) {
    float bfloat = 0.0F;
    if (!Read16BitFloat(tensor, true, bfloat)) {
      return false;
    }
    value = bfloat;
    return true;
  }
  return ReadScalarAsDouble(tensor, value);
}

bool IsSupportedSingleOne(const TensorProto &tensor) {
  for (int64_t dim : tensor.dims()) {
    if (dim != 1) {
      return false;
    }
  }
  const auto type = static_cast<TensorProto::DataType>(tensor.data_type());
  if (type != TensorProto::DataType::FLOAT && type != TensorProto::DataType::FLOAT16 &&
      type != TensorProto::DataType::DOUBLE && type != TensorProto::DataType::INT32 &&
      type != TensorProto::DataType::INT64) {
    return false;
  }
  double value = 0.0;
  return ReadScalar(tensor, value) && value == 1.0;
}

std::vector<int64_t> ProductShape(const TensorProto &left, const TensorProto &right) {
  if (left.dims_size() == 1 || right.dims_size() == 1) {
    return {1};
  }
  return {};
}

TensorProto MakeScalarTensor(const std::string &name, TensorProto::DataType type,
                             const std::vector<int64_t> &dims, double value) {
  TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(type);
  for (int64_t dim : dims) {
    tensor.add_dims(dim);
  }
  switch (type) {
  case TensorProto::DataType::FLOAT16:
    tensor.ref_int32_data().push_back(
        static_cast<int32_t>(core::runtime::FloatToFloat16Bits(static_cast<float>(value))));
    break;
  case TensorProto::DataType::BFLOAT16:
    tensor.ref_int32_data().push_back(
        static_cast<int32_t>(core::runtime::FloatToBfloat16Bits(static_cast<float>(value))));
    break;
  case TensorProto::DataType::FLOAT:
    tensor.ref_float_data().push_back(static_cast<float>(value));
    break;
  case TensorProto::DataType::DOUBLE:
    tensor.ref_double_data().push_back(value);
    break;
  case TensorProto::DataType::INT64:
    tensor.ref_int64_data().push_back(static_cast<int64_t>(value));
    break;
  case TensorProto::DataType::UINT64:
    tensor.ref_uint64_data().push_back(static_cast<uint64_t>(value));
    break;
  case TensorProto::DataType::UINT32:
    tensor.ref_uint64_data().push_back(static_cast<uint32_t>(value));
    break;
  case TensorProto::DataType::INT32:
    tensor.ref_int32_data().push_back(static_cast<int32_t>(value));
    break;
  case TensorProto::DataType::INT16:
    tensor.ref_int32_data().push_back(static_cast<int16_t>(value));
    break;
  case TensorProto::DataType::INT8:
    tensor.ref_int32_data().push_back(static_cast<int8_t>(value));
    break;
  case TensorProto::DataType::UINT16:
    tensor.ref_int32_data().push_back(static_cast<uint16_t>(value));
    break;
  case TensorProto::DataType::UINT8:
  case TensorProto::DataType::BOOL:
    tensor.ref_int32_data().push_back(static_cast<uint8_t>(value));
    break;
  default:
    tensor.ref_int32_data().push_back(static_cast<int32_t>(value));
    break;
  }
  return tensor;
}

TensorProto MakeIntegerScalarTensor(const std::string &name, TensorProto::DataType type,
                                    const std::vector<int64_t> &dims, int64_t left, int64_t right) {
  TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(type);
  for (int64_t dim : dims) {
    tensor.add_dims(dim);
  }
  const uint64_t product = static_cast<uint64_t>(left) * static_cast<uint64_t>(right);
  switch (type) {
  case TensorProto::DataType::INT64:
    tensor.ref_int64_data().push_back(static_cast<int64_t>(product));
    break;
  case TensorProto::DataType::UINT64:
    tensor.ref_uint64_data().push_back(product);
    break;
  case TensorProto::DataType::INT32:
    tensor.ref_int32_data().push_back(static_cast<int32_t>(product));
    break;
  case TensorProto::DataType::UINT32:
    tensor.ref_uint64_data().push_back(static_cast<uint32_t>(product));
    break;
  case TensorProto::DataType::INT16:
    tensor.ref_int32_data().push_back(static_cast<int16_t>(product));
    break;
  case TensorProto::DataType::UINT16:
    tensor.ref_int32_data().push_back(static_cast<uint16_t>(product));
    break;
  case TensorProto::DataType::INT8:
    tensor.ref_int32_data().push_back(static_cast<int8_t>(product));
    break;
  case TensorProto::DataType::UINT8:
    tensor.ref_int32_data().push_back(static_cast<uint8_t>(product));
    break;
  case TensorProto::DataType::BOOL:
    tensor.ref_int32_data().push_back(left != 0 && right != 0 ? 1 : 0);
    break;
  default:
    break;
  }
  return tensor;
}

bool CombineConstants(const TensorProto &left, const TensorProto &right, bool reciprocal_left,
                      bool reciprocal_right, TensorProto &result) {
  const auto type = static_cast<TensorProto::DataType>(left.data_type());
  if (type != static_cast<TensorProto::DataType>(right.data_type())) {
    return false;
  }
  const bool floating =
      type == TensorProto::DataType::FLOAT16 || type == TensorProto::DataType::BFLOAT16 ||
      type == TensorProto::DataType::FLOAT || type == TensorProto::DataType::DOUBLE;
  if (!floating) {
    std::vector<int64_t> left_values;
    std::vector<int64_t> right_values;
    if (reciprocal_left || reciprocal_right || !ReadIntegerValues(left, left_values) ||
        !ReadIntegerValues(right, right_values) || left_values.size() != 1 ||
        right_values.size() != 1) {
      return false;
    }
    result = MakeIntegerScalarTensor("", type, ProductShape(left, right), left_values[0],
                                     right_values[0]);
    return true;
  }
  double left_value = 0.0;
  double right_value = 0.0;
  if (!ReadScalar(left, left_value) || !ReadScalar(right, right_value)) {
    return false;
  }
  if (type == TensorProto::DataType::FLOAT16) {
    float l = static_cast<float>(left_value);
    float r = static_cast<float>(right_value);
    if (reciprocal_left) {
      l = core::runtime::Float16BitsToFloat(core::runtime::FloatToFloat16Bits(1.0F / l));
    }
    if (reciprocal_right) {
      r = core::runtime::Float16BitsToFloat(core::runtime::FloatToFloat16Bits(1.0F / r));
    }
    result = MakeScalarTensor("", type, ProductShape(left, right), static_cast<float>(l * r));
    return true;
  }
  if (type == TensorProto::DataType::BFLOAT16) {
    if (reciprocal_left || reciprocal_right) {
      return false;
    }
    const float product = static_cast<float>(left_value) * static_cast<float>(right_value);
    result = MakeScalarTensor("", type, ProductShape(left, right), product);
    return true;
  }
  if (type == TensorProto::DataType::FLOAT) {
    float l = static_cast<float>(left_value);
    float r = static_cast<float>(right_value);
    if (reciprocal_left) {
      l = 1.0F / l;
    }
    if (reciprocal_right) {
      r = 1.0F / r;
    }
    result = MakeScalarTensor("", type, ProductShape(left, right), static_cast<float>(l * r));
    return true;
  }
  if (type == TensorProto::DataType::DOUBLE) {
    if (reciprocal_left) {
      left_value = 1.0 / left_value;
    }
    if (reciprocal_right) {
      right_value = 1.0 / right_value;
    }
    result = MakeScalarTensor("", type, ProductShape(left, right), left_value * right_value);
    return true;
  }
  return false;
}

bool HasShape(core::builder::GraphGraph &graph, const NodeProto &node) {
  return node.input_size() >= 2 && graph.HasShape(node.input()[0].value()) &&
         graph.HasShape(node.input()[1].value());
}

std::vector<SymDim> AlignShape(const SymShape &shape, std::size_t rank) {
  std::vector<SymDim> aligned(rank - shape.Rank(), SymDim(1));
  for (const SymDim &dim : shape.Dims()) {
    aligned.push_back(dim);
  }
  return aligned;
}

int SwitchOrder(const SymShape &shape_left, const SymShape &shape_right,
                const SymShape &shape_before_left, const SymShape &shape_before_right, int side) {
  if (side == 1) {
    return SwitchOrder(shape_right, shape_left, shape_before_left, shape_before_right, 0);
  }
  const std::size_t rank = std::max(std::max(shape_left.Rank(), shape_right.Rank()),
                                    std::max(shape_before_left.Rank(), shape_before_right.Rank()));
  const std::size_t case0 = std::max(shape_before_left.Rank(), shape_before_right.Rank());
  const std::size_t case1 = std::max(shape_right.Rank(), shape_before_left.Rank());
  const std::size_t case2 = std::max(shape_right.Rank(), shape_before_right.Rank());
  if (case0 < std::min(case1, case2)) {
    return 0;
  }
  if (case1 < std::min(case0, case2)) {
    return 1;
  }
  if (case2 < std::min(case0, case1)) {
    return 2;
  }
  const std::vector<SymDim> right = AlignShape(shape_right, rank);
  const std::vector<SymDim> before_left = AlignShape(shape_before_left, rank);
  const std::vector<SymDim> before_right = AlignShape(shape_before_right, rank);
  for (std::size_t index = 0; index < rank; ++index) {
    const SymDim &a = right[index];
    const SymDim &b = before_left[index];
    const SymDim &c = before_right[index];
    if (a == b && b == c) {
      continue;
    }
    if (a.IsInt() && b.IsInt() && c.IsInt()) {
      const int64_t case0_dim = std::max(b.AsInt(), c.AsInt());
      const int64_t case1_dim = std::max(b.AsInt(), a.AsInt());
      const int64_t case2_dim = std::max(a.AsInt(), c.AsInt());
      if (case0_dim < std::min(case1_dim, case2_dim)) {
        return 0;
      }
      if (case1_dim < std::min(case0_dim, case2_dim)) {
        return 1;
      }
      if (case2_dim < std::min(case0_dim, case1_dim)) {
        return 2;
      }
    }
  }
  return 0;
}

} // namespace

std::set<std::string> DivMulPattern::FastOpType() const { return {"Div"}; }

core::builder::MatchResult DivMulPattern::Match(core::builder::GraphGraph &graph,
                                                const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Div") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain binary Div");
  }
  const std::string &div_output = candidate.output()[0].value();
  if (graph.IsUsedMoreThanOnce(div_output)) {
    return NoMatch(candidate, "the Div output has another use or is a graph output");
  }
  const std::vector<const NodeProto *> &consumers = graph.NextNodes(div_output);
  if (consumers.size() != 1 || !IsDefaultOp(*consumers[0], "Mul") ||
      consumers[0]->input_size() != 2 || consumers[0]->output_size() != 1) {
    return NoMatch(candidate, "the Div is not followed by one default-domain binary Mul");
  }
  const NodeProto *mul = consumers[0];
  const bool is_left_input = mul->input()[0].value() == div_output;
  const bool is_right_input = mul->input()[1].value() == div_output;
  if (is_left_input == is_right_input) {
    return NoMatch(candidate, "the Mul must consume the Div output exactly once");
  }
  const TensorProto *one = graph.GetComputedConstant(candidate.input()[0].value());
  if (one == nullptr || !IsSupportedSingleOne(*one)) {
    return NoMatch(candidate, "the Div numerator is not a supported one-element constant one");
  }
  return core::builder::MatchResult{this, {&candidate, mul}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
DivMulPattern::Apply(core::builder::GraphGraph &graph,
                     const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("DivMulPattern::Apply expects a Div followed by a Mul.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("DivMulPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &div = *nodes[0];
  const NodeProto &mul = *nodes[1];
  const std::string &other = mul.input()[0].value() == div.output()[0].value()
                                 ? mul.input()[1].value()
                                 : mul.input()[0].value();
  NodeProto replacement =
      MakeNode("Div", {other, div.input()[1].value()}, {mul.output()[0].value()}, "",
               ("DivMulPattern--" + mul.name().value()).c_str());
  if (mul.has_doc_string()) {
    replacement.set_doc_string(mul.doc_string().value());
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> MulMulMulScalarPattern::FastOpType() const { return {"Div", "Mul"}; }

core::builder::MatchResult MulMulMulScalarPattern::Match(core::builder::GraphGraph &graph,
                                                         const NodeProto &candidate) const {
  const bool is_div = IsDefaultOp(candidate, "Div");
  const bool is_mul = IsDefaultOp(candidate, "Mul");
  if ((!is_div && !is_mul) || candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain binary Div or Mul");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value()) ||
      graph.IsUsedMoreThanOnce(candidate.input()[1].value())) {
    return NoMatch(candidate, "a binary input has another use");
  }
  const NodeProto *left = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *right = graph.NodeBefore(candidate.input()[1].value());
  if (left == nullptr || right == nullptr ||
      (!IsDefaultOp(*left, "Div") && !IsDefaultOp(*left, "Mul")) ||
      (!IsDefaultOp(*right, "Div") && !IsDefaultOp(*right, "Mul")) || left->input_size() != 2 ||
      right->input_size() != 2 || left->output_size() != 1 || right->output_size() != 1) {
    return NoMatch(candidate, "both binary inputs are not produced by default-domain Div or Mul");
  }
  if (!graph.IsConstant(left->input()[1].value()) || !graph.IsConstant(right->input()[1].value())) {
    return NoMatch(candidate, "the inner right operands are not constants");
  }
  const TensorProto *left_constant = graph.GetComputedConstant(left->input()[1].value());
  const TensorProto *right_constant = graph.GetComputedConstant(right->input()[1].value());
  if (left_constant == nullptr || right_constant == nullptr ||
      !IsScalarOrOneElementVector(*left_constant) || !IsScalarOrOneElementVector(*right_constant)) {
    return NoMatch(candidate, "the inner constants are not scalar or one-element vectors");
  }
  if ((IsDefaultOp(*left, "Div") || IsDefaultOp(*right, "Div")) &&
      (!IsFloatingForDiv(*left_constant) || !IsFloatingForDiv(*right_constant))) {
    return NoMatch(candidate, "Div constants are not float16, float, or double");
  }
  return core::builder::MatchResult{this, {&candidate, left, right}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
MulMulMulScalarPattern::Apply(core::builder::GraphGraph &graph,
                              const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError("MulMulMulScalarPattern::Apply expects three binary nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("MulMulMulScalarPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &node = *nodes[0];
  const NodeProto &left = *nodes[1];
  const NodeProto &right = *nodes[2];
  const TensorProto *left_constant = graph.GetComputedConstant(left.input()[1].value());
  const TensorProto *right_constant = graph.GetComputedConstant(right.input()[1].value());
  TensorProto combined;
  const bool both_div = IsDefaultOp(left, "Div") && IsDefaultOp(right, "Div");
  if (left_constant == nullptr || right_constant == nullptr ||
      !CombineConstants(*left_constant, *right_constant, !both_div && IsDefaultOp(left, "Div"),
                        !both_div && IsDefaultOp(right, "Div"), combined)) {
    throw BuilderError("MulMulMulScalarPattern::Apply could not combine the scalar constants.");
  }
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string combined_name = FreeInitializerName(builder, "MulMulMulScalarPattern.cst");
  combined.set_name(combined_name);
  builder.MakeInitializer(combined);
  const std::string intermediate =
      builder.UniqueName("MulMulMulScalarPattern--" + node.output()[0].value());
  const std::string name = "MulMulMulScalarPattern--" + node.name().value();

  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakeNode(node.op_type().value().c_str(),
                                  {left.input()[0].value(), right.input()[0].value()},
                                  {intermediate}, "", name.c_str()));
  replacements.push_back(MakeNode(both_div ? "Div" : "Mul", {intermediate, combined_name},
                                  {node.output()[0].value()}, "", (name + "-Cst").c_str()));
  return replacements;
}

std::set<std::string> SwitchOrderBinaryPattern::FastOpType() const { return {"Add", "Mul"}; }

core::builder::MatchResult SwitchOrderBinaryPattern::Match(core::builder::GraphGraph &graph,
                                                           const NodeProto &candidate) const {
  const bool is_add = IsDefaultOp(candidate, "Add");
  const bool is_mul = IsDefaultOp(candidate, "Mul");
  if ((!is_add && !is_mul) || candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain binary Add or Mul");
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.input()[1].value())) {
    return NoMatch(candidate, "the candidate input shapes are unknown");
  }
  const NodeProto *left = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *right = graph.NodeBefore(candidate.input()[1].value());
  const std::string &op_type = candidate.op_type().value();
  const bool left_matches = left != nullptr && IsDefaultOp(*left, op_type.c_str());
  const bool right_matches = right != nullptr && IsDefaultOp(*right, op_type.c_str());
  if (!left_matches && !right_matches) {
    return NoMatch(candidate, "neither candidate input is produced by the same binary operator");
  }

  int choose = 0;
  if (left == nullptr) {
    choose = 1;
  } else if (right == nullptr) {
    choose = 0;
  } else if (!left_matches || !HasShape(graph, *left)) {
    if (!right_matches) {
      return NoMatch(candidate, "neither nested binary operation has both input shapes");
    }
    choose = 1;
  } else if (!right_matches || !HasShape(graph, *right)) {
    if (!left_matches) {
      return NoMatch(candidate, "neither nested binary operation has both input shapes");
    }
    choose = 0;
  } else {
    choose = 3;
  }

  const NodeProto *other = choose == 0 ? left : right;
  if (other == nullptr || !HasShape(graph, *other)) {
    return NoMatch(candidate, "the selected nested binary operation has unknown input shapes");
  }
  const SymShape &shape_left = graph.GetShape(candidate.input()[0].value()).Shape();
  const SymShape &shape_right = graph.GetShape(candidate.input()[1].value()).Shape();
  const SymShape &before_left = graph.GetShape(other->input()[0].value()).Shape();
  const SymShape &before_right = graph.GetShape(other->input()[1].value()).Shape();
  if (SwitchOrder(shape_left, shape_right, before_left, before_right, choose) == 0) {
    if (choose < 3) {
      return NoMatch(candidate, "reordering would not reduce the broadcast rank");
    }
    choose = 1;
    other = right;
    if (other == nullptr || !HasShape(graph, *other) ||
        SwitchOrder(shape_left, shape_right, graph.GetShape(other->input()[0].value()).Shape(),
                    graph.GetShape(other->input()[1].value()).Shape(), choose) == 0) {
      return NoMatch(candidate, "neither nested binary order reduces the broadcast rank");
    }
  }
  if (graph.IsUsedMoreThanOnce(other->output()[0].value())) {
    return NoMatch(candidate, "the nested binary output has another use");
  }
  return core::builder::MatchResult{
      this, {&candidate, choose == 0 ? left : nullptr, choose == 1 ? right : nullptr}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
SwitchOrderBinaryPattern::Apply(core::builder::GraphGraph &graph,
                                const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || (nodes[1] == nullptr && nodes[2] == nullptr)) {
    throw BuilderError(
        "SwitchOrderBinaryPattern::Apply expects one outer and one inner binary node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("SwitchOrderBinaryPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &node = *nodes[0];
  const NodeProto *left_node = nodes[1];
  const NodeProto *right_node = nodes[2];
  const int side = left_node == nullptr ? 1 : 0;
  const NodeProto &other = left_node == nullptr ? *right_node : *left_node;
  const SymShape &shape_left = graph.GetShape(node.input()[0].value()).Shape();
  const SymShape &shape_right = graph.GetShape(node.input()[1].value()).Shape();
  const int which =
      SwitchOrder(shape_left, shape_right, graph.GetShape(other.input()[0].value()).Shape(),
                  graph.GetShape(other.input()[1].value()).Shape(), side);
  if (which != 1 && which != 2) {
    throw BuilderError("SwitchOrderBinaryPattern::Apply received a non-beneficial ordering.");
  }
  const std::string &a = side == 0 ? node.input()[1].value() : node.input()[0].value();
  const std::string &b = other.input()[0].value();
  const std::string &c = other.input()[1].value();
  const std::string name = "SwitchOrderBinaryPattern--" + node.name().value();
  const std::vector<std::string> first_inputs =
      which == 1 ? std::vector<std::string>{b, a} : std::vector<std::string>{c, a};
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string intermediate =
      builder.UniqueName("SwitchOrderBinaryPattern--" + node.output()[0].value());
  NodeProto first =
      MakeNode(node.op_type().value().c_str(), first_inputs, {intermediate}, "", name.c_str());
  const std::vector<std::string> final_inputs = which == 1
                                                    ? std::vector<std::string>{intermediate, c}
                                                    : std::vector<std::string>{intermediate, b};
  NodeProto second = MakeNode(node.op_type().value().c_str(), final_inputs,
                              {node.output()[0].value()}, "", name.c_str());
  if (node.has_doc_string()) {
    second.set_doc_string(node.doc_string().value());
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(first));
  replacements.push_back(std::move(second));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
