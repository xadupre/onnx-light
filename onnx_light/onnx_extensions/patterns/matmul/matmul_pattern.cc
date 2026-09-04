// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/matmul/matmul_pattern.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
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

std::vector<std::string> Inputs(const NodeProto &node) {
  std::vector<std::string> inputs;
  inputs.reserve(node.input_size());
  for (int i = 0; i < node.input_size(); ++i) {
    inputs.push_back(node.input()[i].value());
  }
  return inputs;
}

std::vector<std::string> Outputs(const NodeProto &node) {
  std::vector<std::string> outputs;
  outputs.reserve(node.output_size());
  for (int i = 0; i < node.output_size(); ++i) {
    outputs.push_back(node.output()[i].value());
  }
  return outputs;
}

void CopyAttributes(const NodeProto &source, NodeProto &destination) {
  for (int i = 0; i < source.attribute_size(); ++i) {
    *destination.add_attribute() = source.attribute(i);
  }
}

bool ReadConstantInts(core::builder::GraphGraph &graph, const std::string &name,
                      std::vector<int64_t> &values) {
  if (!graph.IsConstant(name)) {
    return false;
  }
  const TensorProto *tensor = graph.GetComputedConstant(name);
  return tensor != nullptr && ReadIntegerValues(*tensor, values);
}

bool ReadStaticShape(core::builder::GraphGraph &graph, const std::string &name,
                     std::vector<int64_t> &shape) {
  if (!graph.HasShape(name)) {
    return false;
  }
  const SymShape &symbolic = graph.GetShape(name).Shape();
  shape.clear();
  shape.reserve(symbolic.Rank());
  for (std::size_t i = 0; i < symbolic.Rank(); ++i) {
    if (!symbolic[i].IsInt()) {
      return false;
    }
    shape.push_back(symbolic[i].AsInt());
  }
  return true;
}

std::vector<int64_t> StaticShapeOrThrow(core::builder::GraphGraph &graph, const std::string &name,
                                        const char *pattern_name) {
  std::vector<int64_t> shape;
  if (!ReadStaticShape(graph, name, shape)) {
    throw BuilderError(std::string(pattern_name) + "::Apply requires a fully-static shape for '" +
                       name + "'.");
  }
  return shape;
}

SymShape Slice(const SymShape &shape, std::size_t begin, std::size_t end) {
  SymShape result;
  for (std::size_t i = begin; i < end; ++i) {
    result.PushBack(shape[i]);
  }
  return result;
}

SymShape Suffix(const SymShape &shape, std::size_t count) {
  return Slice(shape, shape.Rank() - count, shape.Rank());
}

std::vector<int64_t> Prefix(const std::vector<int64_t> &shape, std::size_t trailing) {
  return std::vector<int64_t>(shape.begin(), shape.end() - static_cast<std::ptrdiff_t>(trailing));
}

std::vector<int64_t> Suffix(const std::vector<int64_t> &shape, std::size_t count) {
  return std::vector<int64_t>(shape.end() - static_cast<std::ptrdiff_t>(count), shape.end());
}

std::optional<int64_t> ElementCount(const std::vector<int64_t> &shape) {
  int64_t count = 1;
  for (int64_t dimension : shape) {
    if (dimension < 0) {
      return std::nullopt;
    }
    count *= dimension;
  }
  return count;
}

bool SameElementCount(const std::vector<int64_t> &left, const std::vector<int64_t> &right) {
  const std::optional<int64_t> left_count = ElementCount(left);
  const std::optional<int64_t> right_count = ElementCount(right);
  return left_count.has_value() && right_count.has_value() &&
         left_count.value() == right_count.value();
}

std::optional<std::vector<int64_t>> BroadcastBatch(const std::vector<int64_t> &left,
                                                   const std::vector<int64_t> &right) {
  const std::size_t rank = std::max(left.size(), right.size());
  std::vector<int64_t> result(rank, 1);
  for (std::size_t offset = 0; offset < rank; ++offset) {
    const int64_t left_dim =
        offset < left.size() ? left[left.size() - 1 - offset] : static_cast<int64_t>(1);
    const int64_t right_dim =
        offset < right.size() ? right[right.size() - 1 - offset] : static_cast<int64_t>(1);
    if (left_dim != right_dim && left_dim != 1 && right_dim != 1) {
      return std::nullopt;
    }
    result[rank - 1 - offset] = left_dim == 1 ? right_dim : left_dim;
  }
  return result;
}

std::optional<std::vector<int64_t>> MatMulShape(const std::vector<int64_t> &left,
                                                const std::vector<int64_t> &right) {
  if (left.size() < 2 || right.size() < 2 || left.back() != right[right.size() - 2]) {
    return std::nullopt;
  }
  const std::optional<std::vector<int64_t>> batch =
      BroadcastBatch(Prefix(left, 2), Prefix(right, 2));
  if (!batch.has_value()) {
    return std::nullopt;
  }
  std::vector<int64_t> result = batch.value();
  result.push_back(left[left.size() - 2]);
  result.push_back(right.back());
  return result;
}

std::vector<int64_t> WithMatrixSuffix(const std::vector<int64_t> &batch,
                                      const std::vector<int64_t> &matrix) {
  std::vector<int64_t> result = batch;
  result.insert(result.end(), matrix.end() - 2, matrix.end());
  return result;
}

bool MainOpsetAtLeast(core::builder::GraphGraph &graph, int minimum) {
  const int opset = graph.Builder().OpsetVersion("");
  return opset == core::shapes::kUnknownOpsetVersion || opset >= minimum;
}

std::string AddShapeInitializer(core::builder::GraphGraph &graph, const std::string &base,
                                const std::vector<int64_t> &values) {
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = FreeInitializerName(builder, base);
  return builder.MakeInitializer(MakeInitializerShape(name.c_str(), values));
}

NodeProto MakePatternNode(const char *op_type, const std::vector<std::string> &inputs,
                          const std::vector<std::string> &outputs, const std::string &domain,
                          const std::string &name) {
  return MakeNode(op_type, inputs, outputs, domain.c_str(), name.c_str());
}

bool Read16BitFloat(const TensorProto &tensor, bool bfloat, float &value) {
  std::uint16_t bits = 0;
  if (!tensor.int32_data().empty()) {
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
  if (type == TensorProto::DataType::FLOAT16 || type == TensorProto::DataType::BFLOAT16) {
    float converted = 0.0F;
    if (!Read16BitFloat(tensor, type == TensorProto::DataType::BFLOAT16, converted)) {
      return false;
    }
    value = converted;
    return true;
  }
  return ReadScalarAsDouble(tensor, value);
}

std::vector<int64_t> ScalarProductShape(const TensorProto &left, const TensorProto &right) {
  return left.dims_size() == 1 || right.dims_size() == 1 ? std::vector<int64_t>{1}
                                                         : std::vector<int64_t>{};
}

TensorProto MakeScalarProduct(const TensorProto &left, const TensorProto &right, double product) {
  TensorProto tensor;
  const auto type = static_cast<TensorProto::DataType>(left.data_type());
  tensor.set_data_type(type);
  for (int64_t dim : ScalarProductShape(left, right)) {
    tensor.add_dims(dim);
  }
  switch (type) {
  case TensorProto::DataType::FLOAT16:
    tensor.ref_int32_data().push_back(
        static_cast<int32_t>(core::runtime::FloatToFloat16Bits(static_cast<float>(product))));
    break;
  case TensorProto::DataType::BFLOAT16:
    tensor.ref_int32_data().push_back(
        static_cast<int32_t>(core::runtime::FloatToBfloat16Bits(static_cast<float>(product))));
    break;
  case TensorProto::DataType::FLOAT:
    tensor.ref_float_data().push_back(static_cast<float>(product));
    break;
  case TensorProto::DataType::DOUBLE:
    tensor.ref_double_data().push_back(product);
    break;
  case TensorProto::DataType::INT64:
    tensor.ref_int64_data().push_back(static_cast<int64_t>(product));
    break;
  case TensorProto::DataType::UINT64:
    tensor.ref_uint64_data().push_back(static_cast<uint64_t>(product));
    break;
  case TensorProto::DataType::UINT32:
    tensor.ref_uint64_data().push_back(static_cast<uint32_t>(product));
    break;
  case TensorProto::DataType::INT32:
    tensor.ref_int32_data().push_back(static_cast<int32_t>(product));
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
    tensor.ref_int32_data().push_back(product != 0.0 ? 1 : 0);
    break;
  default:
    break;
  }
  return tensor;
}

bool MakeIntegerScalarProduct(const TensorProto &left, const TensorProto &right, int64_t left_value,
                              int64_t right_value, TensorProto &result) {
  const auto type = static_cast<TensorProto::DataType>(left.data_type());
  result.set_data_type(type);
  for (int64_t dim : ScalarProductShape(left, right)) {
    result.add_dims(dim);
  }
  const uint64_t product = static_cast<uint64_t>(left_value) * static_cast<uint64_t>(right_value);
  switch (type) {
  case TensorProto::DataType::INT64:
    result.ref_int64_data().push_back(static_cast<int64_t>(product));
    break;
  case TensorProto::DataType::UINT64:
    result.ref_uint64_data().push_back(product);
    break;
  case TensorProto::DataType::INT32:
    result.ref_int32_data().push_back(static_cast<int32_t>(product));
    break;
  case TensorProto::DataType::UINT32:
    result.ref_uint64_data().push_back(static_cast<uint32_t>(product));
    break;
  case TensorProto::DataType::INT16:
    result.ref_int32_data().push_back(static_cast<int16_t>(product));
    break;
  case TensorProto::DataType::UINT16:
    result.ref_int32_data().push_back(static_cast<uint16_t>(product));
    break;
  case TensorProto::DataType::INT8:
    result.ref_int32_data().push_back(static_cast<int8_t>(product));
    break;
  case TensorProto::DataType::UINT8:
    result.ref_int32_data().push_back(static_cast<uint8_t>(product));
    break;
  case TensorProto::DataType::BOOL:
    result.ref_int32_data().push_back(left_value != 0 && right_value != 0 ? 1 : 0);
    break;
  default:
    return false;
  }
  return true;
}

bool MultiplyScalarConstants(const TensorProto &left, const TensorProto &right,
                             TensorProto &result) {
  if (left.data_type() != right.data_type()) {
    return false;
  }
  const auto type = static_cast<TensorProto::DataType>(left.data_type());
  if (type != TensorProto::DataType::FLOAT16 && type != TensorProto::DataType::BFLOAT16 &&
      type != TensorProto::DataType::FLOAT && type != TensorProto::DataType::DOUBLE) {
    std::vector<int64_t> left_values;
    std::vector<int64_t> right_values;
    return ReadIntegerValues(left, left_values) && ReadIntegerValues(right, right_values) &&
           left_values.size() == 1 && right_values.size() == 1 &&
           MakeIntegerScalarProduct(left, right, left_values[0], right_values[0], result);
  }
  double left_value = 0.0;
  double right_value = 0.0;
  if (!ReadScalar(left, left_value) || !ReadScalar(right, right_value)) {
    return false;
  }
  const double product = type == TensorProto::DataType::DOUBLE
                             ? left_value * right_value
                             : static_cast<float>(left_value) * static_cast<float>(right_value);
  result = MakeScalarProduct(left, right, product);
  return true;
}

bool IsFloatingType(TensorProto::DataType type) {
  return type == TensorProto::DataType::FLOAT16 || type == TensorProto::DataType::BFLOAT16 ||
         type == TensorProto::DataType::FLOAT || type == TensorProto::DataType::DOUBLE;
}

bool ReadFloatingTensor(const TensorProto &tensor, std::vector<double> &values) {
  const auto type = static_cast<TensorProto::DataType>(tensor.data_type());
  if (type == TensorProto::DataType::FLOAT16 || type == TensorProto::DataType::BFLOAT16) {
    const std::size_t count = !tensor.int32_data().empty() ? tensor.int32_data().size()
                                                           : tensor.ref_raw_data().size() / 2;
    if (count == 0) {
      return false;
    }
    values.clear();
    values.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      std::uint16_t bits = 0;
      if (!tensor.int32_data().empty()) {
        bits = static_cast<std::uint16_t>(tensor.int32_data()[index]);
      } else {
        bits = static_cast<std::uint16_t>(tensor.ref_raw_data()[2 * index]) |
               (static_cast<std::uint16_t>(tensor.ref_raw_data()[2 * index + 1]) << 8);
      }
      const float value = type == TensorProto::DataType::BFLOAT16
                              ? core::runtime::Bfloat16BitsToFloat(bits)
                              : core::runtime::Float16BitsToFloat(bits);
      values.push_back(static_cast<double>(value));
    }
    return true;
  }
  return ReadFloatingValues(tensor, values);
}

TensorProto MakeFloatingTensor(const std::string &name, TensorProto::DataType type,
                               const std::vector<int64_t> &dims,
                               const std::vector<double> &values) {
  TensorProto tensor;
  tensor.set_name(name);
  tensor.set_data_type(type);
  for (int64_t dim : dims) {
    tensor.add_dims(dim);
  }
  for (double value : values) {
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
    default:
      break;
    }
  }
  return tensor;
}

std::vector<int64_t> TensorDims(const TensorProto &tensor) {
  std::vector<int64_t> dims;
  dims.reserve(tensor.dims_size());
  for (int index = 0; index < tensor.dims_size(); ++index) {
    dims.push_back(tensor.dims(index));
  }
  return dims;
}

bool IsRankTwo(core::builder::GraphGraph &graph, const std::string &name) {
  return graph.HasShape(name) && graph.GetShape(name).Shape().Rank() == 2;
}

bool GemmBiasBroadcasts(const SymShape &output, const SymShape &bias) {
  if (output.Rank() != 2 || bias.Rank() > 2) {
    return false;
  }
  const std::size_t offset = 2 - bias.Rank();
  for (std::size_t index = 0; index < bias.Rank(); ++index) {
    const SymDim &dimension = bias[index];
    const SymDim &target = output[index + offset];
    if (dimension != target && (!dimension.IsInt() || dimension.AsInt() != 1)) {
      return false;
    }
  }
  return true;
}

struct ScalarScale {
  const NodeProto *node = nullptr;
  std::string variable;
  double factor = 0.0;
};

std::optional<ScalarScale> ReadScalarScale(core::builder::GraphGraph &graph,
                                           const NodeProto &node) {
  if ((!IsDefaultOp(node, "Mul") && !IsDefaultOp(node, "Div")) || node.input_size() != 2 ||
      node.output_size() != 1) {
    return std::nullopt;
  }
  const bool left_constant = graph.IsConstantScalar(node.input()[0].value());
  const bool right_constant = graph.IsConstantScalar(node.input()[1].value());
  if (left_constant == right_constant) {
    return std::nullopt;
  }
  const int constant_index = left_constant ? 0 : 1;
  if (IsDefaultOp(node, "Div") && constant_index == 0) {
    return std::nullopt;
  }
  const TensorProto *constant = graph.GetComputedConstant(node.input()[constant_index].value());
  double value = 0.0;
  if (constant == nullptr || !ReadScalar(*constant, value) || !std::isfinite(value)) {
    return std::nullopt;
  }
  if (IsDefaultOp(node, "Div") && value == 0.0) {
    return std::nullopt;
  }
  const double factor = IsDefaultOp(node, "Div") ? 1.0 / value : value;
  if (!std::isfinite(factor)) {
    return std::nullopt;
  }
  return ScalarScale{&node, node.input()[1 - constant_index].value(), factor};
}

bool CanUseGemmAlpha(const TensorProto &scalar, double factor) {
  const auto type = static_cast<TensorProto::DataType>(scalar.data_type());
  if (!IsFloatingType(type) || !std::isfinite(factor)) {
    return false;
  }
  return type != TensorProto::DataType::DOUBLE ||
         static_cast<double>(static_cast<float>(factor)) == factor;
}

bool IsActivation(const NodeProto &node) {
  static const std::set<std::string> activations = {
      "Cos",  "Cosh", "Elu", "Erf",  "Exp",      "Gelu", "LeakyRelu",
      "Relu", "Selu", "Sin", "Sinh", "Softplus", "Tan",  "Tanh",
  };
  return node.input_size() == 1 && node.output_size() == 1 &&
         NormaliseDomain(node.domain().value()) == kDefaultOnnxDomain &&
         activations.contains(node.op_type().value());
}

struct TransposeReshapeSide {
  const NodeProto *reshape = nullptr;
  const NodeProto *transpose = nullptr;
};

TransposeReshapeSide FindTransposeReshape(core::builder::GraphGraph &graph,
                                          const std::string &name) {
  if (graph.IsUsedMoreThanOnce(name)) {
    return {};
  }
  const NodeProto *reshape = graph.NodeBefore(name);
  if (reshape == nullptr || !IsDefaultOp(*reshape, "Reshape") || reshape->input_size() != 2 ||
      reshape->output_size() != 1 || graph.IsUsedMoreThanOnce(reshape->input()[0].value())) {
    return {};
  }
  const NodeProto *transpose = graph.NodeBefore(reshape->input()[0].value());
  if (transpose == nullptr || !IsDefaultOp(*transpose, "Transpose") ||
      transpose->input_size() != 1 || transpose->output_size() != 1) {
    return {};
  }
  std::vector<int64_t> perm;
  if (!GetAttributeInts(*transpose, "perm", perm) || perm.size() < 2) {
    return {};
  }
  for (std::size_t i = 0; i + 2 < perm.size(); ++i) {
    if (perm[i] != static_cast<int64_t>(i)) {
      return {};
    }
  }
  const int64_t rank = static_cast<int64_t>(perm.size());
  if (perm[perm.size() - 2] != rank - 1 || perm.back() != rank - 2) {
    return {};
  }
  return {reshape, transpose};
}

bool ValidTransposeReshapeSide(core::builder::GraphGraph &graph, const TransposeReshapeSide &side) {
  if (side.reshape == nullptr || !graph.IsConstant(side.reshape->input()[1].value()) ||
      !graph.HasShape(side.reshape->input()[0].value()) ||
      !graph.HasShape(side.reshape->output()[0].value())) {
    return false;
  }
  const SymShape &before = graph.GetShape(side.reshape->input()[0].value()).Shape();
  const SymShape &after = graph.GetShape(side.reshape->output()[0].value()).Shape();
  return before.Rank() >= 2 && after.Rank() >= 2 && Suffix(before, 2) == Suffix(after, 2);
}

} // namespace

std::set<std::string> GemmSumFusionPattern::FastOpType() const { return {"Sum"}; }

core::builder::MatchResult GemmSumFusionPattern::Match(core::builder::GraphGraph &graph,
                                                       const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Sum") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a two-input default-domain Sum");
  }
  const NodeProto *gemm = nullptr;
  int gemm_input = -1;
  for (int index = 0; index < 2; ++index) {
    const NodeProto *producer = graph.NodeBefore(candidate.input()[index].value());
    if (producer != nullptr && IsDefaultOp(*producer, "Gemm")) {
      if (gemm != nullptr) {
        return NoMatch(candidate, "both Sum inputs are produced by Gemm");
      }
      gemm = producer;
      gemm_input = index;
    }
  }
  if (gemm == nullptr || gemm->input_size() != 2 || gemm->output_size() != 1) {
    return NoMatch(candidate, "the Sum has no unbiased Gemm input");
  }
  if (graph.IsUsedMoreThanOnce(gemm->output()[0].value())) {
    return NoMatch(candidate, "the Gemm output has another consumer, output, or capture");
  }
  if (!graph.HasShape(gemm->output()[0].value()) ||
      !graph.HasShape(candidate.input()[1 - gemm_input].value()) ||
      !GemmBiasBroadcasts(graph.GetShape(gemm->output()[0].value()).Shape(),
                          graph.GetShape(candidate.input()[1 - gemm_input].value()).Shape())) {
    return NoMatch(candidate, "the Sum addend is not a valid Gemm bias");
  }
  return core::builder::MatchResult{this, {gemm, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
GemmSumFusionPattern::Apply(core::builder::GraphGraph &graph,
                            const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("GemmSumFusionPattern::Apply expects Gemm followed by Sum.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("GemmSumFusionPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &gemm = *nodes[0];
  const NodeProto &sum = *nodes[1];
  const std::string &bias =
      sum.input()[sum.input()[0].value() == gemm.output()[0].value() ? 1 : 0].value();
  NodeProto replacement =
      MakePatternNode("Gemm", {gemm.input()[0].value(), gemm.input()[1].value(), bias},
                      Outputs(sum), "", "GemmSumFusionPattern--" + gemm.name().value());
  replacement.set_doc_string(gemm.doc_string().value());
  for (int index = 0; index < gemm.attribute_size(); ++index) {
    if (gemm.attribute(index).name().value() != "beta") {
      *replacement.add_attribute() = gemm.attribute(index);
    }
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> MatMulAddPattern::FastOpType() const { return {"Gemm", "MatMul"}; }

core::builder::MatchResult MatMulAddPattern::Match(core::builder::GraphGraph &graph,
                                                   const NodeProto &candidate) const {
  if ((!IsDefaultOp(candidate, "MatMul") && !IsDefaultOp(candidate, "Gemm")) ||
      candidate.input_size() < 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain MatMul or Gemm");
  }
  if (IsDefaultOp(candidate, "Gemm") && GetAttributeOr<float>(candidate, "beta", 1.0F) != 1.0F) {
    return NoMatch(candidate, "a Gemm with beta different from one cannot absorb an Add");
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.input()[1].value())) {
    return NoMatch(candidate, "the matrix input ranks are unknown");
  }
  const SymShape &left_shape = graph.GetShape(candidate.input()[0].value()).Shape();
  const SymShape &right_shape = graph.GetShape(candidate.input()[1].value()).Shape();
  if (left_shape.Rank() < 2 || right_shape.Rank() != 2) {
    return NoMatch(candidate, "the matrix input ranks are not supported");
  }
  if (!allow_reshape_ && left_shape.Rank() != 2) {
    return NoMatch(candidate, "the left matrix requires a disabled reshape");
  }
  if (left_shape.Rank() > 2 && !left_shape[left_shape.Rank() - 1].IsInt() &&
      !right_shape[0].IsInt()) {
    return NoMatch(candidate, "the reduction dimension is symbolic on both inputs");
  }

  const std::vector<const NodeProto *> &next_nodes = graph.NextNodes(candidate.output()[0].value());
  if (next_nodes.size() != 1 || next_nodes[0] == nullptr || !IsDefaultOp(*next_nodes[0], "Add") ||
      next_nodes[0]->input_size() != 2 || next_nodes[0]->output_size() != 1) {
    return NoMatch(candidate, "the matrix output is not consumed by one Add");
  }
  if (graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
    return NoMatch(candidate, "the matrix output is also a graph output or subgraph capture");
  }
  const NodeProto &add = *next_nodes[0];
  const std::string &bias =
      add.input()[add.input()[1].value() == candidate.output()[0].value() ? 0 : 1].value();
  if (!graph.HasShape(candidate.input()[1].value()) || !graph.HasShape(bias)) {
    return NoMatch(candidate, "the weight or added bias shape is unknown");
  }
  const int64_t trans_b =
      IsDefaultOp(candidate, "Gemm") ? GetAttributeOr<int64_t>(candidate, "transB", 0) : 0;
  const SymDim &last_dimension =
      right_shape[right_shape.Rank() - 1 - static_cast<std::size_t>(trans_b)];
  const SymShape &bias_shape = graph.GetShape(bias).Shape();
  if (bias_shape.Rank() == 0 || last_dimension != bias_shape[bias_shape.Rank() - 1]) {
    return NoMatch(candidate, "the bias last dimension does not match the Gemm output");
  }
  if (bias_shape.Rank() > 1) {
    if (graph.HasShape(candidate.output()[0].value())) {
      if (graph.GetShape(candidate.output()[0].value()).Shape() != bias_shape) {
        return NoMatch(candidate, "the bias shape differs from the matrix output shape");
      }
    } else {
      for (std::size_t i = 0; i + 1 < bias_shape.Rank(); ++i) {
        if (!bias_shape[i].IsInt() || bias_shape[i].AsInt() <= 1) {
          return NoMatch(candidate, "the bias prefix cannot safely stand for a Gemm bias");
        }
      }
    }
  }
  if (add.input()[0].value() == add.input()[1].value()) {
    return NoMatch(candidate, "the Add uses the matrix output twice");
  }
  if (IsDefaultOp(candidate, "MatMul") || candidate.input_size() == 2) {
    return core::builder::MatchResult{this, {&candidate, &add}, &add};
  }
  if (candidate.input_size() < 3 || !graph.HasShape(candidate.input()[2].value()) ||
      graph.GetShape(candidate.input()[2].value()).Shape() != bias_shape) {
    return NoMatch(candidate, "the two Gemm bias shapes differ");
  }
  return core::builder::MatchResult{this, {&candidate, &add}, &add};
}

utils::RepeatedProtoField<NodeProto>
MatMulAddPattern::Apply(core::builder::GraphGraph &graph,
                        const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("MatMulAddPattern::Apply expects a MatMul or Gemm followed by Add.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("MatMulAddPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &matmul = *nodes[0];
  const NodeProto &add = *nodes[1];
  const std::string &added_bias =
      add.input()[add.input()[1].value() == matmul.output()[0].value() ? 0 : 1].value();
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "MatMulAddPattern--" + matmul.name().value();

  if (IsDefaultOp(matmul, "Gemm") && matmul.input_size() > 2) {
    const std::string combined_bias =
        builder.UniqueName("MatMulAddPattern--" + matmul.input()[2].value());
    NodeProto bias_add =
        MakePatternNode("Add", {added_bias, matmul.input()[2].value()}, {combined_bias}, "", name);
    NodeProto gemm = MakePatternNode(
        "Gemm", {matmul.input()[0].value(), matmul.input()[1].value(), combined_bias}, Outputs(add),
        "", name);
    gemm.set_doc_string(matmul.doc_string().value());
    CopyAttributes(matmul, gemm);
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(std::move(bias_add));
    replacements.push_back(std::move(gemm));
    return replacements;
  }

  const SymShape &left_shape = graph.GetShape(matmul.input()[0].value()).Shape();
  if (left_shape.Rank() == 2) {
    NodeProto gemm =
        MakePatternNode("Gemm", {matmul.input()[0].value(), matmul.input()[1].value(), added_bias},
                        Outputs(add), "", name);
    gemm.set_doc_string(matmul.doc_string().value());
    if (IsDefaultOp(matmul, "Gemm")) {
      CopyAttributes(matmul, gemm);
    }
    utils::RepeatedProtoField<NodeProto> replacements;
    replacements.push_back(std::move(gemm));
    return replacements;
  }

  const SymShape &right_shape = graph.GetShape(matmul.input()[1].value()).Shape();
  const int64_t reduction = left_shape[left_shape.Rank() - 1].IsInt()
                                ? left_shape[left_shape.Rank() - 1].AsInt()
                                : right_shape[0].AsInt();
  const std::string matrix_shape =
      AddShapeInitializer(graph, "MatMulAddPattern.new_shape.1", {-1, reduction});
  const std::string reshaped = builder.UniqueName("MatMulAddPattern--" + matmul.input()[0].value());
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(
      MakePatternNode("Reshape", {matmul.input()[0].value(), matrix_shape}, {reshaped}, "", name));

  std::string gemm_bias = added_bias;
  const SymShape &bias_shape = graph.GetShape(added_bias).Shape();
  if (bias_shape.Rank() > 2) {
    std::string flattened_bias_shape;
    if (bias_shape[bias_shape.Rank() - 1].IsInt()) {
      flattened_bias_shape = AddShapeInitializer(graph, "MatMulAddPattern.new_shape.3",
                                                 {-1, bias_shape[bias_shape.Rank() - 1].AsInt()});
    } else {
      const std::string last_bias_shape =
          builder.UniqueName("MatMulAddPattern--" + matmul.input()[0].value());
      NodeProto shape = MakePatternNode("Shape", {added_bias}, {last_bias_shape}, "", name);
      AddAttribute<int64_t>(shape, "start", -1);
      replacements.push_back(std::move(shape));
      flattened_bias_shape = builder.UniqueName("MatMulAddPattern--" + matmul.input()[0].value());
      const std::string minus_one =
          AddShapeInitializer(graph, "MatMulAddPattern.new_shape.7", {-1});
      NodeProto concat =
          MakePatternNode("Concat", {minus_one, last_bias_shape}, {flattened_bias_shape}, "", name);
      AddAttribute<int64_t>(concat, "axis", 0);
      replacements.push_back(std::move(concat));
    }
    gemm_bias = builder.UniqueName("MatMulAddPattern--" + matmul.input()[0].value());
    replacements.push_back(
        MakePatternNode("Reshape", {added_bias, flattened_bias_shape}, {gemm_bias}, "", name));
  }

  const std::string unshaped = builder.UniqueName("MatMulAddPattern--" + matmul.input()[0].value());
  std::string shape_back;
  if (left_shape.IsFullyKnown()) {
    std::vector<int64_t> final_shape;
    final_shape.reserve(left_shape.Rank());
    for (std::size_t i = 0; i + 1 < left_shape.Rank(); ++i) {
      final_shape.push_back(left_shape[i].AsInt());
    }
    final_shape.push_back(-1);
    shape_back = AddShapeInitializer(graph, "MatMulAddPattern.new_shape.2", final_shape);
  } else {
    const std::string prefix_shape =
        builder.UniqueName("MatMulAddPattern--" + matmul.input()[0].value());
    NodeProto shape =
        MakePatternNode("Shape", {matmul.input()[0].value()}, {prefix_shape}, "", name);
    AddAttribute<int64_t>(shape, "start", 0);
    AddAttribute<int64_t>(shape, "end", -1);
    replacements.push_back(std::move(shape));
    shape_back = builder.UniqueName("MatMulAddPattern--" + matmul.input()[0].value());
    const std::string minus_one = AddShapeInitializer(graph, "MatMulAddPattern.new_shape.3", {-1});
    NodeProto concat = MakePatternNode("Concat", {prefix_shape, minus_one}, {shape_back}, "", name);
    AddAttribute<int64_t>(concat, "axis", 0);
    replacements.push_back(std::move(concat));
  }

  NodeProto gemm = MakePatternNode("Gemm", {reshaped, matmul.input()[1].value(), gemm_bias},
                                   {unshaped}, "", name);
  gemm.set_doc_string(matmul.doc_string().value());
  if (IsDefaultOp(matmul, "Gemm")) {
    CopyAttributes(matmul, gemm);
  }
  replacements.push_back(std::move(gemm));
  replacements.push_back(
      MakePatternNode("Reshape", {unshaped, shape_back}, Outputs(add), "", name));
  return replacements;
}

std::set<std::string> GemmTransposePattern::FastOpType() const { return {"Gemm"}; }

core::builder::MatchResult GemmTransposePattern::Match(core::builder::GraphGraph &graph,
                                                       const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Gemm") || candidate.input_size() < 2 ||
      candidate.output_size() == 0) {
    return NoMatch(candidate, "candidate is not a default-domain Gemm");
  }
  if (!graph.IsConstant(candidate.input()[1].value())) {
    return NoMatch(candidate, "the Gemm right input is not constant");
  }
  if (GetAttributeOr<float>(candidate, "beta", 1.0F) != 1.0F ||
      GetAttributeOr<int64_t>(candidate, "transA", 0) != 0 ||
      GetAttributeOr<int64_t>(candidate, "transB", 0) != 0) {
    return NoMatch(candidate, "the Gemm transpose or beta attributes are not supported");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
GemmTransposePattern::Apply(core::builder::GraphGraph &graph,
                            const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("GemmTransposePattern::Apply expects one Gemm node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("GemmTransposePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &gemm = *nodes[0];
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string transposed =
      builder.UniqueName("GemmTransposePattern--" + gemm.input()[1].value());
  const std::string name = "GemmTransposePattern--" + gemm.name().value();
  NodeProto transpose =
      MakePatternNode("Transpose", {gemm.input()[1].value()}, {transposed}, "", name);
  AddAttribute(transpose, "perm", std::vector<int64_t>{1, 0});
  transpose.set_doc_string(gemm.doc_string().value());

  std::vector<std::string> inputs = {gemm.input()[0].value(), transposed};
  for (int i = 2; i < gemm.input_size(); ++i) {
    inputs.push_back(gemm.input()[i].value());
  }
  NodeProto replacement = MakePatternNode("Gemm", inputs, Outputs(gemm), "", name);
  AddAttribute<int64_t>(replacement, "transB", 1);
  for (const char *attribute_name : {"alpha", "beta"}) {
    const AttributeProto *attribute = FindAttribute(gemm, attribute_name);
    if (attribute != nullptr) {
      *replacement.add_attribute() = *attribute;
    }
  }
  replacement.set_doc_string(gemm.doc_string().value());
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(transpose));
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> MatMulReshape2Of3Pattern::FastOpType() const {
  return {"FusedMatMul", "MatMul"};
}

core::builder::MatchResult MatMulReshape2Of3Pattern::Match(core::builder::GraphGraph &graph,
                                                           const NodeProto &candidate) const {
  const bool matmul = IsDefaultOp(candidate, "MatMul");
  const bool fused =
      candidate.op_type().value() == "FusedMatMul" && candidate.domain().value() == "com.microsoft";
  if ((!matmul && !fused) || candidate.input_size() != 2 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not MatMul or com.microsoft FusedMatMul");
  }
  if (fused && (GetAttributeOr<int64_t>(candidate, "transA", 0) != 0 ||
                GetAttributeOr<int64_t>(candidate, "transB", 0) != 0 ||
                GetAttributeOr<int64_t>(candidate, "transBatchA", 0) != 0 ||
                GetAttributeOr<int64_t>(candidate, "transBatchB", 0) != 0)) {
    return NoMatch(candidate, "FusedMatMul transpose attributes are not supported");
  }

  const std::vector<const NodeProto *> &successors = graph.NextNodes(candidate.output()[0].value());
  if (successors.size() > 1 ||
      (successors.empty() && !graph.IsOutput(candidate.output()[0].value()))) {
    return NoMatch(candidate, "the MatMul output has an unsupported number of uses");
  }
  const NodeProto *next = successors.empty() ? nullptr : successors[0];
  const NodeProto *left = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *right = graph.NodeBefore(candidate.input()[1].value());
  const bool left_reshape = left != nullptr && IsDefaultOp(*left, "Reshape") &&
                            left->input_size() == 2 && left->output_size() == 1;
  const bool right_reshape = right != nullptr && IsDefaultOp(*right, "Reshape") &&
                             right->input_size() == 2 && right->output_size() == 1;
  const bool next_reshape = next != nullptr && IsDefaultOp(*next, "Reshape") &&
                            next->input_size() == 2 && next->output_size() == 1 &&
                            next->input()[0].value() == candidate.output()[0].value();
  const int reshape_count = static_cast<int>(left_reshape) + static_cast<int>(right_reshape) +
                            static_cast<int>(next_reshape);
  if (reshape_count < 2) {
    return NoMatch(candidate, "fewer than two of the three adjacent nodes are Reshape");
  }
  left = left_reshape ? left : nullptr;
  right = right_reshape ? right : nullptr;
  next = next_reshape ? next : nullptr;

  std::vector<int64_t> left_shape;
  std::vector<int64_t> right_shape;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> effective_left;
  std::vector<int64_t> effective_right;
  if (!ReadStaticShape(graph, candidate.input()[0].value(), left_shape) ||
      !ReadStaticShape(graph, candidate.input()[1].value(), right_shape) ||
      !ReadStaticShape(graph, candidate.output()[0].value(), output_shape) ||
      (left != nullptr && !ReadStaticShape(graph, left->input()[0].value(), effective_left)) ||
      (right != nullptr && !ReadStaticShape(graph, right->input()[0].value(), effective_right))) {
    return NoMatch(candidate, "the MatMul and adjacent Reshape shapes are not static");
  }
  if (left == nullptr) {
    effective_left = left_shape;
  }
  if (right == nullptr) {
    effective_right = right_shape;
  }
  if (left_shape.size() < 2 || right_shape.size() < 2 || effective_left.size() < 2 ||
      effective_right.size() < 2) {
    return NoMatch(candidate, "an adjacent matrix shape has rank below two");
  }
  if ((left != nullptr && (!SameElementCount(effective_left, left_shape) ||
                           Suffix(effective_left, 2) != Suffix(left_shape, 2))) ||
      (right != nullptr && (!SameElementCount(effective_right, right_shape) ||
                            Suffix(effective_right, 2) != Suffix(right_shape, 2)))) {
    return NoMatch(candidate, "a pre-Reshape changes matrix dimensions or element count");
  }
  const std::optional<std::vector<int64_t>> original_output = MatMulShape(left_shape, right_shape);
  if (!original_output.has_value() || original_output.value() != output_shape) {
    return NoMatch(candidate, "the recorded MatMul output shape is inconsistent with its inputs");
  }

  std::vector<int64_t> final_shape = output_shape;
  if (next != nullptr) {
    if (!ReadStaticShape(graph, next->output()[0].value(), final_shape) ||
        !SameElementCount(output_shape, final_shape) ||
        Suffix(final_shape, 2) != Suffix(output_shape, 2)) {
      return NoMatch(candidate, "the post-Reshape changes result size or matrix dimensions");
    }
    if (final_shape.size() != effective_left.size() &&
        final_shape.size() != effective_right.size()) {
      return NoMatch(candidate, "the post-Reshape rank matches neither effective input rank");
    }
  }
  const std::vector<int64_t> desired_batch = Prefix(final_shape, 2);
  const std::optional<int64_t> desired_batch_size = ElementCount(desired_batch);
  if (!desired_batch_size.has_value() ||
      ElementCount(Prefix(effective_left, 2)) != desired_batch_size ||
      ElementCount(Prefix(effective_right, 2)) != desired_batch_size) {
    return NoMatch(candidate, "the effective and result batch sizes differ");
  }
  const std::vector<int64_t> desired_left = WithMatrixSuffix(desired_batch, left_shape);
  const std::vector<int64_t> desired_right = WithMatrixSuffix(desired_batch, right_shape);
  const std::optional<std::vector<int64_t>> rewritten_output =
      MatMulShape(desired_left, desired_right);
  if (!SameElementCount(effective_left, desired_left) ||
      !SameElementCount(effective_right, desired_right) || !rewritten_output.has_value() ||
      rewritten_output.value() != final_shape) {
    return NoMatch(candidate, "the rewritten MatMul shapes do not produce the final result");
  }
  return core::builder::MatchResult{this, {left, right, &candidate, next}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
MatMulReshape2Of3Pattern::Apply(core::builder::GraphGraph &graph,
                                const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 4 || nodes[2] == nullptr) {
    throw BuilderError(
        "MatMulReshape2Of3Pattern::Apply expects left, right, MatMul, and output slots.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("MatMulReshape2Of3Pattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto *left = nodes[0];
  const NodeProto *right = nodes[1];
  const NodeProto &matmul = *nodes[2];
  const NodeProto *next = nodes[3];
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "MatMulReshape2Of3Pattern--" + matmul.name().value();
  const std::vector<int64_t> left_shape =
      StaticShapeOrThrow(graph, matmul.input()[0].value(), "MatMulReshape2Of3Pattern");
  const std::vector<int64_t> right_shape =
      StaticShapeOrThrow(graph, matmul.input()[1].value(), "MatMulReshape2Of3Pattern");
  const std::vector<int64_t> effective_left =
      left == nullptr
          ? left_shape
          : StaticShapeOrThrow(graph, left->input()[0].value(), "MatMulReshape2Of3Pattern");
  const std::vector<int64_t> effective_right =
      right == nullptr
          ? right_shape
          : StaticShapeOrThrow(graph, right->input()[0].value(), "MatMulReshape2Of3Pattern");
  const std::vector<int64_t> final_shape = StaticShapeOrThrow(
      graph, next == nullptr ? matmul.output()[0].value() : next->output()[0].value(),
      "MatMulReshape2Of3Pattern");
  const std::vector<int64_t> desired_batch = Prefix(final_shape, 2);
  const std::vector<int64_t> desired_left = WithMatrixSuffix(desired_batch, left_shape);
  const std::vector<int64_t> desired_right = WithMatrixSuffix(desired_batch, right_shape);

  utils::RepeatedProtoField<NodeProto> replacements;
  auto prepare_input = [&](const NodeProto *reshape, const std::string &current_name,
                           const std::vector<int64_t> &effective_shape,
                           const std::vector<int64_t> &desired_shape, const char *shape_base,
                           const std::string &name_base) {
    const std::string source_name = reshape == nullptr ? current_name : reshape->input()[0].value();
    if (reshape != nullptr && graph.IsUsedMoreThanOnce(reshape->output()[0].value())) {
      replacements.push_back(*reshape);
    }
    if (effective_shape == desired_shape) {
      return source_name;
    }
    const std::string shape = AddShapeInitializer(graph, shape_base, desired_shape);
    const std::string output = builder.UniqueName(name_base);
    replacements.push_back(MakePatternNode("Reshape", {source_name, shape}, {output}, "", name));
    return output;
  };
  const std::string left_name =
      prepare_input(left, matmul.input()[0].value(), effective_left, desired_left,
                    "MatMulReshape2Of3Pattern.apply.shape.1",
                    "MatMulReshape2Of3PatternL_" + matmul.input()[0].value());
  const std::string right_name =
      prepare_input(right, matmul.input()[1].value(), effective_right, desired_right,
                    "MatMulReshape2Of3Pattern.apply.shape.2",
                    "MatMulReshape2Of3PatternL_" + matmul.input()[1].value());

  auto make_matmul = [&](const std::vector<std::string> &inputs,
                         const std::vector<std::string> &outputs) {
    NodeProto replacement = MakePatternNode(matmul.op_type().value().c_str(), inputs, outputs,
                                            matmul.domain().value(), name);
    CopyAttributes(matmul, replacement);
    return replacement;
  };

  if (next == nullptr) {
    replacements.push_back(make_matmul({left_name, right_name}, {matmul.output()[0].value()}));
    return replacements;
  }

  replacements.push_back(make_matmul({left_name, right_name}, {next->output()[0].value()}));
  if (graph.IsUsedMoreThanOnce(matmul.output()[0].value())) {
    const std::vector<int64_t> previous =
        StaticShapeOrThrow(graph, matmul.output()[0].value(), "MatMulReshape2Of3Pattern");
    const std::string shape =
        AddShapeInitializer(graph, "MatMulReshape2Of3Pattern.apply.shape.4", previous);
    replacements.push_back(MakePatternNode("Reshape", {next->output()[0].value(), shape},
                                           {matmul.output()[0].value()}, "", name));
  }
  return replacements;
}

std::set<std::string> MulMulMatMulPattern::FastOpType() const { return {"MatMul"}; }

core::builder::MatchResult MulMulMatMulPattern::Match(core::builder::GraphGraph &graph,
                                                      const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "MatMul") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain MatMul");
  }
  const NodeProto *left = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *right = graph.NodeBefore(candidate.input()[1].value());
  if (left == nullptr || right == nullptr || !IsDefaultOp(*left, "Mul") ||
      !IsDefaultOp(*right, "Mul") || left->input_size() != 2 || right->input_size() != 2 ||
      left->output_size() != 1 || right->output_size() != 1) {
    return NoMatch(candidate, "both MatMul inputs are not produced by default-domain Mul");
  }
  if (graph.IsUsedMoreThanOnce(left->output()[0].value()) ||
      graph.IsUsedMoreThanOnce(right->output()[0].value())) {
    return NoMatch(candidate, "a Mul output has another consumer, output, or subgraph capture");
  }
  std::vector<std::string> constants;
  for (const NodeProto *mul : {left, right}) {
    for (int i = 0; i < mul->input_size(); ++i) {
      if (graph.IsConstant(mul->input()[i].value())) {
        constants.push_back(mul->input()[i].value());
      }
    }
  }
  if (constants.size() != 2 ||
      !std::all_of(constants.begin(), constants.end(),
                   [&](const std::string &name) { return graph.IsConstantScalar(name); })) {
    return NoMatch(candidate, "the two Mul nodes do not contain exactly two scalar constants");
  }
  return core::builder::MatchResult{this, {left, right, &candidate}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
MulMulMatMulPattern::Apply(core::builder::GraphGraph &graph,
                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError("MulMulMatMulPattern::Apply expects two Mul nodes and one MatMul.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("MulMulMatMulPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &left = *nodes[0];
  const NodeProto &right = *nodes[1];
  const NodeProto &matmul = *nodes[2];
  std::vector<std::string> constants;
  std::vector<std::string> variables;
  for (const NodeProto *mul : {&left, &right}) {
    for (int i = 0; i < mul->input_size(); ++i) {
      const std::string &input = mul->input()[i].value();
      (graph.IsConstant(input) ? constants : variables).push_back(input);
    }
  }
  if (constants.size() != 2 || variables.size() != 2) {
    throw BuilderError("MulMulMatMulPattern::Apply could not separate constants from inputs.");
  }
  const TensorProto *left_constant = graph.GetComputedConstant(constants[0]);
  const TensorProto *right_constant = graph.GetComputedConstant(constants[1]);
  TensorProto product;
  if (left_constant == nullptr || right_constant == nullptr ||
      !MultiplyScalarConstants(*left_constant, *right_constant, product)) {
    throw BuilderError("MulMulMatMulPattern::Apply could not multiply the scalar constants.");
  }
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string constant_name = FreeInitializerName(builder, "MulMulMatMulPattern.apply.ccc");
  product.set_name(constant_name);
  builder.MakeInitializer(product);
  const std::string intermediate =
      builder.UniqueName("MulMulMatMulPattern_" + matmul.output()[0].value());
  const std::string name = "MulMulMatMulPattern--" + matmul.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakePatternNode("MatMul", variables, {intermediate}, "", name + "-1"));
  replacements.push_back(MakePatternNode("Mul", {intermediate, constant_name},
                                         {matmul.output()[0].value()}, "", name + "-2"));
  return replacements;
}

std::set<std::string> MatMulBatchNormalizationFusionPattern::FastOpType() const {
  return {"BatchNormalization"};
}

core::builder::MatchResult
MatMulBatchNormalizationFusionPattern::Match(core::builder::GraphGraph &graph,
                                             const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "BatchNormalization") || candidate.input_size() < 5 ||
      candidate.output_size() < 1) {
    return NoMatch(candidate, "candidate is not an inference BatchNormalization");
  }
  const int opset = graph.Builder().OpsetVersion("");
  if ((opset < 14 && candidate.output_size() != 1) ||
      (opset >= 14 && GetAttributeOr<int64_t>(candidate, "training_mode", 0) != 0)) {
    return NoMatch(candidate, "the BatchNormalization is in training mode");
  }
  for (int index = 1; index < candidate.output_size(); ++index) {
    if (!candidate.output()[index].value().empty() &&
        graph.IsUsed(candidate.output()[index].value())) {
      return NoMatch(candidate, "an auxiliary BatchNormalization output is used");
    }
  }
  const NodeProto *matmul = graph.NodeBefore(candidate.input()[0].value());
  if (matmul == nullptr || !IsDefaultOp(*matmul, "MatMul") || matmul->input_size() != 2 ||
      matmul->output_size() != 1 || !IsRankTwo(graph, matmul->input()[0].value()) ||
      !IsRankTwo(graph, matmul->input()[1].value()) ||
      graph.IsUsedMoreThanOnce(matmul->output()[0].value())) {
    return NoMatch(candidate, "the BatchNormalization input is not a private rank-two MatMul");
  }
  for (int index = 1; index < 5; ++index) {
    if (!graph.IsConstant(candidate.input()[index].value())) {
      return NoMatch(candidate, "a BatchNormalization parameter is not constant");
    }
  }
  if (!graph.IsConstant(matmul->input()[1].value())) {
    return NoMatch(candidate, "the MatMul weight is not constant");
  }
  const TensorProto *weight = graph.GetComputedConstant(matmul->input()[1].value());
  if (weight == nullptr || weight->dims_size() != 2 || weight->dims(1) <= 0) {
    return NoMatch(candidate, "the MatMul weight is not a valid rank-two tensor");
  }
  const auto type = static_cast<TensorProto::DataType>(weight->data_type());
  if (!IsFloatingType(type)) {
    return NoMatch(candidate, "the MatMul weight is not floating-point");
  }
  const int64_t channels = weight->dims(1);
  std::vector<double> weight_values;
  if (!ReadFloatingTensor(*weight, weight_values) ||
      weight_values.size() != static_cast<std::size_t>(weight->dims(0) * weight->dims(1)) ||
      !std::all_of(weight_values.begin(), weight_values.end(),
                   [](double value) { return std::isfinite(value); })) {
    return NoMatch(candidate, "the MatMul weight has invalid values");
  }
  std::vector<std::vector<double>> parameters;
  for (int index = 1; index < 5; ++index) {
    const TensorProto *parameter = graph.GetComputedConstant(candidate.input()[index].value());
    std::vector<double> values;
    if (parameter == nullptr || parameter->data_type() != weight->data_type() ||
        parameter->dims_size() != 1 || parameter->dims(0) != channels ||
        !ReadFloatingTensor(*parameter, values) ||
        values.size() != static_cast<std::size_t>(channels)) {
      return NoMatch(candidate, "a BatchNormalization parameter has an incompatible type or shape");
    }
    parameters.push_back(std::move(values));
  }
  const double epsilon = GetAttributeOr<float>(candidate, "epsilon", 1.0e-5F);
  if (!std::isfinite(epsilon) || epsilon < 0.0) {
    return NoMatch(candidate, "BatchNormalization epsilon is invalid");
  }
  for (int64_t channel = 0; channel < channels; ++channel) {
    const std::size_t index = static_cast<std::size_t>(channel);
    const double denominator = parameters[3][index] + epsilon;
    if (!(denominator > 0.0) || !std::isfinite(denominator)) {
      return NoMatch(candidate, "a BatchNormalization variance is invalid");
    }
    const double factor = parameters[0][index] / std::sqrt(denominator);
    const double folded_bias = parameters[1][index] - parameters[2][index] * factor;
    if (!std::isfinite(factor) || !std::isfinite(folded_bias)) {
      return NoMatch(candidate, "the folded BatchNormalization parameters are non-finite");
    }
  }
  return core::builder::MatchResult{this, {matmul, &candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
MatMulBatchNormalizationFusionPattern::Apply(core::builder::GraphGraph &graph,
                                             const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError(
        "MatMulBatchNormalizationFusionPattern::Apply expects MatMul and BatchNormalization.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("MatMulBatchNormalizationFusionPattern::Apply received an unsafe match.");
  }
  const NodeProto &matmul = *nodes[0];
  const NodeProto &batch = *nodes[1];
  const TensorProto *weight = graph.GetComputedConstant(matmul.input()[1].value());
  if (weight == nullptr) {
    throw BuilderError(
        "MatMulBatchNormalizationFusionPattern::Apply could not read the MatMul weight.");
  }
  std::vector<double> weight_values;
  std::vector<double> scale;
  std::vector<double> bias;
  std::vector<double> mean;
  std::vector<double> variance;
  if (!ReadFloatingTensor(*weight, weight_values) ||
      !ReadFloatingTensor(*graph.GetComputedConstant(batch.input()[1].value()), scale) ||
      !ReadFloatingTensor(*graph.GetComputedConstant(batch.input()[2].value()), bias) ||
      !ReadFloatingTensor(*graph.GetComputedConstant(batch.input()[3].value()), mean) ||
      !ReadFloatingTensor(*graph.GetComputedConstant(batch.input()[4].value()), variance)) {
    throw BuilderError(
        "MatMulBatchNormalizationFusionPattern::Apply could not read constant values.");
  }
  const int64_t channels = weight->dims(1);
  const float epsilon = GetAttributeOr<float>(batch, "epsilon", 1.0e-5F);
  std::vector<double> folded_bias(static_cast<std::size_t>(channels));
  std::vector<double> factors(static_cast<std::size_t>(channels));
  for (int64_t channel = 0; channel < channels; ++channel) {
    const std::size_t index = static_cast<std::size_t>(channel);
    factors[index] = scale[index] / std::sqrt(variance[index] + epsilon);
    folded_bias[index] = bias[index] - mean[index] * factors[index];
    if (!std::isfinite(factors[index]) || !std::isfinite(folded_bias[index])) {
      throw BuilderError(
          "MatMulBatchNormalizationFusionPattern::Apply produced non-finite parameters.");
    }
  }
  for (std::size_t index = 0; index < weight_values.size(); ++index) {
    weight_values[index] *= factors[index % static_cast<std::size_t>(channels)];
  }
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string weight_name =
      FreeInitializerName(builder, "MatMulBatchNormalizationFusionPattern.weight");
  const std::string bias_name =
      FreeInitializerName(builder, "MatMulBatchNormalizationFusionPattern.bias");
  const auto type = static_cast<TensorProto::DataType>(weight->data_type());
  builder.MakeInitializer(
      MakeFloatingTensor(weight_name, type, TensorDims(*weight), weight_values));
  builder.MakeInitializer(MakeFloatingTensor(bias_name, type, {channels}, folded_bias));
  NodeProto replacement = MakePatternNode(
      "Gemm", {matmul.input()[0].value(), weight_name, bias_name}, {batch.output()[0].value()}, "",
      "MatMulBatchNormalizationFusionPattern--" + matmul.name().value());
  replacement.set_doc_string(matmul.doc_string().value());
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> MatMulScaleFusionPattern::FastOpType() const { return {"MatMul"}; }

core::builder::MatchResult MatMulScaleFusionPattern::Match(core::builder::GraphGraph &graph,
                                                           const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "MatMul") || candidate.input_size() != 2 ||
      candidate.output_size() != 1 || !IsRankTwo(graph, candidate.input()[0].value()) ||
      !IsRankTwo(graph, candidate.input()[1].value())) {
    return NoMatch(candidate, "candidate is not a rank-two default-domain MatMul");
  }
  std::vector<const NodeProto *> scales;
  const std::vector<const NodeProto *> &consumers = graph.NextNodes(candidate.output()[0].value());
  if (consumers.size() == 1 && !graph.IsUsedMoreThanOnce(candidate.output()[0].value()) &&
      ReadScalarScale(graph, *consumers[0]).has_value()) {
    scales.push_back(consumers[0]);
  }
  for (int index = 0; index < 2; ++index) {
    const NodeProto *producer = graph.NodeBefore(candidate.input()[index].value());
    if (producer != nullptr && !graph.IsUsedMoreThanOnce(producer->output()[0].value()) &&
        ReadScalarScale(graph, *producer).has_value()) {
      scales.push_back(producer);
    }
  }
  if (scales.size() != 1) {
    return NoMatch(candidate, "the MatMul must have exactly one safe adjacent scalar scale");
  }
  const ScalarScale scale = ReadScalarScale(graph, *scales[0]).value();
  std::vector<std::string> inputs = Inputs(candidate);
  for (std::string &input : inputs) {
    if (input == scale.node->output()[0].value()) {
      input = scale.variable;
    }
  }
  const int fold_index = graph.IsConstant(inputs[1]) ? 1 : (graph.IsConstant(inputs[0]) ? 0 : -1);
  const TensorProto *scalar = nullptr;
  for (int index = 0; index < scale.node->input_size(); ++index) {
    if (graph.IsConstantScalar(scale.node->input()[index].value())) {
      scalar = graph.GetComputedConstant(scale.node->input()[index].value());
      break;
    }
  }
  if (scalar == nullptr) {
    return NoMatch(candidate, "the scalar value is unavailable");
  }
  if (fold_index >= 0) {
    const TensorProto *matrix = graph.GetComputedConstant(inputs[fold_index]);
    std::vector<double> values;
    if (matrix == nullptr || matrix->data_type() != scalar->data_type() ||
        !IsFloatingType(static_cast<TensorProto::DataType>(matrix->data_type())) ||
        !ReadFloatingTensor(*matrix, values)) {
      return NoMatch(candidate, "the constant matrix cannot absorb the scalar");
    }
  } else if (!CanUseGemmAlpha(*scalar, scale.factor)) {
    return NoMatch(candidate, "the scalar cannot be represented safely as Gemm alpha");
  }
  return core::builder::MatchResult{this, {&candidate, scale.node}, scale.node};
}

utils::RepeatedProtoField<NodeProto>
MatMulScaleFusionPattern::Apply(core::builder::GraphGraph &graph,
                                const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("MatMulScaleFusionPattern::Apply expects MatMul and one scalar scale.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("MatMulScaleFusionPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &matmul = *nodes[0];
  const ScalarScale scale = ReadScalarScale(graph, *nodes[1]).value();
  std::vector<std::string> inputs = Inputs(matmul);
  bool post_scale = true;
  for (std::string &input : inputs) {
    if (input == scale.node->output()[0].value()) {
      input = scale.variable;
      post_scale = false;
    }
  }
  const std::vector<std::string> outputs = post_scale ? Outputs(*scale.node) : Outputs(matmul);
  const int fold_index = graph.IsConstant(inputs[1]) ? 1 : (graph.IsConstant(inputs[0]) ? 0 : -1);
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "MatMulScaleFusionPattern--" + matmul.name().value();
  NodeProto replacement;
  if (fold_index >= 0) {
    const TensorProto *matrix = graph.GetComputedConstant(inputs[fold_index]);
    if (matrix == nullptr) {
      throw BuilderError("MatMulScaleFusionPattern::Apply could not read the constant matrix.");
    }
    std::vector<double> values;
    if (!ReadFloatingTensor(*matrix, values)) {
      throw BuilderError("MatMulScaleFusionPattern::Apply could not fold the constant matrix.");
    }
    for (double &value : values) {
      value *= scale.factor;
    }
    const std::string folded_name = FreeInitializerName(builder, "MatMulScaleFusionPattern.weight");
    builder.MakeInitializer(
        MakeFloatingTensor(folded_name, static_cast<TensorProto::DataType>(matrix->data_type()),
                           TensorDims(*matrix), values));
    inputs[fold_index] = folded_name;
    replacement = MakePatternNode("MatMul", inputs, outputs, "", name);
  } else {
    replacement = MakePatternNode("Gemm", inputs, outputs, "", name);
    AddAttribute<float>(replacement, "alpha", static_cast<float>(scale.factor));
  }
  replacement.set_doc_string(matmul.doc_string().value());
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> ReshapeMatMulReshapePattern::FastOpType() const { return {"MatMul"}; }

core::builder::MatchResult ReshapeMatMulReshapePattern::Match(core::builder::GraphGraph &graph,
                                                              const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "MatMul") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain MatMul");
  }
  if (graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
    return NoMatch(candidate, "the MatMul output has another use");
  }
  const std::vector<const NodeProto *> &successors = graph.NextNodes(candidate.output()[0].value());
  if (successors.size() != 1 || successors[0] == nullptr ||
      !IsDefaultOp(*successors[0], "Reshape") || successors[0]->input_size() != 2 ||
      successors[0]->output_size() != 1) {
    return NoMatch(candidate, "the MatMul is not followed by Reshape");
  }
  const NodeProto *left = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *right = graph.NodeBefore(candidate.input()[1].value());
  if (left == nullptr || right == nullptr || !IsDefaultOp(*left, "Reshape") ||
      !IsDefaultOp(*right, "Reshape") || left->input_size() < 2 || right->input_size() < 2) {
    return NoMatch(candidate, "both MatMul inputs are not produced by default-domain Reshape");
  }
  if (!graph.IsConstant(left->input()[1].value()) || !graph.IsConstant(right->input()[1].value()) ||
      !graph.IsConstant(successors[0]->input()[1].value())) {
    return NoMatch(candidate, "a Reshape target is not constant");
  }
  std::vector<int64_t> left_source;
  std::vector<int64_t> right_source;
  std::vector<int64_t> left_reshaped;
  std::vector<int64_t> right_reshaped;
  std::vector<int64_t> matmul_output;
  std::vector<int64_t> final_shape;
  if (!ReadStaticShape(graph, left->input()[0].value(), left_source) ||
      !ReadStaticShape(graph, right->input()[0].value(), right_source) ||
      !ReadStaticShape(graph, left->output()[0].value(), left_reshaped) ||
      !ReadStaticShape(graph, right->output()[0].value(), right_reshaped) ||
      !ReadStaticShape(graph, candidate.output()[0].value(), matmul_output) ||
      !ReadStaticShape(graph, successors[0]->output()[0].value(), final_shape)) {
    return NoMatch(candidate, "a source, reshaped, or result shape is not static");
  }
  if (final_shape.size() < 4 || left_reshaped.size() != 3 || right_reshaped.size() != 3) {
    return NoMatch(candidate, "the Reshape result ranks are not supported");
  }
  if (left_source.size() != final_shape.size() || right_source.size() != final_shape.size()) {
    return NoMatch(candidate, "the original input ranks differ from the restored rank");
  }
  if (!SameElementCount(left_source, left_reshaped) ||
      !SameElementCount(right_source, right_reshaped) ||
      !SameElementCount(matmul_output, final_shape) ||
      Suffix(left_source, 2) != Suffix(left_reshaped, 2) ||
      Suffix(right_source, 2) != Suffix(right_reshaped, 2)) {
    return NoMatch(candidate, "a Reshape changes element count or matrix dimensions");
  }
  const std::optional<std::vector<int64_t>> original_result =
      MatMulShape(left_reshaped, right_reshaped);
  if (!original_result.has_value() || original_result.value() != matmul_output) {
    return NoMatch(candidate, "the reshaped MatMul result is inconsistent");
  }
  const std::optional<std::vector<int64_t>> direct_result = MatMulShape(left_source, right_source);
  if (!direct_result.has_value() || direct_result.value() != final_shape) {
    return NoMatch(candidate, "the original batch prefixes do not broadcast to the final shape");
  }
  return core::builder::MatchResult{this, {left, right, &candidate, successors[0]}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
ReshapeMatMulReshapePattern::Apply(core::builder::GraphGraph &graph,
                                   const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 4 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr ||
      nodes[3] == nullptr) {
    throw BuilderError(
        "ReshapeMatMulReshapePattern::Apply expects two Reshape, MatMul, and Reshape nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ReshapeMatMulReshapePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &left = *nodes[0];
  const NodeProto &right = *nodes[1];
  const NodeProto &matmul = *nodes[2];
  const NodeProto &output_reshape = *nodes[3];
  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.IsUsedMoreThanOnce(left.output()[0].value())) {
    replacements.push_back(left);
  }
  if (graph.IsUsedMoreThanOnce(right.output()[0].value())) {
    replacements.push_back(right);
  }
  NodeProto replacement = MakePatternNode(
      "MatMul", {left.input()[0].value(), right.input()[0].value()}, Outputs(output_reshape), "",
      "ReshapeMatMulReshapePattern--" + matmul.name().value());
  replacement.set_doc_string(output_reshape.doc_string().value());
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> TransposeMatMulPattern::FastOpType() const { return {"Gemm", "MatMul"}; }

core::builder::MatchResult TransposeMatMulPattern::Match(core::builder::GraphGraph &graph,
                                                         const NodeProto &candidate) const {
  if ((!IsDefaultOp(candidate, "MatMul") && !IsDefaultOp(candidate, "Gemm")) ||
      candidate.input_size() < 2 || candidate.output_size() == 0) {
    return NoMatch(candidate, "candidate is not a default-domain MatMul or Gemm");
  }
  if (IsDefaultOp(candidate, "MatMul") && !MainOpsetAtLeast(graph, 11)) {
    return NoMatch(candidate, "Gemm with two inputs is unavailable before opset 11");
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.input()[1].value()) ||
      graph.GetShape(candidate.input()[0].value()).Shape().Rank() != 2 ||
      graph.GetShape(candidate.input()[1].value()).Shape().Rank() != 2) {
    return NoMatch(candidate, "both matrix inputs are not rank two");
  }
  const NodeProto *left = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *right = graph.NodeBefore(candidate.input()[1].value());
  if (left == nullptr || !IsDefaultOp(*left, "Transpose")) {
    left = nullptr;
  }
  if (right == nullptr || !IsDefaultOp(*right, "Transpose")) {
    right = nullptr;
  }
  if (left == nullptr && right == nullptr) {
    return NoMatch(candidate, "neither matrix input is produced by Transpose");
  }
  if (core::symbolic::IsGPU(graph.Builder().device())) {
    if (left != nullptr && graph.IsUsedMoreThanOnce(left->output()[0].value())) {
      left = nullptr;
    }
    if (right != nullptr && graph.IsUsedMoreThanOnce(right->output()[0].value())) {
      right = nullptr;
    }
  }
  for (const NodeProto *transpose : {left, right}) {
    if (transpose == nullptr) {
      continue;
    }
    std::vector<int64_t> perm;
    if (!GetAttributeInts(*transpose, "perm", perm) || perm != std::vector<int64_t>({1, 0})) {
      return NoMatch(candidate, "an input Transpose does not swap the two dimensions");
    }
  }
  if (left == nullptr && right == nullptr) {
    return NoMatch(candidate, "all input Transpose nodes are shared on GPU");
  }
  if (IsDefaultOp(candidate, "Gemm")) {
    const int64_t trans_a = GetAttributeOr<int64_t>(candidate, "transA", 0);
    const int64_t trans_b = GetAttributeOr<int64_t>(candidate, "transB", 0);
    if (right != nullptr && trans_b != trans_a && graph.IsConstant(candidate.input()[1].value())) {
      return NoMatch(candidate, "the constant right transpose should be folded");
    }
    if (left != nullptr && trans_b != trans_a && graph.IsConstant(candidate.input()[0].value())) {
      return NoMatch(candidate, "the constant left transpose should be folded");
    }
  }
  return core::builder::MatchResult{this, {left, right, &candidate}, nullptr};
}

utils::RepeatedProtoField<NodeProto>
TransposeMatMulPattern::Apply(core::builder::GraphGraph &graph,
                              const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[2] == nullptr || (nodes[0] == nullptr && nodes[1] == nullptr)) {
    throw BuilderError(
        "TransposeMatMulPattern::Apply expects optional left and right Transpose slots.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("TransposeMatMulPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto *left = nodes[0];
  const NodeProto *right = nodes[1];
  const NodeProto &matmul = *nodes[2];
  std::vector<std::string> inputs = {
      left == nullptr ? matmul.input()[0].value() : left->input()[0].value(),
      right == nullptr ? matmul.input()[1].value() : right->input()[0].value(),
  };
  for (int i = 2; i < matmul.input_size(); ++i) {
    inputs.push_back(matmul.input()[i].value());
  }
  int64_t trans_a = left == nullptr ? 0 : 1;
  int64_t trans_b = right == nullptr ? 0 : 1;
  std::vector<AttributeProto> kept;
  for (int i = 0; i < matmul.attribute_size(); ++i) {
    const AttributeProto &attribute = matmul.attribute(i);
    if (attribute.name().value() == "alpha" || attribute.name().value() == "beta") {
      kept.push_back(attribute);
    } else if (attribute.name().value() == "transA") {
      trans_a = (attribute.i() + trans_a) % 2;
    } else if (attribute.name().value() == "transB") {
      trans_b = (attribute.i() + trans_b) % 2;
    } else {
      throw BuilderError("TransposeMatMulPattern::Apply found an unexpected Gemm attribute.");
    }
  }
  NodeProto replacement = MakePatternNode("Gemm", inputs, Outputs(matmul), "",
                                          "TransposeMatMulPattern--" + matmul.name().value());
  AddAttribute<int64_t>(replacement, "transA", trans_a);
  AddAttribute<int64_t>(replacement, "transB", trans_b);
  replacement.set_doc_string(matmul.doc_string().value());
  for (const AttributeProto &attribute : kept) {
    *replacement.add_attribute() = attribute;
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  if (left != nullptr && graph.IsUsedMoreThanOnce(left->output()[0].value())) {
    replacements.push_back(*left);
  }
  if (right != nullptr && graph.IsUsedMoreThanOnce(right->output()[0].value())) {
    replacements.push_back(*right);
  }
  return replacements;
}

std::set<std::string> TransposeReshapeMatMulPattern::FastOpType() const { return {"MatMul"}; }

core::builder::MatchResult TransposeReshapeMatMulPattern::Match(core::builder::GraphGraph &graph,
                                                                const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "MatMul") || candidate.input_size() != 2 ||
      candidate.output_size() == 0) {
    return NoMatch(candidate, "candidate is not a default-domain MatMul");
  }
  const TransposeReshapeSide left = FindTransposeReshape(graph, candidate.input()[0].value());
  const TransposeReshapeSide right = FindTransposeReshape(graph, candidate.input()[1].value());
  const TransposeReshapeSide *selected = nullptr;
  bool select_left = false;
  if (left.reshape != nullptr && ValidTransposeReshapeSide(graph, left)) {
    selected = &left;
    select_left = true;
  } else if (right.reshape != nullptr && ValidTransposeReshapeSide(graph, right)) {
    selected = &right;
  }
  if (selected == nullptr) {
    return NoMatch(candidate, "neither MatMul input has a movable Transpose-Reshape sequence");
  }
  if (!graph.HasShape(select_left ? candidate.input()[0].value() : candidate.input()[1].value())) {
    return NoMatch(candidate, "the selected MatMul input rank is unknown");
  }
  return core::builder::MatchResult{this,
                                    {&candidate, select_left ? selected->reshape : nullptr,
                                     select_left ? selected->transpose : nullptr,
                                     select_left ? nullptr : selected->reshape,
                                     select_left ? nullptr : selected->transpose},
                                    &candidate};
}

utils::RepeatedProtoField<NodeProto>
TransposeReshapeMatMulPattern::Apply(core::builder::GraphGraph &graph,
                                     const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 5 || nodes[0] == nullptr ||
      ((nodes[1] == nullptr || nodes[2] == nullptr) &&
       (nodes[3] == nullptr || nodes[4] == nullptr))) {
    throw BuilderError(
        "TransposeReshapeMatMulPattern::Apply expects MatMul and one Transpose-Reshape side.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "TransposeReshapeMatMulPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &matmul = *nodes[0];
  const NodeProto *reshape = nodes[1] == nullptr ? nodes[3] : nodes[1];
  const NodeProto *transpose = nodes[2] == nullptr ? nodes[4] : nodes[2];
  std::vector<int64_t> shape;
  if (!ReadConstantInts(graph, reshape->input()[1].value(), shape) || shape.size() < 2) {
    throw BuilderError("TransposeReshapeMatMulPattern::Apply could not read the Reshape target.");
  }
  std::swap(shape[shape.size() - 2], shape.back());
  const std::string shape_name =
      AddShapeInitializer(graph, "TransposeReshapeMatMulPattern.apply.shape_name", shape);
  const bool left_side = nodes[1] != nullptr;
  const std::string &matrix_input = matmul.input()[left_side ? 0 : 1].value();
  const std::size_t rank = graph.GetShape(matrix_input).Shape().Rank();
  if (rank < 2) {
    throw BuilderError(
        "TransposeReshapeMatMulPattern::Apply requires a matrix input rank of at least two.");
  }
  std::vector<int64_t> perm(rank);
  for (std::size_t i = 0; i < rank; ++i) {
    perm[i] = static_cast<int64_t>(i);
  }
  std::swap(perm[rank - 2], perm.back());
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string intermediate =
      builder.UniqueName("TransposeReshapeMatMulPatternL_" + transpose->input()[0].value());
  const std::string name = "TransposeReshapeMatMulPattern--" + matmul.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(MakePatternNode("Reshape", {transpose->input()[0].value(), shape_name},
                                         {intermediate}, "", name));
  NodeProto new_transpose = MakePatternNode("Transpose", {intermediate}, {matrix_input}, "", name);
  AddAttribute(new_transpose, "perm", perm);
  replacements.push_back(std::move(new_transpose));
  replacements.push_back(matmul);
  return replacements;
}

std::set<std::string> SwitchReshapeActivationPattern::FastOpType() const {
  return {"Cos",  "Cosh", "Elu", "Erf",  "Exp",      "Gelu", "LeakyRelu",
          "Relu", "Selu", "Sin", "Sinh", "Softplus", "Tan",  "Tanh"};
}

core::builder::MatchResult SwitchReshapeActivationPattern::Match(core::builder::GraphGraph &graph,
                                                                 const NodeProto &candidate) const {
  if (!IsActivation(candidate) || candidate.input_size() == 0 || candidate.output_size() == 0) {
    return NoMatch(candidate, "candidate is not a supported default-domain activation");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value())) {
    return NoMatch(candidate, "the layout output has another use");
  }
  const NodeProto *layout = graph.NodeBefore(candidate.input()[0].value());
  if (layout == nullptr || layout->input_size() == 0 ||
      graph.IsUsedMoreThanOnce(layout->input()[0].value())) {
    return NoMatch(candidate, "the layout input has another use or no producer");
  }
  if ((!IsDefaultOp(*layout, "Reshape") && !IsDefaultOp(*layout, "Transpose")) ||
      layout->output_size() != 1) {
    return NoMatch(candidate, "the activation is not preceded by Reshape or Transpose");
  }
  const NodeProto *matmul = graph.NodeBefore(layout->input()[0].value());
  if (matmul == nullptr || (!IsDefaultOp(*matmul, "Gemm") && !IsDefaultOp(*matmul, "MatMul"))) {
    return NoMatch(candidate, "the layout is not preceded by default-domain Gemm or MatMul");
  }
  return core::builder::MatchResult{
      this, {matmul, layout, &candidate}, IsDefaultOp(*layout, "Reshape") ? layout : matmul};
}

utils::RepeatedProtoField<NodeProto>
SwitchReshapeActivationPattern::Apply(core::builder::GraphGraph &graph,
                                      const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError(
        "SwitchReshapeActivationPattern::Apply expects MatMul, layout, and activation.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "SwitchReshapeActivationPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &matmul = *nodes[0];
  const NodeProto &layout = *nodes[1];
  const NodeProto &activation = *nodes[2];
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string matmul_output =
      builder.UniqueName("SwitchReshapeActivationPatternL_" + matmul.output()[0].value());
  const std::string activation_output =
      builder.UniqueName("SwitchReshapeActivationPatternL_" + layout.output()[0].value());
  NodeProto new_matmul = MakePatternNode(
      matmul.op_type().value().c_str(), Inputs(matmul), {matmul_output}, matmul.domain().value(),
      "SwitchReshapeActivationPattern--" + matmul.name().value());
  CopyAttributes(matmul, new_matmul);
  NodeProto new_activation = MakePatternNode(
      activation.op_type().value().c_str(), {matmul_output}, {activation_output},
      activation.domain().value(), "SwitchReshapeActivationPattern--" + activation.name().value());
  CopyAttributes(activation, new_activation);
  std::vector<std::string> layout_inputs = {activation_output};
  for (int i = 1; i < layout.input_size(); ++i) {
    layout_inputs.push_back(layout.input()[i].value());
  }
  NodeProto new_layout = MakePatternNode(
      layout.op_type().value().c_str(), layout_inputs, Outputs(activation), layout.domain().value(),
      "SwitchReshapeActivationPattern--" + layout.name().value());
  CopyAttributes(layout, new_layout);
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(new_matmul));
  replacements.push_back(std::move(new_activation));
  replacements.push_back(std::move(new_layout));
  return replacements;
}

std::set<std::string> ShapeBasedMatMulToMulPattern::FastOpType() const { return {"MatMul"}; }

core::builder::MatchResult ShapeBasedMatMulToMulPattern::Match(core::builder::GraphGraph &graph,
                                                               const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "MatMul") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain MatMul");
  }
  if (!graph.HasShape(candidate.input()[0].value()) ||
      !graph.HasShape(candidate.input()[1].value())) {
    return NoMatch(candidate, "the MatMul input shapes are unknown");
  }
  const SymShape &left = graph.GetShape(candidate.input()[0].value()).Shape();
  const SymShape &right = graph.GetShape(candidate.input()[1].value()).Shape();
  if (left.Rank() < 2 || right.Rank() < 2 || !left[left.Rank() - 1].IsInt() ||
      left[left.Rank() - 1].AsInt() != 1 || !right[right.Rank() - 2].IsInt() ||
      right[right.Rank() - 2].AsInt() != 1) {
    return NoMatch(candidate, "the MatMul reduction dimensions are not both one");
  }
  const std::vector<const NodeProto *> &successors = graph.NextNodes(candidate.output()[0].value());
  if (successors.size() == 1 && successors[0] != nullptr &&
      IsDefaultOp(*successors[0], "Transpose")) {
    std::vector<int64_t> perm;
    if (GetAttributeInts(*successors[0], "perm", perm) && perm.size() >= 2) {
      bool swaps_last_two = true;
      for (std::size_t i = 0; i + 2 < perm.size(); ++i) {
        swaps_last_two = swaps_last_two && perm[i] == static_cast<int64_t>(i);
      }
      const int64_t rank = static_cast<int64_t>(perm.size());
      swaps_last_two =
          swaps_last_two && perm[perm.size() - 2] == rank - 1 && perm.back() == rank - 2;
      if (swaps_last_two) {
        if (left.Rank() <= 2 || right.Rank() <= 2) {
          return NoMatch(candidate, "the Transpose rewrite requires both input ranks above two");
        }
        if (graph.IsUsedMoreThanOnce(candidate.output()[0].value())) {
          return NoMatch(candidate,
                         "the MatMul output has another consumer, output, or subgraph capture");
        }
        return core::builder::MatchResult{this, {&candidate, successors[0]}, &candidate};
      }
    }
  }
  return core::builder::MatchResult{this, {&candidate, nullptr}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
ShapeBasedMatMulToMulPattern::Apply(core::builder::GraphGraph &graph,
                                    const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr) {
    throw BuilderError(
        "ShapeBasedMatMulToMulPattern::Apply expects MatMul and optional Transpose slots.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "ShapeBasedMatMulToMulPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &matmul = *nodes[0];
  const NodeProto *transpose = nodes[1];
  const std::string name = "ShapeBasedMatMulToMulPattern--" + matmul.name().value();
  utils::RepeatedProtoField<NodeProto> replacements;
  if (transpose == nullptr) {
    replacements.push_back(
        MakePatternNode("Mul", Inputs(matmul), {matmul.output()[0].value()}, "", name));
    return replacements;
  }
  const std::size_t left_rank = graph.GetShape(matmul.input()[0].value()).Shape().Rank();
  const std::size_t right_rank = graph.GetShape(matmul.input()[1].value()).Shape().Rank();
  if (left_rank <= 2 || right_rank <= 2) {
    throw BuilderError(
        "ShapeBasedMatMulToMulPattern::Apply requires ranks above two with Transpose.");
  }
  std::vector<int64_t> left_shape(left_rank - 2, 0);
  left_shape.push_back(1);
  left_shape.push_back(-1);
  std::vector<int64_t> right_shape(right_rank - 2, 0);
  right_shape.push_back(-1);
  right_shape.push_back(1);
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string left_shape_name =
      FreeInitializerName(builder, matmul.input()[0].value() + "-ZEROS");
  builder.MakeInitializer(MakeInitializerShape(left_shape_name.c_str(), left_shape));
  const std::string right_shape_name =
      FreeInitializerName(builder, matmul.input()[1].value() + "-ZEROS");
  builder.MakeInitializer(MakeInitializerShape(right_shape_name.c_str(), right_shape));
  const std::string left_output = builder.UniqueName(matmul.input()[0].value());
  const std::string right_output = builder.UniqueName(matmul.input()[1].value());
  replacements.push_back(MakePatternNode("Reshape", {matmul.input()[0].value(), left_shape_name},
                                         {left_output}, "", name));
  replacements.push_back(MakePatternNode("Reshape", {matmul.input()[1].value(), right_shape_name},
                                         {right_output}, "", name));
  replacements.push_back(
      MakePatternNode("Mul", {left_output, right_output}, Outputs(*transpose), "", name));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
