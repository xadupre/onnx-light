// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/normalization/normalization_pattern.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <set>
#include <string>
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
using core::symbolic::SymShape;
using core::symbolic::TensorType;

bool MainOpsetAtLeast(core::builder::GraphGraph &graph, int minimum) {
  const int opset = graph.Builder().OpsetVersion("");
  return opset == core::shapes::kUnknownOpsetVersion || opset >= minimum;
}

const NodeProto *OnlyConsumer(core::builder::GraphGraph &graph, const std::string &name) {
  const std::vector<const NodeProto *> &nodes = graph.NextNodes(name);
  return nodes.size() == 1 ? nodes[0] : nullptr;
}

bool HasOnlyConsumers(core::builder::GraphGraph &graph, const std::string &name,
                      std::initializer_list<const NodeProto *> expected) {
  if (graph.IsOutput(name) || graph.IsUsedBySubgraph(name)) {
    return false;
  }
  const std::vector<const NodeProto *> &actual = graph.NextNodes(name);
  if (actual.size() != expected.size()) {
    return false;
  }
  return std::all_of(expected.begin(), expected.end(), [&](const NodeProto *consumer) {
    return consumer != nullptr && std::find(actual.begin(), actual.end(), consumer) != actual.end();
  });
}

bool HasOnlyConsumers(core::builder::GraphGraph &graph, const NodeProto &node,
                      std::initializer_list<const NodeProto *> expected) {
  return node.output_size() == 1 && HasOnlyConsumers(graph, node.output()[0].value(), expected);
}

bool ReadAxes(core::builder::GraphGraph &graph, const std::string &name,
              std::vector<int64_t> &axes) {
  const TensorProto *tensor = graph.GetComputedConstant(name);
  return tensor != nullptr && ReadIntegerValues(*tensor, axes);
}

bool Read16BitValues(const TensorProto &tensor, bool bfloat, std::vector<double> &values) {
  std::size_t count = tensor.int32_data().size();
  if (count == 0 && tensor.is_raw_data()) {
    count = tensor.ref_raw_data().size() / 2;
  }
  if (count == 0) {
    return false;
  }
  values.clear();
  values.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    std::uint16_t bits = 0;
    if (tensor.int32_data().size() > 0) {
      bits = static_cast<std::uint16_t>(tensor.int32_data()[index]);
    } else {
      bits = static_cast<std::uint16_t>(tensor.ref_raw_data()[2 * index]) |
             (static_cast<std::uint16_t>(tensor.ref_raw_data()[2 * index + 1]) << 8);
    }
    const float value =
        bfloat ? core::runtime::Bfloat16BitsToFloat(bits) : core::runtime::Float16BitsToFloat(bits);
    values.push_back(static_cast<double>(value));
  }
  return true;
}

bool ReadNumericValues(const TensorProto &tensor, std::vector<double> &values) {
  const auto type = static_cast<TensorProto::DataType>(tensor.data_type());
  if (type == TensorProto::DataType::FLOAT16) {
    return Read16BitValues(tensor, false, values);
  }
  if (type == TensorProto::DataType::BFLOAT16) {
    return Read16BitValues(tensor, true, values);
  }
  if (ReadFloatingValues(tensor, values)) {
    return true;
  }
  std::vector<int64_t> integers;
  if (!ReadIntegerValues(tensor, integers)) {
    return false;
  }
  values.clear();
  values.reserve(integers.size());
  for (int64_t value : integers) {
    values.push_back(static_cast<double>(value));
  }
  return true;
}

bool ReadConstantValues(core::builder::GraphGraph &graph, const std::string &name,
                        std::vector<double> &values) {
  const TensorProto *tensor = graph.GetComputedConstant(name);
  if (tensor != nullptr) {
    return ReadNumericValues(*tensor, values);
  }
  const NodeProto *constant = graph.NodeBefore(name);
  if (constant == nullptr || !IsDefaultOp(*constant, "Constant")) {
    return false;
  }
  const AttributeProto *value_int = FindAttribute(*constant, "value_int");
  if (value_int != nullptr && value_int->type() == AttributeProto::AttributeType::INT) {
    values = {static_cast<double>(value_int->i())};
    return true;
  }
  const AttributeProto *value_float = FindAttribute(*constant, "value_float");
  if (value_float != nullptr && value_float->type() == AttributeProto::AttributeType::FLOAT) {
    values = {static_cast<double>(value_float->f())};
    return true;
  }
  return false;
}

bool AllEqual(const std::vector<double> &values, double expected) {
  return !values.empty() && std::all_of(values.begin(), values.end(),
                                        [expected](double value) { return value == expected; });
}

TensorProto MakeNumericTensor(const std::string &name, TensorProto::DataType type,
                              const std::vector<int64_t> &dims, const std::vector<double> &values) {
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
    case TensorProto::DataType::UINT16:
      tensor.ref_int32_data().push_back(static_cast<uint16_t>(value));
      break;
    case TensorProto::DataType::INT8:
      tensor.ref_int32_data().push_back(static_cast<int8_t>(value));
      break;
    case TensorProto::DataType::UINT8:
    case TensorProto::DataType::BOOL:
      tensor.ref_int32_data().push_back(static_cast<uint8_t>(value));
      break;
    default:
      throw BuilderError("Unsupported initializer type in normalization pattern.");
    }
  }
  return tensor;
}

TensorProto MakeFilledTensor(const std::string &name, TensorType type,
                             const std::vector<int64_t> &dims, double value) {
  int64_t count = 1;
  for (int64_t dim : dims) {
    count *= dim;
  }
  return MakeNumericTensor(name, core::symbolic::TensorTypeToDataType(type), dims,
                           std::vector<double>(static_cast<std::size_t>(count), value));
}

void AddTensorAttribute(NodeProto &node, const char *name, TensorProto value) {
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::TENSOR);
  *attribute->mutable_t() = std::move(value);
}

void CopyAttributes(const NodeProto &from, NodeProto &to) {
  for (const AttributeProto &attribute : from.attribute()) {
    to.mutable_attribute()->push_back(attribute);
  }
}

bool CastTarget(const NodeProto &node, int64_t &target) {
  const AttributeProto *attribute = FindAttribute(node, "to");
  if (attribute == nullptr || attribute->type() != AttributeProto::AttributeType::INT) {
    return false;
  }
  target = attribute->i();
  return true;
}

bool HasRank(core::builder::GraphGraph &graph, const std::string &name, std::size_t &rank) {
  if (!graph.HasShape(name)) {
    return false;
  }
  rank = graph.GetShape(name).Shape().Rank();
  return true;
}

bool OtherInput(const NodeProto &node, const std::string &known, std::string &other) {
  if (node.input_size() != 2) {
    return false;
  }
  const bool first = node.input()[0].value() == known;
  const bool second = node.input()[1].value() == known;
  if (first == second) {
    return false;
  }
  other = first ? node.input()[1].value() : node.input()[0].value();
  return true;
}

bool ResolveSuffixAxis(core::builder::GraphGraph &graph, const std::string &input,
                       const std::vector<int64_t> &axes, int64_t &axis) {
  if (axes.empty()) {
    return false;
  }
  if (axes == std::vector<int64_t>{-1} && !graph.HasShape(input)) {
    axis = -1;
    return true;
  }
  if (!graph.HasShape(input)) {
    return false;
  }
  const std::size_t rank = graph.GetShape(input).Shape().Rank();
  if (axes.size() > rank) {
    return false;
  }
  std::vector<int64_t> normalized;
  normalized.reserve(axes.size());
  bool all_negative = true;
  for (int64_t value : axes) {
    all_negative = all_negative && value < 0;
    const int64_t normalized_value = value < 0 ? value + static_cast<int64_t>(rank) : value;
    if (normalized_value < 0 || normalized_value >= static_cast<int64_t>(rank)) {
      return false;
    }
    normalized.push_back(normalized_value);
  }
  std::sort(normalized.begin(), normalized.end());
  if (std::adjacent_find(normalized.begin(), normalized.end()) != normalized.end()) {
    return false;
  }
  const int64_t start = static_cast<int64_t>(rank - normalized.size());
  for (std::size_t index = 0; index < normalized.size(); ++index) {
    if (normalized[index] != start + static_cast<int64_t>(index)) {
      return false;
    }
  }
  axis = all_negative ? start - static_cast<int64_t>(rank) : start;
  return true;
}

TensorProto::DataType ProductType(TensorProto::DataType left, TensorProto::DataType right) {
  if (left == right) {
    return left;
  }
  if (left == TensorProto::DataType::DOUBLE || right == TensorProto::DataType::DOUBLE) {
    return TensorProto::DataType::DOUBLE;
  }
  if (left == TensorProto::DataType::FLOAT || right == TensorProto::DataType::FLOAT ||
      left == TensorProto::DataType::FLOAT16 || right == TensorProto::DataType::FLOAT16 ||
      left == TensorProto::DataType::BFLOAT16 || right == TensorProto::DataType::BFLOAT16) {
    return TensorProto::DataType::FLOAT;
  }
  if (left == TensorProto::DataType::INT64 || right == TensorProto::DataType::INT64) {
    return TensorProto::DataType::INT64;
  }
  return left;
}

bool MultiplyConstants(const TensorProto &left, const TensorProto &right, const std::string &name,
                       TensorProto &product) {
  std::vector<double> left_values;
  std::vector<double> right_values;
  if (!ReadNumericValues(left, left_values) || !ReadNumericValues(right, right_values) ||
      left_values.size() != right_values.size() || left.dims_size() != right.dims_size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.dims().size(); ++index) {
    if (left.dims()[index] != right.dims()[index]) {
      return false;
    }
  }
  std::vector<double> values;
  values.reserve(left_values.size());
  for (std::size_t index = 0; index < left_values.size(); ++index) {
    values.push_back(left_values[index] * right_values[index]);
  }
  std::vector<int64_t> dims;
  dims.reserve(left.dims_size());
  for (int64_t dim : left.dims()) {
    dims.push_back(dim);
  }
  product = MakeNumericTensor(name,
                              ProductType(static_cast<TensorProto::DataType>(left.data_type()),
                                          static_cast<TensorProto::DataType>(right.data_type())),
                              dims, values);
  return true;
}

} // namespace

std::set<std::string> LayerNormalizationPattern::FastOpType() const { return {"ReduceMean"}; }

core::builder::MatchResult LayerNormalizationPattern::Match(core::builder::GraphGraph &graph,
                                                            const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "ReduceMean") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a two-input default-domain ReduceMean");
  }
  std::vector<int64_t> axes;
  if (!graph.IsConstant(candidate.input()[1].value()) ||
      !ReadAxes(graph, candidate.input()[1].value(), axes)) {
    return NoMatch(candidate, "the variance ReduceMean axes are not constant integers");
  }
  if (axes.empty()) {
    return NoMatch(candidate, "the variance ReduceMean axes are empty");
  }
  if (GetAttributeOr<int64_t>(candidate, "keepdims", 1) != 1) {
    return NoMatch(candidate, "the variance ReduceMean does not keep dimensions");
  }
  if (axes != std::vector<int64_t>{-1}) {
    if (!graph.HasShape(candidate.input()[0].value())) {
      return NoMatch(candidate, "non-trivial axes require a known input rank");
    }
    const std::size_t rank = graph.GetShape(candidate.input()[0].value()).Shape().Rank();
    if (axes.size() > rank) {
      return NoMatch(candidate, "the axes contain more entries than the input rank");
    }
    std::vector<int64_t> trailing;
    for (std::size_t index = rank - axes.size(); index < rank; ++index) {
      trailing.push_back(static_cast<int64_t>(index));
    }
    if (axes != trailing) {
      return NoMatch(candidate, "the axes are not the trailing input dimensions");
    }
  }

  const NodeProto *power = graph.NodeBefore(candidate.input()[0].value());
  if (power == nullptr || !IsDefaultOp(*power, "Pow") || power->input_size() != 2 ||
      power->output_size() != 1 || graph.NextNodes(power->output()[0].value()).size() != 1) {
    return NoMatch(candidate, "the variance input is not produced by an unshared Pow");
  }
  if (!graph.IsConstantScalar(power->input()[1].value(), 2.0, true)) {
    return NoMatch(candidate, "the Pow exponent is not the scalar two");
  }
  const NodeProto *sub = graph.NodeBefore(power->input()[0].value());
  if (sub == nullptr || !IsDefaultOp(*sub, "Sub") || sub->input_size() != 2 ||
      sub->output_size() != 1 || graph.NextNodes(sub->output()[0].value()).size() != 2) {
    return NoMatch(candidate, "the Pow input is not produced by a two-use Sub");
  }
  const NodeProto *mean = graph.NodeBefore(sub->input()[1].value());
  if (mean == nullptr || !IsDefaultOp(*mean, "ReduceMean") || mean->input_size() < 2 ||
      mean->output_size() != 1 || graph.NextNodes(mean->output()[0].value()).size() != 1) {
    return NoMatch(candidate, "the Sub mean input is not produced by an unshared ReduceMean");
  }
  std::vector<int64_t> mean_axes;
  if (!graph.IsConstant(mean->input()[1].value()) ||
      !ReadAxes(graph, mean->input()[1].value(), mean_axes) || axes != mean_axes) {
    return NoMatch(candidate, "the two ReduceMean axes differ");
  }
  if (sub->input()[0].value() != mean->input()[0].value()) {
    return NoMatch(candidate, "the centering Sub does not reuse the mean input");
  }
  const AttributeProto *keepdims = FindAttribute(*mean, "keepdims");
  if (keepdims == nullptr || keepdims->type() != AttributeProto::AttributeType::INT ||
      keepdims->i() != 1) {
    return NoMatch(candidate, "the first ReduceMean does not explicitly keep dimensions");
  }

  const std::vector<const NodeProto *> &after_reduce =
      graph.NextNodes(candidate.output()[0].value());
  if (after_reduce.size() != 1) {
    return NoMatch(candidate, "the variance ReduceMean does not have one consumer");
  }
  const NodeProto *add = nullptr;
  const NodeProto *sqrt = nullptr;
  std::string epsilon_name;
  if (IsDefaultOp(*after_reduce[0], "Add")) {
    add = after_reduce[0];
    std::vector<double> epsilon_values;
    if (add->output_size() != 1 || !OtherInput(*add, candidate.output()[0].value(), epsilon_name) ||
        !graph.IsConstantScalar(epsilon_name, true) ||
        !ReadConstantValues(graph, epsilon_name, epsilon_values) || epsilon_values.empty()) {
      return NoMatch(candidate, "the epsilon Add does not combine variance and one scalar");
    }
    sqrt = OnlyConsumer(graph, add->output()[0].value());
  } else {
    sqrt = after_reduce[0];
  }
  if (sqrt == nullptr || !IsDefaultOp(*sqrt, "Sqrt") || sqrt->input_size() != 1 ||
      sqrt->output_size() != 1) {
    return NoMatch(candidate, "the variance path is not followed by one Sqrt");
  }
  const NodeProto *normalization = OnlyConsumer(graph, sqrt->output()[0].value());
  if (normalization == nullptr || normalization->output_size() != 1) {
    return NoMatch(candidate, "the Sqrt is not followed by one normalization node");
  }
  const NodeProto *reciprocal = nullptr;
  const NodeProto *final = normalization;
  if (IsDefaultOp(*normalization, "Div")) {
    if (normalization->input_size() != 2 ||
        normalization->input()[0].value() != sub->output()[0].value() ||
        normalization->input()[1].value() != sqrt->output()[0].value()) {
      return NoMatch(candidate, "the final Div does not divide the centered input by the Sqrt");
    }
  } else if (IsDefaultOp(*normalization, "Reciprocal")) {
    reciprocal = normalization;
    if (reciprocal->input_size() != 1 ||
        reciprocal->input()[0].value() != sqrt->output()[0].value()) {
      return NoMatch(candidate, "the Reciprocal does not consume the Sqrt");
    }
    final = OnlyConsumer(graph, reciprocal->output()[0].value());
    if (final == nullptr || !IsDefaultOp(*final, "Mul") || final->input_size() != 2 ||
        final->output_size() != 1 ||
        std::set<std::string>{final->input()[0].value(), final->input()[1].value()} !=
            std::set<std::string>{sub->output()[0].value(), reciprocal->output()[0].value()}) {
      return NoMatch(candidate, "the Reciprocal is not multiplied by the centered input");
    }
  } else {
    return NoMatch(candidate, "the final normalization node is neither Div nor Reciprocal");
  }
  if (!HasOnlyConsumers(graph, *mean, {sub}) || !HasOnlyConsumers(graph, *sub, {power, final}) ||
      !HasOnlyConsumers(graph, *power, {&candidate}) ||
      !HasOnlyConsumers(graph, candidate, {add == nullptr ? sqrt : add}) ||
      (add != nullptr && !HasOnlyConsumers(graph, *add, {sqrt})) ||
      !HasOnlyConsumers(graph, *sqrt, {normalization}) ||
      (reciprocal != nullptr && !HasOnlyConsumers(graph, *reciprocal, {final}))) {
    return NoMatch(candidate, "a LayerNormalization intermediate is externally used");
  }

  if (reciprocal == nullptr) {
    return core::builder::MatchResult{
        this, {mean, sub, power, &candidate, add, sqrt, final}, &candidate};
  }
  return core::builder::MatchResult{
      this, {mean, sub, power, &candidate, add, sqrt, reciprocal, final}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
LayerNormalizationPattern::Apply(core::builder::GraphGraph &graph,
                                 const std::vector<const NodeProto *> &nodes) const {
  if ((nodes.size() != 7 && nodes.size() != 8) || nodes[0] == nullptr || nodes[1] == nullptr ||
      nodes[2] == nullptr || nodes[3] == nullptr || nodes[5] == nullptr || nodes[6] == nullptr ||
      (nodes.size() == 8 && nodes[7] == nullptr)) {
    throw BuilderError(
        "LayerNormalizationPattern::Apply expects mean, Sub, Pow, mean, optional Add, Sqrt, "
        "and Div or Reciprocal-Mul.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[3]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "LayerNormalizationPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &mean = *nodes[0];
  const NodeProto &variance = *nodes[3];
  const NodeProto *add = nodes[4];
  const NodeProto &final = *nodes.back();
  core::builder::GraphBuilder &builder = graph.Builder();

  std::vector<int64_t> axes;
  if (!ReadAxes(graph, mean.input()[1].value(), axes) || axes.empty() ||
      !graph.HasType(mean.input()[0].value())) {
    throw BuilderError("LayerNormalizationPattern::Apply could not read axes or input type.");
  }
  int64_t layer_axis =
      axes == std::vector<int64_t>{-1} ? -1 : *std::min_element(axes.begin(), axes.end());
  const TensorType input_type = graph.GetType(mean.input()[0].value());
  std::string scale;
  std::string bias;
  utils::RepeatedProtoField<NodeProto> replacements;
  if (axes == std::vector<int64_t>{-1} && graph.HasShape(mean.input()[0].value())) {
    const SymShape &shape = graph.GetShape(mean.input()[0].value()).Shape();
    if (shape.Rank() > 0 && shape[shape.Rank() - 1].IsInt()) {
      const int64_t dimension = shape[shape.Rank() - 1].AsInt();
      scale = FreeInitializerName(builder, "LayerNormalizationPattern.scale");
      bias = FreeInitializerName(builder, "LayerNormalizationPattern.bias");
      builder.MakeInitializer(MakeFilledTensor(scale, input_type, {dimension}, 1.0));
      builder.MakeInitializer(MakeFilledTensor(bias, input_type, {dimension}, 0.0));
    }
  }
  if (scale.empty()) {
    const std::string node_name = "LayerNormalizationPattern--" + mean.name().value();
    const std::string shape =
        builder.UniqueName("LayerNormalizationPattern_Sh_" + mean.input()[0].value());
    NodeProto shape_node =
        MakeNode("Shape", {mean.input()[0].value()}, {shape}, "", node_name.c_str());
    AddAttribute<int64_t>(shape_node, "start", layer_axis);
    replacements.push_back(std::move(shape_node));

    scale = builder.UniqueName("LayerNormalizationPattern_Sc_" + mean.input()[0].value());
    NodeProto scale_node = MakeNode("ConstantOfShape", {shape}, {scale}, "", node_name.c_str());
    AddTensorAttribute(scale_node, "value", MakeFilledTensor("", input_type, {1}, 1.0));
    replacements.push_back(std::move(scale_node));

    bias = builder.UniqueName("LayerNormalizationPattern_Bi_" + mean.input()[0].value());
    NodeProto bias_node = MakeNode("ConstantOfShape", {shape}, {bias}, "", node_name.c_str());
    AddTensorAttribute(bias_node, "value", MakeFilledTensor("", input_type, {1}, 0.0));
    replacements.push_back(std::move(bias_node));
  }

  double epsilon = 0.0;
  if (add != nullptr) {
    std::string epsilon_name;
    std::vector<double> epsilon_values;
    if (!OtherInput(*add, variance.output()[0].value(), epsilon_name) ||
        !ReadConstantValues(graph, epsilon_name, epsilon_values) || epsilon_values.empty()) {
      throw BuilderError("LayerNormalizationPattern::Apply could not read epsilon.");
    }
    epsilon = epsilon_values[0];
  }
  NodeProto layer = MakeNode("LayerNormalization", {mean.input()[0].value(), scale, bias},
                             {final.output()[0].value()}, "",
                             ("LayerNormalizationPattern--" + variance.name().value()).c_str());
  AddAttribute<float>(layer, "epsilon", static_cast<float>(epsilon));
  AddAttribute<int64_t>(layer, "stash_type", static_cast<int64_t>(TensorProto::DataType::FLOAT));
  AddAttribute<int64_t>(layer, "axis", layer_axis);
  if (variance.has_doc_string()) {
    layer.set_doc_string(variance.doc_string().value());
  }
  replacements.push_back(std::move(layer));
  return replacements;
}

std::set<std::string> LayerNormalizationScalePattern::FastOpType() const {
  return {"LayerNormalization"};
}

core::builder::MatchResult LayerNormalizationScalePattern::Match(core::builder::GraphGraph &graph,
                                                                 const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "LayerNormalization") || candidate.input_size() < 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a single-output LayerNormalization");
  }
  const std::vector<const NodeProto *> &mul_nodes = graph.NextNodes(candidate.output()[0].value());
  if (mul_nodes.size() != 1 || !IsDefaultOp(*mul_nodes[0], "Mul") ||
      mul_nodes[0]->input_size() != 2 || mul_nodes[0]->output_size() != 1) {
    return NoMatch(candidate, "LayerNormalization is not followed by one Mul");
  }
  const NodeProto *mul = mul_nodes[0];
  const bool first_is_layer = mul->input()[0].value() == candidate.output()[0].value();
  const bool second_is_layer = mul->input()[1].value() == candidate.output()[0].value();
  if (first_is_layer == second_is_layer) {
    return NoMatch(candidate, "the following Mul does not use the normalized value exactly once");
  }
  if (!HasOnlyConsumers(graph, candidate, {mul})) {
    return NoMatch(candidate, "the normalized value is externally used");
  }

  const std::size_t scale_index = first_is_layer ? 1 : 0;
  if (!graph.HasShape(mul->input()[scale_index].value()) ||
      !graph.HasShape(candidate.input()[1].value()) ||
      graph.GetShape(mul->input()[scale_index].value()).Shape() !=
          graph.GetShape(candidate.input()[1].value()).Shape()) {
    return NoMatch(candidate, "the following Mul scale shape differs from the existing scale");
  }
  const std::vector<const NodeProto *> &after_mul = graph.NextNodes(mul->output()[0].value());
  if (after_mul.empty()) {
    return core::builder::MatchResult{this, {&candidate, mul, nullptr}, mul};
  }
  if (after_mul.size() != 1 || !IsDefaultOp(*after_mul[0], "Add")) {
    return core::builder::MatchResult{
        this, {&candidate, mul, nullptr}, after_mul.empty() ? mul : after_mul[0]};
  }

  const NodeProto *add = after_mul[0];
  if (add->input_size() != 2 || add->output_size() != 1) {
    return NoMatch(candidate, "the following Add does not have two inputs and one output");
  }
  const bool first_is_mul = add->input()[0].value() == mul->output()[0].value();
  const bool second_is_mul = add->input()[1].value() == mul->output()[0].value();
  if (first_is_mul == second_is_mul) {
    return NoMatch(candidate, "the following Add does not use the Mul output exactly once");
  }
  const std::size_t bias_index = first_is_mul ? 1 : 0;
  if (!graph.HasShape(add->input()[bias_index].value()) ||
      !graph.HasShape(candidate.input()[1].value()) ||
      graph.GetShape(add->input()[bias_index].value()).Shape() !=
          graph.GetShape(candidate.input()[1].value()).Shape()) {
    return NoMatch(candidate, "the following Add bias shape differs from the scale shape");
  }
  if (!HasOnlyConsumers(graph, *mul, {add})) {
    return NoMatch(candidate, "the scaled normalized value is externally used");
  }
  return core::builder::MatchResult{this, {&candidate, mul, add}, add};
}

utils::RepeatedProtoField<NodeProto>
LayerNormalizationScalePattern::Apply(core::builder::GraphGraph &graph,
                                      const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError(
        "LayerNormalizationScalePattern::Apply expects LayerNormalization, Mul, optional Add.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "LayerNormalizationScalePattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &layer = *nodes[0];
  const NodeProto &mul = *nodes[1];
  const NodeProto *add = nodes[2];
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "LayerNormalizationScalePattern--" + layer.name().value();

  const std::string scale = mul.input()[0].value() == layer.output()[0].value()
                                ? mul.input()[1].value()
                                : mul.input()[0].value();
  std::string new_scale;
  if (graph.IsConstantScalar(layer.input()[1].value(), 1.0, true)) {
    new_scale = scale;
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  if (new_scale.empty()) {
    new_scale = builder.UniqueName("LayerNormalizationScalePattern_" + layer.input()[1].value());
    replacements.push_back(
        MakeNode("Mul", {layer.input()[1].value(), scale}, {new_scale}, "", name.c_str()));
  }

  std::string new_bias;
  if (add != nullptr) {
    if (layer.input_size() == 2) {
      new_bias = add->input()[0].value() == mul.output()[0].value() ? add->input()[1].value()
                                                                    : add->input()[0].value();
    } else {
      const std::string existing_bias = layer.input()[2].value();
      const std::string mul_constant = mul.input()[1].value() == layer.output()[0].value()
                                           ? mul.input()[0].value()
                                           : mul.input()[1].value();
      const std::string add_constant = add->input()[1].value() == mul.output()[0].value()
                                           ? add->input()[0].value()
                                           : add->input()[1].value();
      const std::string temporary =
          builder.UniqueName("LayerNormalizationScalePattern_" + layer.input()[1].value());
      new_bias = builder.UniqueName("LayerNormalizationScalePattern_" + layer.input()[1].value());
      replacements.push_back(
          MakeNode("Mul", {mul_constant, existing_bias}, {temporary}, "", name.c_str()));
      replacements.push_back(
          MakeNode("Add", {temporary, add_constant}, {new_bias}, "", name.c_str()));
    }
  } else if (layer.input_size() > 2) {
    new_bias = builder.UniqueName("LayerNormalizationScalePattern_" + layer.input()[2].value());
    replacements.push_back(
        MakeNode("Mul", {scale, layer.input()[2].value()}, {new_bias}, "", name.c_str()));
  }

  std::vector<std::string> inputs = {layer.input()[0].value(), new_scale};
  if (!new_bias.empty()) {
    inputs.push_back(new_bias);
  }
  NodeProto replacement = MakeNode(
      "LayerNormalization", inputs,
      {(add == nullptr ? mul.output()[0].value() : add->output()[0].value())}, "", name.c_str());
  for (const char *attribute_name : {"axis", "epsilon", "stash_type"}) {
    const AttributeProto *attribute = FindAttribute(layer, attribute_name);
    if (attribute != nullptr) {
      replacement.mutable_attribute()->push_back(*attribute);
    }
  }
  if (layer.has_doc_string()) {
    replacement.set_doc_string(layer.doc_string().value());
  }
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> CastLayerNormalizationCastPattern::FastOpType() const {
  return {"GroupNormalization", "LayerNormalization", "RMSNormalization",
          "SimplifiedLayerNormalization"};
}

core::builder::MatchResult
CastLayerNormalizationCastPattern::Match(core::builder::GraphGraph &graph,
                                         const NodeProto &candidate) const {
  const std::string &op_type = candidate.op_type().value();
  if (op_type != "GroupNormalization" && op_type != "LayerNormalization" &&
      op_type != "RMSNormalization" && op_type != "SimplifiedLayerNormalization") {
    return NoMatch(candidate, "candidate is not a supported normalization operator");
  }
  const std::string normalized_domain = NormaliseDomain(candidate.domain().value());
  if (normalized_domain != kDefaultOnnxDomain && candidate.domain().value() != "com.microsoft") {
    return NoMatch(candidate, "candidate domain is neither ONNX nor com.microsoft");
  }
  if (candidate.input_size() < 1 || candidate.output_size() < 1) {
    return NoMatch(candidate, "candidate has no data input or output");
  }
  for (std::size_t index = 1; index < candidate.output().size(); ++index) {
    if (!candidate.output()[index].value().empty() &&
        graph.IsUsed(candidate.output()[index].value())) {
      return NoMatch(candidate, "an optional normalization output is used");
    }
  }

  const int64_t stash_type = GetAttributeOr<int64_t>(
      candidate, "stash_type", static_cast<int64_t>(TensorProto::DataType::FLOAT));
  const NodeProto *cast_before = graph.NodeBefore(candidate.input()[0].value());
  int64_t cast_before_target = 0;
  if (cast_before == nullptr || !IsDefaultOp(*cast_before, "Cast") ||
      cast_before->input_size() != 1 || cast_before->output_size() != 1 ||
      !CastTarget(*cast_before, cast_before_target)) {
    return NoMatch(candidate, "the data input is not produced by a default-domain Cast");
  }
  if (cast_before_target != stash_type) {
    return NoMatch(candidate, "the input Cast target differs from stash_type");
  }
  if (!HasOnlyConsumers(graph, *cast_before, {&candidate})) {
    return NoMatch(candidate, "the input Cast output is externally used");
  }

  const std::vector<const NodeProto *> &cast_after_nodes =
      graph.NextNodes(candidate.output()[0].value());
  if (cast_after_nodes.size() != 1) {
    return NoMatch(candidate, "the normalization data output does not have one consumer");
  }
  const NodeProto *cast_after = cast_after_nodes[0];
  int64_t cast_after_target = 0;
  if (!IsDefaultOp(*cast_after, "Cast") || cast_after->input_size() != 1 ||
      cast_after->output_size() != 1 || !CastTarget(*cast_after, cast_after_target)) {
    return NoMatch(candidate, "the data output is not consumed by a default-domain Cast");
  }
  if (!graph.HasType(cast_before->input()[0].value()) ||
      cast_after_target != static_cast<int64_t>(core::symbolic::TensorTypeToDataType(
                               graph.GetType(cast_before->input()[0].value())))) {
    return NoMatch(candidate, "the output Cast does not restore the original element type");
  }
  if (!HasOnlyConsumers(graph, candidate.output()[0].value(), {cast_after})) {
    return NoMatch(candidate, "the normalization data output is externally used");
  }
  return core::builder::MatchResult{this, {cast_before, &candidate, cast_after}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
CastLayerNormalizationCastPattern::Apply(core::builder::GraphGraph &graph,
                                         const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 3 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr) {
    throw BuilderError(
        "CastLayerNormalizationCastPattern::Apply expects Cast, normalization, Cast.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[1]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "CastLayerNormalizationCastPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &cast_before = *nodes[0];
  const NodeProto &normalization = *nodes[1];
  const NodeProto &cast_after = *nodes[2];
  core::builder::GraphBuilder &builder = graph.Builder();
  const int64_t input_type = static_cast<int64_t>(
      core::symbolic::TensorTypeToDataType(graph.GetType(cast_before.input()[0].value())));

  utils::RepeatedProtoField<NodeProto> replacements;
  std::vector<std::string> inputs = {cast_before.input()[0].value()};
  for (std::size_t index = 1; index < normalization.input().size(); ++index) {
    const std::string &input = normalization.input()[index].value();
    const std::string output = builder.UniqueName("CastLayerNormalizationCastPattern_" + input +
                                                  "::C" + std::to_string(input_type));
    NodeProto cast = MakeNode(
        "Cast", {input}, {output}, "",
        ("CastLayerNormalizationCastPattern--cast--" + normalization.name().value()).c_str());
    AddAttribute<int64_t>(cast, "to", input_type);
    replacements.push_back(std::move(cast));
    inputs.push_back(output);
  }

  std::vector<std::string> outputs = {cast_after.output()[0].value()};
  for (std::size_t index = 1; index < normalization.output().size(); ++index) {
    outputs.push_back(normalization.output()[index].value());
  }
  NodeProto replacement =
      MakeNode(normalization.op_type().value().c_str(), inputs, outputs,
               normalization.domain().value().c_str(),
               ("CastLayerNormalizationCastPattern--" + normalization.name().value()).c_str());
  CopyAttributes(normalization, replacement);
  if (normalization.has_doc_string()) {
    replacement.set_doc_string(normalization.doc_string().value());
  }
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> BatchNormalizationPattern::FastOpType() const {
  return {"BatchNormalization"};
}

core::builder::MatchResult BatchNormalizationPattern::Match(core::builder::GraphGraph &graph,
                                                            const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "BatchNormalization") || candidate.input_size() < 5 ||
      candidate.output_size() < 1) {
    return NoMatch(candidate, "candidate is not a valid default-domain BatchNormalization");
  }
  if (candidate.output_size() > 1 && graph.IsUsed(candidate.output()[1].value())) {
    return NoMatch(candidate, "the running mean output is used");
  }
  if (candidate.output_size() > 2 && graph.IsUsed(candidate.output()[2].value())) {
    return NoMatch(candidate, "the running variance output is used");
  }
  const float epsilon = GetAttributeOr<float>(candidate, "epsilon", 1e-5F);
  const int64_t training_mode = GetAttributeOr<int64_t>(candidate, "training_mode", 0);
  if (training_mode != 0) {
    return NoMatch(candidate, "BatchNormalization is not in inference mode");
  }
  if (epsilon != 0.0F) {
    return NoMatch(candidate, "epsilon is not zero");
  }
  if (!graph.IsConstant(candidate.input()[1].value()) ||
      !graph.IsConstant(candidate.input()[2].value()) ||
      !graph.IsConstant(candidate.input()[3].value()) ||
      !graph.IsConstant(candidate.input()[4].value())) {
    return NoMatch(candidate, "a required BatchNormalization parameter is not constant");
  }

  for (std::size_t index : {std::size_t{2}, std::size_t{3}}) {
    std::vector<double> values;
    if (!ReadConstantValues(graph, candidate.input()[index].value(), values) ||
        !AllEqual(values, 0.0)) {
      return NoMatch(candidate, "the bias or input mean is not uniformly zero");
    }
  }
  for (std::size_t index : {std::size_t{1}, std::size_t{4}}) {
    std::vector<double> values;
    if (!ReadConstantValues(graph, candidate.input()[index].value(), values) ||
        !AllEqual(values, 1.0)) {
      return NoMatch(candidate, "the scale or input variance is not uniformly one");
    }
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
BatchNormalizationPattern::Apply(core::builder::GraphGraph &graph,
                                 const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError("BatchNormalizationPattern::Apply expects one BatchNormalization node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "BatchNormalizationPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &batch = *nodes[0];
  NodeProto replacement =
      MakeNode("Identity", {batch.input()[0].value()}, {batch.output()[0].value()}, "",
               ("BatchNormalizationPattern--" + batch.name().value()).c_str());
  if (batch.has_doc_string()) {
    replacement.set_doc_string(batch.doc_string().value());
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

std::set<std::string> BatchNormalizationTrainingPattern::FastOpType() const {
  return {"BatchNormalization"};
}

core::builder::MatchResult
BatchNormalizationTrainingPattern::Match(core::builder::GraphGraph &graph,
                                         const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "BatchNormalization") || candidate.input_size() < 5 ||
      candidate.output_size() < 1) {
    return NoMatch(candidate, "candidate is not a valid default-domain BatchNormalization");
  }
  if (!MainOpsetAtLeast(graph, 18)) {
    return NoMatch(candidate, "the default-domain opset is below 18");
  }
  std::size_t input_rank = 0;
  if (!HasRank(graph, candidate.input()[0].value(), input_rank) || input_rank < 2) {
    return NoMatch(candidate, "the data input rank is unknown or below two");
  }
  std::size_t scale_rank = 0;
  if (!HasRank(graph, candidate.input()[1].value(), scale_rank)) {
    return NoMatch(candidate, "the scale rank is unknown");
  }
  std::size_t bias_rank = 0;
  if (!HasRank(graph, candidate.input()[2].value(), bias_rank)) {
    return NoMatch(candidate, "the bias rank is unknown");
  }
  if (candidate.output_size() > 1 && graph.IsUsed(candidate.output()[1].value())) {
    return NoMatch(candidate, "the running mean output is used");
  }
  if (candidate.output_size() > 2 && graph.IsUsed(candidate.output()[2].value())) {
    return NoMatch(candidate, "the running variance output is used");
  }
  const int64_t training_mode = GetAttributeOr<int64_t>(candidate, "training_mode", 0);
  if (training_mode != 1) {
    return NoMatch(candidate, "BatchNormalization is not in training mode");
  }
  return core::builder::MatchResult{this, {&candidate}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
BatchNormalizationTrainingPattern::Apply(core::builder::GraphGraph &graph,
                                         const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 1 || nodes[0] == nullptr) {
    throw BuilderError(
        "BatchNormalizationTrainingPattern::Apply expects one BatchNormalization node.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "BatchNormalizationTrainingPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &batch = *nodes[0];
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string name = "BatchNormalizationTrainingPattern--" + batch.name().value();
  const std::size_t rank = graph.GetShape(batch.input()[0].value()).Shape().Rank();
  std::vector<int64_t> axes;
  for (std::size_t index = 0; index < rank; ++index) {
    if (index != 1) {
      axes.push_back(static_cast<int64_t>(index));
    }
  }
  const std::string axes_name =
      FreeInitializerName(builder, "BatchNormalizationTrainingPattern.axes");
  builder.MakeInitializer(MakeInitializerShape(axes_name.c_str(), axes));

  const std::string mean_name =
      builder.UniqueName("BatchNormalizationTrainingPattern_mean_" + batch.input()[0].value());
  NodeProto mean =
      MakeNode("ReduceMean", {batch.input()[0].value(), axes_name}, {mean_name}, "", name.c_str());
  AddAttribute<int64_t>(mean, "keepdims", 1);

  const std::string centered_name =
      builder.UniqueName("BatchNormalizationTrainingPattern_center_" + batch.input()[0].value());
  NodeProto sub =
      MakeNode("Sub", {batch.input()[0].value(), mean_name}, {centered_name}, "", name.c_str());

  const std::string squared_name =
      builder.UniqueName("BatchNormalizationTrainingPattern_x2_" + batch.input()[0].value());
  NodeProto square =
      MakeNode("Mul", {centered_name, centered_name}, {squared_name}, "", name.c_str());

  const std::string variance_name =
      builder.UniqueName("BatchNormalizationTrainingPattern_var_" + batch.input()[0].value());
  NodeProto variance =
      MakeNode("ReduceMean", {squared_name, axes_name}, {variance_name}, "", name.c_str());
  AddAttribute<int64_t>(variance, "keepdims", 1);

  if (!graph.HasType(batch.input()[0].value())) {
    throw BuilderError("BatchNormalizationTrainingPattern::Apply needs the input type.");
  }
  const std::string epsilon_name =
      FreeInitializerName(builder, "BatchNormalizationTrainingPattern.epsilon");
  builder.MakeInitializer(
      MakeFilledTensor(epsilon_name, graph.GetType(batch.input()[0].value()), {1},
                       static_cast<double>(GetAttributeOr<float>(batch, "epsilon", 1e-5F))));
  const std::string variance_epsilon_name =
      builder.UniqueName("BatchNormalizationTrainingPattern_vareps_" + batch.input()[0].value());
  NodeProto add =
      MakeNode("Add", {variance_name, epsilon_name}, {variance_epsilon_name}, "", name.c_str());
  const std::string standard_deviation_name =
      builder.UniqueName("BatchNormalizationTrainingPattern_vareps_" + batch.input()[0].value());
  NodeProto sqrt =
      MakeNode("Sqrt", {variance_epsilon_name}, {standard_deviation_name}, "", name.c_str());

  std::vector<int64_t> new_shape(rank, 1);
  new_shape[1] = -1;
  const std::string shape_name =
      FreeInitializerName(builder, "BatchNormalizationTrainingPattern.shape");
  builder.MakeInitializer(MakeInitializerShape(shape_name.c_str(), new_shape));

  if (!graph.HasShape(batch.input()[1].value()) || !graph.HasShape(batch.input()[2].value())) {
    throw BuilderError("BatchNormalizationTrainingPattern::Apply needs scale and bias ranks.");
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(mean));
  replacements.push_back(std::move(sub));
  replacements.push_back(std::move(square));
  replacements.push_back(std::move(variance));
  replacements.push_back(std::move(add));
  replacements.push_back(std::move(sqrt));

  std::string scale_name = batch.input()[1].value();
  if (graph.GetShape(scale_name).Shape().Rank() == 1) {
    const std::string reshaped =
        builder.UniqueName("BatchNormalizationTrainingPattern_scale_" + scale_name);
    replacements.push_back(
        MakeNode("Reshape", {scale_name, shape_name}, {reshaped}, "", name.c_str()));
    scale_name = reshaped;
  }
  std::string bias_name = batch.input()[2].value();
  if (graph.GetShape(bias_name).Shape().Rank() == 1) {
    const std::string reshaped =
        builder.UniqueName("BatchNormalizationTrainingPattern_bias_" + bias_name);
    replacements.push_back(
        MakeNode("Reshape", {bias_name, shape_name}, {reshaped}, "", name.c_str()));
    bias_name = reshaped;
  }

  const std::string scaled_name =
      builder.UniqueName("BatchNormalizationTrainingPattern_scaled_" + batch.input()[1].value());
  replacements.push_back(
      MakeNode("Div", {centered_name, standard_deviation_name}, {scaled_name}, "", name.c_str()));
  const std::string scaled_twice_name =
      builder.UniqueName("BatchNormalizationTrainingPattern_scaled2_" + batch.input()[2].value());
  replacements.push_back(
      MakeNode("Mul", {scaled_name, scale_name}, {scaled_twice_name}, "", name.c_str()));
  replacements.push_back(MakeNode("Add", {scaled_twice_name, bias_name},
                                  {batch.output()[0].value()}, "", name.c_str()));
  return replacements;
}

std::set<std::string> RMSNormalizationPattern::FastOpType() const { return {"ReduceMean"}; }

core::builder::MatchResult RMSNormalizationPattern::Match(core::builder::GraphGraph &graph,
                                                          const NodeProto &candidate) const {
  if (!MainOpsetAtLeast(graph, 23)) {
    return NoMatch(candidate, "the default-domain opset is below 23");
  }
  if (!IsDefaultOp(candidate, "ReduceMean") || candidate.input_size() < 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not a default-domain ReduceMean with axes");
  }
  std::vector<int64_t> axes;
  if (!ReadAxes(graph, candidate.input()[1].value(), axes) || axes.empty()) {
    return NoMatch(candidate, "the ReduceMean axes are not non-empty constant integers");
  }
  if (GetAttributeOr<int64_t>(candidate, "keepdims", 1) != 1) {
    return NoMatch(candidate, "the ReduceMean does not keep dimensions");
  }
  const NodeProto *power = graph.NodeBefore(candidate.input()[0].value());
  if (power == nullptr || !IsDefaultOp(*power, "Pow") || power->input_size() != 2 ||
      power->output_size() != 1) {
    return NoMatch(candidate, "the ReduceMean input is not produced by Pow");
  }
  int64_t axis = 0;
  if (!ResolveSuffixAxis(graph, power->input()[0].value(), axes, axis)) {
    return NoMatch(candidate, "the ReduceMean axes are not a suffix of the normalized input");
  }
  if (!graph.IsConstantScalar(power->input()[1].value(), 2.0, false)) {
    return NoMatch(candidate, "the Pow exponent is not the scalar two");
  }
  const NodeProto *add = OnlyConsumer(graph, candidate.output()[0].value());
  if (add == nullptr || !IsDefaultOp(*add, "Add") || add->input_size() != 2 ||
      add->output_size() != 1) {
    return NoMatch(candidate, "the ReduceMean is not followed by one default-domain Add");
  }
  std::string epsilon_name;
  std::vector<double> epsilon_values;
  if (!OtherInput(*add, candidate.output()[0].value(), epsilon_name) ||
      !graph.IsConstantScalar(epsilon_name, false) ||
      !ReadConstantValues(graph, epsilon_name, epsilon_values) || epsilon_values.empty()) {
    return NoMatch(candidate, "the Add does not combine the reduced value and one scalar epsilon");
  }
  const NodeProto *sqrt = OnlyConsumer(graph, add->output()[0].value());
  if (sqrt == nullptr || !IsDefaultOp(*sqrt, "Sqrt") || sqrt->input_size() != 1 ||
      sqrt->output_size() != 1) {
    return NoMatch(candidate, "the Add is not followed by one default-domain Sqrt");
  }
  const NodeProto *reciprocal = OnlyConsumer(graph, sqrt->output()[0].value());
  if (reciprocal == nullptr ||
      (reciprocal->op_type().value() != "Reciprocal" && reciprocal->op_type().value() != "Div") ||
      NormaliseDomain(reciprocal->domain().value()) != kDefaultOnnxDomain ||
      reciprocal->output_size() != 1) {
    return NoMatch(candidate, "the Sqrt is not followed by default-domain Reciprocal or Div");
  }
  if (reciprocal->op_type().value() == "Reciprocal") {
    if (reciprocal->input_size() != 1 ||
        reciprocal->input()[0].value() != sqrt->output()[0].value()) {
      return NoMatch(candidate, "the Reciprocal does not consume the Sqrt");
    }
  } else if (reciprocal->input_size() != 2 ||
             reciprocal->input()[1].value() != sqrt->output()[0].value() ||
             !graph.IsConstantScalar(reciprocal->input()[0].value(), 1.0, false)) {
    return NoMatch(candidate, "the Div is not one divided by the Sqrt");
  }
  const NodeProto *mul = OnlyConsumer(graph, reciprocal->output()[0].value());
  if (mul == nullptr || !IsDefaultOp(*mul, "Mul") || mul->input_size() != 2 ||
      mul->output_size() != 1) {
    return NoMatch(candidate, "the reciprocal is not followed by one default-domain Mul");
  }
  if (graph.IsUsedMoreThanOnce(power->output()[0].value()) ||
      graph.IsUsedMoreThanOnce(candidate.output()[0].value()) ||
      graph.IsUsedMoreThanOnce(add->output()[0].value()) ||
      graph.IsUsedMoreThanOnce(sqrt->output()[0].value())) {
    return NoMatch(candidate, "an intermediate RMS result has another use");
  }
  const std::set<std::string> mul_inputs = {mul->input()[0].value(), mul->input()[1].value()};
  const std::set<std::string> expected = {power->input()[0].value(),
                                          reciprocal->output()[0].value()};
  if (mul_inputs != expected) {
    return NoMatch(candidate, "the Mul does not combine the Pow input and reciprocal");
  }
  if (!HasOnlyConsumers(graph, *power, {&candidate}) ||
      !HasOnlyConsumers(graph, candidate, {add}) || !HasOnlyConsumers(graph, *add, {sqrt}) ||
      !HasOnlyConsumers(graph, *sqrt, {reciprocal}) ||
      !HasOnlyConsumers(graph, *reciprocal, {mul})) {
    return NoMatch(candidate, "an RMSNormalization intermediate is externally used");
  }

  const NodeProto *cast_before = graph.NodeBefore(power->input()[0].value());
  if (cast_before != nullptr && !IsDefaultOp(*cast_before, "Cast")) {
    cast_before = nullptr;
  }
  const NodeProto *cast_after = nullptr;
  if (cast_before != nullptr) {
    int64_t before_target = 0;
    if (!CastTarget(*cast_before, before_target) || !graph.HasType(candidate.input()[0].value()) ||
        before_target != static_cast<int64_t>(core::symbolic::TensorTypeToDataType(
                             graph.GetType(candidate.input()[0].value())))) {
      cast_before = nullptr;
    } else {
      cast_after = OnlyConsumer(graph, mul->output()[0].value());
      if (cast_after == nullptr || !IsDefaultOp(*cast_after, "Cast")) {
        cast_before = nullptr;
        cast_after = nullptr;
      } else {
        int64_t after_target = 0;
        if (!CastTarget(*cast_after, after_target) || cast_before->input_size() < 1 ||
            !graph.HasType(cast_before->input()[0].value()) ||
            after_target != static_cast<int64_t>(core::symbolic::TensorTypeToDataType(
                                graph.GetType(cast_before->input()[0].value())))) {
          cast_before = nullptr;
          cast_after = nullptr;
        }
      }
    }
  }
  if (cast_before != nullptr && !HasOnlyConsumers(graph, *cast_before, {power, mul})) {
    return NoMatch(candidate, "the input Cast output is externally used");
  }
  if (cast_after != nullptr && !HasOnlyConsumers(graph, *mul, {cast_after})) {
    return NoMatch(candidate, "the pre-output RMS value is externally used");
  }

  return core::builder::MatchResult{
      this, {cast_before, power, &candidate, add, sqrt, reciprocal, mul, cast_after}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
RMSNormalizationPattern::Apply(core::builder::GraphGraph &graph,
                               const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 8 || nodes[1] == nullptr || nodes[2] == nullptr || nodes[3] == nullptr ||
      nodes[4] == nullptr || nodes[5] == nullptr || nodes[6] == nullptr) {
    throw BuilderError(
        "RMSNormalizationPattern::Apply expects optional Cast, Pow, ReduceMean, Add, Sqrt, "
        "reciprocal, Mul, optional Cast.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[2]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError("RMSNormalizationPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto *cast_before = nodes[0];
  const NodeProto &power = *nodes[1];
  const NodeProto &reduce = *nodes[2];
  const NodeProto &add = *nodes[3];
  const NodeProto &mul = *nodes[6];
  const NodeProto *cast_after = nodes[7];
  core::builder::GraphBuilder &builder = graph.Builder();

  std::string epsilon_name;
  std::vector<double> epsilon_values;
  if (!OtherInput(add, reduce.output()[0].value(), epsilon_name) ||
      !ReadConstantValues(graph, epsilon_name, epsilon_values) || epsilon_values.empty()) {
    throw BuilderError("RMSNormalizationPattern::Apply could not read epsilon.");
  }
  std::vector<int64_t> axes;
  int64_t axis = 0;
  if (!ReadAxes(graph, reduce.input()[1].value(), axes) ||
      !ResolveSuffixAxis(graph, power.input()[0].value(), axes, axis) ||
      !graph.HasType(reduce.input()[0].value())) {
    throw BuilderError("RMSNormalizationPattern::Apply could not read axes or stash type.");
  }
  const TensorType stash_type = graph.GetType(reduce.input()[0].value());
  const std::string input_name =
      cast_before == nullptr ? power.input()[0].value() : cast_before->input()[0].value();
  const TensorType scale_type =
      cast_before == nullptr ? stash_type : graph.GetType(cast_before->input()[0].value());

  std::string scale;
  utils::RepeatedProtoField<NodeProto> replacements;
  if (graph.HasShape(input_name)) {
    const SymShape &shape = graph.GetShape(input_name).Shape();
    const int64_t resolved_axis = axis < 0 ? static_cast<int64_t>(shape.Rank()) + axis : axis;
    std::vector<int64_t> scale_shape;
    bool fully_known = resolved_axis >= 0 && resolved_axis < static_cast<int64_t>(shape.Rank());
    for (std::size_t index = static_cast<std::size_t>(std::max<int64_t>(resolved_axis, 0));
         fully_known && index < shape.Rank(); ++index) {
      fully_known = shape[index].IsInt();
      if (fully_known) {
        scale_shape.push_back(shape[index].AsInt());
      }
    }
    if (fully_known && !scale_shape.empty()) {
      std::string suffix;
      for (int64_t dimension : scale_shape) {
        suffix += (suffix.empty() ? "" : "x") + std::to_string(dimension);
      }
      scale = FreeInitializerName(builder, "ONES" + suffix);
      builder.MakeInitializer(MakeFilledTensor(scale, scale_type, scale_shape, 1.0));
    }
  }
  if (scale.empty()) {
    const std::string name = "RMSNormalizationPattern--" + reduce.name().value();
    const std::string shape = builder.UniqueName("RMSNormalizationPattern_Sh_" + input_name);
    NodeProto shape_node = MakeNode("Shape", {input_name}, {shape}, "", name.c_str());
    AddAttribute<int64_t>(shape_node, "start", axis);
    replacements.push_back(std::move(shape_node));
    scale = builder.UniqueName("RMSNormalizationPattern_Sc_" + input_name);
    NodeProto constant = MakeNode("ConstantOfShape", {shape}, {scale}, "", name.c_str());
    AddTensorAttribute(constant, "value", MakeFilledTensor("", scale_type, {1}, 1.0));
    replacements.push_back(std::move(constant));
  }

  NodeProto layer =
      MakeNode("RMSNormalization", {input_name, scale},
               {cast_after == nullptr ? mul.output()[0].value() : cast_after->output()[0].value()},
               "", ("RMSNormalizationPattern--" + reduce.name().value()).c_str());
  AddAttribute<float>(layer, "epsilon", static_cast<float>(epsilon_values[0]));
  AddAttribute<int64_t>(layer, "axis", axis);
  AddAttribute<int64_t>(layer, "stash_type",
                        static_cast<int64_t>(core::symbolic::TensorTypeToDataType(stash_type)));
  replacements.push_back(std::move(layer));
  return replacements;
}

std::set<std::string> RMSNormalizationMulPattern::FastOpType() const {
  return {"RMSNormalization"};
}

core::builder::MatchResult RMSNormalizationMulPattern::Match(core::builder::GraphGraph &graph,
                                                             const NodeProto &candidate) const {
  if (!MainOpsetAtLeast(graph, 23)) {
    return NoMatch(candidate, "the default-domain opset is below 23");
  }
  if (!IsDefaultOp(candidate, "RMSNormalization") || candidate.input_size() < 2 ||
      candidate.output_size() < 1) {
    return NoMatch(candidate, "candidate is not a default-domain RMSNormalization");
  }
  const std::vector<const NodeProto *> &mul_nodes = graph.NextNodes(candidate.output()[0].value());
  if (mul_nodes.size() != 1 || !IsDefaultOp(*mul_nodes[0], "Mul") ||
      mul_nodes[0]->input_size() != 2 || mul_nodes[0]->output_size() != 1) {
    return NoMatch(candidate, "RMSNormalization is not followed by one default-domain Mul");
  }
  const NodeProto *mul = mul_nodes[0];
  const std::string other = mul->input()[0].value() == candidate.output()[0].value()
                                ? mul->input()[1].value()
                                : mul->input()[0].value();
  if (!graph.HasShape(candidate.input()[1].value()) || !graph.HasShape(other)) {
    return NoMatch(candidate, "the two scale shapes are unknown");
  }
  if (graph.GetShape(candidate.input()[1].value()).Shape() != graph.GetShape(other).Shape()) {
    return NoMatch(candidate, "the two scale shapes differ");
  }
  if (!graph.IsConstant(candidate.input()[1].value()) || !graph.IsConstant(other)) {
    return NoMatch(candidate, "both scales are not constant");
  }
  if (!HasOnlyConsumers(graph, candidate, {mul})) {
    return NoMatch(candidate, "the RMSNormalization output is externally used");
  }
  const TensorProto *left = graph.GetComputedConstant(candidate.input()[1].value());
  const TensorProto *right = graph.GetComputedConstant(other);
  TensorProto product;
  if (left == nullptr || right == nullptr || !MultiplyConstants(*left, *right, "", product)) {
    return NoMatch(candidate, "the two constant scales cannot be multiplied");
  }
  return core::builder::MatchResult{this, {&candidate, mul}, mul};
}

utils::RepeatedProtoField<NodeProto>
RMSNormalizationMulPattern::Apply(core::builder::GraphGraph &graph,
                                  const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("RMSNormalizationMulPattern::Apply expects RMSNormalization and Mul.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "RMSNormalizationMulPattern::Apply received an unsafe or inconsistent match.");
  }
  const NodeProto &rms = *nodes[0];
  const NodeProto &mul = *nodes[1];
  const std::string other = mul.input()[0].value() == rms.output()[0].value()
                                ? mul.input()[1].value()
                                : mul.input()[0].value();
  const TensorProto *left = graph.GetComputedConstant(rms.input()[1].value());
  const TensorProto *right = graph.GetComputedConstant(other);
  if (left == nullptr || right == nullptr) {
    throw BuilderError("RMSNormalizationMulPattern::Apply could not read both scales.");
  }
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string scale_name = FreeInitializerName(builder, "RMSNormalizationMulPattern.cst");
  TensorProto product;
  if (!MultiplyConstants(*left, *right, scale_name, product)) {
    throw BuilderError("RMSNormalizationMulPattern::Apply could not multiply both scales.");
  }
  builder.MakeInitializer(product);

  NodeProto replacement =
      MakeNode("RMSNormalization", {rms.input()[0].value(), scale_name}, {mul.output()[0].value()},
               "", ("RMSNormalizationMulPattern--" + rms.name().value()).c_str());
  CopyAttributes(rms, replacement);
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(std::move(replacement));
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
