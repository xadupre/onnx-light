// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/attention/attention_pattern.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
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
using core::symbolic::TensorType;

constexpr const char *kIntermediateDomain = "intermediate";

bool MainOpsetAtLeast(core::builder::GraphGraph &graph, int minimum) {
  const int opset = graph.Builder().OpsetVersion("");
  return opset == core::shapes::kUnknownOpsetVersion || opset >= minimum;
}

int ResolvedMainOpset(core::builder::GraphBuilder &builder, int minimum) {
  const int opset = builder.OpsetVersion("");
  return opset == core::shapes::kUnknownOpsetVersion ? minimum : opset;
}

bool IsNode(const NodeProto *node, const char *op_type, int inputs = -1, int outputs = -1) {
  return node != nullptr && IsDefaultOp(*node, op_type) &&
         (inputs < 0 || node->input_size() == inputs) &&
         (outputs < 0 || node->output_size() == outputs);
}

bool IsDomainNode(const NodeProto *node, const char *op_type, const char *domain, int inputs = -1,
                  int outputs = -1) {
  return node != nullptr && node->op_type().value() == op_type &&
         NormaliseDomain(node->domain().value()) == NormaliseDomain(domain) &&
         (inputs < 0 || node->input_size() == inputs) &&
         (outputs < 0 || node->output_size() == outputs);
}

bool ReadConstantInts(core::builder::GraphGraph &graph, const std::string &name,
                      std::vector<int64_t> &values) {
  if (!graph.IsConstant(name)) {
    return false;
  }
  const TensorProto *tensor = graph.GetComputedConstant(name);
  return tensor != nullptr && ReadIntegerValues(*tensor, values);
}

bool ReadInt64Constant(core::builder::GraphGraph &graph, const std::string &name,
                       std::vector<int64_t> &values) {
  const TensorProto *tensor = graph.GetComputedConstant(name);
  return graph.IsConstant(name) && tensor != nullptr &&
         tensor->data_type() == static_cast<int32_t>(TensorProto::DataType::INT64) &&
         ReadIntegerValues(*tensor, values);
}

bool Read16BitScalar(const TensorProto &tensor, bool bfloat, double &value) {
  std::uint16_t bits = 0;
  if (!tensor.int32_data().empty()) {
    bits = static_cast<std::uint16_t>(tensor.int32_data()[0]);
  } else if (tensor.is_raw_data() && tensor.ref_raw_data().size() >= 2) {
    bits = static_cast<std::uint16_t>(tensor.ref_raw_data()[0]) |
           (static_cast<std::uint16_t>(tensor.ref_raw_data()[1]) << 8);
  } else {
    return false;
  }
  value = static_cast<double>(bfloat ? core::runtime::Bfloat16BitsToFloat(bits)
                                     : core::runtime::Float16BitsToFloat(bits));
  return true;
}

bool ReadConstantScalar(core::builder::GraphGraph &graph, const std::string &name, double &value,
                        const TensorProto **tensor_out = nullptr) {
  if (!graph.IsConstantScalar(name, false)) {
    return false;
  }
  const TensorProto *tensor = graph.GetComputedConstant(name);
  if (tensor == nullptr) {
    return false;
  }
  if (tensor_out != nullptr) {
    *tensor_out = tensor;
  }
  const auto type = static_cast<TensorProto::DataType>(tensor->data_type());
  if (type == TensorProto::DataType::FLOAT16) {
    return Read16BitScalar(*tensor, false, value);
  }
  if (type == TensorProto::DataType::BFLOAT16) {
    return Read16BitScalar(*tensor, true, value);
  }
  return ReadScalarAsDouble(*tensor, value);
}

bool IsScalar(core::builder::GraphGraph &graph, const std::string &name, double expected) {
  double value = 0.0;
  return ReadConstantScalar(graph, name, value) && value == expected;
}

bool IsNegativeInfinity(core::builder::GraphGraph &graph, const std::string &name) {
  double value = 0.0;
  return ReadConstantScalar(graph, name, value) && std::isinf(value) && value < 0.0;
}

std::optional<int> NegativeInfinityInput(core::builder::GraphGraph &graph, const NodeProto &where) {
  if (!IsNode(&where, "Where", 3, 1)) {
    return std::nullopt;
  }
  for (int index : {1, 2}) {
    if (IsNegativeInfinity(graph, where.input()[index].value())) {
      return index;
    }
  }
  return std::nullopt;
}

bool ReadAttributeInts(const NodeProto &node, const char *name, std::vector<int64_t> &values) {
  values.clear();
  return GetAttributeInts(node, name, values);
}

bool HasPerm(const NodeProto &node, const std::vector<int64_t> &expected) {
  std::vector<int64_t> actual;
  return ReadAttributeInts(node, "perm", actual) && actual == expected;
}

int64_t Axis(const NodeProto &node, int64_t default_value = 0) {
  return GetAttributeOr<int64_t>(node, "axis", default_value);
}

bool NormalizedAxis(const NodeProto &node, int64_t rank, int64_t &axis) {
  axis = Axis(node, 0);
  if (axis < 0) {
    axis += rank;
  }
  return axis >= 0 && axis < rank;
}

bool TensorIsZero(const TensorProto &tensor) {
  const auto type = static_cast<TensorProto::DataType>(tensor.data_type());
  if (type == TensorProto::DataType::FLOAT16 || type == TensorProto::DataType::BFLOAT16) {
    double value = 0.0;
    return Read16BitScalar(tensor, type == TensorProto::DataType::BFLOAT16, value) && value == 0.0;
  }
  std::vector<double> floating;
  if (ReadFloatingValues(tensor, floating)) {
    return !floating.empty() &&
           std::all_of(floating.begin(), floating.end(), [](double value) { return value == 0.0; });
  }
  std::vector<int64_t> integers;
  if (ReadIntegerValues(tensor, integers)) {
    return !integers.empty() &&
           std::all_of(integers.begin(), integers.end(), [](int64_t value) { return value == 0; });
  }
  return false;
}

bool IsZeroConstantOfShape(const NodeProto &node) {
  if (!IsNode(&node, "ConstantOfShape", 1, 1)) {
    return false;
  }
  const AttributeProto *value = FindAttribute(node, "value");
  return value == nullptr ||
         (value->type() == AttributeProto::AttributeType::TENSOR && TensorIsZero(value->ref_t()));
}

bool HasRank(core::builder::GraphGraph &graph, const std::string &name, std::size_t rank) {
  return graph.HasShape(name) && graph.GetShape(name).Shape().Rank() == rank;
}

bool StaticDimension(core::builder::GraphGraph &graph, const std::string &name, std::size_t index,
                     int64_t &dimension) {
  if (!graph.HasShape(name)) {
    return false;
  }
  const SymShape &shape = graph.GetShape(name).Shape();
  if (index >= shape.Rank() || !shape[index].IsInt()) {
    return false;
  }
  dimension = shape[index].AsInt();
  return true;
}

bool SameSelectedDimensions(const SymShape &left, const SymShape &right,
                            const std::vector<std::size_t> &indices) {
  if (left.Rank() != right.Rank()) {
    return false;
  }
  for (std::size_t index : indices) {
    if (index >= left.Rank() || left[index] != right[index]) {
      return false;
    }
  }
  return true;
}

bool NodeOutputHasExternalUse(core::builder::GraphGraph &graph, const NodeProto &node,
                              const std::unordered_set<const NodeProto *> &matched,
                              const std::unordered_set<std::string> &replaced_outputs) {
  for (int output_index = 0; output_index < node.output_size(); ++output_index) {
    const std::string &output = node.output()[output_index].value();
    if (output.empty() || replaced_outputs.count(output) != 0) {
      continue;
    }
    if (graph.IsOutput(output) || graph.IsUsedBySubgraph(output)) {
      return true;
    }
    for (const NodeProto *consumer : graph.NextNodes(output)) {
      if (matched.count(consumer) == 0) {
        return true;
      }
    }
  }
  return false;
}

bool HasUnsafeMovedOutput(core::builder::GraphGraph &graph, const std::string &output,
                          const NodeProto &insert_at,
                          const std::unordered_set<const NodeProto *> &removed) {
  if (graph.IsUsedBySubgraph(output)) {
    return true;
  }
  const std::size_t insertion_position = graph.Position(insert_at);
  for (const NodeProto *consumer : graph.NextNodes(output)) {
    if (removed.count(consumer) == 0 && graph.Position(*consumer) < insertion_position) {
      return true;
    }
  }
  return false;
}

bool HasUnsupportedAttributes(const NodeProto &node,
                              const std::unordered_set<std::string> &supported) {
  for (const AttributeProto &attribute : node.attribute()) {
    if (supported.count(attribute.name().value()) == 0) {
      return true;
    }
  }
  return false;
}

bool HasActiveInputsFrom(const NodeProto &node, int start) {
  for (int index = start; index < node.input_size(); ++index) {
    if (!node.input()[index].value().empty()) {
      return true;
    }
  }
  return false;
}

std::vector<const NodeProto *>
PreservedNodes(core::builder::GraphGraph &graph, const std::vector<const NodeProto *> &nodes,
               const std::vector<std::string> &replaced_output_names) {
  std::unordered_set<const NodeProto *> matched;
  for (const NodeProto *node : nodes) {
    if (node != nullptr) {
      matched.insert(node);
    }
  }
  const std::unordered_set<std::string> replaced_outputs(replaced_output_names.begin(),
                                                         replaced_output_names.end());
  std::unordered_set<const NodeProto *> keep;
  for (const NodeProto *node : matched) {
    if (NodeOutputHasExternalUse(graph, *node, matched, replaced_outputs)) {
      keep.insert(node);
    }
  }
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<const NodeProto *> current(keep.begin(), keep.end());
    for (const NodeProto *node : current) {
      for (int input_index = 0; input_index < node->input_size(); ++input_index) {
        const NodeProto *producer = graph.NodeBefore(node->input()[input_index].value());
        if (producer != nullptr && matched.count(producer) != 0 && keep.insert(producer).second) {
          changed = true;
        }
      }
    }
  }
  std::vector<const NodeProto *> ordered(keep.begin(), keep.end());
  std::sort(ordered.begin(), ordered.end(), [&](const NodeProto *left, const NodeProto *right) {
    return graph.Position(*left) < graph.Position(*right);
  });
  return ordered;
}

void AppendOriginalsInGraphOrder(core::builder::GraphGraph &graph,
                                 utils::RepeatedProtoField<NodeProto> &result,
                                 const std::vector<const NodeProto *> &nodes) {
  std::unordered_set<const NodeProto *> unique;
  std::vector<const NodeProto *> ordered;
  for (const NodeProto *node : nodes) {
    if (node != nullptr && unique.insert(node).second) {
      ordered.push_back(node);
    }
  }
  std::sort(ordered.begin(), ordered.end(), [&](const NodeProto *left, const NodeProto *right) {
    return graph.Position(*left) < graph.Position(*right);
  });
  for (const NodeProto *node : ordered) {
    result.push_back(*node);
  }
}

NodeProto MakePatternNode(const char *op_type, const std::vector<std::string> &inputs,
                          const std::vector<std::string> &outputs, const std::string &domain,
                          const std::string &name) {
  return MakeNode(op_type, inputs, outputs, domain.c_str(), name.c_str());
}

void AddIntAttribute(NodeProto &node, const char *name, int64_t value) {
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::INT);
  attribute->set_i(value);
}

void AddFloatAttribute(NodeProto &node, const char *name, float value) {
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::FLOAT);
  attribute->set_f(value);
}

void AddIntsAttribute(NodeProto &node, const char *name, const std::vector<int64_t> &values) {
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::INTS);
  for (int64_t value : values) {
    attribute->ints().push_back(value);
  }
}

void AddTensorAttribute(NodeProto &node, const char *name, TensorProto tensor) {
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name(name);
  attribute->set_type(AttributeProto::AttributeType::TENSOR);
  *attribute->mutable_t() = std::move(tensor);
}

TensorProto MakeInt64Tensor(const std::vector<int64_t> &dims, const std::vector<int64_t> &values) {
  return MakeInitializer<int64_t>("value", dims, values);
}

TensorProto MakeFloatingTensor(TensorProto::DataType type, float value) {
  TensorProto tensor;
  tensor.set_name("value");
  tensor.set_data_type(static_cast<int32_t>(type));
  tensor.add_dims(1);
  switch (type) {
  case TensorProto::DataType::FLOAT:
    tensor.ref_float_data().push_back(value);
    break;
  case TensorProto::DataType::DOUBLE:
    tensor.ref_double_data().push_back(static_cast<double>(value));
    break;
  case TensorProto::DataType::FLOAT16:
    tensor.ref_int32_data().push_back(
        static_cast<int32_t>(core::runtime::FloatToFloat16Bits(value)));
    break;
  case TensorProto::DataType::BFLOAT16:
    tensor.ref_int32_data().push_back(
        static_cast<int32_t>(core::runtime::FloatToBfloat16Bits(value)));
    break;
  default:
    break;
  }
  return tensor;
}

bool IsSupportedAttentionType(TensorProto::DataType type) {
  return type == TensorProto::DataType::FLOAT || type == TensorProto::DataType::DOUBLE ||
         type == TensorProto::DataType::FLOAT16 || type == TensorProto::DataType::BFLOAT16;
}

NodeProto MakeConstantNode(const std::string &output, TensorProto value, const std::string &name) {
  NodeProto node = MakePatternNode("Constant", {}, {output}, "", name);
  AddTensorAttribute(node, "value", std::move(value));
  return node;
}

FunctionProto StartFunction(const std::string &name, const std::vector<std::string> &inputs,
                            const std::vector<std::string> &outputs, int opset) {
  FunctionProto function;
  function.set_name(name);
  function.set_domain(kIntermediateDomain);
  function.add_opset("", opset);
  for (const std::string &input : inputs) {
    function.add_input(input);
  }
  for (const std::string &output : outputs) {
    function.add_output(output);
  }
  return function;
}

NodeProto &AddFunctionNode(FunctionProto &function, const char *op_type,
                           const std::vector<std::string> &inputs,
                           const std::vector<std::string> &outputs, const std::string &name) {
  return function.add_node(op_type, inputs, outputs, "", name);
}

bool CanEnsureFunction(core::builder::GraphBuilder &builder, const std::string &name) {
  return builder.HasLocalFunction(name) || !builder.HasName(name);
}

SymShape LocalShape(const std::string &prefix, std::size_t rank) {
  SymShape shape;
  for (std::size_t index = 0; index < rank; ++index) {
    shape.PushBack(SymDim(prefix + std::to_string(index)));
  }
  return shape;
}

void MakeLocalInput(core::builder::GraphBuilder &local, const std::string &function_name,
                    const std::string &input, std::size_t index) {
  if (function_name == "HalfRotaryEmbedding") {
    local.MakeInput(input, TensorType::kFloat, LocalShape(input, index == 0 ? 4 : 2));
    return;
  }
  if (function_name == "CausalMask" || function_name == "ShiftedCausalMask" ||
      function_name == "CausalMaskMulAdd") {
    local.MakeInput(input, TensorType::kInt64, LocalShape(input, 1));
    return;
  }
  if (function_name.starts_with("CosSinCache")) {
    local.MakeInput(input, input == "weights" ? TensorType::kFloat : TensorType::kInt64,
                    LocalShape(input, input == "weights" ? 3 : 1));
    return;
  }
  if (function_name.starts_with("LocalAttention")) {
    if (input == "expand_shape") {
      local.MakeInput(input, TensorType::kInt64, SymShape({SymDim(5)}));
    } else if (input == "gqa_shape") {
      const bool squeeze = function_name.find("sQ") != std::string::npos;
      local.MakeInput(input, TensorType::kInt64, SymShape({SymDim(squeeze ? 1 : 4)}));
    } else if (input == "scale_sqrt") {
      local.MakeInput(input, TensorType::kFloat, SymShape({SymDim(1)}));
    } else if (input == "mask" || input == "not_mask") {
      local.MakeInput(input, TensorType::kBool, LocalShape(input, 4));
    } else {
      local.MakeInput(input, TensorType::kFloat, LocalShape(input, 4));
    }
    return;
  }
  ValueInfoProto value_info;
  value_info.set_name(input);
  local.MakeInput(value_info);
}

void EnsureFunction(core::builder::GraphBuilder &builder, const FunctionProto &function) {
  const std::string name = function.name().value();
  if (builder.HasLocalFunction(name)) {
    return;
  }
  core::builder::GraphBuilder &local = builder.MakeLocalFunction(name, function.domain().value());
  for (const OperatorSetIdProto &opset : function.opset_import()) {
    local.SetOpsetVersion(opset.domain().value(), static_cast<int>(opset.version()));
  }
  for (std::size_t index = 0; index < function.input().size(); ++index) {
    MakeLocalInput(local, name, function.input(index).value(), index);
  }
  for (const NodeProto &node : function.node()) {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    for (const utils::String &input : node.input()) {
      inputs.push_back(input.value());
    }
    for (const utils::String &output : node.output()) {
      outputs.push_back(output.value());
    }
    utils::RepeatedProtoField<AttributeProto> attributes;
    for (const AttributeProto &attribute : node.attribute()) {
      attributes.push_back(attribute);
    }
    local.MakeNode(node.op_type().value(), inputs, outputs, node.domain().value(),
                   node.name().value(), attributes);
  }
  for (const utils::String &output : function.output()) {
    local.MakeOutput(output.value());
  }
}

FunctionProto MakeHalfRotaryFunction(core::builder::GraphBuilder &builder) {
  const std::string name = "HalfRotaryEmbedding";
  FunctionProto function =
      StartFunction(name, {"X", "cos_cache", "sin_cache"}, {"Y"}, ResolvedMainOpset(builder, 18));
  NodeProto &split = AddFunctionNode(function, "Split", {"X"}, {"left", "right"},
                                     "FunctionHalfRotaryEmbeddingPattern");
  AddIntAttribute(split, "num_outputs", 2);
  AddIntAttribute(split, "axis", -1);
  AddFunctionNode(function, "Neg", {"right"}, {"right_neg"}, "FunctionHalfRotaryEmbeddingPattern");
  NodeProto &concat = AddFunctionNode(function, "Concat", {"right_neg", "left"}, {"rotated"},
                                      "FunctionHalfRotaryEmbeddingPattern");
  AddIntAttribute(concat, "axis", -1);
  AddFunctionNode(function, "Mul", {"X", "cos_cache"}, {"x_cos"},
                  "FunctionHalfRotaryEmbeddingPattern");
  AddFunctionNode(function, "Mul", {"rotated", "sin_cache"}, {"rotated_sin"},
                  "FunctionHalfRotaryEmbeddingPattern");
  AddFunctionNode(function, "Add", {"x_cos", "rotated_sin"}, {"Y"},
                  "FunctionHalfRotaryEmbeddingPattern");
  return function;
}

FunctionProto MakeCausalMaskFunction(core::builder::GraphBuilder &builder, bool shifted) {
  const std::string name = shifted ? "ShiftedCausalMask" : "CausalMask";
  FunctionProto function = StartFunction(name,
                                         shifted ? std::vector<std::string>{"A", "B", "shift"}
                                                 : std::vector<std::string>{"A", "B"},
                                         {"mask"}, ResolvedMainOpset(builder, 13));
  AddFunctionNode(function, "Squeeze", {"A"}, {"sA"}, "FunctionCausalMaskPattern");
  AddFunctionNode(function, "Squeeze", {"B"}, {"sB"}, "FunctionCausalMaskPattern");
  function.ref_node().push_back(
      MakeConstantNode("zero", MakeInt64Tensor({}, {0}), "FunctionCausalMaskPattern"));
  function.ref_node().push_back(
      MakeConstantNode("one", MakeInt64Tensor({}, {1}), "FunctionCausalMaskPattern"));
  AddFunctionNode(function, "Range", {"zero", "sB", "one"}, {"range1"},
                  "FunctionCausalMaskPattern");
  AddFunctionNode(function, "Range", {"sA", "sB", "one"}, {"range2"}, "FunctionCausalMaskPattern");
  function.ref_node().push_back(
      MakeConstantNode("axes1", MakeInt64Tensor({3}, {0, 1, 2}), "FunctionCausalMaskPattern"));
  function.ref_node().push_back(
      MakeConstantNode("axes2", MakeInt64Tensor({3}, {0, 1, 3}), "FunctionCausalMaskPattern"));
  AddFunctionNode(function, "Unsqueeze", {"range1", "axes1"}, {"unsqueezed1"},
                  "FunctionCausalMaskPattern");
  AddFunctionNode(function, "Unsqueeze", {"range2", "axes2"}, {"unsqueezed2"},
                  "FunctionCausalMaskPattern");
  if (shifted) {
    AddFunctionNode(function, "Sub", {"unsqueezed2", "shift"}, {"shifted"},
                    "FunctionCausalMaskPattern");
    AddFunctionNode(function, "Greater", {"unsqueezed1", "shifted"}, {"mask"},
                    "FunctionCausalMaskPattern");
  } else {
    AddFunctionNode(function, "LessOrEqual", {"unsqueezed1", "unsqueezed2"}, {"mask"},
                    "FunctionCausalMaskPattern");
  }
  return function;
}

FunctionProto MakeCausalMaskMulAddFunction(core::builder::GraphBuilder &builder) {
  FunctionProto function =
      StartFunction("CausalMaskMulAdd", {"A", "B", "C"}, {"mask"}, ResolvedMainOpset(builder, 13));
  AddFunctionNode(function, "Squeeze", {"A"}, {"sA"}, "FunctionCausalMaskMulAddPattern");
  AddFunctionNode(function, "Squeeze", {"B"}, {"sB"}, "FunctionCausalMaskMulAddPattern");
  function.ref_node().push_back(
      MakeConstantNode("zero", MakeInt64Tensor({}, {0}), "FunctionCausalMaskMulAddPattern"));
  function.ref_node().push_back(
      MakeConstantNode("one", MakeInt64Tensor({}, {1}), "FunctionCausalMaskMulAddPattern"));
  AddFunctionNode(function, "Range", {"zero", "sA", "one"}, {"range1"},
                  "FunctionCausalMaskMulAddPattern");
  AddFunctionNode(function, "Range", {"zero", "sB", "one"}, {"range2"},
                  "FunctionCausalMaskMulAddPattern");
  function.ref_node().push_back(MakeConstantNode("axes1", MakeInt64Tensor({3}, {0, 1, 2}),
                                                 "FunctionCausalMaskMulAddPattern"));
  function.ref_node().push_back(MakeConstantNode("axes2", MakeInt64Tensor({3}, {1, 2, 3}),
                                                 "FunctionCausalMaskMulAddPattern"));
  AddFunctionNode(function, "Unsqueeze", {"range1", "axes1"}, {"unsqueezed1"},
                  "FunctionCausalMaskMulAddPattern");
  AddFunctionNode(function, "Unsqueeze", {"range2", "axes2"}, {"unsqueezed2"},
                  "FunctionCausalMaskMulAddPattern");
  AddFunctionNode(function, "Mul", {"unsqueezed2", "C"}, {"scaled"},
                  "FunctionCausalMaskMulAddPattern");
  AddFunctionNode(function, "Add", {"scaled", "unsqueezed1"}, {"mask"},
                  "FunctionCausalMaskMulAddPattern");
  return function;
}

FunctionProto MakeCosSinCacheFunction(core::builder::GraphBuilder &builder, const std::string &name,
                                      std::optional<int64_t> to,
                                      const std::vector<int64_t> &position_axes, bool with_range) {
  FunctionProto function =
      StartFunction(name,
                    with_range ? std::vector<std::string>{"dim1", "dim2", "weights"}
                               : std::vector<std::string>{"position_ids", "weights"},
                    {"cos", "sin"}, ResolvedMainOpset(builder, 13));
  std::string positions = "position_ids";
  if (with_range) {
    AddFunctionNode(function, "Squeeze", {"dim1"}, {"sA"}, "FunctionCosSinCachePattern");
    AddFunctionNode(function, "Squeeze", {"dim2"}, {"sB"}, "FunctionCosSinCachePattern");
    function.ref_node().push_back(
        MakeConstantNode("one", MakeInt64Tensor({}, {1}), "FunctionCosSinCachePattern"));
    AddFunctionNode(function, "Range", {"sA", "sB", "one"}, {"positions"},
                    "FunctionCosSinCachePattern");
    positions = "positions";
  }
  function.ref_node().push_back(MakeConstantNode(
      "axes", MakeInt64Tensor({static_cast<int64_t>(position_axes.size())}, position_axes),
      "FunctionCosSinCachePattern"));
  AddFunctionNode(function, "Unsqueeze", {positions, "axes"}, {"unsqueezed"},
                  "FunctionCosSinCachePattern");
  NodeProto &cast =
      AddFunctionNode(function, "Cast", {"unsqueezed"}, {"cast"}, "FunctionCosSinCachePattern");
  AddIntAttribute(cast, "to", static_cast<int64_t>(TensorProto::DataType::FLOAT));
  function.ref_node().push_back(MakeConstantNode("reshape_shape", MakeInt64Tensor({3}, {0, -1, 1}),
                                                 "FunctionCosSinCachePattern"));
  AddFunctionNode(function, "Reshape", {"cast", "reshape_shape"}, {"reshaped"},
                  "FunctionCosSinCachePattern");
  AddFunctionNode(function, "Mul", {"weights", "reshaped"}, {"weighted"},
                  "FunctionCosSinCachePattern");
  if (to.has_value()) {
    AddFunctionNode(function, "Cos", {"weighted"}, {"cos_before_cast"},
                    "FunctionCosSinCachePattern");
    AddFunctionNode(function, "Sin", {"weighted"}, {"sin_before_cast"},
                    "FunctionCosSinCachePattern");
    NodeProto &cos_cast = AddFunctionNode(function, "Cast", {"cos_before_cast"}, {"cos"},
                                          "FunctionCosSinCachePattern");
    AddIntAttribute(cos_cast, "to", to.value());
    NodeProto &sin_cast = AddFunctionNode(function, "Cast", {"sin_before_cast"}, {"sin"},
                                          "FunctionCosSinCachePattern");
    AddIntAttribute(sin_cast, "to", to.value());
  } else {
    AddFunctionNode(function, "Cos", {"weighted"}, {"cos"}, "FunctionCosSinCachePattern");
    AddFunctionNode(function, "Sin", {"weighted"}, {"sin"}, "FunctionCosSinCachePattern");
  }
  return function;
}

FunctionProto MakeAttentionFunction(core::builder::GraphBuilder &builder, const std::string &name,
                                    TensorProto::DataType type, bool gqa, bool switch_where,
                                    bool squeeze_gqa) {
  std::vector<std::string> inputs = {"query", "keys", "values", switch_where ? "not_mask" : "mask",
                                     "scale_sqrt"};
  if (gqa) {
    inputs.push_back("expand_shape");
    inputs.push_back("gqa_shape");
  }
  FunctionProto function = StartFunction(name, inputs, {"Y"}, ResolvedMainOpset(builder, 18));
  AddFunctionNode(function, "Mul", {"keys", "scale_sqrt"}, {"scaled_keys"},
                  "FunctionAttentionPattern");
  std::string scaled_keys = "scaled_keys";
  std::string values = "values";
  if (gqa) {
    function.ref_node().push_back(
        MakeConstantNode("two", MakeInt64Tensor({1}, {2}), "FunctionAttentionPattern"));
    AddFunctionNode(function, "Unsqueeze", {"scaled_keys", "two"}, {"unsqueezed_keys"},
                    "FunctionAttentionPattern");
    AddFunctionNode(function, "Unsqueeze", {"values", "two"}, {"unsqueezed_values"},
                    "FunctionAttentionPattern");
    AddFunctionNode(function, "Expand", {"unsqueezed_keys", "expand_shape"}, {"expanded_keys"},
                    "FunctionAttentionPattern");
    AddFunctionNode(function, "Expand", {"unsqueezed_values", "expand_shape"}, {"expanded_values"},
                    "FunctionAttentionPattern");
    AddFunctionNode(function, squeeze_gqa ? "Squeeze" : "Reshape", {"expanded_keys", "gqa_shape"},
                    {"repeated_keys"}, "FunctionAttentionPattern");
    AddFunctionNode(function, squeeze_gqa ? "Squeeze" : "Reshape", {"expanded_values", "gqa_shape"},
                    {"repeated_values"}, "FunctionAttentionPattern");
    scaled_keys = "repeated_keys";
    values = "repeated_values";
  }
  AddFunctionNode(function, "Mul", {"query", "scale_sqrt"}, {"scaled_query"},
                  "FunctionAttentionPattern");
  NodeProto &transpose = AddFunctionNode(function, "Transpose", {scaled_keys}, {"scaled_keys_t"},
                                         "FunctionAttentionPattern");
  AddIntsAttribute(transpose, "perm", {0, 1, 3, 2});
  AddFunctionNode(function, "MatMul", {"scaled_query", "scaled_keys_t"}, {"qk"},
                  "FunctionAttentionPattern");
  function.ref_node().push_back(
      MakeConstantNode("zero", MakeFloatingTensor(type, 0.0F), "FunctionAttentionPattern"));
  function.ref_node().push_back(
      MakeConstantNode("minfty", MakeFloatingTensor(type, -std::numeric_limits<float>::infinity()),
                       "FunctionAttentionPattern"));
  AddFunctionNode(function, "Where",
                  switch_where ? std::vector<std::string>{"not_mask", "minfty", "qk"}
                               : std::vector<std::string>{"mask", "qk", "minfty"},
                  {"masked_qk"}, "FunctionAttentionPattern");
  NodeProto &softmax =
      AddFunctionNode(function, "Softmax", {"masked_qk"}, {"softmax"}, "FunctionAttentionPattern");
  AddIntAttribute(softmax, "axis", -1);
  AddFunctionNode(function, "IsNaN", {"softmax"}, {"isnan"}, "FunctionAttentionPattern");
  AddFunctionNode(function, "Where", {"isnan", "zero", "softmax"}, {"filtered"},
                  "FunctionAttentionPattern");
  AddFunctionNode(function, "MatMul", {"filtered", values}, {"Y"}, "FunctionAttentionPattern");
  return function;
}

struct GqaBranch {
  const NodeProto *unsqueeze = nullptr;
  const NodeProto *expand = nullptr;
  const NodeProto *reshape = nullptr;
  std::vector<int64_t> unsqueeze_shape;
  std::vector<int64_t> expand_shape;
  std::vector<int64_t> reshape_shape;
};

bool MatchGqaBranch(core::builder::GraphGraph &graph, const std::string &output, bool allow_squeeze,
                    GqaBranch &branch) {
  branch.reshape = graph.NodeBefore(output);
  if (branch.reshape == nullptr ||
      !(IsDefaultOp(*branch.reshape, "Reshape") ||
        (allow_squeeze && IsDefaultOp(*branch.reshape, "Squeeze"))) ||
      branch.reshape->input_size() != 2 || branch.reshape->output_size() != 1) {
    return false;
  }
  branch.expand = graph.NodeBefore(branch.reshape->input()[0].value());
  if (!IsNode(branch.expand, "Expand", 2, 1)) {
    return false;
  }
  branch.unsqueeze = graph.NodeBefore(branch.expand->input()[0].value());
  if (!IsNode(branch.unsqueeze, "Unsqueeze", 2, 1) ||
      !ReadConstantInts(graph, branch.expand->input()[1].value(), branch.expand_shape) ||
      branch.expand_shape.size() != 5 || branch.expand_shape[0] != 1 ||
      branch.expand_shape[1] != 1 || branch.expand_shape[3] != 1 || branch.expand_shape[4] != 1 ||
      !ReadConstantInts(graph, branch.unsqueeze->input()[1].value(), branch.unsqueeze_shape) ||
      branch.unsqueeze_shape != std::vector<int64_t>{2} ||
      !ReadConstantInts(graph, branch.reshape->input()[1].value(), branch.reshape_shape)) {
    return false;
  }
  if (IsDefaultOp(*branch.reshape, "Reshape")) {
    if (branch.reshape_shape.size() != 4) {
      return false;
    }
  } else if (branch.reshape_shape.size() != 1) {
    return false;
  }
  if (!graph.HasShape(branch.unsqueeze->input()[0].value()) ||
      !graph.HasShape(branch.reshape->output()[0].value())) {
    return false;
  }
  const SymShape &before = graph.GetShape(branch.unsqueeze->input()[0].value()).Shape();
  const SymShape &after = graph.GetShape(branch.reshape->output()[0].value()).Shape();
  return before.Rank() >= 4 && after.Rank() >= 4 && before[0] == after[0] &&
         before[2] == after[2] && before[3] == after[3];
}

std::string AttentionFunctionName(TensorProto::DataType type, bool gqa, bool switch_where,
                                  bool no_transpose) {
  std::string name = "LocalAttention";
  if (gqa) {
    name += "GQA";
  }
  if (switch_where) {
    name += "SW";
  }
  if (no_transpose) {
    name += "NoT";
  }
  name += "_to" + std::to_string(static_cast<int32_t>(type));
  return name;
}

utils::RepeatedProtoField<NodeProto>
ApplyRotaryTranspose(core::builder::GraphGraph &graph, const std::vector<const NodeProto *> &nodes);

} // namespace

core::builder::MatchResult FunctionAttentionGQAPattern::Match(core::builder::GraphGraph &graph,
                                                              const NodeProto &candidate) const {
  const std::string &op_type = candidate.op_type().value();
  if (!op_type.starts_with("LocalAttention") || op_type.starts_with("LocalAttentionGQA") ||
      NormaliseDomain(candidate.domain().value()) != kIntermediateDomain ||
      candidate.input_size() < 3 || candidate.output_size() != 1 || !MainOpsetAtLeast(graph, 18)) {
    return NoMatch(candidate, "FunctionAttentionGQAPattern expects a LocalAttention call.");
  }
  GqaBranch keys;
  GqaBranch values;
  if (!MatchGqaBranch(graph, candidate.input()[1].value(), true, keys) ||
      !MatchGqaBranch(graph, candidate.input()[2].value(), true, values) ||
      keys.unsqueeze_shape != values.unsqueeze_shape || keys.expand_shape != values.expand_shape ||
      keys.reshape_shape != values.reshape_shape ||
      IsDefaultOp(*keys.reshape, "Reshape") != IsDefaultOp(*values.reshape, "Reshape")) {
    return NoMatch(candidate, "Key and value repeat-interleave branches must match exactly.");
  }
  const std::vector<const NodeProto *> matched = {keys.unsqueeze,   keys.expand,   keys.reshape,
                                                  values.unsqueeze, values.expand, values.reshape,
                                                  &candidate};
  for (std::size_t index = 0; index + 1 < matched.size(); ++index) {
    const NodeProto *node = matched[index];
    if (graph.IsUsedMoreThanOnce(node->output()[0].value())) {
      return NoMatch(candidate, "A GQA repeat-interleave intermediate is shared.");
    }
  }
  if (!graph.HasType(keys.unsqueeze->input()[0].value())) {
    return NoMatch(candidate, "The GQA key type must be known.");
  }
  const auto type =
      core::symbolic::TensorTypeToDataType(graph.GetType(keys.unsqueeze->input()[0].value()));
  if (!IsSupportedAttentionType(type)) {
    return NoMatch(candidate, "The GQA local function requires floating inputs.");
  }
  const std::string suffix = op_type.substr(std::string("LocalAttention").size());
  const std::string function_name =
      std::string("LocalAttentionGQA") +
      (IsDefaultOp(*keys.reshape, "Reshape") ? std::string() : std::string("sQ")) + suffix;
  if (!CanEnsureFunction(graph.Builder(), function_name)) {
    return NoMatch(candidate, "The GQA local-function name is already reserved.");
  }
  return core::builder::MatchResult{this, matched, &candidate};
}

utils::RepeatedProtoField<NodeProto>
FunctionAttentionGQAPattern::Apply(core::builder::GraphGraph &graph,
                                   const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 7 || std::any_of(nodes.begin(), nodes.end(),
                                       [](const NodeProto *node) { return node == nullptr; })) {
    throw BuilderError(
        "FunctionAttentionGQAPattern::Apply expects two repeat-interleave branches.");
  }
  const NodeProto &key_unsqueeze = *nodes[0];
  const NodeProto &key_expand = *nodes[1];
  const NodeProto &key_reshape = *nodes[2];
  const NodeProto &value_unsqueeze = *nodes[3];
  const NodeProto &value_reshape = *nodes[5];
  const NodeProto &attention = *nodes[6];
  if (!graph.HasType(key_unsqueeze.input()[0].value())) {
    throw BuilderError("FunctionAttentionGQAPattern::Apply needs the GQA input type.");
  }
  const auto type =
      core::symbolic::TensorTypeToDataType(graph.GetType(key_unsqueeze.input()[0].value()));
  const std::string suffix =
      attention.op_type().value().substr(std::string("LocalAttention").size());
  const std::string function_name =
      std::string("LocalAttentionGQA") +
      (IsDefaultOp(key_reshape, "Reshape") ? std::string() : std::string("sQ")) + suffix;
  const std::vector<std::string> inputs = {
      attention.input()[0].value(),
      key_unsqueeze.input()[0].value(),
      value_unsqueeze.input()[0].value(),
      attention.input_size() > 3 ? attention.input()[3].value() : std::string(),
      attention.input_size() > 4 ? attention.input()[4].value() : std::string(),
      key_expand.input()[1].value(),
      key_reshape.input()[1].value()};
  utils::RepeatedProtoField<NodeProto> result;
  result.push_back(MakePatternNode(function_name.c_str(), inputs, {attention.output()[0].value()},
                                   kIntermediateDomain,
                                   "FunctionAttentionGQAPattern--" + attention.name().value()));
  EnsureFunction(graph.Builder(),
                 MakeAttentionFunction(graph.Builder(), function_name, type, true,
                                       attention.op_type().value().find("SW") != std::string::npos,
                                       IsDefaultOp(value_reshape, "Squeeze")));
  return result;
}

namespace {

bool IsLocalAttentionGqaCall(const NodeProto &node) {
  if (NormaliseDomain(node.domain().value()) != kIntermediateDomain || node.input_size() != 7) {
    return false;
  }
  const std::string &name = node.op_type().value();
  return name.starts_with("LocalAttentionGQASW_to") ||
         name.starts_with("LocalAttentionGQASWsQ_to") || name.starts_with("LocalAttentionGQA_to") ||
         name.starts_with("LocalAttentionGQAsQ_to") || name.starts_with("LocalAttentionGQAsQSW_to");
}

} // namespace

core::builder::MatchResult AttentionGQAPattern::Match(core::builder::GraphGraph &graph,
                                                      const NodeProto &candidate) const {
  if (!MainOpsetAtLeast(graph, 23) || candidate.output_size() != 1) {
    return NoMatch(candidate, "AttentionGQAPattern requires opset 23 and one output.");
  }
  const bool onnx_attention = IsNode(&candidate, "Attention", -1, 1);
  const bool local_attention = IsLocalAttentionGqaCall(candidate);
  if ((!onnx_attention && !local_attention) || candidate.input_size() < 3) {
    return NoMatch(candidate, "Candidate is neither ONNX Attention nor LocalAttentionGQA.");
  }
  if ((onnx_attention && (HasUnsupportedAttributes(candidate, {"scale", "is_causal"}) ||
                          HasActiveInputsFrom(candidate, 4))) ||
      (local_attention && candidate.attribute_size() != 0)) {
    return NoMatch(candidate, "Active optional inputs or unsupported Attention attributes exist.");
  }
  if (candidate.input_size() > 3 && !candidate.input()[3].value().empty() &&
      (!graph.HasShape(candidate.input()[3].value()) ||
       graph.GetShape(candidate.input()[3].value()).Shape().Rank() < 2)) {
    return NoMatch(candidate, "Attention mask rank must be at least two.");
  }

  GqaBranch keys;
  GqaBranch values;
  const NodeProto *concat_keys = nullptr;
  const NodeProto *concat_values = nullptr;
  std::vector<const NodeProto *> gqa_nodes(6, nullptr);
  if (onnx_attention) {
    if (!HasRank(graph, candidate.input()[0].value(), 4) ||
        !MatchGqaBranch(graph, candidate.input()[1].value(), true, keys) ||
        !MatchGqaBranch(graph, candidate.input()[2].value(), true, values) ||
        keys.unsqueeze_shape != values.unsqueeze_shape ||
        keys.expand_shape != values.expand_shape || keys.reshape_shape != values.reshape_shape ||
        IsDefaultOp(*keys.reshape, "Reshape") != IsDefaultOp(*values.reshape, "Reshape")) {
      return NoMatch(candidate, "ONNX Attention needs matching rank-four GQA branches.");
    }
    gqa_nodes = {keys.unsqueeze,   keys.expand,   keys.reshape,
                 values.unsqueeze, values.expand, values.reshape};
    concat_keys = graph.NodeBefore(keys.unsqueeze->input()[0].value());
    concat_values = graph.NodeBefore(values.unsqueeze->input()[0].value());
  } else {
    concat_keys = graph.NodeBefore(candidate.input()[1].value());
    concat_values = graph.NodeBefore(candidate.input()[2].value());
    double scale = 0.0;
    std::vector<int64_t> expand_shape;
    std::vector<int64_t> shape_or_axes;
    if (!ReadConstantScalar(graph, candidate.input()[4].value(), scale) ||
        !ReadConstantInts(graph, candidate.input()[5].value(), expand_shape) ||
        expand_shape.size() < 4 || expand_shape[0] != 1 || expand_shape[1] != 1 ||
        expand_shape[expand_shape.size() - 2] != 1 || expand_shape.back() != 1 ||
        !ReadConstantInts(graph, candidate.input()[6].value(), shape_or_axes)) {
      return NoMatch(candidate, "LocalAttentionGQA shape and scale inputs must be constant.");
    }
    if (candidate.op_type().value().find("sQ_to") != std::string::npos) {
      if (!graph.HasShape(candidate.input()[1].value()) || shape_or_axes.size() != 1) {
        return NoMatch(candidate, "Squeeze-based GQA needs a known key shape and one axis.");
      }
    } else if (shape_or_axes.size() < 2 || shape_or_axes[1] <= 0) {
      return NoMatch(candidate, "Reshape-based GQA needs a positive head dimension.");
    }
    if (candidate.op_type().value().find("SW") != std::string::npos) {
      if (candidate.input()[3].value().empty() || !graph.HasType(candidate.input()[3].value()) ||
          graph.GetType(candidate.input()[3].value()) != TensorType::kBool) {
        return NoMatch(candidate, "Switched masks are supported only for BOOL tensors.");
      }
    }
  }
  int64_t key_axis = 0;
  int64_t value_axis = 0;
  if (!IsNode(concat_keys, "Concat", 2, 1) || !IsNode(concat_values, "Concat", 2, 1) ||
      !graph.HasShape(concat_keys->input()[0].value()) ||
      !graph.HasShape(concat_keys->input()[1].value()) ||
      !graph.HasShape(concat_values->input()[0].value()) ||
      !graph.HasShape(concat_values->input()[1].value()) ||
      !NormalizedAxis(*concat_keys, 4, key_axis) ||
      !NormalizedAxis(*concat_values, 4, value_axis) || key_axis != 2 || value_axis != 2) {
    return NoMatch(candidate, "Key and value caches must concatenate on normalized axis 2.");
  }
  const SymShape &past_key_shape = graph.GetShape(concat_keys->input()[0].value()).Shape();
  const SymShape &key_shape = graph.GetShape(concat_keys->input()[1].value()).Shape();
  const SymShape &past_value_shape = graph.GetShape(concat_values->input()[0].value()).Shape();
  const SymShape &value_shape = graph.GetShape(concat_values->input()[1].value()).Shape();
  if (past_key_shape.Rank() != 4 || key_shape.Rank() != 4 || past_value_shape.Rank() != 4 ||
      value_shape.Rank() != 4 || !SameSelectedDimensions(past_key_shape, key_shape, {0, 1, 3}) ||
      !SameSelectedDimensions(past_value_shape, value_shape, {0, 1, 3}) ||
      past_key_shape != past_value_shape || key_shape != value_shape) {
    return NoMatch(candidate, "Key and value cache shapes are incompatible.");
  }
  for (const NodeProto *node : gqa_nodes) {
    if (node != nullptr && graph.IsUsedMoreThanOnce(node->output()[0].value())) {
      return NoMatch(candidate, "A removed GQA repeat-interleave node is shared.");
    }
  }
  const std::vector<const NodeProto *> matched = {concat_keys,  concat_values, gqa_nodes[0],
                                                  gqa_nodes[1], gqa_nodes[2],  gqa_nodes[3],
                                                  gqa_nodes[4], gqa_nodes[5],  &candidate};
  const std::unordered_set<const NodeProto *> removed(matched.begin(), matched.end());
  if (HasUnsafeMovedOutput(graph, concat_keys->output()[0].value(), candidate, removed) ||
      HasUnsafeMovedOutput(graph, concat_values->output()[0].value(), candidate, removed)) {
    return NoMatch(candidate,
                   "A cache output has an earlier external consumer or subgraph capture.");
  }
  return core::builder::MatchResult{this, matched, &candidate};
}

utils::RepeatedProtoField<NodeProto>
AttentionGQAPattern::Apply(core::builder::GraphGraph &graph,
                           const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 9 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[8] == nullptr) {
    throw BuilderError("AttentionGQAPattern::Apply expects cache Concat nodes and Attention.");
  }
  const NodeProto &key_concat = *nodes[0];
  const NodeProto &value_concat = *nodes[1];
  const NodeProto &attention = *nodes[8];
  if (attention.input_size() < 3 || attention.output_size() != 1) {
    throw BuilderError("AttentionGQAPattern::Apply received an invalid Attention node.");
  }
  const bool onnx_attention = IsDefaultOp(attention, "Attention");
  const std::string query = attention.input()[0].value();
  const std::string mask =
      attention.input_size() > 3 ? attention.input()[3].value() : std::string();
  std::string final_mask = mask;
  utils::RepeatedProtoField<NodeProto> result;
  if (!onnx_attention && attention.op_type().value().find("SW") != std::string::npos) {
    if (mask.empty() || !graph.HasType(mask) || graph.GetType(mask) != TensorType::kBool) {
      throw BuilderError("AttentionGQAPattern::Apply supports switched BOOL masks only.");
    }
    final_mask = graph.Builder().UniqueName("AttentionGQAPattern--" + mask);
    result.push_back(MakePatternNode("Not", {mask}, {final_mask}, "",
                                     "AttentionGQAPattern--" + attention.name().value()));
  }
  NodeProto replacement =
      MakePatternNode("Attention",
                      {query, key_concat.input()[1].value(), value_concat.input()[1].value(),
                       final_mask, key_concat.input()[0].value(), value_concat.input()[0].value()},
                      {attention.output()[0].value(), key_concat.output()[0].value(),
                       value_concat.output()[0].value()},
                      "", "AttentionGQAPattern--" + attention.name().value());
  if (onnx_attention) {
    const AttributeProto *scale = FindAttribute(attention, "scale");
    if (scale != nullptr) {
      AddFloatAttribute(replacement, "scale", scale->f());
    }
    AddIntAttribute(replacement, "is_causal", GetAttributeOr<int64_t>(attention, "is_causal", 0));
  } else {
    double scale = 0.0;
    if (!ReadConstantScalar(graph, attention.input()[4].value(), scale)) {
      throw BuilderError("AttentionGQAPattern::Apply could not read the local scale.");
    }
    AddFloatAttribute(replacement, "scale", static_cast<float>(scale * scale));
  }
  result.push_back(std::move(replacement));
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

std::set<std::string> RotaryConcatPartPattern::FastOpType() const { return {"Add"}; }

core::builder::MatchResult RotaryConcatPartPattern::Match(core::builder::GraphGraph &graph,
                                                          const NodeProto &candidate) const {
  if (!IsDefaultOp(candidate, "Add") || candidate.input_size() != 2 ||
      candidate.output_size() != 1) {
    return NoMatch(candidate, "RotaryConcatPartPattern expects a default-domain Add.");
  }
  const NodeProto *left = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *right = graph.NodeBefore(candidate.input()[1].value());
  if (left == nullptr || right == nullptr) {
    return NoMatch(candidate, "Both Add inputs must be produced by nodes.");
  }

  if (IsDefaultOp(*left, "Concat") || IsDefaultOp(*right, "Concat")) {
    const NodeProto *concat_left = left;
    const NodeProto *concat_right = right;
    if (!IsNode(concat_left, "Concat", 2, 1) || !IsNode(concat_right, "Concat", 2, 1) ||
        graph.IsUsedMoreThanOnce(concat_left->output()[0].value()) ||
        graph.IsUsedMoreThanOnce(concat_right->output()[0].value())) {
      return NoMatch(candidate, "The two Concat branches are incompatible or shared.");
    }

    std::vector<const NodeProto *> left_before = {
        graph.NodeBefore(concat_left->input()[0].value()),
        graph.NodeBefore(concat_left->input()[1].value())};
    std::vector<const NodeProto *> right_before = {
        graph.NodeBefore(concat_right->input()[0].value()),
        graph.NodeBefore(concat_right->input()[1].value())};
    if (std::find(left_before.begin(), left_before.end(), nullptr) != left_before.end() ||
        std::find(right_before.begin(), right_before.end(), nullptr) != right_before.end()) {
      return NoMatch(candidate, "Every Concat input must be produced by a node.");
    }
    for (const std::string &input :
         {concat_left->input()[0].value(), concat_left->input()[1].value(),
          concat_right->input()[0].value(), concat_right->input()[1].value()}) {
      if (!graph.HasShape(input)) {
        return NoMatch(candidate, "Every Concat input needs an inferred shape.");
      }
    }
    const std::size_t concat_rank = graph.GetShape(concat_left->input()[0].value()).Shape().Rank();
    int64_t concat_axis_left = 0;
    int64_t concat_axis_right = 0;
    if (concat_rank == 0 ||
        !NormalizedAxis(*concat_left, static_cast<int64_t>(concat_rank), concat_axis_left) ||
        !NormalizedAxis(*concat_right, static_cast<int64_t>(concat_rank), concat_axis_right) ||
        concat_axis_left != concat_axis_right) {
      return NoMatch(candidate, "The two Concat axes must normalize to the same dimension.");
    }

    const auto constant_of_shape = [](const std::vector<const NodeProto *> &nodes) {
      return std::find_if(nodes.begin(), nodes.end(), [](const NodeProto *node) {
        return IsNode(node, "ConstantOfShape", 1, 1);
      });
    };
    const auto left_constant = constant_of_shape(left_before);
    const auto right_constant = constant_of_shape(right_before);
    if (left_constant == left_before.end() || right_constant == right_before.end() ||
        std::distance(left_before.cbegin(), left_constant) ==
            std::distance(right_before.cbegin(), right_constant)) {
      return NoMatch(candidate, "Each Concat needs a ConstantOfShape on opposite input positions.");
    }
    const NodeProto *cst_left = *left_constant;
    const NodeProto *cst_right = *right_constant;
    if (!IsZeroConstantOfShape(*cst_left) || !IsZeroConstantOfShape(*cst_right)) {
      return NoMatch(candidate, "ConstantOfShape padding must be exactly zero.");
    }
    const std::string left_value = concat_left->input()[0].value() == cst_left->output()[0].value()
                                       ? concat_left->input()[1].value()
                                       : concat_left->input()[0].value();
    const std::string right_value =
        concat_right->input()[0].value() == cst_right->output()[0].value()
            ? concat_right->input()[1].value()
            : concat_right->input()[0].value();

    const NodeProto *neg_left = nullptr;
    const NodeProto *neg_right = nullptr;
    const NodeProto *slice_left = nullptr;
    const NodeProto *slice_right = nullptr;
    const NodeProto *split_left = nullptr;
    const NodeProto *split_right = nullptr;
    const auto find_type = [](const std::vector<const NodeProto *> &nodes, const char *op_type) {
      return std::find_if(nodes.begin(), nodes.end(),
                          [&](const NodeProto *node) { return IsDefaultOp(*node, op_type); });
    };
    const auto right_neg = find_type(right_before, "Neg");
    const auto left_neg = find_type(left_before, "Neg");
    if (right_neg != right_before.end()) {
      neg_right = *right_neg;
      const NodeProto *left_data = left_before[0] == cst_left ? left_before[1] : left_before[0];
      if (IsDefaultOp(*left_data, "Slice")) {
        slice_left = left_data;
      } else if (IsDefaultOp(*left_data, "Split")) {
        split_left = left_data;
      } else {
        return NoMatch(candidate, "The positive branch must come from Slice or Split.");
      }
      const NodeProto *right_data = graph.NodeBefore(neg_right->input()[0].value());
      if (IsNode(right_data, "Slice")) {
        slice_right = right_data;
      } else if (IsNode(right_data, "Split")) {
        split_right = right_data;
      } else {
        return NoMatch(candidate, "Neg must consume a Slice or Split output.");
      }
    } else if (left_neg != left_before.end()) {
      neg_left = *left_neg;
      const NodeProto *left_data = graph.NodeBefore(neg_left->input()[0].value());
      if (IsNode(left_data, "Slice")) {
        slice_left = left_data;
      } else if (IsNode(left_data, "Split")) {
        split_left = left_data;
      } else {
        return NoMatch(candidate, "Neg must consume a Slice or Split output.");
      }
      const NodeProto *right_data =
          right_before[0] == cst_right ? right_before[1] : right_before[0];
      if (IsDefaultOp(*right_data, "Slice")) {
        slice_right = right_data;
      } else if (IsDefaultOp(*right_data, "Split")) {
        split_right = right_data;
      } else {
        return NoMatch(candidate, "The positive branch must come from Slice or Split.");
      }
    } else {
      return NoMatch(candidate, "Exactly one Concat branch must contain Neg.");
    }

    const bool slices = slice_left != nullptr && slice_right != nullptr;
    const bool splits = split_left != nullptr && split_right != nullptr;
    if (slices == splits) {
      return NoMatch(candidate, "Both branches must use the same Slice or Split form.");
    }

    int64_t axis = concat_axis_left;
    if (slices) {
      if (!IsNode(slice_left, "Slice", 4, 1) || !IsNode(slice_right, "Slice", 4, 1) ||
          slice_left->input()[0].value() != slice_right->input()[0].value()) {
        return NoMatch(candidate, "The Slice branches must read the same tensor.");
      }
      std::vector<int64_t> left_starts;
      std::vector<int64_t> left_ends;
      std::vector<int64_t> left_axes;
      std::vector<int64_t> right_starts;
      std::vector<int64_t> right_ends;
      std::vector<int64_t> right_axes;
      std::vector<int64_t> left_shape;
      std::vector<int64_t> right_shape;
      if (!ReadConstantInts(graph, slice_left->input()[1].value(), left_starts) ||
          !ReadConstantInts(graph, slice_left->input()[2].value(), left_ends) ||
          !ReadConstantInts(graph, slice_left->input()[3].value(), left_axes) ||
          !ReadConstantInts(graph, slice_right->input()[1].value(), right_starts) ||
          !ReadConstantInts(graph, slice_right->input()[2].value(), right_ends) ||
          !ReadConstantInts(graph, slice_right->input()[3].value(), right_axes) ||
          left_starts.size() != 1 || left_ends.size() != 1 || left_axes.size() != 1 ||
          right_starts.size() != 1 || right_ends.size() != 1 || right_axes.size() != 1 ||
          !ReadConstantInts(graph, cst_left->input()[0].value(), left_shape) ||
          !ReadConstantInts(graph, cst_right->input()[0].value(), right_shape)) {
        return NoMatch(candidate, "Slice bounds and ConstantOfShape shapes must be constant.");
      }
      const SymShape &source_shape = graph.GetShape(slice_left->input()[0].value()).Shape();
      int64_t left_axis = left_axes[0];
      int64_t right_axis = right_axes[0];
      if (left_axis < 0) {
        left_axis += static_cast<int64_t>(source_shape.Rank());
      }
      if (right_axis < 0) {
        right_axis += static_cast<int64_t>(source_shape.Rank());
      }
      if (left_axis != right_axis || left_axis != axis || left_axis < 0 ||
          static_cast<std::size_t>(left_axis) >= source_shape.Rank() ||
          !source_shape[static_cast<std::size_t>(left_axis)].IsInt() ||
          static_cast<std::size_t>(left_axis) >= left_shape.size() ||
          static_cast<std::size_t>(left_axis) >= right_shape.size() || left_starts[0] != 0 ||
          left_ends[0] != right_starts[0] ||
          right_ends[0] != source_shape[static_cast<std::size_t>(left_axis)].AsInt() ||
          left_ends[0] <= left_starts[0] || right_ends[0] <= right_starts[0] ||
          left_ends[0] - left_starts[0] != right_shape[left_axis] ||
          right_ends[0] - right_starts[0] != left_shape[left_axis]) {
        return NoMatch(
            candidate,
            "Slices must be ordered, contiguous, cover the axis, and complement zero tensors.");
      }
    } else {
      const SymShape &split_shape = graph.GetShape(split_left->input()[0].value()).Shape();
      int64_t split_axis = 0;
      const bool canonical_wiring =
          neg_right != nullptr
              ? left_value == split_left->output()[0].value() &&
                    neg_right->input()[0].value() == split_left->output()[1].value()
              : neg_left->input()[0].value() == split_left->output()[0].value() &&
                    right_value == split_left->output()[1].value();
      if (split_left != split_right || !IsNode(split_left, "Split", -1, 2) ||
          !NormalizedAxis(*split_left, static_cast<int64_t>(split_shape.Rank()), split_axis) ||
          split_axis != axis || !canonical_wiring) {
        return NoMatch(candidate, "Both branches must consume the same compatible Split.");
      }
    }

    std::set<int64_t> concrete_dims;
    std::set<std::string> symbolic_dims;
    for (const std::string &input :
         {concat_left->input()[0].value(), concat_left->input()[1].value(),
          concat_right->input()[0].value(), concat_right->input()[1].value()}) {
      const SymShape &shape = graph.GetShape(input).Shape();
      int64_t normalized = axis < 0 ? axis + static_cast<int64_t>(shape.Rank()) : axis;
      if (normalized < 0 || static_cast<std::size_t>(normalized) >= shape.Rank()) {
        return NoMatch(candidate, "Concat axis is invalid for an input shape.");
      }
      const SymDim &dimension = shape[static_cast<std::size_t>(normalized)];
      if (dimension.IsInt()) {
        concrete_dims.insert(dimension.AsInt());
      } else {
        symbolic_dims.insert(dimension.AsExpr());
      }
    }
    if (concrete_dims.size() > 1 || symbolic_dims.size() > 2) {
      return NoMatch(candidate, "Concat dimensions do not prove the rotary cancellation.");
    }

    return core::builder::MatchResult{this,
                                      {cst_left, split_left, slice_left, neg_left, concat_left,
                                       cst_right, slice_right, neg_right, concat_right, &candidate},
                                      nullptr};
  }

  if (!IsNode(left, "Transpose", 1, 1) || !IsNode(right, "Transpose", 1, 1) ||
      graph.IsUsedMoreThanOnce(left->output()[0].value()) ||
      graph.IsUsedMoreThanOnce(right->output()[0].value())) {
    return NoMatch(candidate, "The Add inputs are neither compatible Concat nor Transpose nodes.");
  }
  std::vector<int64_t> perm_left;
  std::vector<int64_t> perm_right;
  if (!ReadAttributeInts(*left, "perm", perm_left) ||
      !ReadAttributeInts(*right, "perm", perm_right) || perm_left != perm_right) {
    return NoMatch(candidate, "The output Transpose permutations must match.");
  }
  const NodeProto *scatter_left = graph.NodeBefore(left->input()[0].value());
  const NodeProto *scatter_right = graph.NodeBefore(right->input()[0].value());
  if (!IsNode(scatter_left, "ScatterND", 3, 1) || !IsNode(scatter_right, "ScatterND", 3, 1)) {
    return NoMatch(candidate, "Both Transpose nodes must follow ScatterND.");
  }
  const NodeProto *tr_data_left = graph.NodeBefore(scatter_left->input()[0].value());
  const NodeProto *tr_data_right = graph.NodeBefore(scatter_right->input()[0].value());
  const NodeProto *tr_update_left = graph.NodeBefore(scatter_left->input()[2].value());
  const NodeProto *tr_update_right = graph.NodeBefore(scatter_right->input()[2].value());
  if (!IsNode(tr_data_left, "Transpose", 1, 1) || !IsNode(tr_data_right, "Transpose", 1, 1) ||
      !IsNode(tr_update_left, "Transpose", 1, 1) || !IsNode(tr_update_right, "Transpose", 1, 1) ||
      !HasPerm(*tr_data_left, perm_left) || !HasPerm(*tr_data_right, perm_left) ||
      !HasPerm(*tr_update_left, perm_left) || !HasPerm(*tr_update_right, perm_left)) {
    return NoMatch(candidate, "ScatterND data and updates need matching Transpose nodes.");
  }
  const NodeProto *cst_left = graph.NodeBefore(tr_data_left->input()[0].value());
  const NodeProto *cst_right = graph.NodeBefore(tr_data_right->input()[0].value());
  if (cst_left == nullptr || cst_right == nullptr || !IsZeroConstantOfShape(*cst_left) ||
      !IsZeroConstantOfShape(*cst_right)) {
    return NoMatch(candidate, "ScatterND data must originate in zero ConstantOfShape.");
  }
  const NodeProto *part_left = graph.NodeBefore(tr_update_left->input()[0].value());
  const NodeProto *part_right = graph.NodeBefore(tr_update_right->input()[0].value());
  if (part_left == nullptr || part_right == nullptr) {
    return NoMatch(candidate, "ScatterND updates need Slice, Split, or Neg producers.");
  }
  const NodeProto *neg_left = nullptr;
  const NodeProto *neg_right = nullptr;
  if (IsDefaultOp(*part_left, "Neg")) {
    neg_left = part_left;
    part_left = graph.NodeBefore(neg_left->input()[0].value());
  } else if (IsDefaultOp(*part_right, "Neg")) {
    neg_right = part_right;
    part_right = graph.NodeBefore(neg_right->input()[0].value());
  } else {
    return NoMatch(candidate, "Exactly one ScatterND update must be negated.");
  }
  if (part_left == nullptr || part_right == nullptr) {
    return NoMatch(candidate, "Neg must consume Slice or Split.");
  }
  const bool split_form = IsDefaultOp(*part_left, "Split") && part_left == part_right;
  if (!split_form && (!IsNode(part_left, "Slice", 5, 1) || !IsNode(part_right, "Slice", 5, 1) ||
                      part_left->input()[0].value() != part_right->input()[0].value() ||
                      !graph.HasShape(part_left->input()[0].value()))) {
    return NoMatch(candidate, "ScatterND updates must share a Split or source tensor.");
  }
  if (!graph.HasShape(scatter_left->input()[0].value()) ||
      !graph.HasShape(scatter_right->input()[0].value()) ||
      graph.GetShape(scatter_left->input()[0].value()).Shape() !=
          graph.GetShape(scatter_right->input()[0].value()).Shape()) {
    return NoMatch(candidate, "ScatterND data shapes must match.");
  }
  std::vector<int64_t> indices_left;
  std::vector<int64_t> indices_right;
  const TensorProto *indices_left_tensor =
      graph.GetComputedConstant(scatter_left->input()[1].value());
  const TensorProto *indices_right_tensor =
      graph.GetComputedConstant(scatter_right->input()[1].value());
  const SymShape &scatter_shape = graph.GetShape(scatter_left->input()[0].value()).Shape();
  if (indices_left_tensor == nullptr || indices_right_tensor == nullptr ||
      indices_left_tensor->dims_size() != 2 || indices_right_tensor->dims_size() != 2 ||
      indices_left_tensor->dims()[1] != 1 || indices_right_tensor->dims()[1] != 1 ||
      !ReadIntegerValues(*indices_left_tensor, indices_left) ||
      !ReadIntegerValues(*indices_right_tensor, indices_right) || scatter_shape.Empty() ||
      !scatter_shape[0].IsInt()) {
    return NoMatch(candidate, "ScatterND indices must be constant column vectors.");
  }
  std::vector<int64_t> indices =
      !indices_left.empty() && indices_left[0] == 0 ? indices_left : indices_right;
  const std::vector<int64_t> &tail =
      !indices_left.empty() && indices_left[0] == 0 ? indices_right : indices_left;
  indices.insert(indices.end(), tail.begin(), tail.end());
  std::vector<int64_t> expected(static_cast<std::size_t>(scatter_shape[0].AsInt()));
  for (std::size_t index = 0; index < expected.size(); ++index) {
    expected[index] = static_cast<int64_t>(index);
  }
  if (indices != expected) {
    return NoMatch(candidate, "ScatterND indices must cover the leading dimension in order.");
  }

  if (split_form) {
    std::vector<int64_t> split;
    const bool canonical_wiring =
        neg_left != nullptr
            ? neg_left->input()[0].value() == part_left->output()[0].value() &&
                  tr_update_right->input()[0].value() == part_left->output()[1].value()
            : tr_update_left->input()[0].value() == part_left->output()[0].value() &&
                  neg_right->input()[0].value() == part_left->output()[1].value();
    if (part_left->input_size() != 2 ||
        !ReadConstantInts(graph, part_left->input()[1].value(), split) || split.empty() ||
        !std::all_of(split.begin(), split.end(),
                     [&](int64_t value) { return value == split.front(); }) ||
        !canonical_wiring) {
      return NoMatch(candidate, "Split parts must be equal.");
    }
  } else {
    std::vector<int64_t> left_starts;
    std::vector<int64_t> left_ends;
    std::vector<int64_t> left_axes;
    std::vector<int64_t> left_steps;
    std::vector<int64_t> right_starts;
    std::vector<int64_t> right_ends;
    std::vector<int64_t> right_axes;
    std::vector<int64_t> right_steps;
    std::vector<int64_t> left_shape;
    std::vector<int64_t> right_shape;
    if (!ReadConstantInts(graph, part_left->input()[1].value(), left_starts) ||
        !ReadConstantInts(graph, part_left->input()[2].value(), left_ends) ||
        !ReadConstantInts(graph, part_left->input()[3].value(), left_axes) ||
        !ReadConstantInts(graph, part_left->input()[4].value(), left_steps) ||
        !ReadConstantInts(graph, part_right->input()[1].value(), right_starts) ||
        !ReadConstantInts(graph, part_right->input()[2].value(), right_ends) ||
        !ReadConstantInts(graph, part_right->input()[3].value(), right_axes) ||
        !ReadConstantInts(graph, part_right->input()[4].value(), right_steps) ||
        left_starts.size() != 1 || left_ends.size() != 1 || left_axes.size() != 1 ||
        left_steps != std::vector<int64_t>{1} || right_starts.size() != 1 ||
        right_ends.size() != 1 || right_axes.size() != 1 ||
        right_steps != std::vector<int64_t>{1} ||
        !ReadConstantInts(graph, cst_left->input()[0].value(), left_shape) ||
        !ReadConstantInts(graph, cst_right->input()[0].value(), right_shape)) {
      return NoMatch(candidate, "Slice bounds, steps, and zero shapes must be constant.");
    }
    const SymShape &source_shape = graph.GetShape(part_left->input()[0].value()).Shape();
    int64_t left_axis = left_axes[0];
    int64_t right_axis = right_axes[0];
    if (left_axis < 0) {
      left_axis += static_cast<int64_t>(source_shape.Rank());
    }
    if (right_axis < 0) {
      right_axis += static_cast<int64_t>(source_shape.Rank());
    }
    if (left_axis != right_axis || left_axis < 0 ||
        static_cast<std::size_t>(left_axis) >= source_shape.Rank() ||
        !source_shape[static_cast<std::size_t>(left_axis)].IsInt() ||
        static_cast<std::size_t>(left_axis) >= left_shape.size() ||
        static_cast<std::size_t>(left_axis) >= right_shape.size() || left_starts[0] != 0 ||
        left_ends[0] != right_starts[0] ||
        right_ends[0] != source_shape[static_cast<std::size_t>(left_axis)].AsInt() ||
        left_ends[0] <= left_starts[0] || right_ends[0] <= right_starts[0] ||
        left_shape[left_axis] != right_shape[left_axis] ||
        left_ends[0] - left_starts[0] + right_ends[0] - right_starts[0] != left_shape[left_axis]) {
      return NoMatch(candidate, "The two slices must be ordered and cover the zero dimension.");
    }
  }

  const std::vector<const NodeProto *> matched = {
      cst_left,       part_left,       neg_left,  cst_right, part_right,   neg_right,
      scatter_left,   scatter_right,   left,      right,     tr_data_left, tr_data_right,
      tr_update_left, tr_update_right, &candidate};
  const std::unordered_set<const NodeProto *> matched_set(matched.begin(), matched.end());
  const std::unordered_set<std::string> replaced{candidate.output()[0].value()};
  for (const NodeProto *node : matched) {
    if (node != nullptr && node != cst_left && node != cst_right && node != part_left &&
        NodeOutputHasExternalUse(graph, *node, matched_set, replaced)) {
      return NoMatch(candidate, "A removed ScatterND branch node has an external use.");
    }
  }
  return core::builder::MatchResult{this, matched, nullptr};
}

utils::RepeatedProtoField<NodeProto>
RotaryConcatPartPattern::Apply(core::builder::GraphGraph &graph,
                               const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 10 && nodes.size() != 15) {
    throw BuilderError("RotaryConcatPartPattern::Apply expects a concat or transpose match.");
  }
  utils::RepeatedProtoField<NodeProto> result;
  if (nodes.size() == 10) {
    const NodeProto *cst_left = nodes[0];
    const NodeProto *split = nodes[1];
    const NodeProto *slice_left = nodes[2];
    const NodeProto *neg_left = nodes[3];
    const NodeProto *concat_left = nodes[4];
    const NodeProto *cst_right = nodes[5];
    const NodeProto *slice_right = nodes[6];
    const NodeProto *neg_right = nodes[7];
    const NodeProto *concat_right = nodes[8];
    const NodeProto *add = nodes[9];
    if (cst_left == nullptr || concat_left == nullptr || cst_right == nullptr ||
        concat_right == nullptr || add == nullptr ||
        (split == nullptr) == (slice_left == nullptr) ||
        (neg_left == nullptr) == (neg_right == nullptr)) {
      throw BuilderError("RotaryConcatPartPattern::Apply received an inconsistent concat match.");
    }
    const bool split_form = split != nullptr;
    int64_t axis = split_form ? Axis(*split, 0) : 0;
    if (!split_form) {
      std::vector<int64_t> axes;
      if (!ReadConstantInts(graph, slice_left->input()[3].value(), axes) || axes.size() != 1) {
        throw BuilderError("RotaryConcatPartPattern::Apply could not read the Slice axis.");
      }
      axis = axes[0];
    }
    const NodeProto *neg = neg_left == nullptr ? neg_right : neg_left;
    std::vector<std::string> concat_inputs;
    if (neg_left == nullptr) {
      const bool neg_first = concat_right->input()[0].value() == neg_right->output()[0].value();
      const std::string positive =
          split_form ? split->output()[0].value() : slice_left->output()[0].value();
      concat_inputs = neg_first ? std::vector<std::string>{neg->output()[0].value(), positive}
                                : std::vector<std::string>{positive, neg->output()[0].value()};
    } else {
      const bool neg_first = concat_left->input()[0].value() == neg_left->output()[0].value();
      const std::string positive =
          split_form ? split->output()[1].value() : slice_right->output()[0].value();
      concat_inputs = neg_first ? std::vector<std::string>{neg->output()[0].value(), positive}
                                : std::vector<std::string>{positive, neg->output()[0].value()};
    }
    std::vector<const NodeProto *> originals =
        split_form ? std::vector<const NodeProto *>{cst_left, split, neg}
                   : std::vector<const NodeProto *>{cst_left, slice_left, slice_right, neg};
    const auto preserved = PreservedNodes(graph, nodes, {add->output()[0].value()});
    originals.insert(originals.end(), preserved.begin(), preserved.end());
    std::unordered_set<const NodeProto *> copied;
    for (const NodeProto *original : originals) {
      if (original != nullptr && copied.insert(original).second) {
        result.push_back(*original);
      }
    }
    NodeProto concat = MakePatternNode("Concat", concat_inputs, {add->output()[0].value()}, "",
                                       "RotaryConcatPartPattern--" + add->name().value());
    AddIntAttribute(concat, "axis", axis);
    concat.set_doc_string(add->doc_string().value());
    result.push_back(std::move(concat));
    return result;
  }

  return ApplyRotaryTranspose(graph, nodes);
}

std::set<std::string> FunctionHalfRotaryEmbeddingPattern::FastOpType() const { return {"Split"}; }

core::builder::MatchResult
FunctionHalfRotaryEmbeddingPattern::Match(core::builder::GraphGraph &graph,
                                          const NodeProto &candidate) const {
  if (!MainOpsetAtLeast(graph, 18) || !IsNode(&candidate, "Split", -1, 2) ||
      !HasRank(graph, candidate.input()[0].value(), 4)) {
    return NoMatch(candidate,
                   "FunctionHalfRotaryEmbeddingPattern expects an opset-18 rank-four Split.");
  }
  int64_t axis = 0;
  if (!NormalizedAxis(candidate, 4, axis) || axis != 3) {
    return NoMatch(candidate, "Split must act on the last dimension.");
  }
  if (candidate.input_size() == 2) {
    std::vector<int64_t> split;
    if (!ReadInt64Constant(graph, candidate.input()[1].value(), split) || split.size() != 2 ||
        split[0] != split[1]) {
      return NoMatch(candidate, "Split sizes must be an equal INT64 pair.");
    }
  } else if (candidate.input_size() != 1 ||
             GetAttributeOr<int64_t>(candidate, "num_outputs", 0) != 2) {
    return NoMatch(candidate, "Split must use equal sizes or num_outputs=2.");
  }

  const auto &neg_consumers = graph.NextNodes(candidate.output()[1].value());
  if (neg_consumers.size() != 1 || !IsNode(neg_consumers[0], "Neg", 1, 1)) {
    return NoMatch(candidate, "The second Split output must feed one Neg.");
  }
  const NodeProto *neg = neg_consumers[0];
  const auto &concat_consumers = graph.NextNodes(neg->output()[0].value());
  if (concat_consumers.size() != 1 || !IsNode(concat_consumers[0], "Concat", 2, 1)) {
    return NoMatch(candidate, "Neg must feed one two-input Concat.");
  }
  const NodeProto *concat = concat_consumers[0];
  int64_t concat_axis = 0;
  if (concat->input()[0].value() != neg->output()[0].value() ||
      concat->input()[1].value() != candidate.output()[0].value() ||
      !NormalizedAxis(*concat, 4, concat_axis) || concat_axis != 3) {
    return NoMatch(candidate,
                   "Concat must canonically rebuild [Neg(second), first] on the last axis.");
  }
  const auto &mul1_consumers = graph.NextNodes(concat->output()[0].value());
  if (mul1_consumers.size() != 1 || !IsNode(mul1_consumers[0], "Mul", 2, 1)) {
    return NoMatch(candidate, "The rotated half must feed one Mul.");
  }
  const NodeProto *mul1 = mul1_consumers[0];
  const auto &input_consumers = graph.NextNodes(candidate.input()[0].value());
  if (input_consumers.size() != 2) {
    return NoMatch(candidate, "The rotary input must feed only Split and one Mul.");
  }
  const NodeProto *mul2 =
      input_consumers[0] == &candidate ? input_consumers[1] : input_consumers[0];
  if (!IsNode(mul2, "Mul", 2, 1) || std::find(input_consumers.begin(), input_consumers.end(),
                                              &candidate) == input_consumers.end()) {
    return NoMatch(candidate, "The unsplit input branch must be a Mul.");
  }
  const auto &add1 = graph.NextNodes(mul1->output()[0].value());
  const auto &add2 = graph.NextNodes(mul2->output()[0].value());
  if (add1.size() != 1 || add2.size() != 1 || add1[0] != add2[0] || !IsNode(add1[0], "Add", 2, 1)) {
    return NoMatch(candidate, "Both Mul branches must feed the same Add.");
  }
  const NodeProto *add = add1[0];
  std::set<std::string> caches;
  for (const NodeProto *mul : {mul1, mul2}) {
    for (int index = 0; index < mul->input_size(); ++index) {
      const std::string &input = mul->input()[index].value();
      if (input != candidate.input()[0].value() && input != concat->output()[0].value()) {
        caches.insert(input);
      }
    }
  }
  if (caches.size() != 2 || !CanEnsureFunction(graph.Builder(), "HalfRotaryEmbedding")) {
    return NoMatch(candidate, "The cache inputs or local-function name are not usable.");
  }
  const std::vector<const NodeProto *> matched = {&candidate, neg, concat, mul1, mul2, add};
  const std::unordered_set<const NodeProto *> matched_set(matched.begin(), matched.end());
  const std::unordered_set<std::string> replaced{add->output()[0].value()};
  for (const NodeProto *node : matched) {
    if (node != add && NodeOutputHasExternalUse(graph, *node, matched_set, replaced)) {
      return NoMatch(candidate, "A half-rotary intermediate has an external use.");
    }
  }
  return core::builder::MatchResult{this, matched, add};
}

utils::RepeatedProtoField<NodeProto>
FunctionHalfRotaryEmbeddingPattern::Apply(core::builder::GraphGraph &graph,
                                          const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 6 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[2] == nullptr ||
      nodes[3] == nullptr || nodes[4] == nullptr || nodes[5] == nullptr) {
    throw BuilderError(
        "FunctionHalfRotaryEmbeddingPattern::Apply expects the six-node decomposition.");
  }
  const NodeProto &split = *nodes[0];
  const NodeProto &concat = *nodes[2];
  const NodeProto &mul1 = *nodes[3];
  const NodeProto &mul2 = *nodes[4];
  const NodeProto &add = *nodes[5];
  const auto other_input = [](const NodeProto &mul, const std::string &known) -> std::string {
    if (mul.input()[0].value() == known) {
      return mul.input()[1].value();
    }
    if (mul.input()[1].value() == known) {
      return mul.input()[0].value();
    }
    return {};
  };
  const std::string sin_cache = other_input(mul1, concat.output()[0].value());
  const std::string cos_cache = other_input(mul2, split.input()[0].value());
  if (sin_cache.empty() || cos_cache.empty()) {
    throw BuilderError("FunctionHalfRotaryEmbeddingPattern::Apply could not identify both caches.");
  }
  utils::RepeatedProtoField<NodeProto> result;
  result.push_back(MakePatternNode("HalfRotaryEmbedding",
                                   {split.input()[0].value(), cos_cache, sin_cache},
                                   {add.output()[0].value()}, kIntermediateDomain,
                                   "FunctionHalfRotaryEmbeddingPattern--" + split.name().value()));
  EnsureFunction(graph.Builder(), MakeHalfRotaryFunction(graph.Builder()));
  return result;
}

std::set<std::string> RotaryEmbeddingPattern::FastOpType() const { return {"HalfRotaryEmbedding"}; }

core::builder::MatchResult RotaryEmbeddingPattern::Match(core::builder::GraphGraph &graph,
                                                         const NodeProto &candidate) const {
  if (!MainOpsetAtLeast(graph, 23) ||
      !IsDomainNode(&candidate, "HalfRotaryEmbedding", kIntermediateDomain, 3, 1) ||
      !HasRank(graph, candidate.input()[0].value(), 4) ||
      !graph.HasShape(candidate.input()[1].value()) ||
      !graph.HasShape(candidate.input()[2].value())) {
    return NoMatch(candidate,
                   "RotaryEmbeddingPattern expects a rank-four HalfRotaryEmbedding at opset 23.");
  }
  const SymShape &cos_shape = graph.GetShape(candidate.input()[1].value()).Shape();
  const SymShape &sin_shape = graph.GetShape(candidate.input()[2].value()).Shape();
  if (cos_shape != sin_shape || cos_shape.Rank() != 4 || !cos_shape[1].IsInt() ||
      cos_shape[1].AsInt() != 1) {
    return NoMatch(candidate, "Cosine and sine caches need equal [*,1,*,*] shapes.");
  }
  const NodeProto *concat_cos = graph.NodeBefore(candidate.input()[1].value());
  const NodeProto *concat_sin = graph.NodeBefore(candidate.input()[2].value());
  int64_t concat_cos_axis = 0;
  int64_t concat_sin_axis = 0;
  if (!IsNode(concat_cos, "Concat", 2, 1) || !IsNode(concat_sin, "Concat", 2, 1) ||
      concat_cos->input()[0].value() != concat_cos->input()[1].value() ||
      concat_sin->input()[0].value() != concat_sin->input()[1].value() ||
      !NormalizedAxis(*concat_cos, 4, concat_cos_axis) || concat_cos_axis != 3 ||
      !NormalizedAxis(*concat_sin, 4, concat_sin_axis) || concat_sin_axis != 3 ||
      graph.NextNodes(concat_cos->output()[0].value()) !=
          std::vector<const NodeProto *>{&candidate} ||
      graph.NextNodes(concat_sin->output()[0].value()) !=
          std::vector<const NodeProto *>{&candidate} ||
      graph.IsOutput(concat_cos->output()[0].value()) ||
      graph.IsOutput(concat_sin->output()[0].value()) ||
      graph.IsUsedBySubgraph(concat_cos->output()[0].value()) ||
      graph.IsUsedBySubgraph(concat_sin->output()[0].value())) {
    return NoMatch(candidate, "Both caches must be doubled by last-axis Concat.");
  }
  const SymShape &half_shape = graph.GetShape(candidate.input()[0].value()).Shape();
  if (half_shape[2] != cos_shape[2] || half_shape[3] != cos_shape[3] ||
      !graph.HasType(candidate.input()[0].value()) ||
      !graph.HasType(candidate.input()[1].value()) ||
      !graph.HasType(candidate.input()[2].value()) ||
      graph.GetType(candidate.input()[0].value()) != graph.GetType(candidate.input()[1].value()) ||
      graph.GetType(candidate.input()[0].value()) != graph.GetType(candidate.input()[2].value())) {
    return NoMatch(candidate, "Rotary input and caches need compatible sequence, width, and type.");
  }
  if (graph.IsUsedMoreThanOnce(candidate.input()[0].value())) {
    return NoMatch(candidate, "The HalfRotaryEmbedding input must not be shared.");
  }
  const NodeProto *split = graph.NodeBefore(candidate.input()[0].value());
  if (split != nullptr && !IsDefaultOp(*split, "Split")) {
    split = nullptr;
  }
  const NodeProto *concat = nullptr;
  if (split != nullptr) {
    std::vector<int64_t> split_sizes;
    int64_t split_axis = 0;
    if (!IsNode(split, "Split", 2, 2) ||
        !ReadConstantInts(graph, split->input()[1].value(), split_sizes) ||
        split_sizes.size() != 2 || split_sizes[0] <= 0 || split_sizes[1] < 0 ||
        split_sizes[0] % 2 != 0 || candidate.input()[0].value() != split->output()[0].value() ||
        !graph.HasShape(split->input()[0].value()) ||
        !NormalizedAxis(
            *split, static_cast<int64_t>(graph.GetShape(split->input()[0].value()).Shape().Rank()),
            split_axis) ||
        split_axis != 3 || !half_shape[3].IsInt() || half_shape[3].AsInt() != split_sizes[0]) {
      return NoMatch(candidate, "The partial-rotation Split needs two constant sizes.");
    }
    const auto &consumers = graph.NextNodes(candidate.output()[0].value());
    int64_t output_concat_axis = 0;
    if (consumers.size() != 1 || !IsNode(consumers[0], "Concat", 2, 1) ||
        consumers[0]->input()[0].value() != candidate.output()[0].value() ||
        consumers[0]->input()[1].value() != split->output()[1].value() ||
        !NormalizedAxis(*consumers[0], 4, output_concat_axis) || output_concat_axis != 3) {
      return NoMatch(candidate, "Partial rotation must concatenate the untouched Split output.");
    }
    const SymShape &main_shape = graph.GetShape(split->input()[0].value()).Shape();
    if (main_shape[3].IsInt() && main_shape[3].AsInt() != split_sizes[0] + split_sizes[1]) {
      return NoMatch(candidate, "Partial Split sizes must cover the full last dimension.");
    }
    concat = consumers[0];
  }
  const std::string &main_input =
      split == nullptr ? candidate.input()[0].value() : split->input()[0].value();
  int64_t num_heads = 0;
  if (!StaticDimension(graph, main_input, 1, num_heads)) {
    return NoMatch(candidate, "The number of attention heads must be static.");
  }
  const std::vector<const NodeProto *> matched = {concat_cos, concat_sin, split, &candidate,
                                                  concat};
  const std::unordered_set<const NodeProto *> matched_set(matched.begin(), matched.end());
  const std::string final_output =
      concat == nullptr ? candidate.output()[0].value() : concat->output()[0].value();
  const std::unordered_set<std::string> replaced{final_output};
  for (const NodeProto *node : matched) {
    if (node != nullptr && NodeOutputHasExternalUse(graph, *node, matched_set, replaced)) {
      return NoMatch(candidate, "A rotary node removed by the rewrite has an external use.");
    }
  }
  return core::builder::MatchResult{this, matched, concat == nullptr ? &candidate : concat};
}

utils::RepeatedProtoField<NodeProto>
RotaryEmbeddingPattern::Apply(core::builder::GraphGraph &graph,
                              const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 5 || nodes[0] == nullptr || nodes[1] == nullptr || nodes[3] == nullptr) {
    throw BuilderError("RotaryEmbeddingPattern::Apply expects both cache Concat nodes.");
  }
  const NodeProto &concat_cos = *nodes[0];
  const NodeProto &concat_sin = *nodes[1];
  const NodeProto *split = nodes[2];
  const NodeProto &half = *nodes[3];
  const NodeProto *concat = nodes[4];
  if ((split == nullptr) != (concat == nullptr)) {
    throw BuilderError("RotaryEmbeddingPattern::Apply received an incomplete partial rotation.");
  }
  const std::string main_input =
      split == nullptr ? half.input()[0].value() : split->input()[0].value();
  const std::string main_output =
      split == nullptr ? half.output()[0].value() : concat->output()[0].value();
  int64_t num_heads = 0;
  if (!StaticDimension(graph, main_input, 1, num_heads)) {
    throw BuilderError("RotaryEmbeddingPattern::Apply needs a static head count.");
  }
  std::optional<int64_t> rotary_dimension;
  if (split != nullptr) {
    std::vector<int64_t> split_sizes;
    if (!ReadConstantInts(graph, split->input()[1].value(), split_sizes) ||
        split_sizes.size() != 2) {
      throw BuilderError("RotaryEmbeddingPattern::Apply could not read Split sizes.");
    }
    rotary_dimension = split_sizes[0];
  }

  utils::RepeatedProtoField<NodeProto> result;
  const auto preserved = PreservedNodes(graph, nodes, {main_output});
  AppendOriginalsInGraphOrder(graph, result, preserved);
  core::builder::GraphBuilder &builder = graph.Builder();
  const std::string cos_name =
      builder.UniqueName("RotaryEmbeddingPattern--" + half.input()[1].value());
  const std::string sin_name =
      builder.UniqueName("RotaryEmbeddingPattern--" + half.input()[2].value());
  const std::string cos_expanded =
      builder.UniqueName("RotaryEmbeddingPattern--" + half.input()[1].value());
  const std::string sin_expanded =
      builder.UniqueName("RotaryEmbeddingPattern--" + half.input()[2].value());
  const std::string batch_name =
      builder.UniqueName("RotaryEmbeddingPattern--" + half.input()[0].value() + "--dim");
  const std::string shape_name =
      builder.UniqueName("RotaryEmbeddingPattern--" + half.input()[0].value() + "::Shape");
  const std::string one_name = FreeInitializerName(builder, "RotaryEmbeddingPattern--1");
  const std::string ones_name = FreeInitializerName(builder, "RotaryEmbeddingPattern--11");
  builder.MakeInitializer(MakeInitializer<int64_t>(one_name.c_str(), {1}, {1}));
  builder.MakeInitializer(MakeInitializer<int64_t>(ones_name.c_str(), {2}, {1, 1}));
  const std::string node_name = "RotaryEmbeddingPattern--" + half.name().value();
  NodeProto shape = MakePatternNode("Shape", {main_input}, {batch_name}, "", node_name);
  AddIntAttribute(shape, "start", 0);
  AddIntAttribute(shape, "end", 1);
  result.push_back(std::move(shape));
  NodeProto concat_shape =
      MakePatternNode("Concat", {batch_name, ones_name}, {shape_name}, "", node_name);
  AddIntAttribute(concat_shape, "axis", 0);
  result.push_back(std::move(concat_shape));
  result.push_back(MakePatternNode("Squeeze", {concat_cos.input()[0].value(), one_name}, {cos_name},
                                   "", node_name));
  result.push_back(MakePatternNode("Squeeze", {concat_sin.input()[0].value(), one_name}, {sin_name},
                                   "", node_name));
  result.push_back(
      MakePatternNode("Expand", {cos_name, shape_name}, {cos_expanded}, "", node_name));
  result.push_back(
      MakePatternNode("Expand", {sin_name, shape_name}, {sin_expanded}, "", node_name));
  NodeProto rotary = MakePatternNode("RotaryEmbedding", {main_input, cos_expanded, sin_expanded},
                                     {main_output}, "", node_name);
  if (rotary_dimension.has_value()) {
    AddIntAttribute(rotary, "rotary_embedding_dim", rotary_dimension.value());
  }
  AddIntAttribute(rotary, "num_heads", num_heads);
  result.push_back(std::move(rotary));
  return result;
}

std::set<std::string> FunctionCausalMaskPattern::FastOpType() const {
  return {"Greater", "LessOrEqual"};
}

core::builder::MatchResult FunctionCausalMaskPattern::Match(core::builder::GraphGraph &graph,
                                                            const NodeProto &candidate) const {
  const bool shifted = IsDefaultOp(candidate, "Greater");
  if ((!shifted && !IsDefaultOp(candidate, "LessOrEqual")) || candidate.input_size() != 2 ||
      candidate.output_size() != 1 || !MainOpsetAtLeast(graph, 13)) {
    return NoMatch(candidate, "FunctionCausalMaskPattern expects Greater or LessOrEqual.");
  }
  const NodeProto *unsqueeze1 = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *second = graph.NodeBefore(candidate.input()[1].value());
  const NodeProto *sub = nullptr;
  if (!IsNode(unsqueeze1, "Unsqueeze", 2, 1) || second == nullptr) {
    return NoMatch(candidate, "Both comparison branches need Unsqueeze.");
  }
  if (shifted) {
    if (!IsNode(second, "Sub", 2, 1)) {
      return NoMatch(candidate, "Shifted causal mask needs Sub on the second branch.");
    }
    sub = second;
    second = graph.NodeBefore(sub->input()[0].value());
  }
  const NodeProto *unsqueeze2 = second;
  if (!IsNode(unsqueeze2, "Unsqueeze", 2, 1)) {
    return NoMatch(candidate, "The second comparison branch needs Unsqueeze.");
  }
  std::vector<int64_t> axes1;
  std::vector<int64_t> axes2;
  if (!ReadConstantInts(graph, unsqueeze1->input()[1].value(), axes1) ||
      !ReadConstantInts(graph, unsqueeze2->input()[1].value(), axes2) ||
      axes1 != std::vector<int64_t>({0, 1, 2}) || axes2 != std::vector<int64_t>({0, 1, 3})) {
    return NoMatch(candidate, "Causal-mask Unsqueeze axes must be [0,1,2] and [0,1,3].");
  }
  const NodeProto *range1 = graph.NodeBefore(unsqueeze1->input()[0].value());
  const NodeProto *range2 = graph.NodeBefore(unsqueeze2->input()[0].value());
  if (!IsNode(range1, "Range", 3, 1) || !IsNode(range2, "Range", 3, 1) ||
      range1->input()[1].value() != range2->input()[1].value() ||
      !IsScalar(graph, range1->input()[0].value(), 0.0) ||
      !IsScalar(graph, range1->input()[2].value(), 1.0) ||
      !IsScalar(graph, range2->input()[2].value(), 1.0)) {
    return NoMatch(candidate, "Causal-mask Range bounds and steps are incompatible.");
  }
  const NodeProto *squeeze1 = graph.NodeBefore(range2->input()[0].value());
  const NodeProto *squeeze2 = graph.NodeBefore(range2->input()[1].value());
  if (!IsNode(squeeze1, "Squeeze", 1, 1) || !IsNode(squeeze2, "Squeeze", 1, 1)) {
    return NoMatch(candidate, "Range bounds must come from one-input Squeeze nodes.");
  }
  const std::string function_name = shifted ? "ShiftedCausalMask" : "CausalMask";
  if (!CanEnsureFunction(graph.Builder(), function_name)) {
    return NoMatch(candidate, "The causal-mask local-function name is already reserved.");
  }
  const std::vector<const NodeProto *> matched = {squeeze1,   squeeze2,   range1, range2,
                                                  unsqueeze1, unsqueeze2, sub,    &candidate};
  const auto preserved = PreservedNodes(graph, matched, {candidate.output()[0].value()});
  const std::unordered_set<const NodeProto *> preserved_set(preserved.begin(), preserved.end());
  std::vector<const NodeProto *> removed;
  for (const NodeProto *node : matched) {
    if (node != nullptr && preserved_set.count(node) == 0) {
      removed.push_back(node);
    }
  }
  return core::builder::MatchResult{this, removed, &candidate};
}

utils::RepeatedProtoField<NodeProto>
FunctionCausalMaskPattern::Apply(core::builder::GraphGraph &graph,
                                 const std::vector<const NodeProto *> &nodes) const {
  const NodeProto *comparison = nullptr;
  for (const NodeProto *node : nodes) {
    if (node != nullptr && (IsDefaultOp(*node, "Greater") || IsDefaultOp(*node, "LessOrEqual"))) {
      comparison = node;
      break;
    }
  }
  if (comparison == nullptr || comparison->input_size() != 2 || comparison->output_size() != 1) {
    throw BuilderError("FunctionCausalMaskPattern::Apply expects the causal-mask decomposition.");
  }
  const NodeProto *unsqueeze1 = graph.NodeBefore(comparison->input()[0].value());
  const NodeProto *second = graph.NodeBefore(comparison->input()[1].value());
  const NodeProto *sub = nullptr;
  if (IsDefaultOp(*comparison, "Greater")) {
    if (!IsNode(second, "Sub", 2, 1)) {
      throw BuilderError("FunctionCausalMaskPattern::Apply expects the causal-mask decomposition.");
    }
    sub = second;
    second = graph.NodeBefore(sub->input()[0].value());
  }
  const NodeProto *unsqueeze2 = second;
  const NodeProto *range2 =
      unsqueeze2 == nullptr ? nullptr : graph.NodeBefore(unsqueeze2->input()[0].value());
  const NodeProto *squeeze1 =
      range2 == nullptr ? nullptr : graph.NodeBefore(range2->input()[0].value());
  const NodeProto *squeeze2 =
      range2 == nullptr ? nullptr : graph.NodeBefore(range2->input()[1].value());
  if (!IsNode(unsqueeze1, "Unsqueeze", 2, 1) || !IsNode(unsqueeze2, "Unsqueeze", 2, 1) ||
      !IsNode(range2, "Range", 3, 1) || !IsNode(squeeze1, "Squeeze", 1, 1) ||
      !IsNode(squeeze2, "Squeeze", 1, 1)) {
    throw BuilderError("FunctionCausalMaskPattern::Apply expects the causal-mask decomposition.");
  }
  const bool shifted = sub != nullptr;
  const std::string function_name = shifted ? "ShiftedCausalMask" : "CausalMask";
  utils::RepeatedProtoField<NodeProto> result;
  std::vector<std::string> inputs = {squeeze1->input()[0].value(), squeeze2->input()[0].value()};
  if (shifted) {
    inputs.push_back(sub->input()[1].value());
  }
  result.push_back(MakePatternNode(function_name.c_str(), inputs, {comparison->output()[0].value()},
                                   kIntermediateDomain,
                                   "FunctionCausalMaskPattern--" + comparison->name().value()));
  EnsureFunction(graph.Builder(), MakeCausalMaskFunction(graph.Builder(), shifted));
  return result;
}

std::set<std::string> FunctionCausalMaskMulAddPattern::FastOpType() const { return {"Add"}; }

core::builder::MatchResult
FunctionCausalMaskMulAddPattern::Match(core::builder::GraphGraph &graph,
                                       const NodeProto &candidate) const {
  if (!IsNode(&candidate, "Add", 2, 1) || !MainOpsetAtLeast(graph, 13)) {
    return NoMatch(candidate, "FunctionCausalMaskMulAddPattern expects a default-domain Add.");
  }
  const NodeProto *unsqueeze1 = graph.NodeBefore(candidate.input()[0].value());
  const NodeProto *mul = graph.NodeBefore(candidate.input()[1].value());
  if (!IsNode(unsqueeze1, "Unsqueeze", 2, 1) || !IsNode(mul, "Mul", 2, 1)) {
    return NoMatch(candidate, "Add must combine Unsqueeze and Mul.");
  }
  const NodeProto *unsqueeze2 = graph.NodeBefore(mul->input()[0].value());
  if (!IsNode(unsqueeze2, "Unsqueeze", 2, 1)) {
    return NoMatch(candidate, "Mul first input must come from Unsqueeze.");
  }
  std::vector<int64_t> axes1;
  std::vector<int64_t> axes2;
  if (!ReadConstantInts(graph, unsqueeze1->input()[1].value(), axes1) ||
      !ReadConstantInts(graph, unsqueeze2->input()[1].value(), axes2) ||
      axes1 != std::vector<int64_t>({0, 1, 2}) || axes2 != std::vector<int64_t>({1, 2, 3})) {
    return NoMatch(candidate, "Mul-add mask Unsqueeze axes are not canonical.");
  }
  const NodeProto *range1 = graph.NodeBefore(unsqueeze1->input()[0].value());
  const NodeProto *range2 = graph.NodeBefore(unsqueeze2->input()[0].value());
  if (!IsNode(range1, "Range", 3, 1) || !IsNode(range2, "Range", 3, 1) ||
      !IsScalar(graph, range1->input()[0].value(), 0.0) ||
      !IsScalar(graph, range1->input()[2].value(), 1.0) ||
      !IsScalar(graph, range2->input()[0].value(), 0.0) ||
      !IsScalar(graph, range2->input()[2].value(), 1.0)) {
    return NoMatch(candidate, "Both Range nodes must be zero-based with unit steps.");
  }
  const NodeProto *squeeze1 = graph.NodeBefore(range1->input()[1].value());
  const NodeProto *squeeze2 = graph.NodeBefore(range2->input()[1].value());
  if (!IsNode(squeeze1, "Squeeze", 1, 1) || !IsNode(squeeze2, "Squeeze", 1, 1) ||
      !CanEnsureFunction(graph.Builder(), "CausalMaskMulAdd")) {
    return NoMatch(candidate, "Range limits or local-function name are invalid.");
  }
  const std::vector<const NodeProto *> matched = {squeeze1,   squeeze2,   range1, range2,
                                                  unsqueeze1, unsqueeze2, mul,    &candidate};
  const std::unordered_set<const NodeProto *> matched_set(matched.begin(), matched.end());
  const std::unordered_set<std::string> replaced{candidate.output()[0].value()};
  for (const NodeProto *node : matched) {
    if (NodeOutputHasExternalUse(graph, *node, matched_set, replaced)) {
      return NoMatch(candidate, "A causal-mask mul-add intermediate has an external use.");
    }
  }
  return core::builder::MatchResult{this, matched, &candidate};
}

utils::RepeatedProtoField<NodeProto>
FunctionCausalMaskMulAddPattern::Apply(core::builder::GraphGraph &graph,
                                       const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 8 || std::any_of(nodes.begin(), nodes.end(),
                                       [](const NodeProto *node) { return node == nullptr; })) {
    throw BuilderError(
        "FunctionCausalMaskMulAddPattern::Apply expects the eight-node decomposition.");
  }
  const NodeProto &squeeze1 = *nodes[0];
  const NodeProto &squeeze2 = *nodes[1];
  const NodeProto &mul = *nodes[6];
  const NodeProto &add = *nodes[7];
  utils::RepeatedProtoField<NodeProto> result;
  const auto preserved = PreservedNodes(graph, nodes, {add.output()[0].value()});
  AppendOriginalsInGraphOrder(graph, result, preserved);
  result.push_back(MakePatternNode(
      "CausalMaskMulAdd",
      {squeeze1.input()[0].value(), squeeze2.input()[0].value(), mul.input()[1].value()},
      {add.output()[0].value()}, kIntermediateDomain,
      "FunctionCausalMaskMulAddPattern--" + add.name().value()));
  EnsureFunction(graph.Builder(), MakeCausalMaskMulAddFunction(graph.Builder()));
  return result;
}

std::set<std::string> FunctionCosSinCachePattern::FastOpType() const { return {"Cos"}; }

core::builder::MatchResult FunctionCosSinCachePattern::Match(core::builder::GraphGraph &graph,
                                                             const NodeProto &candidate) const {
  if (!IsNode(&candidate, "Cos", 1, 1) || !MainOpsetAtLeast(graph, 13)) {
    return NoMatch(candidate, "FunctionCosSinCachePattern expects a default-domain Cos.");
  }
  const auto &trig_consumers = graph.NextNodes(candidate.input()[0].value());
  if (trig_consumers.size() != 2) {
    return NoMatch(candidate, "The trigonometric input must feed exactly Cos and Sin.");
  }
  const NodeProto *sin = nullptr;
  for (const NodeProto *consumer : trig_consumers) {
    if (consumer != &candidate && IsNode(consumer, "Sin", 1, 1)) {
      sin = consumer;
    }
  }
  if (sin == nullptr) {
    return NoMatch(candidate, "The second trigonometric branch must be Sin.");
  }
  const auto branch_cast = [&](const NodeProto &trig, const NodeProto *&cast) {
    const auto &consumers = graph.NextNodes(trig.output()[0].value());
    if (consumers.empty()) {
      cast = nullptr;
      return graph.IsOutput(trig.output()[0].value()) &&
             !graph.IsUsedBySubgraph(trig.output()[0].value());
    }
    if (consumers.size() == 1 && IsNode(consumers[0], "Cast", 1, 1)) {
      cast = consumers[0];
      return true;
    }
    return false;
  };
  const NodeProto *cos_cast = nullptr;
  const NodeProto *sin_cast = nullptr;
  if (!branch_cast(candidate, cos_cast) || !branch_cast(*sin, sin_cast) ||
      (cos_cast == nullptr) != (sin_cast == nullptr)) {
    return NoMatch(candidate, "Cos and Sin must have matching direct or Cast branches.");
  }
  if (cos_cast != nullptr && GetAttributeOr<int64_t>(*cos_cast, "to", -1) !=
                                 GetAttributeOr<int64_t>(*sin_cast, "to", -2)) {
    return NoMatch(candidate, "Cos and Sin Cast destinations must match.");
  }
  const NodeProto *mul = graph.NodeBefore(candidate.input()[0].value());
  if (!IsNode(mul, "Mul", 2, 1) || graph.IsUsedMoreThanOnce(mul->input()[0].value())) {
    return NoMatch(candidate, "Cos and Sin must consume the expected unshared Mul branch.");
  }
  const NodeProto *reshape = graph.NodeBefore(mul->input()[1].value());
  std::vector<int64_t> reshape_shape;
  if (!IsNode(reshape, "Reshape", 2, 1) ||
      !ReadConstantInts(graph, reshape->input()[1].value(), reshape_shape) ||
      reshape_shape != std::vector<int64_t>({0, -1, 1}) ||
      graph.IsUsedMoreThanOnce(reshape->input()[0].value())) {
    return NoMatch(candidate, "Mul second input must be canonical Reshape.");
  }
  const NodeProto *later = graph.NodeBefore(reshape->input()[0].value());
  const NodeProto *earlier =
      later == nullptr ? nullptr : graph.NodeBefore(later->input()[0].value());
  if (later == nullptr || earlier == nullptr ||
      !((IsNode(later, "Cast", 1, 1) && IsNode(earlier, "Unsqueeze", 2, 1)) ||
        (IsNode(later, "Unsqueeze", 2, 1) && IsNode(earlier, "Cast", 1, 1)))) {
    return NoMatch(candidate, "Reshape must follow Cast and Unsqueeze in either order.");
  }
  const NodeProto *unsqueeze = IsDefaultOp(*later, "Unsqueeze") ? later : earlier;
  const NodeProto *position_cast = IsDefaultOp(*later, "Cast") ? later : earlier;
  if (GetAttributeOr<int64_t>(*position_cast, "to", -1) !=
          static_cast<int64_t>(TensorProto::DataType::FLOAT) ||
      !graph.HasType(mul->input()[0].value()) ||
      graph.GetType(mul->input()[0].value()) != TensorType::kFloat) {
    return NoMatch(candidate, "Position ids must be cast to FLOAT and weights must be FLOAT.");
  }
  std::vector<int64_t> position_axes;
  if (!ReadConstantInts(graph, unsqueeze->input()[1].value(), position_axes) ||
      (position_axes != std::vector<int64_t>({0, 1}) &&
       position_axes != std::vector<int64_t>({1}))) {
    return NoMatch(candidate, "Position Unsqueeze axes must be [0,1] or [1].");
  }
  const NodeProto *range = graph.NodeBefore(earlier->input()[0].value());
  const NodeProto *squeeze1 = nullptr;
  const NodeProto *squeeze2 = nullptr;
  if (range == nullptr) {
    if (position_axes != std::vector<int64_t>({1})) {
      return NoMatch(candidate, "Scalar position axes require a Range producer.");
    }
  } else {
    if (!IsNode(range, "Range", 3, 1) || !IsScalar(graph, range->input()[2].value(), 1.0)) {
      return NoMatch(candidate, "Position Range must use a unit step.");
    }
    squeeze1 = graph.NodeBefore(range->input()[0].value());
    squeeze2 = graph.NodeBefore(range->input()[1].value());
    if (!IsNode(squeeze1, "Squeeze", 1, 1) || !IsNode(squeeze2, "Squeeze", 1, 1)) {
      return NoMatch(candidate, "Position Range bounds must come from Squeeze.");
    }
  }
  std::optional<int64_t> to;
  if (cos_cast != nullptr) {
    to = GetAttributeOr<int64_t>(*cos_cast, "to", -1);
    if (to.value() < 0) {
      return NoMatch(candidate, "Output Cast needs a valid to attribute.");
    }
  }
  std::string function_name = "CosSinCache";
  if (range != nullptr) {
    function_name += "WithRange";
  }
  if (to.has_value()) {
    function_name += "_to" + std::to_string(to.value());
  }
  if (position_axes != std::vector<int64_t>({0, 1})) {
    function_name += "_p";
    for (int64_t value : position_axes) {
      function_name += std::to_string(value);
    }
  }
  if (!CanEnsureFunction(graph.Builder(), function_name)) {
    return NoMatch(candidate, "The cos-sin local-function name is already reserved.");
  }
  const std::vector<const NodeProto *> matched = {
      squeeze1, squeeze2, range, earlier, later, reshape, mul, &candidate, cos_cast, sin, sin_cast};
  const std::string cos_output =
      cos_cast == nullptr ? candidate.output()[0].value() : cos_cast->output()[0].value();
  const std::string sin_output =
      sin_cast == nullptr ? sin->output()[0].value() : sin_cast->output()[0].value();
  const std::unordered_set<const NodeProto *> matched_set(matched.begin(), matched.end());
  const std::unordered_set<std::string> replaced{cos_output, sin_output};
  for (const NodeProto *node : matched) {
    if (node != nullptr && NodeOutputHasExternalUse(graph, *node, matched_set, replaced)) {
      return NoMatch(candidate, "A cos-sin cache intermediate has an external use.");
    }
  }
  return core::builder::MatchResult{this, matched, &candidate};
}

utils::RepeatedProtoField<NodeProto>
FunctionCosSinCachePattern::Apply(core::builder::GraphGraph &graph,
                                  const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 11 || nodes[3] == nullptr || nodes[4] == nullptr || nodes[5] == nullptr ||
      nodes[6] == nullptr || nodes[7] == nullptr || nodes[9] == nullptr) {
    throw BuilderError("FunctionCosSinCachePattern::Apply expects the cache decomposition.");
  }
  const NodeProto *squeeze1 = nodes[0];
  const NodeProto *squeeze2 = nodes[1];
  const NodeProto *range = nodes[2];
  const NodeProto &earlier = *nodes[3];
  const NodeProto &later = *nodes[4];
  const NodeProto &mul = *nodes[6];
  const NodeProto &cos = *nodes[7];
  const NodeProto *cos_cast = nodes[8];
  const NodeProto &sin = *nodes[9];
  const NodeProto *sin_cast = nodes[10];
  const NodeProto &unsqueeze = IsDefaultOp(earlier, "Unsqueeze") ? earlier : later;
  std::vector<int64_t> position_axes;
  if (!ReadConstantInts(graph, unsqueeze.input()[1].value(), position_axes)) {
    throw BuilderError("FunctionCosSinCachePattern::Apply could not read position axes.");
  }
  std::optional<int64_t> to;
  if (cos_cast != nullptr) {
    to = GetAttributeOr<int64_t>(*cos_cast, "to", -1);
  }
  std::string function_name = "CosSinCache";
  if (range != nullptr) {
    function_name += "WithRange";
  }
  if (to.has_value()) {
    function_name += "_to" + std::to_string(to.value());
  }
  if (position_axes != std::vector<int64_t>({0, 1})) {
    function_name += "_p";
    for (int64_t value : position_axes) {
      function_name += std::to_string(value);
    }
  }
  const std::string cos_output =
      cos_cast == nullptr ? cos.output()[0].value() : cos_cast->output()[0].value();
  const std::string sin_output =
      sin_cast == nullptr ? sin.output()[0].value() : sin_cast->output()[0].value();
  utils::RepeatedProtoField<NodeProto> result;
  const auto preserved = PreservedNodes(graph, nodes, {cos_output, sin_output});
  AppendOriginalsInGraphOrder(graph, result, preserved);
  const std::vector<std::string> inputs =
      range == nullptr
          ? std::vector<std::string>{earlier.input()[0].value(), mul.input()[0].value()}
          : std::vector<std::string>{squeeze1->input()[0].value(), squeeze2->input()[0].value(),
                                     mul.input()[0].value()};
  result.push_back(MakePatternNode(function_name.c_str(), inputs, {cos_output, sin_output},
                                   kIntermediateDomain,
                                   "FunctionCosSinCachePattern--" + mul.name().value()));
  EnsureFunction(graph.Builder(), MakeCosSinCacheFunction(graph.Builder(), function_name, to,
                                                          position_axes, range != nullptr));
  return result;
}

std::set<std::string> FunctionAttentionPattern::FastOpType() const { return {"Softmax"}; }

core::builder::MatchResult FunctionAttentionPattern::Match(core::builder::GraphGraph &graph,
                                                           const NodeProto &candidate) const {
  if (!IsNode(&candidate, "Softmax", 1, 1) || !MainOpsetAtLeast(graph, 18) ||
      GetAttributeOr<int64_t>(candidate, "axis", -1) != -1) {
    return NoMatch(candidate, "FunctionAttentionPattern expects last-axis Softmax at opset 18.");
  }
  const NodeProto *before = graph.NodeBefore(candidate.input()[0].value());
  if (before == nullptr) {
    return NoMatch(candidate, "Softmax input must be produced by Add or Where.");
  }
  const NodeProto *add = nullptr;
  const NodeProto *where = nullptr;
  const NodeProto *mat_qk = nullptr;
  std::optional<int> infinity_index;
  if (IsNode(before, "Add", 2, 1)) {
    add = before;
    where = graph.NodeBefore(add->input()[1].value());
    if (!IsNode(where, "Where", 3, 1) || !IsScalar(graph, where->input()[1].value(), 0.0) ||
        !IsNegativeInfinity(graph, where->input()[2].value())) {
      return NoMatch(candidate, "Add mask must be Where(mask, zero, negative infinity).");
    }
    infinity_index = 2;
    mat_qk = graph.NodeBefore(add->input()[0].value());
  } else if (IsNode(before, "Where", 3, 1)) {
    where = before;
    infinity_index = NegativeInfinityInput(graph, *where);
    if (!infinity_index.has_value()) {
      return NoMatch(candidate, "Direct Where mask needs a negative infinity branch.");
    }
    mat_qk = graph.NodeBefore(where->input()[3 - infinity_index.value()].value());
  } else {
    return NoMatch(candidate, "Softmax must follow Add or Where masking.");
  }
  const bool fused_matmul = IsDomainNode(mat_qk, "FusedMatMul", "com.microsoft", 2, 1);
  if (!IsNode(mat_qk, "MatMul", 2, 1) && !fused_matmul) {
    return NoMatch(candidate, "Masked scores must originate in MatMul or FusedMatMul.");
  }

  const NodeProto *mul1 = graph.NodeBefore(mat_qk->input()[0].value());
  const NodeProto *transpose_mul1 = nullptr;
  const NodeProto *reshape_mul1 = nullptr;
  if (IsNode(mul1, "Transpose", 1, 1)) {
    transpose_mul1 = mul1;
    if (!HasPerm(*transpose_mul1, {0, 2, 1, 3})) {
      return NoMatch(candidate, "The 3D query Transpose permutation is not canonical.");
    }
    reshape_mul1 = graph.NodeBefore(transpose_mul1->input()[0].value());
    if (!IsNode(reshape_mul1, "Reshape", 2, 1)) {
      return NoMatch(candidate, "The 3D query Transpose must follow Reshape.");
    }
    mul1 = graph.NodeBefore(reshape_mul1->input()[0].value());
  }
  if (!IsNode(mul1, "Mul", 2, 1) || !graph.IsConstantScalar(mul1->input()[1].value(), false)) {
    return NoMatch(candidate, "Query must be multiplied by a scalar scale.");
  }

  const NodeProto *transpose = nullptr;
  const NodeProto *reshape_mul2 = nullptr;
  const NodeProto *mul2 = nullptr;
  if (!fused_matmul) {
    transpose = graph.NodeBefore(mat_qk->input()[1].value());
    if (!IsNode(transpose, "Transpose", 1, 1)) {
      return NoMatch(candidate, "MatMul keys must be transposed.");
    }
    if (transpose_mul1 == nullptr) {
      if (!HasPerm(*transpose, {0, 1, 3, 2})) {
        return NoMatch(candidate, "The key Transpose permutation is not canonical.");
      }
      mul2 = graph.NodeBefore(transpose->input()[0].value());
    } else {
      if (!HasPerm(*transpose, {0, 2, 3, 1})) {
        return NoMatch(candidate, "The 3D key Transpose permutation is not canonical.");
      }
      reshape_mul2 = graph.NodeBefore(transpose->input()[0].value());
      if (!IsNode(reshape_mul2, "Reshape", 2, 1)) {
        return NoMatch(candidate, "The 3D key Transpose must follow Reshape.");
      }
      mul2 = graph.NodeBefore(reshape_mul2->input()[0].value());
      std::vector<int64_t> query_shape;
      std::vector<int64_t> key_shape;
      if (!ReadConstantInts(graph, reshape_mul1->input()[1].value(), query_shape) ||
          !ReadConstantInts(graph, reshape_mul2->input()[1].value(), key_shape) ||
          query_shape != key_shape) {
        return NoMatch(candidate, "The 3D query and key Reshape targets must match.");
      }
    }
  } else {
    if (GetAttributeOr<int64_t>(*mat_qk, "transA", 0) != 0 ||
        GetAttributeOr<int64_t>(*mat_qk, "transB", 0) != 1 ||
        GetAttributeOr<float>(*mat_qk, "alpha", 1.0F) != 1.0F ||
        GetAttributeOr<int64_t>(*mat_qk, "transBatchA", 0) != 0 ||
        GetAttributeOr<int64_t>(*mat_qk, "transBatchB", 0) != 0) {
      return NoMatch(candidate,
                     "FusedMatMul needs transA=0, transB=1, alpha=1, and no batch transposition.");
    }
    mul2 = graph.NodeBefore(mat_qk->input()[1].value());
  }
  if (mul2 == nullptr) {
    return NoMatch(candidate, "The key scale branch is missing.");
  }

  const NodeProto *gqa_unsqueeze = nullptr;
  const NodeProto *gqa_expand = nullptr;
  const NodeProto *gqa_reshape = nullptr;
  if (IsDefaultOp(*mul2, "Reshape")) {
    gqa_reshape = mul2;
    if (!IsNode(gqa_reshape, "Reshape", 2, 1)) {
      return NoMatch(candidate, "The GQA key Reshape is malformed.");
    }
    gqa_expand = graph.NodeBefore(gqa_reshape->input()[0].value());
    if (!IsNode(gqa_expand, "Expand", 2, 1)) {
      return NoMatch(candidate, "The GQA key Reshape must follow Expand.");
    }
    mul2 = graph.NodeBefore(gqa_expand->input()[0].value());
    if (!IsNode(mul2, "Mul", 2, 1)) {
      return NoMatch(candidate, "The GQA key Expand must follow scaled Mul.");
    }
    gqa_unsqueeze = graph.NodeBefore(mul2->input()[0].value());
    std::vector<int64_t> expand_shape;
    std::vector<int64_t> unsqueeze_shape;
    std::vector<int64_t> reshape_shape;
    if (!IsNode(gqa_unsqueeze, "Unsqueeze", 2, 1) ||
        !ReadConstantInts(graph, gqa_expand->input()[1].value(), expand_shape) ||
        expand_shape.size() != 5 || expand_shape[0] != 1 || expand_shape[1] != 1 ||
        expand_shape[3] != 1 || expand_shape[4] != 1 ||
        !ReadConstantInts(graph, gqa_unsqueeze->input()[1].value(), unsqueeze_shape) ||
        unsqueeze_shape != std::vector<int64_t>({2}) ||
        !ReadConstantInts(graph, gqa_reshape->input()[1].value(), reshape_shape) ||
        reshape_shape.size() != 4 || !graph.HasShape(gqa_unsqueeze->input()[0].value()) ||
        !graph.HasShape(gqa_reshape->output()[0].value()) ||
        !SameSelectedDimensions(graph.GetShape(gqa_unsqueeze->input()[0].value()).Shape(),
                                graph.GetShape(gqa_reshape->output()[0].value()).Shape(),
                                {0, 2, 3})) {
      return NoMatch(candidate, "The GQA key repeat-interleave shape is incompatible.");
    }
  } else if (!IsNode(mul2, "Mul", 2, 1)) {
    return NoMatch(candidate, "Keys must use Mul or the GQA repeat-interleave form.");
  }
  if (!graph.IsConstantScalar(mul2->input()[1].value(), false)) {
    return NoMatch(candidate, "Keys must be multiplied by a scalar scale.");
  }
  double scale1 = 0.0;
  double scale2 = 0.0;
  if (!ReadConstantScalar(graph, mul1->input()[1].value(), scale1) ||
      !ReadConstantScalar(graph, mul2->input()[1].value(), scale2) || scale1 != scale2) {
    return NoMatch(candidate, "Query and key scales must be equal.");
  }

  const auto &softmax_consumers = graph.NextNodes(candidate.output()[0].value());
  if (softmax_consumers.size() != 2 || graph.IsOutput(candidate.output()[0].value()) ||
      graph.IsUsedBySubgraph(candidate.output()[0].value())) {
    return NoMatch(candidate, "Softmax must feed only IsNaN and Where.");
  }
  const NodeProto *isnan = nullptr;
  const NodeProto *where2 = nullptr;
  for (const NodeProto *consumer : softmax_consumers) {
    if (IsNode(consumer, "IsNaN", 1, 1)) {
      isnan = consumer;
    } else if (IsNode(consumer, "Where", 3, 1)) {
      where2 = consumer;
    }
  }
  if (isnan == nullptr || where2 == nullptr ||
      where2->input()[0].value() != isnan->output()[0].value() ||
      where2->input()[2].value() != candidate.output()[0].value() ||
      !IsScalar(graph, where2->input()[1].value(), 0.0)) {
    return NoMatch(candidate, "Softmax NaNs must be replaced by zero.");
  }
  const auto &matmul_consumers = graph.NextNodes(where2->output()[0].value());
  if (matmul_consumers.size() != 1 || !IsNode(matmul_consumers[0], "MatMul", 2, 1)) {
    return NoMatch(candidate, "Filtered probabilities must feed one value MatMul.");
  }
  const NodeProto *mat_qkv = matmul_consumers[0];

  const NodeProto *gqa_unsqueeze_v = nullptr;
  const NodeProto *gqa_expand_v = nullptr;
  const NodeProto *gqa_reshape_v = nullptr;
  if (gqa_reshape != nullptr) {
    gqa_reshape_v = graph.NodeBefore(mat_qkv->input()[1].value());
    if (!IsNode(gqa_reshape_v, "Reshape", 2, 1)) {
      return NoMatch(candidate, "The GQA value branch needs Reshape.");
    }
    gqa_expand_v = graph.NodeBefore(gqa_reshape_v->input()[0].value());
    gqa_unsqueeze_v =
        gqa_expand_v == nullptr ? nullptr : graph.NodeBefore(gqa_expand_v->input()[0].value());
    std::vector<int64_t> key_expand;
    std::vector<int64_t> value_expand;
    std::vector<int64_t> key_unsqueeze;
    std::vector<int64_t> value_unsqueeze;
    std::vector<int64_t> key_reshape;
    std::vector<int64_t> value_reshape;
    if (!IsNode(gqa_expand_v, "Expand", 2, 1) || !IsNode(gqa_unsqueeze_v, "Unsqueeze", 2, 1) ||
        !ReadConstantInts(graph, gqa_expand->input()[1].value(), key_expand) ||
        !ReadConstantInts(graph, gqa_expand_v->input()[1].value(), value_expand) ||
        key_expand != value_expand ||
        !ReadConstantInts(graph, gqa_unsqueeze->input()[1].value(), key_unsqueeze) ||
        !ReadConstantInts(graph, gqa_unsqueeze_v->input()[1].value(), value_unsqueeze) ||
        key_unsqueeze != value_unsqueeze ||
        !ReadConstantInts(graph, gqa_reshape->input()[1].value(), key_reshape) ||
        !ReadConstantInts(graph, gqa_reshape_v->input()[1].value(), value_reshape) ||
        key_reshape != value_reshape) {
      return NoMatch(candidate, "The GQA key and value repeat-interleave shapes must match.");
    }
  }

  const std::vector<const NodeProto *> matched = {
      mul1,        transpose_mul1,  reshape_mul1, gqa_unsqueeze, mul2,   reshape_mul2, gqa_expand,
      gqa_reshape, transpose,       mat_qk,       where,         add,    &candidate,   isnan,
      where2,      gqa_unsqueeze_v, gqa_expand_v, gqa_reshape_v, mat_qkv};
  for (std::size_t index = 0; index + 1 < matched.size(); ++index) {
    const NodeProto *node = matched[index];
    if (node == nullptr || node == &candidate) {
      continue;
    }
    if (node->output_size() != 1 || graph.IsUsedMoreThanOnce(node->output()[0].value())) {
      return NoMatch(candidate, "An attention intermediate is shared outside the pattern.");
    }
  }
  if (!graph.HasType(mul1->input()[1].value())) {
    return NoMatch(candidate, "The attention scale type must be known.");
  }
  const auto type = core::symbolic::TensorTypeToDataType(graph.GetType(mul1->input()[1].value()));
  if (!IsSupportedAttentionType(type)) {
    return NoMatch(candidate, "The attention local function requires a floating scale type.");
  }
  const bool switch_where = infinity_index.value() == 1;
  const std::string function_name =
      AttentionFunctionName(type, gqa_reshape != nullptr, switch_where, fused_matmul);
  if (!CanEnsureFunction(graph.Builder(), function_name)) {
    return NoMatch(candidate, "The attention local-function name is already reserved.");
  }
  return core::builder::MatchResult{this, matched, mat_qkv};
}

utils::RepeatedProtoField<NodeProto>
FunctionAttentionPattern::Apply(core::builder::GraphGraph &graph,
                                const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 19 || nodes[0] == nullptr || nodes[4] == nullptr || nodes[9] == nullptr ||
      nodes[10] == nullptr || nodes[12] == nullptr || nodes[13] == nullptr ||
      nodes[14] == nullptr || nodes[18] == nullptr) {
    throw BuilderError("FunctionAttentionPattern::Apply expects the attention decomposition.");
  }
  const NodeProto &mul1 = *nodes[0];
  const NodeProto *transpose_mul1 = nodes[1];
  const NodeProto *reshape_mul1 = nodes[2];
  const NodeProto *gqa_unsqueeze = nodes[3];
  const NodeProto &mul2 = *nodes[4];
  const NodeProto *reshape_mul2 = nodes[5];
  const NodeProto *gqa_expand = nodes[6];
  const NodeProto *gqa_reshape = nodes[7];
  const NodeProto *transpose = nodes[8];
  const NodeProto &where = *nodes[10];
  const NodeProto &softmax = *nodes[12];
  const NodeProto *gqa_unsqueeze_v = nodes[15];
  const NodeProto &mat_qkv = *nodes[18];
  if (!graph.HasType(mul1.input()[1].value())) {
    throw BuilderError("FunctionAttentionPattern::Apply needs the scale type.");
  }
  const auto type = core::symbolic::TensorTypeToDataType(graph.GetType(mul1.input()[1].value()));
  const std::optional<int> infinity_index = NegativeInfinityInput(graph, where);
  if (!infinity_index.has_value()) {
    throw BuilderError("FunctionAttentionPattern::Apply could not locate negative infinity.");
  }
  const bool switch_where = infinity_index.value() == 1;
  const bool gqa = gqa_reshape != nullptr;
  const bool no_transpose = transpose == nullptr;
  const std::string function_name = AttentionFunctionName(type, gqa, switch_where, no_transpose);

  utils::RepeatedProtoField<NodeProto> result;
  const auto preserved = PreservedNodes(graph, nodes, {mat_qkv.output()[0].value()});
  AppendOriginalsInGraphOrder(graph, result, preserved);
  core::builder::GraphBuilder &builder = graph.Builder();
  std::string query;
  std::string keys;
  if (reshape_mul1 != nullptr) {
    if (reshape_mul2 == nullptr || transpose_mul1 == nullptr || transpose == nullptr ||
        gqa_unsqueeze != nullptr) {
      throw BuilderError("FunctionAttentionPattern::Apply received an inconsistent 3D match.");
    }
    const std::string query_reshaped =
        builder.UniqueName("FunctionAttentionPattern--" + mul1.input()[0].value());
    const std::string keys_reshaped =
        builder.UniqueName("FunctionAttentionPattern--" + mul2.input()[0].value());
    query = builder.UniqueName("FunctionAttentionPattern--" + transpose_mul1->input()[0].value());
    keys = builder.UniqueName("FunctionAttentionPattern--" + transpose->input()[0].value());
    result.push_back(MakePatternNode(
        "Reshape", {mul1.input()[0].value(), reshape_mul1->input()[1].value()}, {query_reshaped},
        "", "FunctionAttentionPattern--" + reshape_mul1->name().value()));
    result.push_back(MakePatternNode(
        "Reshape", {mul2.input()[0].value(), reshape_mul2->input()[1].value()}, {keys_reshaped}, "",
        "FunctionAttentionPattern--" + reshape_mul2->name().value()));
    NodeProto query_transpose =
        MakePatternNode("Transpose", {query_reshaped}, {query}, "",
                        "FunctionAttentionPattern--" + transpose_mul1->name().value());
    AddIntsAttribute(query_transpose, "perm", {0, 2, 1, 3});
    result.push_back(std::move(query_transpose));
    NodeProto key_transpose =
        MakePatternNode("Transpose", {keys_reshaped}, {keys}, "",
                        "FunctionAttentionPattern--" + transpose->name().value());
    AddIntsAttribute(key_transpose, "perm", {0, 2, 1, 3});
    result.push_back(std::move(key_transpose));
  } else {
    query = mul1.input()[0].value();
    keys = gqa ? gqa_unsqueeze->input()[0].value() : mul2.input()[0].value();
  }
  std::vector<std::string> inputs = {
      query, keys, gqa ? gqa_unsqueeze_v->input()[0].value() : mat_qkv.input()[1].value(),
      where.input()[0].value(), mul1.input()[1].value()};
  if (gqa) {
    inputs.push_back(gqa_expand->input()[1].value());
    inputs.push_back(gqa_reshape->input()[1].value());
  }
  result.push_back(MakePatternNode(function_name.c_str(), inputs, {mat_qkv.output()[0].value()},
                                   kIntermediateDomain,
                                   "FunctionAttentionPattern--" + softmax.name().value()));
  EnsureFunction(graph.Builder(), MakeAttentionFunction(graph.Builder(), function_name, type, gqa,
                                                        switch_where, false));
  return result;
}

namespace {

utils::RepeatedProtoField<NodeProto>
ApplyRotaryTranspose(core::builder::GraphGraph &graph,
                     const std::vector<const NodeProto *> &nodes) {
  utils::RepeatedProtoField<NodeProto> result;
  const NodeProto *cst_left = nodes[0];
  const NodeProto *part_left = nodes[1];
  const NodeProto *neg_left = nodes[2];
  const NodeProto *cst_right = nodes[3];
  const NodeProto *part_right = nodes[4];
  const NodeProto *neg_right = nodes[5];
  const NodeProto *add = nodes[14];
  if (cst_left == nullptr || part_left == nullptr || cst_right == nullptr ||
      part_right == nullptr || add == nullptr || (neg_left == nullptr) == (neg_right == nullptr)) {
    throw BuilderError("RotaryConcatPartPattern::Apply received an inconsistent transpose match.");
  }
  const bool split_form = IsDefaultOp(*part_left, "Split");
  NodeProto split;
  const NodeProto *split_source = nullptr;
  int64_t axis = 0;
  if (split_form) {
    split_source = part_left;
    axis = Axis(*part_left, 0);
  } else {
    std::vector<int64_t> left_starts;
    std::vector<int64_t> left_ends;
    std::vector<int64_t> axes;
    std::vector<int64_t> right_starts;
    std::vector<int64_t> right_ends;
    if (!ReadConstantInts(graph, part_left->input()[1].value(), left_starts) ||
        !ReadConstantInts(graph, part_left->input()[2].value(), left_ends) ||
        !ReadConstantInts(graph, part_left->input()[3].value(), axes) ||
        !ReadConstantInts(graph, part_right->input()[1].value(), right_starts) ||
        !ReadConstantInts(graph, part_right->input()[2].value(), right_ends) ||
        left_starts.size() != 1 || left_ends.size() != 1 || axes.size() != 1 ||
        right_starts.size() != 1 || right_ends.size() != 1) {
      throw BuilderError("RotaryConcatPartPattern::Apply could not read Slice bounds.");
    }
    axis = axes[0];
    const std::string split_name = FreeInitializerName(
        graph.Builder(), "RotaryConcatPartPattern--" + add->output()[0].value() + "--split");
    graph.Builder().MakeInitializer(MakeInitializer<int64_t>(
        split_name.c_str(), {2}, {left_ends[0] - left_starts[0], right_ends[0] - right_starts[0]}));
    split = MakePatternNode(
        "Split", {part_left->input()[0].value(), split_name},
        {graph.Builder().UniqueName("RotaryConcatPartPattern--" + add->output()[0].value()),
         graph.Builder().UniqueName("RotaryConcatPartPattern--" + add->output()[0].value())},
        "", "RotaryConcatPartPattern--" + add->name().value());
    AddIntAttribute(split, "axis", axis);
  }
  const std::string first =
      split_form ? split_source->output()[0].value() : split.output()[0].value();
  const std::string second =
      split_form ? split_source->output()[1].value() : split.output()[1].value();
  const std::string neg_output =
      graph.Builder().UniqueName("RotaryConcatPartPattern--" + add->output()[0].value());
  NodeProto neg = MakePatternNode("Neg", {neg_left == nullptr ? second : first}, {neg_output}, "",
                                  "RotaryConcatPartPattern--" + add->name().value());
  const std::vector<std::string> concat_inputs = neg_left == nullptr
                                                     ? std::vector<std::string>{first, neg_output}
                                                     : std::vector<std::string>{neg_output, second};
  NodeProto concat = MakePatternNode("Concat", concat_inputs, {add->output()[0].value()}, "",
                                     "RotaryConcatPartPattern--" + add->name().value());
  AddIntAttribute(concat, "axis", axis);
  concat.set_doc_string(add->doc_string().value());

  std::vector<const NodeProto *> originals;
  if (graph.IsUsedMoreThanOnce(cst_left->output()[0].value())) {
    originals.push_back(cst_left);
  }
  if (cst_right != cst_left && graph.IsUsedMoreThanOnce(cst_right->output()[0].value())) {
    originals.push_back(cst_right);
  }
  if (split_form) {
    originals.push_back(split_source);
  }
  const auto preserved = PreservedNodes(graph, nodes, {add->output()[0].value()});
  originals.insert(originals.end(), preserved.begin(), preserved.end());
  AppendOriginalsInGraphOrder(graph, result, originals);
  if (!split_form) {
    result.push_back(std::move(split));
  }
  result.push_back(std::move(neg));
  result.push_back(std::move(concat));
  return result;
}

} // namespace

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
