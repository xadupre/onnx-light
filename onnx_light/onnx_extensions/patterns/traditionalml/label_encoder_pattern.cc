// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/patterns/traditionalml/label_encoder_pattern.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "onnx_core/builder/graph_graph.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_patterns {

namespace {

using core::builder::BuilderError;

bool IsLabelEncoder(const NodeProto &node) {
  return node.op_type().value() == "LabelEncoder" &&
         NormaliseDomain(node.domain().value()) == "ai.onnx.ml";
}

enum class ValueKind { kNone, kInt64, kString };

struct EncoderAttributes {
  ValueKind key_kind = ValueKind::kNone;
  ValueKind value_kind = ValueKind::kNone;
  const AttributeProto *keys = nullptr;
  const AttributeProto *values = nullptr;
  const AttributeProto *default_value = nullptr;
  int64_t default_int64 = -1;
  std::string default_string = "_Unused";
};

EncoderAttributes GetEncoderAttributes(const NodeProto &node) {
  EncoderAttributes attrs;
  for (const char *unsupported : {"keys_floats", "keys_tensor", "values_floats", "values_tensor",
                                  "default_float", "default_tensor"}) {
    if (FindAttribute(node, unsupported) != nullptr) {
      return {};
    }
  }
  if (const AttributeProto *keys = FindAttribute(node, "keys_int64s");
      keys != nullptr && keys->type() == AttributeProto::AttributeType::INTS) {
    attrs.key_kind = ValueKind::kInt64;
    attrs.keys = keys;
  }
  if (const AttributeProto *keys = FindAttribute(node, "keys_strings");
      keys != nullptr && keys->type() == AttributeProto::AttributeType::STRINGS) {
    if (attrs.keys != nullptr) {
      return {};
    }
    attrs.key_kind = ValueKind::kString;
    attrs.keys = keys;
  }
  if (const AttributeProto *values = FindAttribute(node, "values_int64s");
      values != nullptr && values->type() == AttributeProto::AttributeType::INTS) {
    attrs.value_kind = ValueKind::kInt64;
    attrs.values = values;
    attrs.default_value = FindAttribute(node, "default_int64");
    if (attrs.default_value != nullptr) {
      if (attrs.default_value->type() != AttributeProto::AttributeType::INT) {
        return {};
      }
      attrs.default_int64 = attrs.default_value->i();
    }
  }
  if (const AttributeProto *values = FindAttribute(node, "values_strings");
      values != nullptr && values->type() == AttributeProto::AttributeType::STRINGS) {
    if (attrs.values != nullptr) {
      return {};
    }
    attrs.value_kind = ValueKind::kString;
    attrs.values = values;
    attrs.default_value = FindAttribute(node, "default_string");
    if (attrs.default_value != nullptr) {
      if (attrs.default_value->type() != AttributeProto::AttributeType::STRING) {
        return {};
      }
      attrs.default_string = attrs.default_value->s().value();
    }
  }
  if (attrs.keys == nullptr || attrs.values == nullptr) {
    return {};
  }
  const std::size_t key_count = attrs.key_kind == ValueKind::kInt64
                                    ? static_cast<std::size_t>(attrs.keys->ints_size())
                                    : static_cast<std::size_t>(attrs.keys->strings_size());
  const std::size_t value_count = attrs.value_kind == ValueKind::kInt64
                                      ? static_cast<std::size_t>(attrs.values->ints_size())
                                      : static_cast<std::size_t>(attrs.values->strings_size());
  if (key_count != value_count) {
    return {};
  }
  return attrs;
}

template <typename T> std::vector<T> ReadList(const AttributeProto &attribute);

template <> std::vector<int64_t> ReadList(const AttributeProto &attribute) {
  return {attribute.ints().begin(), attribute.ints().end()};
}

template <> std::vector<std::string> ReadList(const AttributeProto &attribute) {
  return {attribute.strings().begin(), attribute.strings().end()};
}

template <typename T> T ReadDefault(const EncoderAttributes &attributes);

template <> int64_t ReadDefault(const EncoderAttributes &attributes) {
  return attributes.default_int64;
}

template <> std::string ReadDefault(const EncoderAttributes &attributes) {
  return attributes.default_string;
}

template <typename T> void AddValues(NodeProto &node, const std::vector<T> &values);

template <> void AddValues(NodeProto &node, const std::vector<int64_t> &values) {
  AttributeProto &attribute = *node.add_attribute();
  attribute.set_name("values_int64s");
  attribute.set_type(AttributeProto::AttributeType::INTS);
  for (int64_t value : values) {
    attribute.add_ints(value);
  }
}

template <> void AddValues(NodeProto &node, const std::vector<std::string> &values) {
  AttributeProto &attribute = *node.add_attribute();
  attribute.set_name("values_strings");
  attribute.set_type(AttributeProto::AttributeType::STRINGS);
  for (const std::string &value : values) {
    attribute.add_strings(value);
  }
}

template <typename T> void AddDefault(NodeProto &node, const T &value);

template <> void AddDefault(NodeProto &node, const int64_t &value) {
  AddAttribute<int64_t>(node, "default_int64", value);
}

template <> void AddDefault(NodeProto &node, const std::string &value) {
  AddAttribute<std::string>(node, "default_string", value);
}

template <typename MiddleT, typename OutputT>
NodeProto Compose(const NodeProto &first, const NodeProto &second, const EncoderAttributes &a,
                  const EncoderAttributes &b, const std::string &name) {
  const std::vector<MiddleT> first_values = ReadList<MiddleT>(*a.values);
  const MiddleT first_default = ReadDefault<MiddleT>(a);
  const std::vector<MiddleT> second_keys = ReadList<MiddleT>(*b.keys);
  const std::vector<OutputT> second_values = ReadList<OutputT>(*b.values);
  const OutputT second_default = ReadDefault<OutputT>(b);
  std::map<MiddleT, OutputT> mapping;
  for (std::size_t i = 0; i < second_keys.size(); ++i) {
    mapping[second_keys[i]] = second_values[i];
  }
  const auto mapped = [&mapping, &second_default](const MiddleT &value) {
    const auto found = mapping.find(value);
    return found == mapping.end() ? second_default : found->second;
  };
  std::vector<OutputT> values;
  values.reserve(first_values.size());
  for (const MiddleT &value : first_values) {
    values.push_back(mapped(value));
  }

  NodeProto replacement = MakeNode("LabelEncoder", {first.input()[0].value()},
                                   {second.output()[0].value()}, "ai.onnx.ml", name.c_str());
  for (const AttributeProto &attribute : first.attribute()) {
    if (&attribute != a.values && (a.default_value == nullptr || &attribute != a.default_value)) {
      *replacement.add_attribute() = attribute;
    }
  }
  AddValues<OutputT>(replacement, values);
  AddDefault<OutputT>(replacement, mapped(first_default));
  return replacement;
}

} // namespace

std::set<std::string> LabelEncoderFusionPattern::FastOpType() const { return {"LabelEncoder"}; }

core::builder::MatchResult LabelEncoderFusionPattern::Match(core::builder::GraphGraph &graph,
                                                            const NodeProto &candidate) const {
  if (graph.Builder().OpsetVersion("ai.onnx.ml") < 2) {
    return NoMatch(candidate, "the ai.onnx.ml opset is older than 2");
  }
  if (!IsLabelEncoder(candidate) || candidate.input_size() != 1 || candidate.output_size() != 1) {
    return NoMatch(candidate, "candidate is not an ai.onnx.ml LabelEncoder");
  }
  const std::string &intermediate = candidate.output()[0].value();
  if (graph.IsOutput(intermediate) || graph.IsUsedBySubgraph(intermediate) ||
      graph.NextNodes(intermediate).size() != 1) {
    return NoMatch(candidate, "the first LabelEncoder output is not exclusively internal");
  }
  const NodeProto *next = graph.NextNodes(intermediate)[0];
  if (!IsLabelEncoder(*next) || next->input_size() != 1 || next->output_size() != 1 ||
      next->input()[0].value() != intermediate) {
    return NoMatch(candidate, "the sole consumer is not a chained LabelEncoder");
  }
  const EncoderAttributes first = GetEncoderAttributes(candidate);
  const EncoderAttributes second = GetEncoderAttributes(*next);
  if (first.keys == nullptr || second.keys == nullptr || first.value_kind != second.key_kind) {
    return NoMatch(candidate, "the LabelEncoder mappings or defaults are incompatible");
  }
  return core::builder::MatchResult{this, {&candidate, next}, &candidate};
}

utils::RepeatedProtoField<NodeProto>
LabelEncoderFusionPattern::Apply(core::builder::GraphGraph &graph,
                                 const std::vector<const NodeProto *> &nodes) const {
  if (nodes.size() != 2 || nodes[0] == nullptr || nodes[1] == nullptr) {
    throw BuilderError("LabelEncoderFusionPattern::Apply expects two LabelEncoder nodes.");
  }
  const core::builder::MatchResult verified = Match(graph, *nodes[0]);
  if (verified.pattern == nullptr || verified.nodes != nodes) {
    throw BuilderError(
        "LabelEncoderFusionPattern::Apply received an unsafe or inconsistent match.");
  }
  const EncoderAttributes first = GetEncoderAttributes(*nodes[0]);
  const EncoderAttributes second = GetEncoderAttributes(*nodes[1]);
  const std::string name = "LabelEncoderFusionPattern--" + nodes[1]->name().value();
  NodeProto replacement;
  if (first.value_kind == ValueKind::kInt64 && second.value_kind == ValueKind::kInt64) {
    replacement = Compose<int64_t, int64_t>(*nodes[0], *nodes[1], first, second, name);
  } else if (first.value_kind == ValueKind::kInt64 && second.value_kind == ValueKind::kString) {
    replacement = Compose<int64_t, std::string>(*nodes[0], *nodes[1], first, second, name);
  } else if (first.value_kind == ValueKind::kString && second.value_kind == ValueKind::kInt64) {
    replacement = Compose<std::string, int64_t>(*nodes[0], *nodes[1], first, second, name);
  } else {
    replacement = Compose<std::string, std::string>(*nodes[0], *nodes[1], first, second, name);
  }
  utils::RepeatedProtoField<NodeProto> replacements;
  replacements.push_back(replacement);
  return replacements;
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_patterns
