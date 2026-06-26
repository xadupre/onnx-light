// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/annotations/value_tags.h"

#include <algorithm>
#include <string_view>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace annotations {

namespace {

std::string ReadMetadataValueFromProps(const std::vector<StringStringEntryProto> &props,
                                       const char *key) {
  for (const auto &entry : props) {
    if (entry.key().as_string() == key) {
      return entry.value().as_string();
    }
  }
  return {};
}

template <typename T> std::string ReadMetadataValue(const T &obj, const char *key) {
  return ReadMetadataValueFromProps(obj.metadata_props().values(), key);
}

template <typename T> void SetMetadataValue(T &obj, const char *key, const std::string &value) {
  for (int i = 0; i < obj.metadata_props().size(); ++i) {
    if (obj.metadata_props()[i].key().as_string() == key) {
      obj.mutable_metadata_props(static_cast<std::size_t>(i))->set_value(value);
      return;
    }
  }
  auto *entry = obj.add_metadata_props();
  entry->set_key(key);
  entry->set_value(value);
}

std::string NormalizeValueTag(std::string_view tag) {
  std::string lower;
  lower.reserve(tag.size());
  for (char c : tag) {
    if (c >= 'A' && c <= 'Z') {
      lower.push_back(static_cast<char>(c - 'A' + 'a'));
    } else {
      lower.push_back(c);
    }
  }
  while (!lower.empty() && lower.front() == ' ') {
    lower.erase(lower.begin());
  }
  while (!lower.empty() && lower.back() == ' ') {
    lower.pop_back();
  }
  if (lower == "shape" || lower == "axes" || lower == "weight") {
    return lower;
  }
  return {};
}

bool IsFloatRank2Tensor(const ValueInfoProto &value) {
  if (!value.has_type() || !value.type().has_tensor_type()) {
    return false;
  }
  const auto &tensor_type = value.type().tensor_type();
  if (tensor_type.elem_type() != TensorProto::DataType::FLOAT) {
    return false;
  }
  if (!tensor_type.has_shape()) {
    return false;
  }
  return tensor_type.shape().dim_size() == 2;
}

void SetValueTag(std::unordered_map<std::string, std::string> &value_tags, const std::string &name,
                 const std::string &tag) {
  if (name.empty()) {
    return;
  }
  const std::string norm = NormalizeValueTag(tag);
  if (!norm.empty()) {
    value_tags[name] = norm;
  }
}

void CollectGraphSeedTags(const GraphProto &graph,
                          std::unordered_map<std::string, std::string> &value_tags) {
  for (int i = 0; i < graph.input().size(); ++i) {
    const auto &value = graph.input()[i];
    std::string tag = ReadMetadataValue(value, kValueTagMetadataKey);
    if (tag.empty() && IsFloatRank2Tensor(value)) {
      tag = "weight";
    }
    SetValueTag(value_tags, value.name().as_string(), tag);
  }
  for (int i = 0; i < graph.initializer().size(); ++i) {
    const auto &value = graph.initializer()[i];
    std::string tag = ReadMetadataValue(value, kValueTagMetadataKey);
    if (tag.empty()) {
      tag = "weight";
    }
    SetValueTag(value_tags, value.name().as_string(), tag);
  }
  for (int i = 0; i < graph.value_info().size(); ++i) {
    const auto &value = graph.value_info()[i];
    SetValueTag(value_tags, value.name().as_string(),
                ReadMetadataValue(value, kValueTagMetadataKey));
  }
  for (int i = 0; i < graph.output().size(); ++i) {
    const auto &value = graph.output()[i];
    SetValueTag(value_tags, value.name().as_string(),
                ReadMetadataValue(value, kValueTagMetadataKey));
  }
}

void InferNodesTags(const std::vector<const NodeProto *> &nodes,
                    std::unordered_map<std::string, std::string> &value_tags,
                    std::vector<std::string> &node_tags) {
  node_tags.clear();
  node_tags.reserve(nodes.size());
  for (const NodeProto *node : nodes) {
    const std::string op_type = node->op_type().as_string();
    std::string explicit_output_tag;
    if (op_type == "Shape" || op_type == "Size") {
      explicit_output_tag = "shape";
    } else if (op_type == "Constant") {
      explicit_output_tag = "weight";
    }

    if (node->input().size() >= 2) {
      if (op_type == "Reshape" || op_type == "Expand" || op_type == "Slice") {
        SetValueTag(value_tags, node->input(1).as_string(), "shape");
      } else if (op_type == "Squeeze" || op_type == "Unsqueeze" || op_type == "ReduceSum" ||
                 op_type == "ReduceMean" || op_type == "ReduceMax" || op_type == "ReduceMin") {
        SetValueTag(value_tags, node->input(1).as_string(), "axes");
      }
    }
    if (op_type == "Slice") {
      if (node->input().size() > 2) {
        SetValueTag(value_tags, node->input(2).as_string(), "shape");
      }
      if (node->input().size() > 3) {
        SetValueTag(value_tags, node->input(3).as_string(), "axes");
      }
      if (node->input().size() > 4) {
        SetValueTag(value_tags, node->input(4).as_string(), "shape");
      }
    }

    std::string inherited_tag;
    if (!node->input().empty()) {
      auto it = value_tags.find(node->input(0).as_string());
      if (it != value_tags.end()) {
        inherited_tag = it->second;
      }
    }
    const std::string node_tag = explicit_output_tag.empty() ? inherited_tag : explicit_output_tag;
    node_tags.push_back(node_tag);
    if (!node_tag.empty()) {
      for (int o = 0; o < node->output().size(); ++o) {
        SetValueTag(value_tags, node->output(o).as_string(), node_tag);
      }
    }
  }
}

std::string EscapeJson(std::string_view text) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(text.size() + 8);
  for (unsigned char c : text) {
    if (c == '\\' || c == '"') {
      out.push_back('\\');
      out.push_back(static_cast<char>(c));
    } else if (c == '\b') {
      out.append("\\b");
    } else if (c == '\f') {
      out.append("\\f");
    } else if (c == '\n') {
      out.append("\\n");
    } else if (c == '\r') {
      out.append("\\r");
    } else if (c == '\t') {
      out.append("\\t");
    } else if (c < 0x20) {
      out.append("\\u00");
      out.push_back(kHex[c >> 4]);
      out.push_back(kHex[c & 0x0f]);
    } else {
      out.push_back(static_cast<char>(c));
    }
  }
  return out;
}

std::string DumpValueTagsAsJson(const std::unordered_map<std::string, std::string> &value_tags) {
  std::vector<std::pair<std::string, std::string>> entries;
  entries.reserve(value_tags.size());
  for (const auto &kv : value_tags) {
    entries.push_back(kv);
  }
  std::sort(entries.begin(), entries.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  std::string out = "{";
  bool first = true;
  for (const auto &entry : entries) {
    if (!first) {
      out.append(",");
    }
    first = false;
    out.append("\"");
    out.append(EscapeJson(entry.first));
    out.append("\":\"");
    out.append(EscapeJson(entry.second));
    out.append("\"");
  }
  out.append("}");
  return out;
}

void RecurseSubgraphs(NodeProto &node) {
  for (int i = 0; i < node.attribute().size(); ++i) {
    AttributeProto *attr = node.mutable_attribute(static_cast<std::size_t>(i));
    if (attr->has_g()) {
      WriteValueAndNodeTagsToMetadata(*attr->mutable_g());
    }
    for (int g = 0; g < attr->graphs().size(); ++g) {
      WriteValueAndNodeTagsToMetadata(*attr->mutable_graphs(static_cast<std::size_t>(g)));
    }
  }
}

} // namespace

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const GraphProto &graph) {
  std::unordered_map<std::string, std::string> value_tags;
  std::vector<std::string> node_tags;
  CollectGraphSeedTags(graph, value_tags);
  std::vector<const NodeProto *> nodes;
  nodes.reserve(graph.node().size());
  for (int i = 0; i < graph.node().size(); ++i) {
    nodes.push_back(&graph.node()[i]);
  }
  InferNodesTags(nodes, value_tags, node_tags);
  return {value_tags, node_tags};
}

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const FunctionProto &function) {
  std::unordered_map<std::string, std::string> value_tags;
  std::vector<std::string> node_tags;
  std::vector<const NodeProto *> nodes;
  nodes.reserve(function.node().size());
  for (int i = 0; i < function.node().size(); ++i) {
    nodes.push_back(&function.node()[i]);
  }
  InferNodesTags(nodes, value_tags, node_tags);
  return {value_tags, node_tags};
}

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const std::vector<NodeProto> &nodes) {
  std::unordered_map<std::string, std::string> value_tags;
  std::vector<std::string> node_tags;
  std::vector<const NodeProto *> ptrs;
  ptrs.reserve(nodes.size());
  for (const NodeProto &node : nodes) {
    ptrs.push_back(&node);
  }
  InferNodesTags(ptrs, value_tags, node_tags);
  return {value_tags, node_tags};
}

void WriteValueAndNodeTagsToMetadata(GraphProto &graph) {
  const auto inferred = InferValueAndNodeTags(graph);
  const auto &value_tags = inferred.first;
  const auto &node_tags = inferred.second;
  const std::size_t node_limit =
      std::min(node_tags.size(), static_cast<std::size_t>(graph.node().size()));
  for (std::size_t i = 0; i < node_limit; ++i) {
    if (!node_tags[i].empty()) {
      SetMetadataValue(*graph.mutable_node(i), kNodeTagMetadataKey, node_tags[i]);
    }
  }
  SetMetadataValue(graph, kValueTagsMetadataKey, DumpValueTagsAsJson(value_tags));
  for (int i = 0; i < graph.input().size(); ++i) {
    auto it = value_tags.find(graph.input(i).name().as_string());
    if (it != value_tags.end()) {
      SetMetadataValue(*graph.mutable_input(static_cast<std::size_t>(i)), kValueTagMetadataKey,
                       it->second);
    }
  }
  for (int i = 0; i < graph.value_info().size(); ++i) {
    auto it = value_tags.find(graph.value_info(i).name().as_string());
    if (it != value_tags.end()) {
      SetMetadataValue(*graph.mutable_value_info(static_cast<std::size_t>(i)), kValueTagMetadataKey,
                       it->second);
    }
  }
  for (int i = 0; i < graph.output().size(); ++i) {
    auto it = value_tags.find(graph.output(i).name().as_string());
    if (it != value_tags.end()) {
      SetMetadataValue(*graph.mutable_output(static_cast<std::size_t>(i)), kValueTagMetadataKey,
                       it->second);
    }
  }
  for (int i = 0; i < graph.initializer().size(); ++i) {
    auto it = value_tags.find(graph.initializer(i).name().as_string());
    if (it != value_tags.end()) {
      SetMetadataValue(*graph.mutable_initializer(static_cast<std::size_t>(i)),
                       kValueTagMetadataKey, it->second);
    }
  }
  for (int i = 0; i < graph.node().size(); ++i) {
    RecurseSubgraphs(*graph.mutable_node(static_cast<std::size_t>(i)));
  }
}

void WriteValueAndNodeTagsToMetadata(FunctionProto &function) {
  const auto inferred = InferValueAndNodeTags(function);
  const auto &value_tags = inferred.first;
  const auto &node_tags = inferred.second;
  const std::size_t node_limit =
      std::min(node_tags.size(), static_cast<std::size_t>(function.node().size()));
  for (std::size_t i = 0; i < node_limit; ++i) {
    if (!node_tags[i].empty()) {
      SetMetadataValue(*function.mutable_node(i), kNodeTagMetadataKey, node_tags[i]);
    }
  }
  SetMetadataValue(function, kValueTagsMetadataKey, DumpValueTagsAsJson(value_tags));
  for (int i = 0; i < function.node().size(); ++i) {
    RecurseSubgraphs(*function.mutable_node(static_cast<std::size_t>(i)));
  }
}

void WriteValueAndNodeTagsToMetadata(ModelProto &model) {
  if (model.has_graph()) {
    WriteValueAndNodeTagsToMetadata(*model.mutable_graph());
  }
}

} // namespace annotations
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
