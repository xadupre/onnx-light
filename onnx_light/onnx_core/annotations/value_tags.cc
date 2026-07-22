// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/annotations/value_tags.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "onnx_core/annotations/compute_context.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace annotations {

namespace {

template <typename T> void SetMetadataValue(T &obj, const char *key, const std::string &value) {
  for (int i = 0; i < obj.metadata_props().size(); ++i) {
    if (obj.metadata_props()[i].key() == key) {
      obj.mutable_metadata_props(static_cast<std::size_t>(i))->set_value(value);
      return;
    }
  }
  auto *entry = obj.add_metadata_props();
  entry->set_key(key);
  entry->set_value(value);
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
  ComputeContext ctx;
  return ctx.ComputeValueAndNodeTags(graph);
}

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const FunctionProto &function) {
  ComputeContext ctx;
  return ctx.ComputeValueAndNodeTags(function);
}

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const utils::RepeatedProtoField<NodeProto> &nodes) {
  ComputeContext ctx;
  return ctx.ComputeValueAndNodeTags(nodes);
}

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
InferValueAndNodeTags(const std::vector<NodeProto> &nodes) {
  ComputeContext ctx;
  return ctx.ComputeValueAndNodeTags(nodes);
}

void WriteValueAndNodeTagsToMetadata(GraphProto &graph) {
  ComputeContext ctx;
  const auto inferred = ctx.ComputeValueAndNodeTags(graph);
  const auto &value_tags = inferred.first;
  const auto &node_tags = inferred.second;
  const std::size_t node_limit =
      std::min(node_tags.size(), static_cast<std::size_t>(graph.node().size()));
  for (std::size_t i = 0; i < node_limit; ++i) {
    if (!node_tags[i].empty()) {
      SetMetadataValue(*graph.mutable_node(i), kNodeTagMetadataKey, node_tags[i]);
      SetMetadataValue(*graph.mutable_node(i), kValueTagMetadataKey, node_tags[i]);
    }
  }
  SetMetadataValue(graph, kValueTagsMetadataKey, DumpValueTagsAsJson(value_tags));
  for (int i = 0; i < graph.input().size(); ++i) {
    auto it = value_tags.find(graph.input(i).name());
    if (it != value_tags.end()) {
      SetMetadataValue(*graph.mutable_input(static_cast<std::size_t>(i)), kValueTagMetadataKey,
                       it->second);
    }
  }
  for (int i = 0; i < graph.value_info().size(); ++i) {
    auto it = value_tags.find(graph.value_info(i).name());
    if (it != value_tags.end()) {
      SetMetadataValue(*graph.mutable_value_info(static_cast<std::size_t>(i)), kValueTagMetadataKey,
                       it->second);
    }
  }
  for (int i = 0; i < graph.output().size(); ++i) {
    auto it = value_tags.find(graph.output(i).name());
    if (it != value_tags.end()) {
      SetMetadataValue(*graph.mutable_output(static_cast<std::size_t>(i)), kValueTagMetadataKey,
                       it->second);
    }
  }
  for (int i = 0; i < graph.initializer().size(); ++i) {
    auto it = value_tags.find(graph.initializer(i).name());
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
  ComputeContext ctx;
  const auto inferred = ctx.ComputeValueAndNodeTags(function);
  const auto &value_tags = inferred.first;
  const auto &node_tags = inferred.second;
  const std::size_t node_limit =
      std::min(node_tags.size(), static_cast<std::size_t>(function.node().size()));
  for (std::size_t i = 0; i < node_limit; ++i) {
    if (!node_tags[i].empty()) {
      SetMetadataValue(*function.mutable_node(i), kNodeTagMetadataKey, node_tags[i]);
      SetMetadataValue(*function.mutable_node(i), kValueTagMetadataKey, node_tags[i]);
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
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
