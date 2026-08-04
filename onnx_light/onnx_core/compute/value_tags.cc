// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/value_tags.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "onnx_core/compute/compute_context.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace compute {

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

// ── Helpers shared by ComputeValueAndNodeTags overloads ─────────────────────

std::string
ReadMetadataValueFromProps(const utils::RepeatedProtoField<StringStringEntryProto> &props,
                           const char *key) {
  for (const auto &entry : props) {
    if (entry.key() == key) {
      return entry.value();
    }
  }
  return {};
}

template <typename T> std::string ReadMetadataValue(const T &obj, const char *key) {
  return ReadMetadataValueFromProps(obj.metadata_props(), key);
}

void SetValueTag(std::unordered_map<std::string, std::string> &value_tags, const std::string &name,
                 const std::string &tag) {
  static_cast<void>(TrySetValueTag(value_tags, name, tag));
}

// Returns the input indices that can safely inherit the given output tag when
// running backward propagation from consumers to producers.
std::vector<int> BackwardTagInputIndices(const NodeProto &node, const std::string &output_tag) {
  const std::string op_type = node.op_type();
  if (op_type == "Concat") {
    // Do not propagate "ambiguous" backward to avoid overwriting inputs that
    // already carry a more specific tag (e.g. "shape" or "axes").
    if (output_tag == "ambiguous") {
      return {};
    }
    std::vector<int> all_inputs;
    all_inputs.reserve(static_cast<std::size_t>(node.input().size()));
    for (int i = 0; i < node.input().size(); ++i) {
      all_inputs.push_back(i);
    }
    return all_inputs;
  }
  // Reshape and Cast preserve the semantics of their first input: the output
  // carries the same tag as input[0].  Back-propagate so that an untagged
  // producer can inherit the tag that the consumer determines.
  if (op_type == "Identity" || op_type == "Cast" || op_type == "Reshape" || op_type == "Squeeze" ||
      op_type == "Unsqueeze" || op_type == "Gather" || op_type == "Slice") {
    return {0};
  }
  // Element-wise binary ops: propagate backward only when the output is a
  // "weight" tensor (both operands belong to the same semantic category).
  // Shape or axes tags must NOT flow back: Mul(shape, weight) → shape should
  // not retag the weight scalar as "shape".
  static const std::unordered_set<std::string> kBinaryElemwiseOps = {
      "Add",   "And",     "BitAnd",         "BitOr", "BitShift",    "BitXor", "Div",
      "Equal", "Greater", "GreaterOrEqual", "Less",  "LessOrEqual", "Mod",    "Mul",
      "Or",    "Pow",     "PRelu",          "Sub",   "Xor"};
  if (kBinaryElemwiseOps.count(op_type) && output_tag == "weight") {
    return {0, 1};
  }
  return {};
}

} // namespace

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
  if (lower == "shape" || lower == "axes" || lower == "weight" || lower == "ambiguous") {
    return lower;
  }
  return {};
}

bool TrySetValueTag(std::unordered_map<std::string, std::string> &value_tags,
                    const std::string &name, const std::string &tag) {
  if (name.empty()) {
    return false;
  }
  const std::string norm = NormalizeValueTag(tag);
  if (!norm.empty()) {
    auto it = value_tags.find(name);
    if (it != value_tags.end()) {
      if (it->second == norm) {
        return false;
      }
      if (it->second == "ambiguous") {
        return false;
      }
      // "shape" and "axes" are more specific tags and win over "weight".
      if (norm == "weight" && (it->second == "shape" || it->second == "axes")) {
        return false;
      }
      if (it->second == "weight" && (norm == "shape" || norm == "axes")) {
        it->second = norm;
        return true;
      }
      it->second = "ambiguous";
      return true;
    }
    value_tags[name] = norm;
    return true;
  }
  return false;
}

void CollectGraphSeedTags(const GraphProto &graph,
                          std::unordered_map<std::string, std::string> &value_tags) {
  for (int i = 0; i < graph.input().size(); ++i) {
    const auto &value = graph.input()[i];
    std::string tag = ReadMetadataValue(value, kValueTagMetadataKey);
    if (tag.empty()) {
      // All graph inputs are unconditionally seeded as "weight": they represent
      // model data or parameters, and this tag is always known at graph level.
      tag = "weight";
    }
    SetValueTag(value_tags, value.name(), tag);
  }
  for (int i = 0; i < graph.initializer().size(); ++i) {
    const auto &value = graph.initializer()[i];
    std::string tag = ReadMetadataValue(value, kValueTagMetadataKey);
    if (tag.empty()) {
      tag = "weight";
    }
    SetValueTag(value_tags, value.name(), tag);
  }
  for (int i = 0; i < graph.value_info().size(); ++i) {
    const auto &value = graph.value_info()[i];
    SetValueTag(value_tags, value.name(), ReadMetadataValue(value, kValueTagMetadataKey));
  }
  for (int i = 0; i < graph.output().size(); ++i) {
    const auto &value = graph.output()[i];
    SetValueTag(value_tags, value.name(), ReadMetadataValue(value, kValueTagMetadataKey));
  }
}

bool ProcessNodeTags(const NodeProto &node, std::size_t n,
                     std::unordered_map<std::string, std::string> &value_tags,
                     std::vector<std::string> &node_tags,
                     std::vector<char> &has_custom_node_tag_override, ComputeContext *ctx,
                     std::vector<std::string> *changed_values) {
  bool changed = false;
  // Applies a single value-tag update, recording the affected value name when it
  // actually changed so incremental callers can re-queue dependent nodes.
  auto note = [&](const std::string &name, bool did) {
    if (did) {
      changed = true;
      if (changed_values != nullptr) {
        changed_values->push_back(name);
      }
    }
  };

  const std::string op_type = node.op_type();
  std::string explicit_output_tag;
  if (op_type == "Shape" || op_type == "Size") {
    explicit_output_tag = "shape";
  } else if (op_type == "Constant") {
    explicit_output_tag = "weight";
  }

  if (node.input().size() >= 2) {
    if (op_type == "Reshape" || op_type == "Expand" || op_type == "Slice") {
      note(node.input(1), TrySetValueTag(value_tags, node.input(1), "shape"));
    } else if (op_type == "Squeeze" || op_type == "Unsqueeze" || op_type == "ReduceSum" ||
               op_type == "ReduceMean" || op_type == "ReduceMax" || op_type == "ReduceMin") {
      note(node.input(1), TrySetValueTag(value_tags, node.input(1), "axes"));
    }
  }
  if (op_type == "Slice") {
    if (node.input().size() > 2) {
      note(node.input(2), TrySetValueTag(value_tags, node.input(2), "shape"));
    }
    if (node.input().size() > 3) {
      note(node.input(3), TrySetValueTag(value_tags, node.input(3), "axes"));
    }
    if (node.input().size() > 4) {
      note(node.input(4), TrySetValueTag(value_tags, node.input(4), "shape"));
    }
  }
  // Custom callbacks run for every node type.
  bool custom_callback_set_node_tag = false;
  if (ctx != nullptr) {
    const auto *custom = ctx->GetCustomValueTagFunction(node.domain(), op_type);
    if (custom != nullptr) {
      // Snapshot before callback to detect whether this callback
      // produced a node-tag override for this node.
      const std::string node_tag_before_callback = node_tags[n];
      ctx->ClearCustomValueTagChangedFlag();
      (*custom)(*ctx, node, n);
      // Only non-empty node tags are treated as callback overrides.
      // The non-empty check remains defensive for future callback-side
      // mutation paths.
      const bool callback_changed_node_tag = node_tags[n] != node_tag_before_callback;
      const bool callback_set_non_empty_node_tag = !node_tags[n].empty();
      if (callback_changed_node_tag && callback_set_non_empty_node_tag) {
        has_custom_node_tag_override[n] = 1;
      }
      custom_callback_set_node_tag = has_custom_node_tag_override[n] != 0 ||
                                     (callback_changed_node_tag && callback_set_non_empty_node_tag);
      if (ctx->ConsumeCustomValueTagChangedFlag()) {
        changed = true;
      }
    }
  }

  std::string current_output_tag;
  bool output_tags_are_consistent = true;
  for (int o = 0; o < node.output().size(); ++o) {
    auto it = value_tags.find(node.output(o));
    if (it != value_tags.end()) {
      if (current_output_tag.empty()) {
        current_output_tag = it->second;
      } else if (current_output_tag != it->second) {
        output_tags_are_consistent = false;
        break;
      }
    }
  }
  if (op_type == "Constant" && output_tags_are_consistent && !current_output_tag.empty()) {
    explicit_output_tag = current_output_tag;
  }

  std::string inherited_tag;
  if (op_type == "Concat") {
    // Concat output tag is determined by examining all inputs:
    //   * if any input carries "weight"                     → "weight" wins
    //   * if all tagged inputs share the same tag          → that tag
    //   * if tagged inputs have different (non-weight) tags → "ambiguous"
    //   * if no input has a known tag                      → no tag
    bool any_weight = false;
    bool has_tag = false;
    bool all_same = true;
    std::string first_known_tag;
    for (int i = 0; i < node.input().size(); ++i) {
      auto it = value_tags.find(node.input(i));
      if (it != value_tags.end() && !it->second.empty()) {
        if (it->second == "weight") {
          any_weight = true;
        }
        if (first_known_tag.empty()) {
          first_known_tag = it->second;
          has_tag = true;
        } else if (first_known_tag != it->second) {
          all_same = false;
        }
      }
    }
    if (any_weight) {
      inherited_tag = "weight";
    } else if (has_tag && all_same) {
      inherited_tag = first_known_tag;
    } else if (has_tag && !all_same) {
      inherited_tag = "ambiguous";
    }
  } else if (!node.input().empty()) {
    auto it = value_tags.find(node.input(0));
    if (it != value_tags.end()) {
      inherited_tag = it->second;
    }
  }
  std::string node_tag;
  if (custom_callback_set_node_tag) {
    node_tag = node_tags[n];
  } else {
    // Without a callback override, built-in explicit/inherited inference
    // remains authoritative over previously inferred tags.
    if (!explicit_output_tag.empty()) {
      node_tag = explicit_output_tag;
    } else if (!inherited_tag.empty()) {
      node_tag = inherited_tag;
    } else {
      node_tag = node_tags[n];
    }
  }
  if (node_tags[n] != node_tag) {
    node_tags[n] = node_tag;
    changed = true;
  }
  if (!node_tag.empty()) {
    for (int o = 0; o < node.output().size(); ++o) {
      note(node.output(o), TrySetValueTag(value_tags, node.output(o), node_tag));
    }
  }

  const std::vector<int> backward_inputs = BackwardTagInputIndices(node, current_output_tag);
  if (!backward_inputs.empty()) {
    if (output_tags_are_consistent && !current_output_tag.empty()) {
      for (int idx : backward_inputs) {
        if (idx >= 0 && idx < node.input().size()) {
          note(node.input(idx), TrySetValueTag(value_tags, node.input(idx), current_output_tag));
        }
      }
    }
  }
  return changed;
}

void InferNodesTags(const std::vector<const NodeProto *> &nodes,
                    std::unordered_map<std::string, std::string> &value_tags,
                    std::vector<std::string> &node_tags, ComputeContext *ctx) {
  node_tags.assign(nodes.size(), std::string());
  std::vector<char> has_custom_node_tag_override(nodes.size(), 0);
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t n = 0; n < nodes.size(); ++n) {
      changed |= ProcessNodeTags(*nodes[n], n, value_tags, node_tags, has_custom_node_tag_override,
                                 ctx, nullptr);
    }
  }
}

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
  for (int i = 0; i < function.value_info().size(); ++i) {
    auto it = value_tags.find(function.value_info(i).name());
    if (it != value_tags.end()) {
      SetMetadataValue(*function.mutable_value_info(static_cast<std::size_t>(i)),
                       kValueTagMetadataKey, it->second);
    }
  }
  for (int i = 0; i < function.node().size(); ++i) {
    RecurseSubgraphs(*function.mutable_node(static_cast<std::size_t>(i)));
  }
}

void WriteValueAndNodeTagsToMetadata(ModelProto &model) {
  if (model.has_graph()) {
    WriteValueAndNodeTagsToMetadata(*model.mutable_graph());
  }
}

} // namespace compute
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
