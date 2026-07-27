// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/compute_context.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "onnx_core/compute/inplace_reuse.h"
#include "onnx_core/compute/result_lifetime.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/symbolic/symbolic_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace compute {

namespace {

// ── Helpers shared by ComputeValueAndNodeTags overloads ─────────────────────

std::string ReadMetadataValueFromProps(const utils::RepeatedField<StringStringEntryProto> &props,
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

// Sets ``name`` to ``tag`` (after normalization) and returns whether the map
// content changed (new key or updated tag value).
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

void InferNodesTags(const std::vector<const NodeProto *> &nodes,
                    std::unordered_map<std::string, std::string> &value_tags,
                    std::vector<std::string> &node_tags, ComputeContext *ctx) {
  node_tags.assign(nodes.size(), std::string());
  std::vector<char> has_custom_node_tag_override(nodes.size(), 0);
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t n = 0; n < nodes.size(); ++n) {
      const NodeProto *node = nodes[n];
      const std::string op_type = node->op_type();
      std::string explicit_output_tag;
      if (op_type == "Shape" || op_type == "Size") {
        explicit_output_tag = "shape";
      } else if (op_type == "Constant") {
        explicit_output_tag = "weight";
      }

      if (node->input().size() >= 2) {
        if (op_type == "Reshape" || op_type == "Expand" || op_type == "Slice") {
          changed |= TrySetValueTag(value_tags, node->input(1), "shape");
        } else if (op_type == "Squeeze" || op_type == "Unsqueeze" || op_type == "ReduceSum" ||
                   op_type == "ReduceMean" || op_type == "ReduceMax" || op_type == "ReduceMin") {
          changed |= TrySetValueTag(value_tags, node->input(1), "axes");
        }
      }
      if (op_type == "Slice") {
        if (node->input().size() > 2) {
          changed |= TrySetValueTag(value_tags, node->input(2), "shape");
        }
        if (node->input().size() > 3) {
          changed |= TrySetValueTag(value_tags, node->input(3), "axes");
        }
        if (node->input().size() > 4) {
          changed |= TrySetValueTag(value_tags, node->input(4), "shape");
        }
      }
      // Custom callbacks run for every node type.
      bool custom_callback_set_node_tag = false;
      if (ctx != nullptr) {
        const auto *custom = ctx->GetCustomValueTagFunction(node->domain(), op_type);
        if (custom != nullptr) {
          // Snapshot before callback to detect whether this callback
          // produced a node-tag override for this node.
          const std::string node_tag_before_callback = node_tags[n];
          ctx->ClearCustomValueTagChangedFlag();
          (*custom)(*ctx, *node, n);
          // Only non-empty node tags are treated as callback overrides.
          // The non-empty check remains defensive for future callback-side
          // mutation paths.
          const bool callback_changed_node_tag = node_tags[n] != node_tag_before_callback;
          const bool callback_set_non_empty_node_tag = !node_tags[n].empty();
          if (callback_changed_node_tag && callback_set_non_empty_node_tag) {
            has_custom_node_tag_override[n] = 1;
          }
          custom_callback_set_node_tag =
              has_custom_node_tag_override[n] != 0 ||
              (callback_changed_node_tag && callback_set_non_empty_node_tag);
          if (ctx->ConsumeCustomValueTagChangedFlag()) {
            changed = true;
          }
        }
      }

      std::string current_output_tag;
      bool output_tags_are_consistent = true;
      for (int o = 0; o < node->output().size(); ++o) {
        auto it = value_tags.find(node->output(o));
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
        for (int i = 0; i < node->input().size(); ++i) {
          auto it = value_tags.find(node->input(i));
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
      } else if (!node->input().empty()) {
        auto it = value_tags.find(node->input(0));
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
        for (int o = 0; o < node->output().size(); ++o) {
          changed |= TrySetValueTag(value_tags, node->output(o), node_tag);
        }
      }

      const std::vector<int> backward_inputs = BackwardTagInputIndices(*node, current_output_tag);
      if (!backward_inputs.empty()) {
        if (output_tags_are_consistent && !current_output_tag.empty()) {
          for (int idx : backward_inputs) {
            if (idx >= 0 && idx < node->input().size()) {
              changed |= TrySetValueTag(value_tags, node->input(idx), current_output_tag);
            }
          }
        }
      }
    }
  }
}

// ── Helpers for ComputeInPlaceReuseGraph ─────────────────────────────────────

enum class MemoryValueSource : uint8_t {
  kInput,
  kInitializer,
  kIntermediate,
};

struct LiveAllocation {
  expressions::DimType bytes = int64_t{0};
  MemoryValueSource source = MemoryValueSource::kIntermediate;
  ShapeTag tag;
};

ShapeTag ValueTag(const std::unordered_map<std::string, std::string> &value_tags,
                  const std::string &name) {
  auto it = value_tags.find(name);
  if (it == value_tags.end()) {
    return {};
  }
  return it->second;
}

// Accessors for the scalar / bucket entries of a NodeMemoryProfile. Only used
// while assembling the per-node profiles below, so they live here rather than
// in the header.
expressions::DimType &NodeMemoryProfileScalar(NodeMemoryProfile &profile, const std::string &key) {
  auto it = profile.find(key);
  if (it == profile.end()) {
    it = profile.emplace(key, expressions::DimType{int64_t{0}}).first;
  }
  return std::get<expressions::DimType>(it->second);
}

const expressions::DimType &NodeMemoryProfileScalar(const NodeMemoryProfile &profile,
                                                    const std::string &key) {
  static const expressions::DimType zero = int64_t{0};
  auto it = profile.find(key);
  if (it == profile.end()) {
    return zero;
  }
  return std::get<expressions::DimType>(it->second);
}

TaggedMemory &NodeMemoryProfileBucket(NodeMemoryProfile &profile, const std::string &key) {
  auto it = profile.find(key);
  if (it == profile.end()) {
    it = profile.emplace(key, TaggedMemory{}).first;
  }
  return std::get<TaggedMemory>(it->second);
}

const TaggedMemory &NodeMemoryProfileBucket(const NodeMemoryProfile &profile,
                                            const std::string &key) {
  static const TaggedMemory empty;
  auto it = profile.find(key);
  if (it == profile.end()) {
    return empty;
  }
  return std::get<TaggedMemory>(it->second);
}

// Accumulates a sum of DimType terms while collapsing string-identical byte-size
// expressions into integer coefficients (and folding integer constants). A node
// memory profile sums the byte sizes of every live allocation, and many of those
// allocations share the exact same symbolic expression (e.g. identically shaped
// weights). expressions::simplify_expression is super-linear in the number of
// additive terms, so simplifying the fully expanded sum once per node dominates
// the cost of profiling large graphs. Because simplify_expression is canonical,
// pre-merging identical terms into "coeff*(term)" before simplifying yields a
// byte-identical result while feeding the simplifier only as many terms as there
// are *distinct* byte-size expressions. This grouping is performed by
// expressions::DimSum (see onnx_core/expressions/dim_sum.h).

void FillTaggedMemory(TaggedMemory &dst, const std::map<ShapeTag, expressions::DimSum> &sums,
                      expressions::SimplifiedExpressionCache *cache) {
  for (const auto &kv : sums) {
    dst[kv.first] = kv.second.Build(cache);
  }
}

NodeMemoryProfile MakeEmptyNodeMemoryProfile() {
  NodeMemoryProfile profile;
  NodeMemoryProfileScalar(profile, kNodeMemoryTotalBytesKey) = int64_t{0};
  NodeMemoryProfileScalar(profile, kNodeMemoryAlreadyAllocatedBytesKey) = int64_t{0};
  NodeMemoryProfileScalar(profile, kNodeMemoryOutputAllocationBytesKey) = int64_t{0};
  NodeMemoryProfileBucket(profile, kNodeMemoryInputsKey);
  NodeMemoryProfileBucket(profile, kNodeMemoryInitializersKey);
  NodeMemoryProfileBucket(profile, kNodeMemoryIntermediatesKey);
  NodeMemoryProfileBucket(profile, kNodeMemoryOutputsKey);
  return profile;
}

} // namespace

// ── ComputeContext method implementations ────────────────────────────────────

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
ComputeContext::ComputeValueAndNodeTags(const GraphProto &graph) {
  value_tags_.clear();
  node_tags_.clear();
  CollectGraphSeedTags(graph, value_tags_);
  std::vector<const NodeProto *> nodes;
  nodes.reserve(graph.node().size());
  for (int i = 0; i < graph.node().size(); ++i) {
    nodes.push_back(&graph.node()[i]);
  }
  InferNodesTags(nodes, value_tags_, node_tags_, this);
  return {value_tags_, node_tags_};
}

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
ComputeContext::ComputeValueAndNodeTags(const FunctionProto &function) {
  value_tags_.clear();
  node_tags_.clear();
  std::vector<const NodeProto *> nodes;
  nodes.reserve(function.node().size());
  for (int i = 0; i < function.node().size(); ++i) {
    nodes.push_back(&function.node()[i]);
  }
  InferNodesTags(nodes, value_tags_, node_tags_, this);
  return {value_tags_, node_tags_};
}

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
ComputeContext::ComputeValueAndNodeTags(const utils::RepeatedProtoField<NodeProto> &nodes) {
  value_tags_.clear();
  node_tags_.clear();
  std::vector<const NodeProto *> ptrs;
  ptrs.reserve(nodes.size());
  for (const NodeProto &node : nodes) {
    ptrs.push_back(&node);
  }
  InferNodesTags(ptrs, value_tags_, node_tags_, this);
  return {value_tags_, node_tags_};
}

std::pair<std::unordered_map<std::string, std::string>, std::vector<std::string>>
ComputeContext::ComputeValueAndNodeTags(const std::vector<NodeProto> &nodes) {
  value_tags_.clear();
  node_tags_.clear();
  std::vector<const NodeProto *> ptrs;
  ptrs.reserve(nodes.size());
  for (const NodeProto &node : nodes) {
    ptrs.push_back(&node);
  }
  InferNodesTags(ptrs, value_tags_, node_tags_, this);
  return {value_tags_, node_tags_};
}

bool ComputeContext::TrySetValueTag(const std::string &name, const std::string &tag) {
  const bool changed =
      ::ONNX_LIGHT_NAMESPACE::core::compute::TrySetValueTag(value_tags_, name, tag);
  if (changed) {
    custom_value_tags_changed_ = true;
  }
  return changed;
}

bool ComputeContext::SetNodeTag(std::size_t node_index, const std::string &tag) {
  if (node_index >= node_tags_.size()) {
    throw std::out_of_range("SetNodeTag: node_index out of bounds.");
  }
  const std::string norm = NormalizeValueTag(tag);
  if (norm.empty()) {
    return false;
  }
  if (node_tags_[node_index] == norm) {
    return false;
  }
  node_tags_[node_index] = norm;
  custom_value_tags_changed_ = true;
  return true;
}

void ComputeContext::ComputeInPlaceReuseGraph(
    const GraphProto &graph, const ShapesContext &ctx, bool allow_input_overwrite,
    const std::unordered_map<std::string, std::string> &value_tags) {
  const int num_nodes = graph.node().size();
  std::vector<NodeMemoryProfile> memory(static_cast<std::size_t>(num_nodes),
                                        MakeEmptyNodeMemoryProfile());
  expressions::SimplifiedExpressionCache simplified_dim_cache;
  std::unordered_map<std::string, std::optional<expressions::DimType>> byte_size_expr_cache;

  ResultLifetimeInfo lifetime = ComputeResultLifetimeInfo(graph, allow_input_overwrite);
  const std::unordered_set<std::string> &graph_inputs = lifetime.graph_inputs;
  const std::unordered_set<std::string> &graph_outputs = lifetime.graph_outputs;
  const std::unordered_map<std::string, int> &last_use = lifetime.last_use;
  if (events_enabled_) {
    for (std::size_t i = 0; i < lifetime.size(); ++i) {
      for (const std::string &name : lifetime[i].release_after) {
        ComputeEvent ev;
        ev.action = ComputeEventAction::kRelease;
        ev.node_index = static_cast<int>(i);
        ev.name = name;
        events_.push_back(std::move(ev));
      }
    }
  }

  std::vector<std::vector<InPlaceReuse>> result = ComputeInPlaceReuseMatches(graph, ctx, lifetime);
  if (events_enabled_) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      for (const InPlaceReuse &reuse : result[i]) {
        ComputeEvent ev;
        ev.action = ComputeEventAction::kInPlace;
        ev.node_index = static_cast<int>(i);
        ev.output_index = static_cast<int>(reuse.output_index);
        ev.input_index = static_cast<int>(reuse.input_index);
        ev.kind = reuse.kind;
        events_.push_back(std::move(ev));
      }
    }
  }

  std::unordered_map<std::string, LiveAllocation> alive;
  for (int i = 0; i < graph.initializer().size(); ++i) {
    const std::string name = graph.initializer()[i].name();
    if (name.empty() || !ctx.Has(name)) {
      continue;
    }
    const std::optional<expressions::DimType> &bytes =
        GetCachedByteSizeExpr(ctx, name, byte_size_expr_cache, &simplified_dim_cache);
    if (!bytes.has_value()) {
      continue;
    }
    alive[name] = LiveAllocation{expressions::simplify_dim_type(*bytes, &simplified_dim_cache),
                                 MemoryValueSource::kInitializer, ValueTag(value_tags, name)};
  }
  for (int i = 0; i < graph.input().size(); ++i) {
    const std::string name = graph.input()[i].name();
    if (name.empty() || alive.find(name) != alive.end() || !ctx.Has(name)) {
      continue;
    }
    const std::optional<expressions::DimType> &bytes =
        GetCachedByteSizeExpr(ctx, name, byte_size_expr_cache, &simplified_dim_cache);
    if (!bytes.has_value()) {
      continue;
    }
    alive[name] = LiveAllocation{expressions::simplify_dim_type(*bytes, &simplified_dim_cache),
                                 MemoryValueSource::kInput, ValueTag(value_tags, name)};
  }

  for (int i = 0; i < num_nodes; ++i) {
    NodeMemoryProfile &profile = memory[static_cast<std::size_t>(i)];

    // Aggregate the byte sizes of every live allocation, grouping identical
    // expressions via DimSum so the symbolic simplifier only sees the distinct
    // terms. This is byte-identical to summing and simplifying each allocation
    // individually but avoids re-simplifying an ever-growing sum for every node.
    expressions::DimSum already_sum;
    std::map<ShapeTag, expressions::DimSum> inputs_bucket;
    std::map<ShapeTag, expressions::DimSum> initializers_bucket;
    std::map<ShapeTag, expressions::DimSum> intermediates_bucket;
    for (const auto &kv : alive) {
      const LiveAllocation &alloc = kv.second;
      already_sum.Add(alloc.bytes);
      if (expressions::is_zero_dim(alloc.bytes)) {
        continue;
      }
      switch (alloc.source) {
      case MemoryValueSource::kInput:
        inputs_bucket[alloc.tag].Add(alloc.bytes);
        break;
      case MemoryValueSource::kInitializer:
        initializers_bucket[alloc.tag].Add(alloc.bytes);
        break;
      case MemoryValueSource::kIntermediate:
        intermediates_bucket[alloc.tag].Add(alloc.bytes);
        break;
      }
    }

    const NodeProto &node = graph.node()[i];
    std::unordered_map<int, int> reuse_by_output;
    for (const InPlaceReuse &reuse : result[static_cast<std::size_t>(i)]) {
      reuse_by_output[static_cast<int>(reuse.output_index)] = static_cast<int>(reuse.input_index);
    }

    expressions::DimSum output_sum;
    std::map<ShapeTag, expressions::DimSum> outputs_bucket;
    for (int o = 0; o < node.output_size(); ++o) {
      if (reuse_by_output.find(o) != reuse_by_output.end()) {
        continue;
      }
      const std::string out_name = node.output(o);
      if (out_name.empty() || !ctx.Has(out_name)) {
        continue;
      }
      const std::optional<expressions::DimType> &out_bytes =
          GetCachedByteSizeExpr(ctx, out_name, byte_size_expr_cache, &simplified_dim_cache);
      if (!out_bytes.has_value()) {
        continue;
      }
      const expressions::DimType out_simplified =
          expressions::simplify_dim_type(*out_bytes, &simplified_dim_cache);
      output_sum.Add(out_simplified);
      if (!expressions::is_zero_dim(out_simplified)) {
        outputs_bucket[ValueTag(value_tags, out_name)].Add(out_simplified);
      }
    }

    const expressions::DimType already_allocated = already_sum.Build(&simplified_dim_cache);
    const expressions::DimType output_allocation = output_sum.Build(&simplified_dim_cache);
    NodeMemoryProfileScalar(profile, kNodeMemoryAlreadyAllocatedBytesKey) = already_allocated;
    NodeMemoryProfileScalar(profile, kNodeMemoryOutputAllocationBytesKey) = output_allocation;
    NodeMemoryProfileScalar(profile, kNodeMemoryTotalBytesKey) = expressions::simplify_dim_type(
        expressions::dim_add(already_allocated, output_allocation), &simplified_dim_cache);
    FillTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryInputsKey), inputs_bucket,
                     &simplified_dim_cache);
    FillTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryInitializersKey),
                     initializers_bucket, &simplified_dim_cache);
    FillTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryIntermediatesKey),
                     intermediates_bucket, &simplified_dim_cache);
    FillTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryOutputsKey), outputs_bucket,
                     &simplified_dim_cache);

    std::unordered_map<std::string, LiveAllocation> outputs_to_add;
    std::unordered_set<std::string> inputs_reused_from_graph_input;
    for (int o = 0; o < node.output_size(); ++o) {
      const std::string out_name = node.output(o);
      if (out_name.empty() || !ctx.Has(out_name)) {
        continue;
      }
      expressions::DimType alloc_bytes = int64_t{0};
      auto reuse_it = reuse_by_output.find(o);
      if (reuse_it != reuse_by_output.end()) {
        const std::string in_name = node.input(reuse_it->second);
        auto alive_it = alive.find(in_name);
        if (alive_it == alive.end()) {
          continue;
        }
        alloc_bytes = alive_it->second.bytes;
        if (graph_inputs.find(in_name) != graph_inputs.end()) {
          inputs_reused_from_graph_input.insert(in_name);
        }
      } else {
        const std::optional<expressions::DimType> &out_bytes =
            GetCachedByteSizeExpr(ctx, out_name, byte_size_expr_cache, &simplified_dim_cache);
        if (!out_bytes.has_value()) {
          continue;
        }
        alloc_bytes = expressions::simplify_dim_type(*out_bytes, &simplified_dim_cache);
      }
      const auto use_it = last_use.find(out_name);
      const bool survives_after_node = graph_outputs.find(out_name) != graph_outputs.end() ||
                                       (use_it != last_use.end() && use_it->second > i);
      if (!survives_after_node) {
        continue;
      }
      outputs_to_add.emplace(
          out_name,
          LiveAllocation{expressions::simplify_dim_type(alloc_bytes, &simplified_dim_cache),
                         MemoryValueSource::kIntermediate, ValueTag(value_tags, out_name)});
    }

    for (const std::string &name : lifetime[static_cast<std::size_t>(i)].release_after) {
      alive.erase(name);
    }
    for (const std::string &name : inputs_reused_from_graph_input) {
      alive.erase(name);
    }
    for (auto &entry : outputs_to_add) {
      alive[entry.first] = std::move(entry.second);
    }
  }

  reuse_ = std::move(result);
  release_after_ = lifetime.MoveReleaseAfter();
  not_used_after_ = lifetime.MoveNotUsedAfter();
  memory_ = std::move(memory);

  // Populate the shape-tagged subset from value_tags (when provided).
  // Only allocate the per-node sub-vectors when value_tags is actually non-empty
  // so that WriteToMetadata can use release_after_shape_tagged_.size() ==
  // reuse_.size() to detect whether shape-tag info was supplied.
  const auto &effective_value_tags = value_tags.empty() ? value_tags_ : value_tags;
  release_after_shape_tagged_.clear();
  if (!effective_value_tags.empty()) {
    const std::size_t n = release_after_.size();
    release_after_shape_tagged_.assign(n, {});
    for (std::size_t i = 0; i < n; ++i) {
      for (const std::string &name : release_after_[i]) {
        auto it = effective_value_tags.find(name);
        if (it != effective_value_tags.end() && it->second == "shape") {
          release_after_shape_tagged_[i].push_back(name);
          if (events_enabled_) {
            ComputeEvent ev;
            ev.action = ComputeEventAction::kReleaseShapeTag;
            ev.node_index = i;
            ev.name = name;
            events_.push_back(std::move(ev));
          }
        }
      }
    }
  }
}

void ComputeContext::WriteToMetadata(GraphProto &graph) const {
  EXT_ENFORCE_INVALID(static_cast<std::size_t>(graph.node().size()) == reuse_.size(),
                      "ComputeContext::WriteToMetadata: graph has ", graph.node().size(),
                      " node(s) but the computed reuse result has ", reuse_.size(),
                      " entry(ies); call WriteToMetadata on the same graph passed to ",
                      "ComputeInPlaceReuseGraph.");
  const bool has_shape_tag_info = release_after_shape_tagged_.size() == reuse_.size();
  for (std::size_t i = 0; i < reuse_.size(); ++i) {
    const bool has_shape_tagged = has_shape_tag_info && !release_after_shape_tagged_[i].empty();
    if (reuse_[i].empty() && release_after_[i].empty() && not_used_after_[i].empty() &&
        !has_shape_tagged) {
      continue;
    }
    NodeProto &node = (*graph.mutable_node())[i];
    if (!reuse_[i].empty()) {
      std::ostringstream value;
      for (std::size_t j = 0; j < reuse_[i].size(); ++j) {
        const InPlaceReuse &r = reuse_[i][j];
        if (j != 0) {
          value << ";";
        }
        value << r.output_index << ":" << r.input_index << ":"
              << (r.kind == InPlaceReuseKind::kEqual ? "equal" : "greater");
      }
      node.add_metadata(kInPlaceReuseMetadataKey, value.str());
    }
    if (!release_after_[i].empty()) {
      std::ostringstream value;
      for (std::size_t j = 0; j < release_after_[i].size(); ++j) {
        if (j != 0) {
          value << ";";
        }
        value << release_after_[i][j];
      }
      node.add_metadata(kReleaseAfterMetadataKey, value.str());
    }
    if (!not_used_after_[i].empty()) {
      std::ostringstream value;
      for (std::size_t j = 0; j < not_used_after_[i].size(); ++j) {
        if (j != 0) {
          value << ";";
        }
        value << not_used_after_[i][j];
      }
      node.add_metadata(kNotUsedAfterMetadataKey, value.str());
    }
    if (has_shape_tagged) {
      std::ostringstream value;
      for (std::size_t j = 0; j < release_after_shape_tagged_[i].size(); ++j) {
        if (j != 0) {
          value << ";";
        }
        value << release_after_shape_tagged_[i][j];
      }
      node.add_metadata(kReleaseAfterShapeTagMetadataKey, value.str());
    }
  }
}

const ShapesContext &ComputeContext::ComputeShapes(const GraphProto &graph) {
  shapes_.Clear();
  shapes_.ComputeShapeGraph(graph);
  return shapes_;
}

const ShapesContext &ComputeContext::ComputeShapes(const ModelProto &model,
                                                   bool prefill_with_value_info_output) {
  shapes_.Clear();
  shapes_.ComputeShapeModel(model, prefill_with_value_info_output);
  return shapes_;
}

const ShapesContext &ComputeContext::ComputeShapes(const FunctionProto &function,
                                                   const InputShapes &input_shapes) {
  shapes_.Clear();
  for (int i = 0; i < function.opset_import().size(); ++i) {
    const OperatorSetIdProto &osi = function.opset_import()[i];
    shapes_.SetOpsetVersion(osi.domain(), static_cast<int>(osi.version()));
  }
  for (const auto &entry : input_shapes) {
    shapes_.Set(entry.first, SymTensor(entry.second));
  }
  shapes_.ComputeShapes(function.node());
  return shapes_;
}

const ShapesContext &
ComputeContext::ComputeShapes(const utils::RepeatedProtoField<NodeProto> &nodes,
                              const InputShapes &input_shapes) {
  shapes_.Clear();
  for (const auto &entry : input_shapes) {
    shapes_.Set(entry.first, SymTensor(entry.second));
  }
  shapes_.ComputeShapes(nodes);
  return shapes_;
}

const std::vector<int64_t> &ComputeContext::ComputePeakMemory(const GraphProto &graph,
                                                              Device device) {
  peak_memory_.assign(static_cast<std::size_t>(graph.node().size()), 0);
  for (int i = 0; i < graph.node().size(); ++i) {
    const NodeProto &node = graph.node()[i];
    std::vector<SymShape> input_shapes;
    input_shapes.reserve(node.input().size());
    for (const auto &input_name : node.input()) {
      if (!input_name.empty() && shapes_.Has(input_name)) {
        input_shapes.push_back(shapes_.Get(input_name).Shape());
      } else {
        input_shapes.emplace_back();
      }
    }
    peak_memory_[static_cast<std::size_t>(i)] =
        shapes::ComputePeakMemory(node.domain(), node.op_type(), device, input_shapes);
  }
  return peak_memory_;
}

void ComputeContext::Compute(const GraphProto &graph, Device device, bool allow_input_overwrite) {
  ComputeShapes(graph);
  const auto tags = ComputeValueAndNodeTags(graph);
  ComputeInPlaceReuseGraph(graph, shapes_, allow_input_overwrite, tags.first);
  ComputePeakMemory(graph, device);
}

void ComputeContext::Compute(const ModelProto &model, Device device, bool allow_input_overwrite,
                             bool prefill_with_value_info_output) {
  ComputeShapes(model, prefill_with_value_info_output);
  const GraphProto &graph = model.graph();
  const auto tags = ComputeValueAndNodeTags(graph);
  ComputeInPlaceReuseGraph(graph, shapes_, allow_input_overwrite, tags.first);
  ComputePeakMemory(graph, device);
}

void ComputeContext::WriteToGraph(GraphProto &graph) const {
  shapes_.ApplyInferredShapesToGraph(graph);
  WriteToMetadata(graph);
  if (peak_memory_.size() == static_cast<std::size_t>(graph.node().size())) {
    for (int i = 0; i < graph.node().size(); ++i) {
      const int64_t peak = peak_memory_[static_cast<std::size_t>(i)];
      if (peak > 0) {
        (*graph.mutable_node())[i].add_metadata(kNodePeakMemoryMetadataKey, std::to_string(peak));
      }
    }
  }
}

void ComputeContext::WriteToModel(ModelProto &model) const { WriteToGraph(*model.mutable_graph()); }

runtime::ExecutionPlan ComputeContext::BuildExecutionPlan(GraphProto &graph) const {
  WriteToGraph(graph);
  return runtime::ExecutionPlan(graph);
}

runtime::ExecutionPlan ComputeContext::BuildExecutionPlan(ModelProto &model) const {
  return BuildExecutionPlan(*model.mutable_graph());
}

} // namespace compute
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
