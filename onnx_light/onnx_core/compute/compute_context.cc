// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/compute_context.h"

#include <algorithm>
#include <deque>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "onnx_core/compute/constant_info.h"
#include "onnx_core/compute/inplace_reuse.h"
#include "onnx_core/compute/result_lifetime.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/symbolic/symbolic_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::compute {

namespace {

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

TaggedMemory &NodeMemoryProfileBucket(NodeMemoryProfile &profile, const std::string &key) {
  auto it = profile.find(key);
  if (it == profile.end()) {
    it = profile.emplace(key, TaggedMemory{}).first;
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
  for (std::size_t i = 0; i < graph.node().size(); ++i) {
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
  for (std::size_t i = 0; i < function.node().size(); ++i) {
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

void ComputeContext::SeedValueTag(const std::string &name, const std::string &tag) {
  ::ONNX_LIGHT_NAMESPACE::core::compute::TrySetValueTag(value_tags_, name, tag);
}

void ComputeContext::SeedConstant(const std::string &name) {
  if (!name.empty()) {
    constant_values_.insert(name);
  }
}

void ComputeContext::AppendNodeConstant(const NodeProto &node, std::size_t node_index) {
  if (node_constant_.size() <= node_index) {
    node_constant_.resize(node_index + 1, ConstantInfo::kNotConstant);
  }
  const bool is_constant =
      ::ONNX_LIGHT_NAMESPACE::core::compute::IsNodeConstant(node, constant_values_);
  node_constant_[node_index] = is_constant ? ConstantInfo::kConstant : ConstantInfo::kNotConstant;
  if (is_constant) {
    for (std::size_t o = 0; o < node.output().size(); ++o) {
      const std::string &out = node.output(static_cast<std::size_t>(o));
      if (!out.empty()) {
        constant_values_.insert(out);
      }
    }
  }
}

void ComputeContext::AppendNodeTags(const utils::RepeatedProtoField<NodeProto> &nodes,
                                    std::size_t node_index) {
  if (node_tags_.size() <= node_index) {
    node_tags_.resize(node_index + 1);
  }
  if (node_tag_custom_override_.size() <= node_index) {
    node_tag_custom_override_.resize(node_index + 1, 0);
  }
  const NodeProto &new_node = nodes[node_index];
  // Register the new node's producer/consumer edges so tag changes flowing out
  // of it (forward) or back into its inputs (backward) can re-queue the exact
  // set of dependent nodes without scanning the whole graph.
  for (std::size_t i = 0; i < new_node.input().size(); ++i) {
    const std::string &in = new_node.input(static_cast<std::size_t>(i));
    if (!in.empty()) {
      tag_consumers_[in].push_back(static_cast<int>(node_index));
    }
  }
  for (std::size_t o = 0; o < new_node.output().size(); ++o) {
    const std::string &out = new_node.output(static_cast<std::size_t>(o));
    if (!out.empty()) {
      tag_producer_node_.emplace(out, static_cast<int>(node_index));
    }
  }

  // Monotone worklist: process the new node, then any node that produces or
  // consumes a value whose tag changed, until nothing changes. Tags only ever
  // escalate (weight → shape/axes → ambiguous), so this terminates at the same
  // least fixed point a whole-graph pass would reach.
  std::deque<int> queue;
  std::unordered_set<int> queued;
  queue.push_back(static_cast<int>(node_index));
  queued.insert(static_cast<int>(node_index));
  std::vector<std::string> changed;
  while (!queue.empty()) {
    const int n = queue.front();
    queue.pop_front();
    queued.erase(n);
    changed.clear();
    ProcessNodeTags(nodes[static_cast<std::size_t>(n)], static_cast<std::size_t>(n), value_tags_,
                    node_tags_, node_tag_custom_override_, this, &changed);
    for (const std::string &value : changed) {
      auto producer_it = tag_producer_node_.find(value);
      if (producer_it != tag_producer_node_.end() && producer_it->second != n &&
          queued.insert(producer_it->second).second) {
        queue.push_back(producer_it->second);
      }
      auto consumers_it = tag_consumers_.find(value);
      if (consumers_it != tag_consumers_.end()) {
        for (int consumer : consumers_it->second) {
          if (consumer != n && queued.insert(consumer).second) {
            queue.push_back(consumer);
          }
        }
      }
    }
  }
}

void ComputeContext::SeedReuseInput(const std::string &name, bool is_graph_input,
                                    bool is_initializer, bool allow_input_overwrite) {
  if (name.empty()) {
    return;
  }
  if (is_graph_input) {
    incr_graph_inputs_.insert(name);
    if (allow_input_overwrite) {
      // Available before the first node so it can be reused at its last use.
      incr_producer_[name] = -1;
    } else {
      incr_keep_.insert(name);
    }
  }
  if (is_initializer) {
    incr_graph_initializers_.insert(name);
    incr_keep_.insert(name);
  }
}

namespace {

// Removes every occurrence of ``value`` from ``list``.
void EraseValueFromList(std::vector<std::string> &list, const std::string &value) {
  list.erase(std::remove(list.begin(), list.end(), value), list.end());
}

} // namespace

void ComputeContext::SeedReuseOutput(const std::string &name) {
  if (name.empty()) {
    return;
  }
  incr_keep_.insert(name);
  incr_graph_outputs_.insert(name);
  // A graph output must survive the run, so undo any earlier node's decision to
  // release it (declared graph outputs are usually added after the nodes).
  auto last_use_it = incr_last_use_.find(name);
  if (last_use_it != incr_last_use_.end() && last_use_it->second >= 0) {
    const std::size_t idx = static_cast<std::size_t>(last_use_it->second);
    if (idx < release_after_.size()) {
      EraseValueFromList(release_after_[idx], name);
    }
    if (idx < release_after_shape_tagged_.size()) {
      EraseValueFromList(release_after_shape_tagged_[idx], name);
    }
  }
}

void ComputeContext::AppendNodeReuse(const NodeProto &node, std::size_t node_index,
                                     const ShapesContext &ctx) {
  const int i = static_cast<int>(node_index);
  const std::vector<std::string> referenced =
      ::ONNX_LIGHT_NAMESPACE::core::graph::CollectNodeInputs(node);

  // Advance the last-use of every referenced value to this node, remembering
  // the previous last-use so the earlier node's release bookkeeping can be
  // corrected: a value used again here is no longer releasable back there.
  std::unordered_map<std::string, int> previous_last_use;
  for (const std::string &name : referenced) {
    if (name.empty()) {
      continue;
    }
    auto it = incr_last_use_.find(name);
    previous_last_use.emplace(name, it != incr_last_use_.end() ? it->second : -1);
    incr_last_use_[name] = i;
  }
  for (const auto &entry : previous_last_use) {
    const int prev = entry.second;
    if (prev < 0 || static_cast<std::size_t>(prev) >= release_after_.size()) {
      continue;
    }
    const std::size_t p = static_cast<std::size_t>(prev);
    EraseValueFromList(release_after_[p], entry.first);
    if (p < not_used_after_.size()) {
      EraseValueFromList(not_used_after_[p], entry.first);
    }
    if (p < release_after_shape_tagged_.size()) {
      EraseValueFromList(release_after_shape_tagged_[p], entry.first);
    }
  }

  // In-place reuse matches for this node, from the running lifetime maps. This
  // is the very same per-node core the whole-graph analysis runs.
  std::vector<InPlaceReuse> matches =
      ComputeSingleNodeReuse(node, i, ctx, incr_keep_, incr_producer_, incr_last_use_,
                             incr_byte_size_expr_cache_, incr_simplified_dim_cache_);

  // Register this node's outputs as producers (first producer wins), matching
  // ComputeResultLifetimeInfo.
  for (int o = 0; o < node.output_size(); ++o) {
    const std::string name = node.output(o);
    if (!name.empty()) {
      incr_producer_.emplace(name, i);
    }
  }

  // Release-after / not-used-after for this node (mirrors the second pass of
  // ComputeResultLifetimeInfo, restricted to the current node).
  std::vector<std::string> release_after;
  std::vector<std::string> not_used_after;
  for (const std::string &name : referenced) {
    if (name.empty()) {
      continue;
    }
    auto use_it = incr_last_use_.find(name);
    if (use_it == incr_last_use_.end() || use_it->second != i) {
      continue;
    }
    if (incr_graph_inputs_.count(name) || incr_graph_initializers_.count(name)) {
      not_used_after.push_back(name);
    }
    if (incr_keep_.count(name)) {
      continue;
    }
    auto prod_it = incr_producer_.find(name);
    if (prod_it == incr_producer_.end() || prod_it->second < 0 || prod_it->second >= i) {
      continue;
    }
    release_after.push_back(name);
  }

  // Shape-tagged subset of the release list, from the tags computed so far.
  std::vector<std::string> release_after_shape_tagged;
  for (const std::string &name : release_after) {
    auto tag_it = value_tags_.find(name);
    if (tag_it != value_tags_.end() && tag_it->second == "shape") {
      release_after_shape_tagged.push_back(name);
    }
  }

  // Append exactly one entry to every per-node vector so they stay aligned with
  // reuse_ (Size()). The per-node memory profile is only meaningful once the
  // whole graph is known, so it is left empty here and recomputed by the
  // finalizers' whole-graph ComputeInPlaceReuseGraph pass.
  reuse_.push_back(std::move(matches));
  release_after_.push_back(std::move(release_after));
  not_used_after_.push_back(std::move(not_used_after));
  release_after_shape_tagged_.push_back(std::move(release_after_shape_tagged));
  memory_.push_back(MakeEmptyNodeMemoryProfile());
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
  for (std::size_t i = 0; i < graph.initializer().size(); ++i) {
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
  for (std::size_t i = 0; i < graph.input().size(); ++i) {
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
  for (std::size_t i = 0; i < function.opset_import().size(); ++i) {
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
  for (std::size_t i = 0; i < graph.node().size(); ++i) {
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
    for (std::size_t i = 0; i < graph.node().size(); ++i) {
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

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
