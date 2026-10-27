// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/compute_context.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "onnx_core/compute/constant_info.h"
#include "onnx_core/compute/inplace_reuse.h"
#include "onnx_core/compute/result_lifetime.h"
#include "onnx_core/compute/value_tags.h"
#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_core/symbolic/symbolic_helper.h"
#include "onnx_proto/simple_span.h"

namespace ONNX_LIGHT_NAMESPACE::core::compute {

namespace {

std::string EncodeLifetimeName(const std::string &name, bool encode) {
  if (!encode) {
    return name;
  }
  std::string result;
  for (char c : name) {
    if (c == '%') {
      result += "%25";
    } else if (c == ';') {
      result += "%3B";
    } else {
      result += c;
    }
  }
  return result;
}

void EncodeLifetimeNamesMetadata(NodeProto &node, const char *key) {
  std::string raw;
  for (const auto &entry : node.metadata_props()) {
    if (entry.key() == key) {
      raw = entry.value();
      break;
    }
  }
  if (raw.empty()) {
    return;
  }
  std::string encoded;
  std::size_t start = 0;
  while (start <= raw.size()) {
    const std::size_t end = raw.find(';', start);
    if (!encoded.empty() || start != 0) {
      encoded += ';';
    }
    encoded += EncodeLifetimeName(
        raw.substr(start, end == std::string::npos ? std::string::npos : end - start), true);
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  node.add_metadata(key, encoded);
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

struct FunctionBodyMemoryEstimate {
  expressions::DimType internal_peak = int64_t{0};
  int64_t workspace_peak = 0;
};

using FunctionAttributeMap = std::unordered_map<std::string, const AttributeProto *>;

void BindFunctionAttributes(GraphProto &graph, const FunctionAttributeMap &attributes);

void BindFunctionAttributes(NodeProto &node, const FunctionAttributeMap &attributes) {
  utils::RepeatedProtoField<AttributeProto> bound_attributes;
  for (const AttributeProto &attribute : node.attribute()) {
    AttributeProto bound = utils::CopyWithSharedTensorStorage(attribute);
    if (!attribute.ref_attr_name().empty()) {
      const auto found = attributes.find(attribute.ref_attr_name());
      if (found == attributes.end())
        continue;
      const std::string name = attribute.name();
      bound = utils::CopyWithSharedTensorStorage(*found->second);
      bound.set_name(name);
    } else {
      if (bound.has_g())
        BindFunctionAttributes(*bound.mutable_g(), attributes);
      for (GraphProto &graph : bound.ref_graphs())
        BindFunctionAttributes(graph, attributes);
    }
    bound_attributes.push_back(std::move(bound));
  }
  node.ref_attribute() = std::move(bound_attributes);
}

void BindFunctionAttributes(GraphProto &graph, const FunctionAttributeMap &attributes) {
  for (NodeProto &node : graph.ref_node())
    BindFunctionAttributes(node, attributes);
}

FunctionBodyMemoryEstimate
EstimateFunctionBodyMemory(const NodeProto &node, const ShapesContext &ctx, Device device,
                           std::unordered_set<std::string> &active_functions) {
  const std::string function_key =
      ShapesContext::LocalFunctionKey(node.domain(), node.op_type(), node.overload());
  const FunctionProto *function = ctx.GetLocalFunction(function_key);
  if (function == nullptr || !active_functions.insert(function_key).second)
    return {};
  struct PopFunction {
    std::unordered_set<std::string> &active;
    const std::string &key;
    ~PopFunction() { active.erase(key); }
  } pop_function{active_functions, function_key};

  ShapesContext body_shapes = ctx.MakeChildContext();
  body_shapes.set_events_enabled(false);
  for (const OperatorSetIdProto &opset : function->opset_import())
    body_shapes.SetOpsetVersion(opset.domain(), static_cast<int>(opset.version()));
  FunctionAttributeMap attributes;
  for (const AttributeProto &attribute : function->attribute_proto())
    attributes[attribute.name()] = &attribute;
  for (const AttributeProto &attribute : node.attribute())
    attributes[attribute.name()] = &attribute;
  utils::RepeatedProtoField<NodeProto> bound_nodes;
  for (const NodeProto &inner : function->node()) {
    NodeProto bound = utils::CopyWithSharedTensorStorage(inner);
    BindFunctionAttributes(bound, attributes);
    bound_nodes.push_back(std::move(bound));
  }
  const std::size_t input_count = std::min(function->input().size(), node.input().size());
  for (std::size_t i = 0; i < input_count; ++i) {
    const std::string &actual = node.input(i);
    if (actual.empty())
      continue;
    if (!(ctx.Has(actual) || ctx.HasType(actual) || ctx.HasSequence(actual) ||
          ctx.HasEncodedValue(actual))) {
      return {};
    }
    body_shapes.CopyValueFrom(function->input(i), ctx, actual);
  }
  const auto contains_unbound_attributes = [](const auto &self, const NodeProto &candidate) {
    for (const AttributeProto &attribute : candidate.attribute()) {
      if (!attribute.ref_attr_name().empty())
        return true;
      const auto graph_contains_unbound = [&](const GraphProto &graph) {
        return std::any_of(graph.node().begin(), graph.node().end(),
                           [&](const NodeProto &nested) { return self(self, nested); });
      };
      if ((attribute.has_g() && graph_contains_unbound(attribute.g())) ||
          std::any_of(attribute.graphs().begin(), attribute.graphs().end(), graph_contains_unbound))
        return true;
    }
    return false;
  };
  for (const NodeProto &inner : bound_nodes) {
    if (shapes::ShapeInputsAvailable(body_shapes, inner) &&
        !contains_unbound_attributes(contains_unbound_attributes, inner))
      body_shapes.ComputeShapeNode(inner);
  }
  for (const ValueInfoProto &value_info : function->value_info()) {
    if (!body_shapes.Has(value_info.name().value()) && value_info.has_type() &&
        value_info.type().has_tensor_type() && value_info.type().tensor_type().has_shape()) {
      SymTensor descriptor;
      if (SymTensorFromValueInfo(value_info, descriptor))
        body_shapes.Set(value_info.name().value(), std::move(descriptor));
    }
  }

  std::unordered_set<std::string> formal_inputs(function->input().begin(), function->input().end());
  std::unordered_set<std::string> boundary_outputs(function->output().begin(),
                                                   function->output().end());
  std::unordered_map<std::string, int64_t> producers;
  std::unordered_map<std::string, int64_t> last_uses;
  const int64_t node_count = bound_nodes.size();
  for (std::size_t index = 0; index < bound_nodes.size(); ++index) {
    const NodeProto &inner = bound_nodes[index];
    for (const std::string &input : core::graph::CollectNodeInputs(inner)) {
      if (!input.empty())
        last_uses[input] = static_cast<int64_t>(index);
    }
    for (const std::string &output : inner.output()) {
      if (!output.empty()) {
        producers[output] = static_cast<int64_t>(index);
        last_uses[output] = static_cast<int64_t>(index);
      }
    }
  }

  auto &expression_cache = body_shapes.ExpressionCache();
  std::unordered_map<std::string, std::optional<expressions::DimType>> byte_sizes;
  const auto byte_size = [&](const std::string &name) -> std::optional<expressions::DimType> {
    if (!body_shapes.Has(name))
      return std::nullopt;
    return GetCachedByteSizeExpr(body_shapes, name, byte_sizes, &expression_cache);
  };
  std::vector<expressions::DimSum> allocations_starting(static_cast<std::size_t>(node_count));
  std::vector<expressions::DimSum> allocations_ending(static_cast<std::size_t>(node_count));
  for (const auto &[name, producer] : producers) {
    if (formal_inputs.count(name) != 0 || boundary_outputs.count(name) != 0)
      continue;
    const auto last_use = last_uses.find(name);
    if (last_use == last_uses.end() || last_use->second < producer)
      continue;
    const auto bytes = byte_size(name);
    if (!bytes.has_value())
      continue;
    allocations_starting[static_cast<std::size_t>(producer)].Add(*bytes);
    if (last_use->second < node_count)
      allocations_ending[static_cast<std::size_t>(last_use->second)].Add(*bytes);
  }

  FunctionBodyMemoryEstimate estimate;
  expressions::DimType live = int64_t{0};
  for (std::size_t index = 0; index < bound_nodes.size(); ++index) {
    const NodeProto &inner = bound_nodes[index];
    std::vector<SymShape> input_shapes;
    input_shapes.reserve(inner.input().size());
    for (const std::string &input : inner.input()) {
      if (!input.empty() && body_shapes.Has(input))
        input_shapes.push_back(body_shapes.Get(input).Shape());
      else
        input_shapes.emplace_back();
    }
    int64_t workspace =
        shapes::ComputePeakMemory(inner.domain(), inner.op_type(), device, input_shapes);
    const FunctionBodyMemoryEstimate nested =
        EstimateFunctionBodyMemory(inner, body_shapes, device, active_functions);
    workspace = std::max(workspace, nested.workspace_peak);
    live = expressions::dim_add(live, allocations_starting[index].Build(&expression_cache),
                                &expression_cache);
    const expressions::DimType peak =
        expressions::dim_add(live, nested.internal_peak, &expression_cache);
    estimate.internal_peak = expressions::dim_max(estimate.internal_peak, peak, &expression_cache);
    live = expressions::dim_sub(live, allocations_ending[index].Build(&expression_cache),
                                &expression_cache);
    estimate.workspace_peak = std::max(estimate.workspace_peak, workspace);
  }
  return estimate;
}

FunctionBodyMemoryEstimate EstimateFunctionBodyMemory(const NodeProto &node,
                                                      const ShapesContext &ctx, Device device,
                                                      const NodeMaterializerFn &materialize = {}) {
  std::unordered_set<std::string> active_functions;
  if (materialize &&
      std::any_of(node.attribute().begin(), node.attribute().end(),
                  [](const AttributeProto &attribute) {
                    const std::string &name = attribute.name();
                    return name.size() > 4 && name.compare(name.size() - 4, 4, "_ref") == 0 &&
                           (attribute.type() == AttributeProto::STRING ||
                            attribute.type() == AttributeProto::STRINGS);
                  }) &&
      ctx.GetLocalFunction(ShapesContext::LocalFunctionKey(node.domain(), node.op_type(),
                                                           node.overload())) != nullptr) {
    NodeProto resolved = utils::CopyWithSharedTensorStorage(node);
    materialize(resolved);
    return EstimateFunctionBodyMemory(resolved, ctx, device, active_functions);
  }
  return EstimateFunctionBodyMemory(node, ctx, device, active_functions);
}

void AddFunctionBodyMemory(NodeMemoryProfile &profile, const FunctionBodyMemoryEstimate &estimate,
                           expressions::SimplifiedExpressionCache *cache) {
  const expressions::DimType &additional = estimate.internal_peak;
  if (expressions::is_zero_dim(additional))
    return;
  NodeMemoryProfileScalar(profile, kNodeMemoryAlreadyAllocatedBytesKey) = expressions::dim_add(
      NodeMemoryProfileScalar(profile, kNodeMemoryAlreadyAllocatedBytesKey), additional, cache);
  NodeMemoryProfileScalar(profile, kNodeMemoryTotalBytesKey) = expressions::dim_add(
      NodeMemoryProfileScalar(profile, kNodeMemoryTotalBytesKey), additional, cache);
  TaggedMemory &intermediates = NodeMemoryProfileBucket(profile, kNodeMemoryIntermediatesKey);
  intermediates[""] = expressions::dim_add(
      intermediates.contains("") ? intermediates.at("") : expressions::DimType{int64_t{0}},
      additional, cache);
}

} // namespace

// ── ComputeContext method implementations ────────────────────────────────────

void ComputeContext::Clear() noexcept {
  value_tags_.clear();
  node_tags_.clear();
  constant_values_.clear();
  node_constant_.clear();
  reuse_.clear();
  release_after_.clear();
  not_used_after_.clear();
  release_after_shape_tagged_.clear();
  memory_.clear();
  peak_memory_.clear();
  shapes_.Clear();
  node_tag_custom_override_.clear();
  tag_producer_node_.clear();
  tag_consumers_.clear();
  incr_producer_.clear();
  incr_last_use_.clear();
  incr_keep_.clear();
  incr_graph_inputs_.clear();
  incr_graph_initializers_.clear();
  incr_graph_outputs_.clear();
  incr_node_references_.clear();
  incr_byte_size_expr_cache_.clear();
  incr_allocation_bytes_.clear();
  incr_memory_alive_.clear();
  incr_memory_alive_before_refresh_.clear();
  incr_memory_refresh_from_ = static_cast<std::size_t>(-1);
  incr_memory_refresh_prepared_ = false;
}

void ComputeContext::ResetMetadata() noexcept {
  value_tags_.clear();
  node_tags_.clear();
  constant_values_.clear();
  node_constant_.clear();
  reuse_.clear();
  release_after_.clear();
  not_used_after_.clear();
  release_after_shape_tagged_.clear();
  memory_.clear();
  peak_memory_.clear();
  custom_value_tags_changed_ = false;
  node_tag_custom_override_.clear();
  tag_producer_node_.clear();
  tag_consumers_.clear();
  incr_producer_.clear();
  incr_last_use_.clear();
  incr_keep_.clear();
  incr_graph_inputs_.clear();
  incr_graph_initializers_.clear();
  incr_graph_outputs_.clear();
  incr_node_references_.clear();
  incr_byte_size_expr_cache_.clear();
  incr_allocation_bytes_.clear();
  incr_memory_alive_.clear();
  incr_memory_alive_before_refresh_.clear();
  incr_memory_refresh_from_ = static_cast<std::size_t>(-1);
  incr_memory_refresh_prepared_ = false;
}

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
  CollectFunctionSeedTags(function, value_tags_);
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

ComputeContext::ValueTagPreservationGuard::ValueTagPreservationGuard(ComputeContext &context)
    : context_(context) {
  EXT_ENFORCE_INVALID(!context_.new_rewrite_tags_.has_value(),
                      "Value-tag preservation scopes must not overlap.");
  context_.new_rewrite_tags_.emplace();
}

ComputeContext::ValueTagPreservationGuard::~ValueTagPreservationGuard() {
  context_.new_rewrite_tags_.reset();
}

bool ComputeContext::TrySetValueTag(const std::string &name, const std::string &tag) {
  if (IsValueTagPreserved(name)) {
    return false;
  }
  const bool changed =
      ::ONNX_LIGHT_NAMESPACE::core::compute::TrySetValueTag(value_tags_, name, tag);
  if (changed) {
    if (new_rewrite_tags_.has_value()) {
      new_rewrite_tags_->insert(name);
    }
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
  TrySetValueTag(name, tag);
}

void ComputeContext::SeedValueTag(const std::string &name, const std::string &tag,
                                  const utils::RepeatedProtoField<NodeProto> &nodes) {
  if (!TrySetValueTag(name, tag)) {
    return;
  }
  std::vector<int> seeds;
  auto producer = tag_producer_node_.find(name);
  if (producer != tag_producer_node_.end()) {
    seeds.push_back(producer->second);
  }
  auto consumers = tag_consumers_.find(name);
  if (consumers != tag_consumers_.end()) {
    seeds.insert(seeds.end(), consumers->second.begin(), consumers->second.end());
  }
  PropagateNodeTags(nodes, seeds, {name});
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

void ComputeContext::ComputeConstants(const GraphProto &graph) {
  auto constants = InferConstants(graph);
  constant_values_ = std::move(constants.first);
  node_constant_ = std::move(constants.second);
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
  PropagateNodeTags(nodes, {static_cast<int>(node_index)});
}

void ComputeContext::PropagateNodeTags(const utils::RepeatedProtoField<NodeProto> &nodes,
                                       const std::vector<int> &seeds,
                                       const std::vector<std::string> &initial_changes) {
  std::deque<int> queue;
  std::unordered_set<int> queued;
  std::unordered_set<std::size_t> changed_release_nodes;
  const auto track_release = [&](const std::string &name) {
    const auto use = incr_last_use_.find(name);
    if (use != incr_last_use_.end() && use->second >= 0 &&
        static_cast<std::size_t>(use->second) < release_after_shape_tagged_.size()) {
      changed_release_nodes.insert(static_cast<std::size_t>(use->second));
    }
  };
  const auto mark_tag_memory_refresh = [&](const std::string &name) {
    const auto allocation_producer = incr_producer_.find(name);
    if (allocation_producer != incr_producer_.end()) {
      MarkMemoryRefresh(allocation_producer->second < 0
                            ? 0
                            : static_cast<std::size_t>(allocation_producer->second));
      return;
    }
    const auto tag_producer = tag_producer_node_.find(name);
    MarkMemoryRefresh(tag_producer == tag_producer_node_.end() || tag_producer->second < 0
                          ? 0
                          : static_cast<std::size_t>(tag_producer->second));
  };
  for (const std::string &name : initial_changes) {
    track_release(name);
    mark_tag_memory_refresh(name);
  }
  for (int seed : seeds) {
    if (queued.insert(seed).second) {
      queue.push_back(seed);
    }
  }
  std::vector<std::string> changed;
  while (!queue.empty()) {
    const int n = queue.front();
    queue.pop_front();
    queued.erase(n);
    changed.clear();
    ProcessNodeTags(nodes[static_cast<std::size_t>(n)], static_cast<std::size_t>(n), value_tags_,
                    node_tags_, node_tag_custom_override_, this, &changed);
    for (const std::string &value : changed) {
      track_release(value);
      mark_tag_memory_refresh(value);
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
  for (std::size_t index : changed_release_nodes) {
    auto &shape_releases = release_after_shape_tagged_[index];
    shape_releases.clear();
    for (const std::string &name : release_after_[index]) {
      const auto tag = value_tags_.find(name);
      if (tag != value_tags_.end() && tag->second == "shape") {
        shape_releases.push_back(name);
      }
    }
  }
}

void ComputeContext::RefreshValueTags(const utils::RepeatedProtoField<NodeProto> &nodes,
                                      const std::unordered_set<std::string> &changed_values) {
  std::unordered_set<int> seeds;
  for (const auto &name : changed_values) {
    const auto producer = tag_producer_node_.find(name);
    if (producer != tag_producer_node_.end()) {
      seeds.insert(producer->second);
    }
    const auto consumers = tag_consumers_.find(name);
    if (consumers != tag_consumers_.end()) {
      seeds.insert(consumers->second.begin(), consumers->second.end());
    }
  }
  PropagateNodeTags(nodes, {seeds.begin(), seeds.end()},
                    {changed_values.begin(), changed_values.end()});
}

void ComputeContext::RestoreTagState(const TagState &state,
                                     const utils::RepeatedProtoField<NodeProto> &nodes,
                                     std::size_t node_offset,
                                     const NodeMaterializerFn &materialize) {
  EXT_ENFORCE_INVALID(
      node_offset <= nodes.size() && state.nodes.size() == nodes.size() - node_offset &&
          state.custom_overrides.size() == state.nodes.size() &&
          node_tags_.size() == nodes.size() && node_tag_custom_override_.size() == nodes.size(),
      "RestoreTagState: inconsistent node coordinates.");
  std::vector<std::string> changed;
  for (const auto &[name, tag] : value_tags_) {
    const auto found = state.values.find(name);
    if (found == state.values.end() || found->second != tag)
      changed.push_back(name);
  }
  for (const auto &[name, tag] : state.values)
    if (value_tags_.count(name) == 0)
      changed.push_back(name);
  value_tags_ = state.values;
  for (std::size_t index = 0; index < state.nodes.size(); ++index) {
    node_tags_[node_offset + index] = state.nodes[index];
    node_tag_custom_override_[node_offset + index] = state.custom_overrides[index];
  }
  // Empty seeds update tag-dependent bookkeeping without re-inferring captured tags.
  PropagateNodeTags(nodes, {}, changed);
  RefreshIncrementalMemory(nodes, shapes_, Device::kUndefined, materialize);
}

void ComputeContext::SeedReuseInput(const std::string &name, bool is_graph_input,
                                    bool is_initializer, bool allow_input_overwrite) {
  EXT_ENFORCE_INVALID(!allow_input_overwrite,
                      "SeedReuseInput: graph inputs cannot be overwritten or shadowed.");
  if (name.empty()) {
    return;
  }
  if (is_graph_input) {
    incr_graph_inputs_.insert(name);
    incr_keep_.insert(name);
  }
  if (is_initializer) {
    incr_graph_initializers_.insert(name);
    incr_keep_.insert(name);
  }
  MarkMemoryRefresh(0);
}

namespace {

// Removes every occurrence of ``value`` from ``list``.
void EraseValueFromList(std::vector<std::string> &list, const std::string &value) {
  list.erase(std::remove(list.begin(), list.end(), value), list.end());
}

} // namespace

void ComputeContext::RefreshNodeReuse(const utils::RepeatedProtoField<NodeProto> &nodes,
                                      std::size_t index, const ShapesContext &ctx) {
  auto &expression_cache = ctx.ExpressionCache();
  reuse_[index] =
      ComputeSingleNodeReuse(nodes[index], static_cast<int>(index), ctx, incr_keep_, incr_producer_,
                             incr_last_use_, incr_byte_size_expr_cache_, expression_cache);
}

void ComputeContext::SeedReuseOutput(const std::string &name,
                                     const utils::RepeatedProtoField<NodeProto> &nodes,
                                     const ShapesContext &ctx) {
  if (name.empty()) {
    return;
  }
  const auto producer = incr_producer_.find(name);
  MarkMemoryRefresh(producer == incr_producer_.end() || producer->second < 0
                        ? 0
                        : static_cast<std::size_t>(producer->second));
  PrepareMemoryRefresh(nodes, nodes.size());
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
    if (idx < reuse_.size()) {
      RefreshNodeReuse(nodes, idx, ctx);
    }
  }
}

void ComputeContext::AppendNodeReuse(const utils::RepeatedProtoField<NodeProto> &nodes,
                                     std::size_t node_index, const ShapesContext &ctx,
                                     const std::vector<std::string> &referenced) {
  const NodeProto &node = nodes[node_index];
  const int i = static_cast<int>(node_index);

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
    if (incr_keep_.count(name) == 0) {
      if (it != incr_last_use_.end() && it->second >= 0) {
        MarkMemoryRefresh(static_cast<std::size_t>(it->second));
      } else {
        const auto producer = incr_producer_.find(name);
        if (producer != incr_producer_.end()) {
          MarkMemoryRefresh(producer->second < 0 ? 0 : static_cast<std::size_t>(producer->second));
        }
      }
    }
  }
  MarkMemoryRefresh(node_index);
  PrepareMemoryRefresh(nodes, node_index);
  for (const std::string &name : referenced) {
    if (name.empty()) {
      continue;
    }
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
    if (incr_keep_.count(entry.first) == 0) {
      RefreshNodeReuse(nodes, p, ctx);
    }
  }

  // In-place reuse matches for this node, from the running lifetime maps. This
  // is the very same per-node core the whole-graph analysis runs.
  auto &expression_cache = ctx.ExpressionCache();
  std::vector<InPlaceReuse> matches =
      ComputeSingleNodeReuse(node, i, ctx, incr_keep_, incr_producer_, incr_last_use_,
                             incr_byte_size_expr_cache_, expression_cache);

  // Register this node's outputs as producers (first producer wins), matching
  // ComputeResultLifetimeInfo.
  for (int o = 0; o < node.output_size(); ++o) {
    const std::string name = node.output(o);
    if (!name.empty()) {
      incr_producer_.emplace(name, i);
      incr_last_use_.emplace(name, i);
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
  for (const std::string &name : node.output()) {
    if (!name.empty() && incr_keep_.count(name) == 0 && incr_producer_.at(name) == i &&
        incr_last_use_.at(name) == i) {
      release_after.push_back(name);
    }
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
  // reuse_ (Size()). GraphBuilder refreshes complete memory profiles before the
  // mutating method returns.
  reuse_.push_back(std::move(matches));
  release_after_.push_back(std::move(release_after));
  not_used_after_.push_back(std::move(not_used_after));
  release_after_shape_tagged_.push_back(std::move(release_after_shape_tagged));
  memory_.push_back(MakeEmptyNodeMemoryProfile());
  incr_node_references_.push_back(referenced);
}

void ComputeContext::RefreshNode(const utils::RepeatedProtoField<NodeProto> &nodes,
                                 std::size_t node_index, const ShapesContext &ctx,
                                 const std::vector<std::string> &references, Device device,
                                 const NodeMaterializerFn &materialize) {
  EXT_ENFORCE_INVALID(node_index < nodes.size() && node_index < incr_node_references_.size(),
                      "RefreshNode: node index is out of bounds.");
  const auto &outputs = nodes[node_index].output();
  RefreshNodes(nodes, {{node_index, references}}, ctx, {outputs.begin(), outputs.end()}, device,
               materialize);
}

void ComputeContext::RefreshNodes(const utils::RepeatedProtoField<NodeProto> &nodes,
                                  const std::map<std::size_t, std::vector<std::string>> &references,
                                  const ShapesContext &ctx,
                                  const std::unordered_set<std::string> &changed_values,
                                  Device device, const NodeMaterializerFn &materialize) {
  std::size_t refresh_from = nodes.size();
  std::unordered_set<std::string> changed_references;
  for (const auto &[index, current] : references) {
    EXT_ENFORCE_INVALID(index < nodes.size() && index < incr_node_references_.size(),
                        "RefreshNodes: node index is out of bounds.");
    const auto &previous = incr_node_references_[index];
    if (previous == current) {
      continue;
    }
    refresh_from = std::min(refresh_from, index);
    const std::unordered_set<std::string> old_names(previous.begin(), previous.end());
    const std::unordered_set<std::string> new_names(current.begin(), current.end());
    for (const auto &name : old_names) {
      if (new_names.count(name) == 0) {
        changed_references.insert(name);
      }
    }
    for (const auto &name : new_names) {
      if (old_names.count(name) == 0) {
        changed_references.insert(name);
      }
    }
  }
  for (const auto &name : changed_values) {
    if (name.empty()) {
      continue;
    }
    incr_byte_size_expr_cache_.erase(name);
    const auto producer = incr_producer_.find(name);
    refresh_from = std::min(refresh_from, producer == incr_producer_.end() || producer->second < 0
                                              ? 0
                                              : static_cast<std::size_t>(producer->second));
  }

  const auto updated_references = [&](std::size_t index) -> const std::vector<std::string> & {
    const auto updated = references.find(index);
    return updated == references.end() ? incr_node_references_[index] : updated->second;
  };
  std::unordered_map<std::string, int> updated_last_use;
  for (const std::string &name : changed_references) {
    if (name.empty()) {
      continue;
    }
    const auto old = incr_last_use_.find(name);
    if (old != incr_last_use_.end() && old->second >= 0) {
      refresh_from = std::min(refresh_from, static_cast<std::size_t>(old->second));
    }
    const auto value_producer = incr_producer_.find(name);
    int last = value_producer == incr_producer_.end() ? -1 : value_producer->second;
    for (std::size_t index = 0; index < incr_node_references_.size(); ++index) {
      const auto &node_references = updated_references(index);
      if (std::find(node_references.begin(), node_references.end(), name) !=
          node_references.end()) {
        last = static_cast<int>(index);
      }
    }
    updated_last_use.emplace(name, last);
    if (last >= 0) {
      refresh_from = std::min(refresh_from, static_cast<std::size_t>(last));
    }
    const auto producer = incr_producer_.find(name);
    if (producer != incr_producer_.end() && producer->second >= 0) {
      refresh_from = std::min(refresh_from, static_cast<std::size_t>(producer->second));
    }
  }
  if (refresh_from == nodes.size()) {
    return;
  }
  MarkMemoryRefresh(refresh_from);
  PrepareMemoryRefresh(nodes, nodes.size());
  for (const auto &name : changed_values) {
    incr_allocation_bytes_.erase(name);
  }
  for (const auto &[index, current] : references) {
    if (incr_node_references_[index] != current) {
      incr_node_references_[index] = current;
    }
  }

  for (const auto &[name, last] : updated_last_use) {
    const auto previous = incr_last_use_.find(name);
    if (previous != incr_last_use_.end() && previous->second >= 0) {
      const std::size_t index = static_cast<std::size_t>(previous->second);
      EraseValueFromList(release_after_[index], name);
      EraseValueFromList(not_used_after_[index], name);
      EraseValueFromList(release_after_shape_tagged_[index], name);
    }
    if (last < 0) {
      incr_last_use_.erase(name);
      continue;
    }
    incr_last_use_[name] = last;
    const std::size_t index = static_cast<std::size_t>(last);
    if (incr_graph_inputs_.count(name) != 0 || incr_graph_initializers_.count(name) != 0) {
      not_used_after_[index].push_back(name);
    }
    const auto producer = incr_producer_.find(name);
    if (incr_keep_.count(name) == 0 && producer != incr_producer_.end() && producer->second >= 0 &&
        producer->second <= last) {
      release_after_[index].push_back(name);
      const auto tag = value_tags_.find(name);
      if (tag != value_tags_.end() && tag->second == "shape") {
        release_after_shape_tagged_[index].push_back(name);
      }
    }
  }

  std::unordered_set<std::size_t> reclassified_last_uses;
  for (const auto &name : changed_values) {
    const auto last = incr_last_use_.find(name);
    if (last != incr_last_use_.end() && last->second >= 0 &&
        (incr_graph_inputs_.count(name) != 0 || incr_graph_initializers_.count(name) != 0)) {
      const auto index = static_cast<std::size_t>(last->second);
      reclassified_last_uses.insert(index);
      EraseValueFromList(release_after_[index], name);
      EraseValueFromList(release_after_shape_tagged_[index], name);
    }
  }
  for (const auto index : reclassified_last_uses) {
    not_used_after_[index].clear();
    for (const auto &name : incr_node_references_[index]) {
      const auto last = incr_last_use_.find(name);
      if (last != incr_last_use_.end() && last->second == static_cast<int>(index) &&
          (incr_graph_inputs_.count(name) != 0 || incr_graph_initializers_.count(name) != 0)) {
        not_used_after_[index].push_back(name);
      }
    }
  }

  for (std::size_t index = refresh_from; index < nodes.size(); ++index) {
    RefreshNodeReuse(nodes, index, ctx);
    AppendNodePeakMemory(nodes[index], index, device, materialize);
  }
  RefreshIncrementalMemory(nodes, ctx, device, materialize);
}

void ComputeContext::EraseValue(const std::string &name) {
  if (incr_graph_inputs_.count(name) != 0) {
    incr_graph_initializers_.erase(name);
    return;
  }
  value_tags_.erase(name);
  constant_values_.erase(name);
  incr_graph_initializers_.erase(name);
  if (incr_graph_outputs_.count(name) == 0) {
    incr_keep_.erase(name);
  }
  incr_producer_.erase(name);
  tag_producer_node_.erase(name);
  incr_byte_size_expr_cache_.erase(name);
  incr_allocation_bytes_.erase(name);
  incr_memory_alive_.erase(name);
  incr_memory_alive_before_refresh_.erase(name);
  MarkMemoryRefresh(0);
}

void ComputeContext::UpdateNodes(const utils::RepeatedProtoField<NodeProto> &nodes,
                                 const std::vector<std::size_t> &old_indices,
                                 const std::vector<std::vector<std::string>> &references,
                                 const std::unordered_set<std::string> &changed_values,
                                 Device device, const NodeMaterializerFn &materialize) {
  const std::size_t old_count = Size();
  EXT_ENFORCE_INVALID(old_indices.size() == nodes.size() && references.size() == nodes.size(),
                      "UpdateNodes: inconsistent node coordinates.");
  EXT_ENFORCE_INVALID(nodes.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                      "UpdateNodes: too many nodes for incremental indices.");
  std::vector<int> new_indices(old_count, -1);
  for (std::size_t index = 0; index < old_indices.size(); ++index) {
    EXT_ENFORCE_INVALID(old_indices[index] <= old_count, "UpdateNodes: invalid old node index.");
    if (old_indices[index] < old_count) {
      EXT_ENFORCE_INVALID(new_indices[old_indices[index]] == -1,
                          "UpdateNodes: duplicate old node index.");
      new_indices[old_indices[index]] = static_cast<int>(index);
    }
  }
  std::unordered_set<std::string> changed_references;
  for (std::size_t index = 0; index < old_count; ++index) {
    if (new_indices[index] != static_cast<int>(index)) {
      changed_references.insert(incr_node_references_[index].begin(),
                                incr_node_references_[index].end());
      if (new_indices[index] >= 0) {
        for (const std::string &output :
             nodes[static_cast<std::size_t>(new_indices[index])].output()) {
          if (!output.empty()) {
            changed_references.insert(output);
          }
        }
      }
    }
  }
  const auto remap = [&](auto &values) {
    using Values = std::decay_t<decltype(values)>;
    Values updated(nodes.size());
    for (std::size_t index = 0; index < old_indices.size(); ++index) {
      if (old_indices[index] < values.size()) {
        updated[index] = std::move(values[old_indices[index]]);
      }
    }
    values = std::move(updated);
  };
  remap(node_tags_);
  remap(node_tag_custom_override_);
  remap(node_constant_);
  remap(reuse_);
  remap(release_after_);
  remap(not_used_after_);
  remap(release_after_shape_tagged_);
  remap(memory_);
  remap(peak_memory_);
  remap(incr_node_references_);
  const auto remap_producers = [&](auto &producers) {
    for (auto it = producers.begin(); it != producers.end();) {
      if (it->second >= 0) {
        const int updated = new_indices.at(static_cast<std::size_t>(it->second));
        if (updated < 0) {
          it = producers.erase(it);
          continue;
        }
        it->second = updated;
      }
      ++it;
    }
  };
  remap_producers(incr_producer_);
  remap_producers(tag_producer_node_);
  remap_producers(incr_last_use_);
  for (auto &[name, consumers] : tag_consumers_) {
    std::vector<int> updated;
    for (int index : consumers) {
      if (new_indices.at(static_cast<std::size_t>(index)) >= 0) {
        updated.push_back(new_indices[static_cast<std::size_t>(index)]);
      }
    }
    consumers = std::move(updated);
  }
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    if (old_indices[index] == old_count || incr_node_references_[index] != references[index]) {
      changed_references.insert(incr_node_references_[index].begin(),
                                incr_node_references_[index].end());
      changed_references.insert(references[index].begin(), references[index].end());
      for (const auto &name : incr_node_references_[index]) {
        auto &consumers = tag_consumers_[name];
        consumers.erase(std::remove(consumers.begin(), consumers.end(), static_cast<int>(index)),
                        consumers.end());
      }
      for (const auto &input : nodes[index].input()) {
        if (!input.empty()) {
          tag_consumers_[input.value()].push_back(static_cast<int>(index));
        }
      }
    }
    if (old_indices[index] == old_count ||
        std::any_of(nodes[index].output().begin(), nodes[index].output().end(),
                    [&](const auto &name) { return changed_values.count(name) != 0; })) {
      node_tags_[index].clear();
      node_tag_custom_override_[index] = 0;
    }
    for (const std::string &output : nodes[index].output()) {
      if (!output.empty()) {
        incr_producer_[output] = static_cast<int>(index);
        tag_producer_node_[output] = static_cast<int>(index);
        if (old_indices[index] == old_count || changed_values.count(output) != 0) {
          changed_references.insert(output);
        }
      }
    }
    incr_node_references_[index] = references[index];
  }
  for (const auto &name : changed_values) {
    constant_values_.erase(name);
    incr_byte_size_expr_cache_.erase(name);
    incr_allocation_bytes_.erase(name);
  }
  for (const auto &name : changed_references) {
    incr_last_use_.erase(name);
  }
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    for (const std::string &output : nodes[index].output()) {
      if (!output.empty() && changed_references.count(output) != 0) {
        incr_last_use_.emplace(output, static_cast<int>(index));
      }
    }
    for (const auto &name : references[index]) {
      if (changed_references.count(name) != 0) {
        incr_last_use_[name] = static_cast<int>(index);
      }
    }
  }
  for (const auto &name : changed_values) {
    // Retained names keep their existing tags, including inferred input tags.
    if (incr_producer_.count(name) == 0 && incr_graph_inputs_.count(name) == 0 &&
        incr_graph_initializers_.count(name) == 0 && incr_last_use_.count(name) == 0) {
      value_tags_.erase(name);
    }
  }
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    if (old_indices[index] != old_count &&
        std::none_of(references[index].begin(), references[index].end(),
                     [&](const auto &name) { return changed_references.count(name) != 0; }) &&
        std::none_of(
            nodes[index].output().begin(), nodes[index].output().end(),
            [&](const std::string &name) { return changed_references.count(name) != 0; })) {
      continue;
    }
    release_after_[index].clear();
    not_used_after_[index].clear();
    release_after_shape_tagged_[index].clear();
    std::vector<std::string> release_candidates = references[index];
    for (const std::string &output : nodes[index].output()) {
      if (!output.empty()) {
        release_candidates.push_back(output);
      }
    }
    for (const auto &name : release_candidates) {
      const auto last = incr_last_use_.find(name);
      if (last == incr_last_use_.end() || last->second != static_cast<int>(index)) {
        continue;
      }
      if (incr_graph_inputs_.count(name) != 0 || incr_graph_initializers_.count(name) != 0) {
        not_used_after_[index].push_back(name);
      }
      const auto producer = incr_producer_.find(name);
      if (incr_keep_.count(name) == 0 && producer != incr_producer_.end() &&
          producer->second >= 0 && producer->second <= last->second) {
        release_after_[index].push_back(name);
        const auto tag = value_tags_.find(name);
        if (tag != value_tags_.end() && tag->second == "shape") {
          release_after_shape_tagged_[index].push_back(name);
        }
      }
    }
  }
  shapes_.ReindexSubgraphContexts(new_indices);
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    if (old_indices[index] != index ||
        std::any_of(references[index].begin(), references[index].end(),
                    [&](const auto &name) { return changed_references.count(name) != 0; }) ||
        std::any_of(nodes[index].output().begin(), nodes[index].output().end(),
                    [&](const std::string &name) { return changed_references.count(name) != 0; })) {
      RefreshNodeReuse(nodes, index, shapes_);
      AppendNodePeakMemory(nodes[index], index, device, materialize);
      MarkMemoryRefresh(index);
    }
  }
}

void ComputeContext::MarkMemoryRefresh(std::size_t node_index) {
  if (node_index < incr_memory_refresh_from_) {
    incr_memory_refresh_from_ = node_index;
    incr_memory_refresh_prepared_ = false;
  }
}

void ComputeContext::PrepareMemoryRefresh(const utils::RepeatedProtoField<NodeProto> &nodes,
                                          std::size_t end_exclusive) {
  if (incr_memory_refresh_from_ == static_cast<std::size_t>(-1) || incr_memory_refresh_prepared_) {
    return;
  }
  incr_memory_alive_before_refresh_ = incr_memory_alive_;
  if (incr_memory_refresh_from_ == 0) {
    incr_memory_alive_before_refresh_.clear();
    incr_memory_refresh_prepared_ = true;
    return;
  }
  for (std::size_t position = end_exclusive; position-- > incr_memory_refresh_from_;) {
    const NodeProto &node = nodes[position];
    if (position < release_after_.size()) {
      for (const std::string &name : release_after_[position]) {
        if (incr_allocation_bytes_.count(name) != 0) {
          incr_memory_alive_before_refresh_.insert(name);
        }
      }
    }
    for (const std::string &output : node.output()) {
      if (!output.empty()) {
        incr_memory_alive_before_refresh_.erase(output);
      }
    }
  }
  incr_memory_refresh_prepared_ = true;
}

void ComputeContext::RefreshIncrementalMemory(const utils::RepeatedProtoField<NodeProto> &nodes,
                                              const ShapesContext &ctx, Device device,
                                              const NodeMaterializerFn &materialize) {
  if (incr_memory_refresh_from_ == static_cast<std::size_t>(-1)) {
    return;
  }
  PrepareMemoryRefresh(nodes, nodes.size());
  const std::size_t start = std::min(incr_memory_refresh_from_, nodes.size());
  std::unordered_set<std::string> alive = incr_memory_alive_before_refresh_;
  auto &expression_cache = ctx.ExpressionCache();
  static const std::optional<expressions::DimType> missing_allocation;
  const auto allocation_bytes =
      [&](const std::string &name) -> const std::optional<expressions::DimType> & {
    if (!ctx.Has(name)) {
      return missing_allocation;
    }
    return GetCachedByteSizeExpr(ctx, name, incr_byte_size_expr_cache_, &expression_cache);
  };
  if (start == 0) {
    alive.clear();
    for (const std::string &name : incr_graph_initializers_) {
      const auto &bytes = allocation_bytes(name);
      if (bytes.has_value()) {
        incr_allocation_bytes_[name] = expressions::simplify_dim_type(*bytes, &expression_cache);
        alive.insert(name);
      }
    }
    for (const std::string &name : incr_graph_inputs_) {
      if (alive.count(name) != 0) {
        continue;
      }
      const auto &bytes = allocation_bytes(name);
      if (bytes.has_value()) {
        incr_allocation_bytes_[name] = expressions::simplify_dim_type(*bytes, &expression_cache);
        alive.insert(name);
      }
    }
  }

  if (memory_.size() < nodes.size()) {
    memory_.resize(nodes.size(), MakeEmptyNodeMemoryProfile());
  }
  for (std::size_t index = start; index < nodes.size(); ++index) {
    NodeMemoryProfile profile = MakeEmptyNodeMemoryProfile();
    expressions::DimSum already_sum;
    std::map<ShapeTag, expressions::DimSum> inputs_bucket;
    std::map<ShapeTag, expressions::DimSum> initializers_bucket;
    std::map<ShapeTag, expressions::DimSum> intermediates_bucket;
    for (const std::string &name : alive) {
      const auto allocation = incr_allocation_bytes_.find(name);
      if (allocation == incr_allocation_bytes_.end()) {
        continue;
      }
      const expressions::DimType &bytes = allocation->second;
      already_sum.Add(bytes);
      if (expressions::is_zero_dim(bytes)) {
        continue;
      }
      auto &bucket = incr_graph_initializers_.count(name) != 0 ? initializers_bucket
                     : incr_graph_inputs_.count(name) != 0     ? inputs_bucket
                                                               : intermediates_bucket;
      bucket[ValueTag(value_tags_, name)].Add(bytes);
    }

    const NodeProto &node = nodes[index];
    std::unordered_map<int, int> reuse_by_output;
    if (index < reuse_.size()) {
      for (const InPlaceReuse &reuse : reuse_[index]) {
        reuse_by_output[static_cast<int>(reuse.output_index)] = static_cast<int>(reuse.input_index);
      }
    }
    expressions::DimSum output_sum;
    std::map<ShapeTag, expressions::DimSum> outputs_bucket;
    std::unordered_set<std::string> outputs_to_add;
    for (int output_index = 0; output_index < node.output_size(); ++output_index) {
      const std::string output = node.output(output_index);
      if (output.empty()) {
        continue;
      }
      auto reused = reuse_by_output.find(output_index);
      if (reused != reuse_by_output.end()) {
        const std::string input = node.input(reused->second);
        auto allocation = incr_allocation_bytes_.find(input);
        if (allocation != incr_allocation_bytes_.end()) {
          incr_allocation_bytes_[output] = allocation->second;
        }
      } else {
        const auto &bytes = allocation_bytes(output);
        if (bytes.has_value()) {
          const expressions::DimType simplified =
              expressions::simplify_dim_type(*bytes, &expression_cache);
          incr_allocation_bytes_[output] = simplified;
          output_sum.Add(simplified);
          if (!expressions::is_zero_dim(simplified)) {
            outputs_bucket[ValueTag(value_tags_, output)].Add(simplified);
          }
        }
      }
      const auto last_use = incr_last_use_.find(output);
      if (incr_graph_outputs_.count(output) != 0 ||
          (last_use != incr_last_use_.end() && last_use->second > static_cast<int>(index))) {
        outputs_to_add.insert(output);
      }
    }

    const expressions::DimType already_allocated = already_sum.Build(&expression_cache);
    const expressions::DimType output_allocation = output_sum.Build(&expression_cache);
    NodeMemoryProfileScalar(profile, kNodeMemoryAlreadyAllocatedBytesKey) = already_allocated;
    NodeMemoryProfileScalar(profile, kNodeMemoryOutputAllocationBytesKey) = output_allocation;
    NodeMemoryProfileScalar(profile, kNodeMemoryTotalBytesKey) =
        expressions::dim_add(already_allocated, output_allocation, &expression_cache);
    FillTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryInputsKey), inputs_bucket,
                     &expression_cache);
    FillTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryInitializersKey),
                     initializers_bucket, &expression_cache);
    FillTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryIntermediatesKey),
                     intermediates_bucket, &expression_cache);
    FillTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryOutputsKey), outputs_bucket,
                     &expression_cache);
    AddFunctionBodyMemory(profile, EstimateFunctionBodyMemory(node, ctx, device, materialize),
                          &expression_cache);
    memory_[index] = std::move(profile);

    if (index < release_after_.size()) {
      for (const std::string &name : release_after_[index]) {
        alive.erase(name);
      }
    }
    for (const std::string &output : node.output()) {
      alive.erase(output);
    }
    alive.insert(outputs_to_add.begin(), outputs_to_add.end());
  }
  incr_memory_alive_ = std::move(alive);
  incr_memory_alive_before_refresh_.clear();
  incr_memory_refresh_from_ = static_cast<std::size_t>(-1);
  incr_memory_refresh_prepared_ = false;
}

void ComputeContext::AppendNodePeakMemory(const NodeProto &node, std::size_t node_index,
                                          Device device, const NodeMaterializerFn &materialize) {
  if (peak_memory_.size() <= node_index) {
    peak_memory_.resize(node_index + 1, 0);
  }
  std::vector<SymShape> input_shapes;
  input_shapes.reserve(node.input().size());
  for (const auto &input_name : node.input()) {
    if (!input_name.empty() && shapes_.Has(input_name)) {
      input_shapes.push_back(shapes_.Get(input_name).Shape());
    } else {
      input_shapes.emplace_back();
    }
  }
  peak_memory_[node_index] =
      shapes::ComputePeakMemory(node.domain(), node.op_type(), device, input_shapes);
  peak_memory_[node_index] =
      std::max(peak_memory_[node_index],
               EstimateFunctionBodyMemory(node, shapes_, device, materialize).workspace_peak);
}

void ComputeContext::ComputeInPlaceReuseGraph(
    const GraphProto &graph, const ShapesContext &ctx, bool allow_input_overwrite,
    const std::unordered_map<std::string, std::string> &value_tags) {
  ComputeInPlaceReuseGraphImpl(graph, ctx, allow_input_overwrite, value_tags);
}

void ComputeContext::ComputeInPlaceReuseGraph(
    const FunctionProto &function, const ShapesContext &ctx, bool allow_input_overwrite,
    const std::unordered_map<std::string, std::string> &value_tags) {
  ComputeInPlaceReuseGraphImpl(function, ctx, allow_input_overwrite, value_tags);
}

template <typename GraphOrFunction>
void ComputeContext::ComputeInPlaceReuseGraphImpl(
    const GraphOrFunction &graph, const ShapesContext &ctx, bool allow_input_overwrite,
    const std::unordered_map<std::string, std::string> &value_tags) {
  const int num_nodes = graph.node().size();
  std::vector<NodeMemoryProfile> memory(static_cast<std::size_t>(num_nodes),
                                        MakeEmptyNodeMemoryProfile());
  auto &expression_cache = ctx.ExpressionCache();
  std::unordered_map<std::string, std::optional<expressions::DimType>> byte_size_expr_cache;
  incr_allocation_bytes_.clear();

  ResultLifetimeInfo lifetime = ComputeResultLifetimeInfo(graph, allow_input_overwrite);
  const std::unordered_set<std::string> &graph_inputs = lifetime.graph_inputs;
  const std::unordered_set<std::string> &graph_outputs = lifetime.graph_outputs;
  const std::unordered_map<std::string, int> &last_use = lifetime.last_use;
  incr_node_references_.clear();
  incr_node_references_.reserve(graph.node().size());
  for (const NodeProto &node : graph.node()) {
    incr_node_references_.push_back(core::graph::CollectNodeInputs(node));
  }
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
  for (const std::string &name : lifetime.graph_initializers) {
    if (name.empty() || !ctx.Has(name)) {
      continue;
    }
    const std::optional<expressions::DimType> &bytes =
        GetCachedByteSizeExpr(ctx, name, byte_size_expr_cache, &expression_cache);
    if (!bytes.has_value()) {
      continue;
    }
    alive[name] = LiveAllocation{expressions::simplify_dim_type(*bytes, &expression_cache),
                                 MemoryValueSource::kInitializer, ValueTag(value_tags, name)};
  }
  for (const std::string &name : graph_inputs) {
    if (name.empty() || alive.find(name) != alive.end() || !ctx.Has(name)) {
      continue;
    }
    const std::optional<expressions::DimType> &bytes =
        GetCachedByteSizeExpr(ctx, name, byte_size_expr_cache, &expression_cache);
    if (!bytes.has_value()) {
      continue;
    }
    alive[name] = LiveAllocation{expressions::simplify_dim_type(*bytes, &expression_cache),
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
          GetCachedByteSizeExpr(ctx, out_name, byte_size_expr_cache, &expression_cache);
      if (!out_bytes.has_value()) {
        continue;
      }
      const expressions::DimType out_simplified =
          expressions::simplify_dim_type(*out_bytes, &expression_cache);
      output_sum.Add(out_simplified);
      if (!expressions::is_zero_dim(out_simplified)) {
        outputs_bucket[ValueTag(value_tags, out_name)].Add(out_simplified);
      }
    }

    const expressions::DimType already_allocated = already_sum.Build(&expression_cache);
    const expressions::DimType output_allocation = output_sum.Build(&expression_cache);
    NodeMemoryProfileScalar(profile, kNodeMemoryAlreadyAllocatedBytesKey) = already_allocated;
    NodeMemoryProfileScalar(profile, kNodeMemoryOutputAllocationBytesKey) = output_allocation;
    NodeMemoryProfileScalar(profile, kNodeMemoryTotalBytesKey) =
        expressions::dim_add(already_allocated, output_allocation, &expression_cache);
    FillTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryInputsKey), inputs_bucket,
                     &expression_cache);
    FillTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryInitializersKey),
                     initializers_bucket, &expression_cache);
    FillTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryIntermediatesKey),
                     intermediates_bucket, &expression_cache);
    FillTaggedMemory(NodeMemoryProfileBucket(profile, kNodeMemoryOutputsKey), outputs_bucket,
                     &expression_cache);
    AddFunctionBodyMemory(profile, EstimateFunctionBodyMemory(node, ctx, Device::kUndefined),
                          &expression_cache);
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
            GetCachedByteSizeExpr(ctx, out_name, byte_size_expr_cache, &expression_cache);
        if (!out_bytes.has_value()) {
          continue;
        }
        alloc_bytes = expressions::simplify_dim_type(*out_bytes, &expression_cache);
      }
      incr_allocation_bytes_[out_name] = alloc_bytes;
      const auto use_it = last_use.find(out_name);
      const bool survives_after_node = graph_outputs.find(out_name) != graph_outputs.end() ||
                                       (use_it != last_use.end() && use_it->second > i);
      if (!survives_after_node) {
        continue;
      }
      outputs_to_add.emplace(
          out_name,
          LiveAllocation{expressions::simplify_dim_type(alloc_bytes, &expression_cache),
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
  incr_memory_alive_.clear();
  for (const auto &entry : alive) {
    incr_memory_alive_.insert(entry.first);
    incr_allocation_bytes_[entry.first] = entry.second.bytes;
  }
  incr_memory_alive_before_refresh_.clear();
  incr_memory_refresh_from_ = static_cast<std::size_t>(-1);
  incr_memory_refresh_prepared_ = false;

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

void ComputeContext::WriteToMetadata(GraphProto &graph) const { WriteToMetadataImpl(graph); }

template <typename GraphOrFunction>
void ComputeContext::WriteToMetadataImpl(GraphOrFunction &graph) const {
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
    const auto needs_encoding = [](const std::vector<std::string> &names) {
      return std::any_of(names.begin(), names.end(), [](const std::string &name) {
        return name.find_first_of("%;") != std::string::npos;
      });
    };
    bool encode_names = needs_encoding(release_after_[i]) || needs_encoding(not_used_after_[i]) ||
                        (has_shape_tagged && needs_encoding(release_after_shape_tagged_[i]));
    bool has_lifetime_encoding = false;
    for (const auto &entry : node.metadata_props()) {
      if (entry.key() == kLifetimeNamesEncodingMetadataKey) {
        encode_names = true;
        has_lifetime_encoding = true;
      }
    }
    if (encode_names) {
      if (!has_lifetime_encoding) {
        if (release_after_[i].empty()) {
          EncodeLifetimeNamesMetadata(node, kReleaseAfterMetadataKey);
        }
        if (not_used_after_[i].empty()) {
          EncodeLifetimeNamesMetadata(node, kNotUsedAfterMetadataKey);
        }
        if (!has_shape_tagged) {
          EncodeLifetimeNamesMetadata(node, kReleaseAfterShapeTagMetadataKey);
        }
      }
      node.add_metadata(kLifetimeNamesEncodingMetadataKey, "percent");
    }
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
        value << EncodeLifetimeName(release_after_[i][j], encode_names);
      }
      node.add_metadata(kReleaseAfterMetadataKey, value.str());
    }
    if (!not_used_after_[i].empty()) {
      std::ostringstream value;
      for (std::size_t j = 0; j < not_used_after_[i].size(); ++j) {
        if (j != 0) {
          value << ";";
        }
        value << EncodeLifetimeName(not_used_after_[i][j], encode_names);
      }
      node.add_metadata(kNotUsedAfterMetadataKey, value.str());
    }
    if (has_shape_tagged) {
      std::ostringstream value;
      for (std::size_t j = 0; j < release_after_shape_tagged_[i].size(); ++j) {
        if (j != 0) {
          value << ";";
        }
        value << EncodeLifetimeName(release_after_shape_tagged_[i][j], encode_names);
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
  return ComputePeakMemoryNodes(graph.node(), device);
}

const std::vector<int64_t> &ComputeContext::ComputePeakMemory(const FunctionProto &function,
                                                              Device device) {
  return ComputePeakMemoryNodes(function.node(), device);
}

const std::vector<int64_t> &
ComputeContext::ComputePeakMemoryNodes(const utils::RepeatedProtoField<NodeProto> &nodes,
                                       Device device) {
  peak_memory_.assign(static_cast<std::size_t>(nodes.size()), 0);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const NodeProto &node = nodes[i];
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
    peak_memory_[static_cast<std::size_t>(i)] =
        std::max(peak_memory_[static_cast<std::size_t>(i)],
                 EstimateFunctionBodyMemory(node, shapes_, device).workspace_peak);
  }
  return peak_memory_;
}

void ComputeContext::Compute(const GraphProto &graph, Device device, bool allow_input_overwrite) {
  ComputeShapes(graph);
  const auto tags = ComputeValueAndNodeTags(graph);
  ComputeConstants(graph);
  ComputeInPlaceReuseGraph(graph, shapes_, allow_input_overwrite, tags.first);
  ComputePeakMemory(graph, device);
}

void ComputeContext::Compute(const ModelProto &model, Device device, bool allow_input_overwrite,
                             bool prefill_with_value_info_output) {
  ComputeShapes(model, prefill_with_value_info_output);
  const GraphProto &graph = model.graph();
  const auto tags = ComputeValueAndNodeTags(graph);
  ComputeConstants(graph);
  ComputeInPlaceReuseGraph(graph, shapes_, allow_input_overwrite, tags.first);
  ComputePeakMemory(graph, device);
}

void ComputeContext::WriteToGraph(GraphProto &graph) const {
  shapes_.ApplyInferredShapesToGraph(graph);
  WriteToMetadata(graph);
  WritePeakMemoryToNodes(*graph.mutable_node());
}

void ComputeContext::WriteToFunction(FunctionProto &function) const {
  shapes_.ApplyInferredShapesToFunction(function);
  WriteToMetadataImpl(function);
  WritePeakMemoryToNodes(*function.mutable_node());
}

void ComputeContext::WritePeakMemoryToNodes(utils::RepeatedProtoField<NodeProto> &nodes) const {
  if (peak_memory_.size() == static_cast<std::size_t>(nodes.size())) {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      const int64_t peak = peak_memory_[static_cast<std::size_t>(i)];
      if (peak > 0) {
        nodes[i].add_metadata(kNodePeakMemoryMetadataKey, std::to_string(peak));
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
