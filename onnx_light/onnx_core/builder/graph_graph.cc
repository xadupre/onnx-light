// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "onnx_core/builder/pattern_registry.h"
#include "onnx_core/compute/constant_info.h"
#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_lib/common/scoped_resource.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

namespace {

struct PendingReplacement {
  std::size_t position;
  utils::RepeatedProtoField<NodeProto> nodes;
  LocalRewritingProto rewriting;
};

struct AcceptedMatch {
  MatchResult match;
  const NodeProto *candidate;
  std::size_t pattern_index;
};

std::string RewriteDomain(const std::string &domain) {
  return domain.empty() || domain == "onnx.ai" ? "ai.onnx" : domain;
}

using FunctionKey = std::tuple<std::string, std::string, std::string>;

FunctionKey Identity(const FunctionProto &function) {
  return {RewriteDomain(function.domain().value()), function.name().value(),
          function.overload().value()};
}

FunctionKey Identity(const NodeProto &call) {
  return {RewriteDomain(call.domain().value()), call.op_type().value(), call.overload().value()};
}

void MergeFunction(utils::RepeatedProtoField<FunctionProto> &functions,
                   const FunctionProto &function) {
  if (function.name().empty()) {
    throw BuilderError("ApplyRewriting: an added function must have a name.");
  }
  for (const auto &existing : functions) {
    if (Identity(existing) == Identity(function)) {
      if (!existing.Equals(function)) {
        throw BuilderError("ApplyRewriting: conflicting function definition '" +
                           function.name().value() + "'.");
      }
      return;
    }
    const auto runtime_key = [](const FunctionProto &value) {
      return RewriteDomain(value.domain().value()) + ":" + value.name().value() + ":" +
             value.overload().value();
    };
    if (runtime_key(existing) == runtime_key(function)) {
      throw BuilderError("ApplyRewriting: colliding function names '" + existing.name().value() +
                         "' and '" + function.name().value() + "'.");
    }
  }
  functions.push_back(function);
}

void MergeImport(utils::RepeatedProtoField<OperatorSetIdProto> &imports,
                 const OperatorSetIdProto &opset) {
  if (opset.version() <= 0 || opset.version() > std::numeric_limits<int>::max()) {
    throw BuilderError("ApplyRewriting: invalid opset version.");
  }
  for (const auto &existing : imports) {
    if (RewriteDomain(existing.domain().value()) == RewriteDomain(opset.domain().value())) {
      if (existing.version() != opset.version()) {
        throw BuilderError("ApplyRewriting: conflicting opset import for '" +
                           opset.domain().value() + "'.");
      }
      return;
    }
  }
  imports.push_back(opset);
}

NodeProto CaptureInitializerCall(const NodeProto &node, const GraphBuilder &builder,
                                 RewritingProto &collection) {
  FunctionProto function;
  function.set_domain("onnx.light.rewriting");
  NodeProto body = node;
  body.clear_name();
  body.clr_metadata_props();
  std::unordered_map<std::string, std::string> names;
  NodeProto call;
  call.set_domain(function.domain().value());
  for (std::size_t i = 0; i < node.input().size(); ++i) {
    const std::string input = node.input(i).value();
    if (input.empty()) {
      continue;
    }
    const auto [entry, inserted] = names.emplace(input, "input_" + std::to_string(names.size()));
    if (inserted) {
      function.add_input(entry->second);
      call.add_input(input);
    }
    body.ref_input()[i] = entry->second;
  }
  for (std::size_t i = 0; i < node.output().size(); ++i) {
    if (!node.output(i).empty()) {
      const std::string output = "output_" + std::to_string(i);
      function.add_output(output);
      call.add_output(node.output(i).value());
      body.ref_output()[i] = output;
    }
  }
  function.ref_node().push_back(std::move(body));
  const std::string domain = node.domain().value();
  const int version = builder.OpsetVersion(domain);
  if (version <= 0) {
    throw BuilderError("GraphGraph::Optimize: a recipe requires an explicit opset.");
  }
  function.add_opset(domain, version);
  // The unnamed, canonical body allows identical recipes to share one definition.
  for (const auto &existing : collection.added_functions()) {
    FunctionProto candidate = existing;
    candidate.clear_name();
    if (candidate.Equals(function)) {
      call.set_op_type(existing.name().value());
      return call;
    }
  }
  const std::string name =
      "PrepareInitializer_" + std::to_string(collection.added_functions().size());
  if (builder.HasLocalFunction(name) ||
      std::any_of(collection.added_functions().begin(), collection.added_functions().end(),
                  [&](const auto &existing) { return existing.name() == name; })) {
    throw BuilderError("GraphGraph::Optimize: preparation function name collision: '" + name +
                       "'.");
  }
  function.set_name(name);
  call.set_op_type(function.name().value());
  collection.ref_added_functions().push_back(std::move(function));
  OperatorSetIdProto opset;
  opset.set_domain(call.domain().value());
  opset.set_version(1);
  MergeImport(collection.ref_required_opset_imports(), opset);
  return call;
}

using FunctionIndex = std::map<FunctionKey, const FunctionProto *>;

void ValidateFunctionCalls(const NodeProto &node, const FunctionIndex &functions) {
  const auto function = functions.find(Identity(node));
  if (function != functions.end() && (node.input_size() != function->second->input_size() ||
                                      node.output_size() != function->second->output_size())) {
    throw BuilderError("ApplyRewriting: call to function '" + node.op_type().value() +
                       "' has an incompatible signature.");
  }
  for (const auto &attribute : node.attribute()) {
    const auto validate_graph = [&](const GraphProto &graph) {
      for (const auto &child : graph.node()) {
        ValidateFunctionCalls(child, functions);
      }
    };
    if (attribute.has_g()) {
      validate_graph(attribute.g());
    }
    for (const auto &graph : attribute.graphs()) {
      validate_graph(graph);
    }
  }
}

bool IsPreparationPrimitive(const NodeProto &node) {
  if (core::compute::IsNonDeterministicOp(node.domain(), node.op_type()) ||
      node.op_type() == "Dropout" || node.op_type() == "If" || node.op_type() == "Loop" ||
      node.op_type() == "Scan" || node.op_type() == "SequenceMap")
    return false;
  return std::none_of(node.attribute().begin(), node.attribute().end(), [](const auto &attribute) {
    return attribute.has_g() || !attribute.graphs().empty() || !attribute.ref_attr_name().empty();
  });
}

// Validates the complete dependency tree before any preparation code executes.
void ValidatePreparation(const NodeProto &call, const FunctionIndex &functions,
                         std::set<FunctionKey> &active) {
  const auto found = functions.find(Identity(call));
  if (found == functions.end()) {
    throw BuilderError("ApplyRewriting: initializer calls must reference a defined function.");
  }
  const FunctionProto &function = *found->second;
  if (!call.attribute().empty() || !function.attribute().empty() ||
      !function.attribute_proto().empty() || call.input().size() != function.input().size() ||
      call.output().size() != function.output().size()) {
    throw BuilderError("ApplyRewriting: invalid initializer function signature or attributes.");
  }
  if (active.size() >= 64 || !active.insert(Identity(function)).second) {
    throw BuilderError("ApplyRewriting: recursive initializer preparation is not allowed.");
  }
  std::unordered_set<std::string> available;
  for (const auto &input : function.input()) {
    if (input.empty() || !available.insert(input.value()).second) {
      throw BuilderError("ApplyRewriting: invalid initializer function input.");
    }
  }
  utils::RepeatedProtoField<OperatorSetIdProto> imports;
  for (const auto &opset : function.opset_import())
    MergeImport(imports, opset);
  for (const auto &node : function.node()) {
    for (const auto &input : node.input()) {
      if (!input.empty() && available.count(input.value()) == 0) {
        throw BuilderError("ApplyRewriting: initializer function captures an unavailable value.");
      }
    }
    for (const auto &attribute : node.attribute()) {
      if (attribute.has_g() || !attribute.graphs().empty() || !attribute.ref_attr_name().empty()) {
        throw BuilderError(
            "ApplyRewriting: preparation subgraphs and attribute references are unsupported.");
      }
    }
    if (functions.count(Identity(node)) != 0) {
      ValidatePreparation(node, functions, active);
    } else {
      if (!IsPreparationPrimitive(node)) {
        throw BuilderError(
            "ApplyRewriting: initializer preparation must be deterministic and stateless.");
      }
      const auto imported = std::find_if(imports.begin(), imports.end(), [&](const auto &opset) {
        return RewriteDomain(opset.domain().value()) == RewriteDomain(node.domain().value());
      });
      if (imported == imports.end()) {
        throw BuilderError("ApplyRewriting: initializer operation has no opset import.");
      }
      const std::string key = RewriteDomain(node.domain().value()) + ":" + node.op_type().value();
      if (core::runtime::KernelDispatchTable().count(
              key + core::symbolic::DeviceKeySuffix(core::symbolic::Device::kCPU)) == 0 &&
          core::runtime::GlobalCustomKernels().count(key) == 0) {
        throw BuilderError("ApplyRewriting: no implementation for initializer operation '" + key +
                           "'.");
      }
    }
    for (const auto &output : node.output()) {
      if (!output.empty() && !available.insert(output.value()).second) {
        throw BuilderError(
            "ApplyRewriting: initializer function output is defined more than once.");
      }
    }
  }
  for (const auto &output : function.output()) {
    if (output.empty() || available.count(output.value()) == 0) {
      throw BuilderError("ApplyRewriting: initializer function output is undefined.");
    }
  }
  active.erase(Identity(function));
}

std::size_t CheckedPosition(uint64_t position) {
  if (position > std::numeric_limits<std::size_t>::max()) {
    throw BuilderError("ApplyRewriting: position exceeds size_t.");
  }
  return static_cast<std::size_t>(position);
}

void ValidateInputDefault(const utils::RepeatedProtoField<ValueInfoProto> &inputs,
                          const TensorProto &initializer) {
  for (const auto &input : inputs) {
    if (input.name() != initializer.name() || !input.has_type()) {
      continue;
    }
    if (!input.type().has_tensor_type()) {
      throw BuilderError(
          "ApplyRewriting: an initializer cannot supply a non-tensor input default.");
    }
    const auto &declared = input.type().tensor_type();
    if (declared.elem_type() != TensorProto::UNDEFINED &&
        declared.elem_type() != initializer.data_type()) {
      throw BuilderError(
          "ApplyRewriting: initializer default element type differs from its input.");
    }
    if (!declared.has_shape()) {
      continue;
    }
    if (declared.shape().dim().size() != initializer.dims().size()) {
      throw BuilderError("ApplyRewriting: initializer default rank differs from its input.");
    }
    for (std::size_t i = 0; i < initializer.dims().size(); ++i) {
      const auto &dimension = declared.shape().dim(i);
      if (dimension.has_dim_value() && dimension.dim_value() != initializer.dims(i)) {
        throw BuilderError("ApplyRewriting: initializer default dimension differs from its input.");
      }
    }
  }
}

void RecordRenames(LocalRewritingProto &rewrite,
                   const std::unordered_map<std::string, std::string> &renames) {
  std::vector<std::pair<std::string, std::string>> sorted(renames.begin(), renames.end());
  std::sort(sorted.begin(), sorted.end());
  for (const auto &[key, value] : sorted) {
    auto *entry = rewrite.add_value_renames();
    entry->set_key(key);
    entry->set_value(value);
  }
}

// A Fenwick tree counts live entries before a fixed ordering rank.
class LiveRanks {
public:
  explicit LiveRanks(std::size_t size) : counts_(size + 1, 0) {}

  void Update(std::size_t rank, bool added) {
    for (std::size_t i = rank + 1; i < counts_.size(); i += i & (~i + 1)) {
      if (added)
        ++counts_[i];
      else
        --counts_[i];
    }
  }

  std::size_t Before(std::size_t rank) const {
    std::size_t count = 0;
    for (std::size_t i = rank; i != 0; i -= i & (~i + 1))
      count += counts_[i];
    return count;
  }

  std::size_t Find(std::size_t position) const {
    std::size_t rank = 0;
    for (std::size_t step = std::bit_floor(counts_.size() - 1); step != 0; step >>= 1) {
      const std::size_t next = rank + step;
      if (next < counts_.size() && counts_[next] <= position) {
        rank = next;
        position -= counts_[next];
      }
    }
    return rank;
  }

private:
  std::vector<std::size_t> counts_;
};

void OrderPreparationDependencies(std::vector<PendingReplacement> &replacements) {
  std::unordered_map<std::string, std::size_t> owners;
  for (std::size_t i = 0; i < replacements.size(); ++i) {
    const auto &rewrite = replacements[i].rewriting;
    for (const auto &literal : rewrite.added_initializers())
      owners.emplace(literal.name().value(), i);
    for (const auto &call : rewrite.initializer_calls())
      for (const auto &output : call.output())
        owners.emplace(output.value(), i);
  }
  std::vector<std::vector<std::size_t>> consumers(replacements.size());
  std::vector<std::size_t> dependencies(replacements.size(), 0);
  std::set<std::size_t> ready;
  for (std::size_t i = 0; i < replacements.size(); ++i) {
    std::unordered_set<std::size_t> producers;
    for (const auto &call : replacements[i].rewriting.initializer_calls()) {
      for (const auto &input : call.input()) {
        const auto producer = owners.find(input.value());
        if (producer != owners.end() && producer->second != i)
          producers.insert(producer->second);
      }
    }
    dependencies[i] = producers.size();
    for (std::size_t producer : producers)
      consumers[producer].push_back(i);
    if (producers.empty())
      ready.insert(i);
  }
  std::vector<PendingReplacement> ordered;
  ordered.reserve(replacements.size());
  while (!ready.empty()) {
    const std::size_t index = *ready.begin();
    ready.erase(ready.begin());
    ordered.push_back(std::move(replacements[index]));
    for (std::size_t consumer : consumers[index])
      if (--dependencies[consumer] == 0)
        ready.insert(consumer);
  }
  if (ordered.size() != replacements.size())
    throw BuilderError("GraphGraph::Optimize: cyclic initializer preparation dependencies.");
  replacements = std::move(ordered);
}

// Fixed slots reserve room for every batch addition. Independent edits only
// change live counts; dependency adjustments rewrite the affected slots.
class LiveNodeOrder {
public:
  LiveNodeOrder(std::vector<std::shared_ptr<NodeProto>> slots,
                const std::unordered_set<const NodeProto *> &originals)
      : slots_(std::move(slots)), counts_(slots_.size()) {
    for (std::size_t rank = 0; rank < slots_.size(); ++rank) {
      ranks_.emplace(slots_[rank].get(), rank);
      if (originals.count(slots_[rank].get()) != 0)
        Activate(slots_[rank].get());
    }
  }

  std::size_t Rank(const NodeProto *node) const { return ranks_.at(node); }
  std::size_t Position(const NodeProto *node) const { return counts_.Before(Rank(node)); }
  bool Contains(const NodeProto *node) const { return active_.count(node) != 0; }

  void Activate(const NodeProto *node) {
    const auto rank = Rank(node);
    if (!active_.insert(node).second)
      throw BuilderError("GraphGraph::Optimize: duplicate live batch node.");
    entries_.emplace(rank, slots_[rank]);
    counts_.Update(rank, true);
  }

  void Remove(const NodeProto *node) {
    if (active_.erase(node) == 0)
      throw BuilderError("GraphGraph::Optimize: missing live batch node.");
    const auto rank = Rank(node);
    entries_.erase(rank);
    counts_.Update(rank, false);
  }

  void InsertBefore(const std::vector<const NodeProto *> &added, const NodeProto *anchor) {
    if (added.empty())
      return;
    const auto anchor_rank = anchor == nullptr ? slots_.size() : Rank(anchor);
    const auto position = counts_.Before(anchor_rank);
    if (std::all_of(added.begin(), added.end(),
                    [&](const auto *node) { return Position(node) == position; }))
      return;
    std::size_t first = anchor_rank, last = anchor_rank;
    std::unordered_set<const NodeProto *> selected(added.begin(), added.end());
    std::vector<std::shared_ptr<NodeProto>> additions;
    for (const auto *node : added) {
      const auto rank = Rank(node);
      first = std::min(first, rank);
      last = std::max(last, rank + 1);
      additions.push_back(slots_[rank]);
    }
    std::vector<std::shared_ptr<NodeProto>> interval;
    for (std::size_t rank = first; rank < last; ++rank) {
      if (rank == anchor_rank)
        interval.insert(interval.end(), additions.begin(), additions.end());
      if (selected.count(slots_[rank].get()) == 0)
        interval.push_back(slots_[rank]);
    }
    if (anchor_rank == last)
      interval.insert(interval.end(), additions.begin(), additions.end());
    for (std::size_t offset = 0; offset < interval.size(); ++offset) {
      const auto rank = first + offset;
      if (Contains(slots_[rank].get())) {
        entries_.erase(rank);
        counts_.Update(rank, false);
      }
    }
    for (std::size_t offset = 0; offset < interval.size(); ++offset) {
      const auto rank = first + offset;
      slots_[rank] = std::move(interval[offset]);
      ranks_[slots_[rank].get()] = rank;
      if (Contains(slots_[rank].get())) {
        entries_.emplace(rank, slots_[rank]);
        counts_.Update(rank, true);
      }
    }
  }

  std::vector<std::shared_ptr<NodeProto>> Interval(std::size_t first, std::size_t last) const {
    std::vector<std::shared_ptr<NodeProto>> result;
    for (auto it = entries_.lower_bound(first); it != entries_.end() && it->first <= last; ++it)
      result.push_back(it->second);
    return result;
  }

  void ReplaceInterval(std::size_t first, const std::vector<std::shared_ptr<NodeProto>> &nodes) {
    auto it = entries_.lower_bound(first);
    for (const auto &node : nodes) {
      EXT_ENFORCE_INVALID(it != entries_.end(),
                          "GraphGraph::Optimize: replacement interval exceeds the live order.");
      const auto rank = it->first;
      slots_[rank] = node;
      ranks_[node.get()] = rank;
      it->second = node;
      ++it;
    }
  }

  utils::RepeatedProtoField<NodeProto> Materialize() const {
    utils::RepeatedProtoField<NodeProto> result;
    result.reserve(entries_.size());
    for (const auto &[rank, node] : entries_) {
      result.add();
      result.get(result.size() - 1) = node;
    }
    return result;
  }

private:
  std::vector<std::shared_ptr<NodeProto>> slots_;
  std::unordered_map<const NodeProto *, std::size_t> ranks_;
  std::unordered_set<const NodeProto *> active_;
  std::map<std::size_t, std::shared_ptr<NodeProto>> entries_;
  LiveRanks counts_;
};

// Rebases sequential coordinates without materializing the live graph per edit.
utils::RepeatedProtoField<NodeProto> NormalizeBatch(
    std::vector<PendingReplacement> &replacements,
    const utils::RepeatedProtoField<NodeProto> &original_nodes, std::size_t initializer_count,
    const std::unordered_set<std::string> &available,
    const std::function<std::vector<std::string>(const NodeProto &)> &node_references,
    const std::function<std::vector<std::size_t>(
        const utils::RepeatedProtoField<NodeProto> &, const std::unordered_set<std::string> &,
        const std::vector<std::vector<std::string>> &, const std::vector<bool> &)> &insert_nodes) {
  const std::size_t node_count = original_nodes.size();
  std::unordered_map<std::string, const NodeProto *> producers;
  std::unordered_map<std::string, std::vector<const NodeProto *>> consumers;
  std::unordered_map<const NodeProto *, std::vector<std::string>> references;
  const auto index_node = [&](const NodeProto &node) {
    references[&node] = node_references(node);
    for (const auto &input : references.at(&node))
      consumers[input].push_back(&node);
    for (const auto &output : node.output())
      if (!output.empty())
        producers[output.value()] = &node;
  };
  std::vector<uint64_t> initializer_positions;
  std::unordered_map<std::size_t, std::vector<std::shared_ptr<NodeProto>>> insertions;
  std::size_t total_nodes = node_count;
  for (std::size_t i = 0; i < replacements.size(); ++i) {
    const auto &replacement = replacements[i];
    if (replacement.position > node_count ||
        replacement.rewriting.added_nodes().size() >=
            std::numeric_limits<std::size_t>::max() - total_nodes) {
      throw BuilderError("GraphGraph::Optimize: invalid batch insertion size or position.");
    }
    total_nodes += replacement.rewriting.added_nodes().size();
    for (std::size_t added = 0; added < replacement.rewriting.added_nodes().size(); ++added)
      insertions[replacement.position].push_back(
          replacement.rewriting.added_nodes().shared_at(added));
    const auto &positions = replacement.rewriting.added_initializer_positions();
    initializer_positions.insert(initializer_positions.end(), positions.begin(), positions.end());
    const auto &computed = replacement.rewriting.computed_initializer_positions();
    initializer_positions.insert(initializer_positions.end(), computed.begin(), computed.end());
  }
  std::vector<std::shared_ptr<NodeProto>> slots;
  slots.reserve(total_nodes);
  std::unordered_set<const NodeProto *> originals;
  std::set<std::size_t> retained_originals;
  for (std::size_t index = 0; index <= node_count; ++index) {
    const auto additions = insertions.find(index);
    if (additions != insertions.end())
      slots.insert(slots.end(), additions->second.begin(), additions->second.end());
    if (index < node_count) {
      slots.push_back(original_nodes.shared_at(index));
      originals.insert(&original_nodes[index]);
      retained_originals.insert(index);
      index_node(original_nodes[index]);
    }
  }
  LiveNodeOrder nodes(std::move(slots), originals);
  std::vector<bool> removed(node_count, false);
  std::sort(initializer_positions.begin(), initializer_positions.end());
  if (initializer_positions.size() > std::numeric_limits<std::size_t>::max() - initializer_count) {
    throw BuilderError("GraphGraph::Optimize: batch initializer count overflow.");
  }
  LiveRanks live_initializers(initializer_positions.size());
  const auto initializer_rank = [&](uint64_t position) {
    return static_cast<std::size_t>(
        std::lower_bound(initializer_positions.begin(), initializer_positions.end(), position) -
        initializer_positions.begin());
  };
  for (std::size_t i = 0; i < replacements.size(); ++i) {
    auto &rewrite = replacements[i].rewriting;
    const std::size_t added_node_count = rewrite.added_nodes().size();
    std::vector<std::size_t> matched_positions;
    std::vector<const NodeProto *> matched;
    std::unordered_set<std::string> changed_outputs;
    const auto insertion = retained_originals.lower_bound(replacements[i].position);
    const auto *anchor =
        insertion == retained_originals.end() ? nullptr : &original_nodes[*insertion];
    for (uint64_t &position : rewrite.ref_matched_nodes()) {
      const std::size_t original = CheckedPosition(position);
      if (original >= node_count || removed[original]) {
        throw BuilderError("GraphGraph::Optimize: overlapping batch matches.");
      }
      removed[original] = true;
      const NodeProto *node = &original_nodes[original];
      matched.push_back(node);
      position = nodes.Position(node);
      matched_positions.push_back(position);
      retained_originals.erase(original);
      for (const auto &output : node->output())
        if (!output.empty()) {
          producers.erase(output.value());
          changed_outputs.insert(output.value());
        }
    }
    for (const auto *node : matched)
      nodes.Remove(node);
    std::vector<const NodeProto *> additions;
    for (const auto &node : rewrite.added_nodes())
      additions.push_back(&node);
    nodes.InsertBefore(additions, anchor);
    for (const auto *node : additions) {
      nodes.Activate(node);
      index_node(*node);
      for (const auto &output : node->output())
        if (!output.empty())
          changed_outputs.insert(output.value());
    }
    std::size_t first = total_nodes, last = 0;
    const auto check_reference = [&](const NodeProto *consumer, const std::string &input) {
      const auto producer = producers.find(input);
      if (producer == producers.end()) {
        if (available.count(input) == 0)
          throw BuilderError("GraphGraph::Optimize: unknown rewrite input '" + input + "'.");
      } else if (nodes.Rank(producer->second) >= nodes.Rank(consumer)) {
        first = std::min(first, nodes.Rank(consumer));
        last = std::max(last, nodes.Rank(producer->second));
      }
    };
    for (const auto &added : rewrite.added_nodes())
      for (const auto &input : references.at(&added))
        check_reference(&added, input);
    for (const auto &output : changed_outputs)
      for (const auto *consumer : consumers[output])
        if (nodes.Contains(consumer))
          check_reference(consumer, output);
    if (first < total_nodes) {
      utils::RepeatedProtoField<NodeProto> interval;
      std::vector<std::vector<std::string>> interval_references;
      std::vector<bool> interval_additions;
      std::unordered_set<const NodeProto *> added_nodes;
      for (const auto &node : rewrite.added_nodes())
        added_nodes.insert(&node);
      std::unordered_set<std::string> external;
      const auto previous = nodes.Interval(first, last);
      for (const auto &node : previous) {
        interval.add();
        interval.get(interval.size() - 1) = node;
        interval_references.push_back(references.at(node.get()));
        interval_additions.push_back(added_nodes.count(node.get()) != 0);
        for (const auto &input : references.at(node.get())) {
          const auto producer = producers.find(input);
          if (producer == producers.end() || nodes.Rank(producer->second) < first ||
              nodes.Rank(producer->second) > last)
            external.insert(input);
        }
      }
      const auto order = insert_nodes(interval, external, interval_references, interval_additions);
      EXT_ENFORCE_INVALID(order.size() == previous.size(),
                          "GraphGraph::Optimize: incomplete dependency insertion.");
      std::vector<std::shared_ptr<NodeProto>> ordered;
      std::vector<const NodeProto *> retained_before, retained_after;
      for (const auto &node : previous)
        if (added_nodes.count(node.get()) == 0)
          retained_before.push_back(node.get());
      for (const auto index : order) {
        ordered.push_back(previous[index]);
        if (added_nodes.count(previous[index].get()) == 0)
          retained_after.push_back(previous[index].get());
      }
      if (retained_before != retained_after) {
        std::sort(matched_positions.begin(), matched_positions.end());
        std::vector<std::size_t> added_ranks;
        for (const auto *node : additions)
          added_ranks.push_back(nodes.Rank(node));
        std::sort(added_ranks.begin(), added_ranks.end());
        std::size_t removed_before = 0;
        for (const auto &node : previous) {
          if (added_nodes.count(node.get()) != 0)
            continue;
          const auto added_before = static_cast<std::size_t>(
              std::lower_bound(added_ranks.begin(), added_ranks.end(), nodes.Rank(node.get())) -
              added_ranks.begin());
          auto position = nodes.Position(node.get()) - added_before;
          while (removed_before < matched_positions.size() &&
                 matched_positions[removed_before] <= position + removed_before)
            ++removed_before;
          position += removed_before;
          rewrite.add_matched_nodes(position);
          rewrite.ref_added_nodes().add();
          rewrite.ref_added_nodes().get(rewrite.added_nodes().size() - 1) = node;
        }
      }
      nodes.ReplaceInterval(first, ordered);
    }
    rewrite.clr_added_nodes_positions();
    for (const auto &added : rewrite.added_nodes()) {
      rewrite.add_added_nodes_positions(nodes.Position(&added));
    }
    // Cleanup moves live nodes; recorded retained nodes must own independent snapshots.
    for (std::size_t index = added_node_count; index < rewrite.added_nodes().size(); ++index) {
      rewrite.ref_added_nodes().get(index) =
          std::make_shared<NodeProto>(rewrite.added_nodes(index));
    }
    const auto final_positions = rewrite.added_initializer_positions();
    const auto computed_positions = rewrite.computed_initializer_positions();
    for (uint64_t position : final_positions)
      live_initializers.Update(initializer_rank(position), true);
    for (uint64_t position : computed_positions)
      live_initializers.Update(initializer_rank(position), true);
    for (std::size_t j = 0; j < final_positions.size(); ++j)
      rewrite.ref_added_initializer_positions()[j] =
          initializer_count + live_initializers.Before(initializer_rank(final_positions[j]));
    for (std::size_t j = 0; j < computed_positions.size(); ++j)
      rewrite.ref_computed_initializer_positions()[j] =
          initializer_count + live_initializers.Before(initializer_rank(computed_positions[j]));
  }
  return nodes.Materialize();
}

bool IsNodeOnlyRewrite(const LocalRewritingProto &rewrite) {
  return !rewrite.matched_nodes().empty() && rewrite.removed_initializers().empty() &&
         rewrite.removed_encoded_initializers().empty() &&
         rewrite.removed_paged_cache_initializers().empty() &&
         rewrite.added_initializers().empty() && rewrite.added_initializer_positions().empty() &&
         rewrite.value_renames().empty() && rewrite.initializer_calls().empty() &&
         rewrite.computed_initializer_positions().empty();
}

bool SameGraphPath(const LocalRewritingProto &left, const LocalRewritingProto &right) {
  return left.graph_path().size() == right.graph_path().size() &&
         std::equal(left.graph_path().begin(), left.graph_path().end(), right.graph_path().begin());
}

bool HasNestedGraph(const NodeProto &node) {
  return std::any_of(node.attribute().begin(), node.attribute().end(), [](const auto &attribute) {
    return attribute.has_g() || !attribute.graphs().empty() ||
           attribute.type() == AttributeProto::GRAPH ||
           attribute.type() == AttributeProto::GRAPHS || attribute.name().value().ends_with("_ref");
  });
}

// Batches edits only when each replacement preserves its input/output dependencies.
// Buckets precede original nodes; live counts translate sequential coordinates.
class RewritingNodeBatch {
public:
  explicit RewritingNodeBatch(const utils::RepeatedProtoField<NodeProto> &nodes)
      : nodes_(nodes), removed_(nodes.size(), false), added_(nodes.size() + 1),
        live_(nodes.size() + 1) {
    for (std::size_t i = 0; i < removed_.size(); ++i)
      live_.Update(i, true);
  }

  bool Append(const LocalRewritingProto &rewrite) {
    if (!IsNodeOnlyRewrite(rewrite) ||
        rewrite.added_nodes().size() != rewrite.added_nodes_positions().size())
      return false;
    const std::size_t node_count = live_.Before(added_.size());
    std::vector<std::size_t> originals;
    std::unordered_set<const NodeProto *> matched;
    std::size_t insertion_position = node_count;
    for (uint64_t encoded : rewrite.matched_nodes()) {
      const std::size_t position = CheckedPosition(encoded);
      if (position >= node_count)
        return false;
      const std::size_t original = live_.Find(position);
      if (position - live_.Before(original) < added_[original].size() ||
          original == removed_.size() || removed_[original] ||
          !matched.insert(&nodes_[original]).second)
        return false;
      if (HasNestedGraph(nodes_[original]))
        return false;
      originals.push_back(original);
      insertion_position = std::min(insertion_position, position);
    }
    std::vector<std::pair<std::size_t, const NodeProto *>> additions;
    for (std::size_t i = 0; i < rewrite.added_nodes().size(); ++i)
      additions.emplace_back(CheckedPosition(rewrite.added_nodes_positions(i)),
                             &rewrite.added_nodes(i));
    std::sort(additions.begin(), additions.end(),
              [](const auto &left, const auto &right) { return left.first < right.first; });
    if (additions.size() >
        std::numeric_limits<std::size_t>::max() - (node_count - originals.size()))
      throw BuilderError("ApplyRewriting: batch node count overflow.");
    for (std::size_t i = 0; i < additions.size(); ++i) {
      if (additions[i].first != insertion_position + i || HasNestedGraph(*additions[i].second))
        return false;
    }
    if (originals.size() != additions.size())
      return false;
    std::sort(originals.begin(), originals.end());
    for (std::size_t i = 0; i < originals.size(); ++i) {
      const auto &original = nodes_[originals[i]];
      const auto &replacement = *additions[i].second;
      if (!std::equal(original.input().begin(), original.input().end(), replacement.input().begin(),
                      replacement.input().end()) ||
          !std::equal(original.output().begin(), original.output().end(),
                      replacement.output().begin(), replacement.output().end()))
        return false;
    }
    for (std::size_t original : originals) {
      removed_[original] = true;
      live_.Update(original, false);
    }
    bool committed = false;
    ScopeExit restore([&]() noexcept {
      if (!committed) {
        for (std::size_t original : originals) {
          removed_[original] = false;
          live_.Update(original, true);
        }
      }
    });
    const std::size_t remaining = node_count - originals.size();
    const std::size_t anchor =
        insertion_position == remaining ? removed_.size() : live_.Find(insertion_position);
    if (insertion_position - live_.Before(anchor) != added_[anchor].size())
      return false;
    added_[anchor].reserve(added_[anchor].size() + additions.size());
    for (const auto &[position, node] : additions) {
      (void)position;
      added_[anchor].push_back(node);
      live_.Update(anchor, true);
    }
    committed = true;
    return true;
  }

  LocalRewritingProto Rewriting() const {
    LocalRewritingProto rewrite;
    std::size_t position = 0;
    for (std::size_t original = 0; original < added_.size(); ++original) {
      for (const NodeProto *node : added_[original]) {
        rewrite.add_added_nodes(*node);
        rewrite.add_added_nodes_positions(position++);
      }
      if (original < removed_.size()) {
        if (removed_[original])
          rewrite.add_matched_nodes(original);
        else
          ++position;
      }
    }
    return rewrite;
  }

private:
  const utils::RepeatedProtoField<NodeProto> &nodes_;
  std::vector<bool> removed_;
  std::vector<std::vector<const NodeProto *>> added_;
  LiveRanks live_;
};

int64_t ElapsedNanoseconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                              start)
      .count();
}

void RecordNoMatch(PatternOptimizationStatistics &statistics, std::string_view source_file,
                   std::uint_least32_t source_line, std::string_view reason,
                   std::size_t occurrences = 1) {
  const auto existing = std::find_if(statistics.no_matches.begin(), statistics.no_matches.end(),
                                     [&](const PatternNoMatchStatistics &candidate) {
                                       return candidate.source_file == source_file &&
                                              candidate.source_line == source_line &&
                                              candidate.reason == reason;
                                     });
  if (existing != statistics.no_matches.end()) {
    existing->occurrences += occurrences;
    return;
  }
  statistics.no_matches.push_back(
      {std::string(source_file), source_line, std::string(reason), occurrences});
}

} // namespace

GraphGraph::GraphGraph(GraphBuilder &builder)
    : GraphGraph(builder, CreateRegisteredPatterns(symbolic::Device::kUndefined)) {}

GraphGraph::GraphGraph(GraphBuilder &builder,
                       std::vector<std::unique_ptr<PatternOptimization>> patterns,
                       DoNotRemovePredicate do_not_remove)
    : builder_(builder), do_not_remove_(std::move(do_not_remove)) {
  patterns_.reserve(patterns.size());
  for (std::unique_ptr<PatternOptimization> &pattern : patterns) {
    if (pattern == nullptr) {
      throw BuilderError("GraphGraph: a pattern must not be null.");
    }
    if (pattern->Name().empty()) {
      throw BuilderError("GraphGraph: a pattern must have a stable diagnostic name.");
    }
    patterns_.push_back(std::move(pattern));
  }
  std::stable_sort(patterns_.begin(), patterns_.end(), [](const auto &left, const auto &right) {
    return left->priority < right->priority;
  });
  Rebuild();
}

GraphGraph::GraphGraph(GraphBuilder &builder,
                       std::vector<std::shared_ptr<PatternOptimization>> patterns)
    : builder_(builder), patterns_(std::move(patterns)) {
  for (const std::shared_ptr<PatternOptimization> &pattern : patterns_) {
    if (pattern == nullptr) {
      throw BuilderError("GraphGraph: a pattern must not be null.");
    }
    if (pattern->Name().empty()) {
      throw BuilderError("GraphGraph: a pattern must have a stable diagnostic name.");
    }
  }
  std::stable_sort(patterns_.begin(), patterns_.end(), [](const auto &left, const auto &right) {
    return left->priority < right->priority;
  });
  Rebuild();
}

GraphGraph::GraphGraph(GraphBuilder &builder,
                       const std::vector<std::shared_ptr<PatternOptimization>> &patterns,
                       DoNotRemovePredicate do_not_remove, const GraphGraph *parent_graph,
                       std::size_t parent_position_limit)
    : builder_(builder), patterns_(patterns), do_not_remove_(std::move(do_not_remove)),
      parent_graph_(parent_graph), parent_position_limit_(parent_position_limit) {
  if (builder_.device() == symbolic::Device::kUndefined)
    builder_.set_device(parent_graph_->builder_.device());
  Rebuild();
}

void GraphGraph::SetTargetDevice(symbolic::Device device) {
  if (device != symbolic::Device::kCPU && !symbolic::IsGPU(device))
    throw std::invalid_argument("Pattern selection requires a concrete device.");
  std::vector<GraphBuilder *> builders{&builder_};
  for (size_t index = 0; index < builders.size(); ++index) {
    const auto &builder = *builders[index];
    if (builder.device() != symbolic::Device::kUndefined && builder.device() != device)
      throw std::invalid_argument("Pattern device conflicts with builder device in '" +
                                  builder.name() + "'.");
    for (const auto &subgraph : builder.subgraphs_)
      builders.push_back(subgraph.get());
  }
  for (auto *builder : builders)
    builder->set_device(device);
}

void GraphGraph::Rebuild() {
  predecessors_.clear();
  successors_.clear();
  positions_.clear();
  output_names_.clear();
  subgraph_captured_.clear();
  initializers_.clear();
  computed_constants_.clear();

  std::unordered_set<std::string> input_names;
  for (const ValueInfoProto &input : builder_.Inputs()) {
    input_names.insert(input.name().value());
  }
  for (const TensorProto &initializer : builder_.Initializers()) {
    if (!input_names.count(initializer.name().value())) {
      initializers_.emplace(initializer.name().value(), &initializer);
    }
  }
  for (const ValueInfoProto &output : builder_.Outputs()) {
    output_names_.insert(output.name().value());
  }

  const utils::RepeatedProtoField<NodeProto> &nodes = builder_.Nodes();
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const NodeProto &node = nodes[i];
    positions_.emplace(&node, i);
    for (std::size_t o = 0; o < node.output().size(); ++o) {
      std::string name(node.output(static_cast<std::size_t>(o)));
      if (!name.empty()) {
        predecessors_[std::move(name)] = &node;
      }
    }
  }
  RebuildSuccessors();
}

RewritingProto GraphGraph::Optimize(int max_iter, OptimizationReport *report,
                                    const ConstantFoldingOptions &folding_options,
                                    bool remove_unused) {
  GraphBuilder *root = &builder_;
  while (root->parent_ != nullptr)
    root = root->parent_;
  const auto original_opsets = root->opsets_;
  RewritingProto collection;
  const auto capture_functions = [&](const auto &self, const GraphBuilder &scope) -> void {
    for (const auto &function : scope.local_functions_) {
      MergeFunction(collection.ref_added_functions(),
                    function->BuildFunction(function->function_domain_));
      self(self, *function);
    }
    for (const auto &subgraph : scope.subgraphs_)
      self(self, *subgraph);
  };
  capture_functions(capture_functions, *root);
  const std::size_t original_functions = collection.added_functions().size();
  auto result = OptimizeImpl(max_iter, report, {}, folding_options, remove_unused, collection);
  capture_functions(capture_functions, *root);
  FunctionIndex functions;
  for (const auto &function : collection.added_functions())
    functions.emplace(Identity(function), &function);
  for (const auto &rewrite : result.rewrites()) {
    for (const auto &call : rewrite.initializer_calls()) {
      for (const auto &node : functions.at(Identity(call))->node()) {
        if (functions.count(Identity(node)) != 0)
          throw BuilderError("GraphGraph::Optimize: a later function overrides a captured "
                             "initializer operation '" +
                             node.op_type().value() + "'.");
      }
    }
  }
  utils::RepeatedProtoField<OperatorSetIdProto> graph_imports;
  const auto capture_imports = [&](const auto &self, const GraphBuilder &scope) -> void {
    for (const auto &[domain, version] : scope.opsets_) {
      OperatorSetIdProto opset;
      opset.set_domain(domain);
      opset.set_version(version);
      MergeImport(graph_imports, opset);
    }
    for (const auto &subgraph : scope.subgraphs_)
      self(self, *subgraph);
  };
  capture_imports(capture_imports, *root);
  for (const auto &opset : graph_imports) {
    const std::string domain = RewriteDomain(opset.domain().value());
    const int64_t version = opset.version();
    const auto previous = original_opsets.find(domain);
    if (previous != original_opsets.end()) {
      if (previous->second != version)
        throw BuilderError("GraphGraph::Optimize: a rewrite changed an existing opset.");
      continue;
    }
    MergeImport(collection.ref_required_opset_imports(), opset);
  }
  for (const auto &opset : collection.required_opset_imports())
    MergeImport(graph_imports, opset);
  for (std::size_t i = 0; i < original_functions; ++i)
    collection.ref_added_functions().erase(collection.ref_added_functions().begin());
  result.ref_added_functions() = std::move(collection.ref_added_functions());
  result.ref_required_opset_imports() = std::move(collection.ref_required_opset_imports());
  return result;
}

RewritingProto GraphGraph::OptimizeImpl(int max_iter, OptimizationReport *report,
                                        const std::vector<std::string> &graph_path,
                                        const ConstantFoldingOptions &folding_options,
                                        bool remove_unused, RewritingProto &collection) {
  if (max_iter < -1) {
    throw BuilderError("GraphGraph::Optimize: max_iter must be at least -1.");
  }
  if (report != nullptr) {
    *report = OptimizationReport{};
    report->patterns.reserve(patterns_.size());
    for (const std::shared_ptr<PatternOptimization> &pattern : patterns_) {
      report->patterns.push_back(PatternOptimizationStatistics{.pattern_name = pattern->Name()});
    }
  }

  std::vector<std::set<std::string>> fast_op_types_by_pattern;
  fast_op_types_by_pattern.reserve(patterns_.size());
  for (const std::shared_ptr<PatternOptimization> &pattern : patterns_) {
    fast_op_types_by_pattern.push_back(pattern->FastOpType());
  }

  RewritingProto result;
  auto &applied = result.ref_rewrites();
  std::unordered_set<GraphBuilder *> optimized_subgraphs;
  const auto optimize_subgraph = [&](GraphBuilder &subgraph, std::size_t position_limit) {
    std::vector<std::string> child_path = graph_path;
    child_path.push_back(subgraph.name());
    OptimizationReport child_report;
    std::chrono::steady_clock::time_point subgraph_start;
    if (report != nullptr) {
      subgraph_start = std::chrono::steady_clock::now();
    }
    GraphGraph child_graph(subgraph, patterns_, do_not_remove_, this, position_limit);
    RewritingProto child_rewrites =
        child_graph.OptimizeImpl(max_iter, report == nullptr ? nullptr : &child_report, child_path,
                                 folding_options, remove_unused, collection);

    applied.extend(std::move(child_rewrites.ref_rewrites()));

    if (report != nullptr) {
      const int64_t elapsed_time_ns = ElapsedNanoseconds(subgraph_start);
      report->iterations += child_report.iterations;
      report->rewrites += child_report.rewrites;
      report->subgraph_optimization_time_ns += elapsed_time_ns;
      for (std::size_t i = 0; i < report->patterns.size(); ++i) {
        report->patterns[i].attempts += child_report.patterns[i].attempts;
        report->patterns[i].matches += child_report.patterns[i].matches;
        report->patterns[i].match_time_ns += child_report.patterns[i].match_time_ns;
        report->patterns[i].apply_time_ns += child_report.patterns[i].apply_time_ns;
        for (const PatternNoMatchStatistics &no_match : child_report.patterns[i].no_matches) {
          RecordNoMatch(report->patterns[i], no_match.source_file, no_match.source_line,
                        no_match.reason, no_match.occurrences);
        }
      }
      report->subgraphs.push_back(
          {child_path, child_report.iterations, child_report.rewrites, elapsed_time_ns});
      report->subgraphs.insert(report->subgraphs.end(), child_report.subgraphs.begin(),
                               child_report.subgraphs.end());
    }
  };
  for (std::size_t position = 0; position < builder_.nodes_.size(); ++position) {
    for (GraphBuilder *subgraph : builder_.ReferencedSubgraphs(builder_.nodes_[position])) {
      if (optimized_subgraphs.insert(subgraph).second) {
        optimize_subgraph(*subgraph, position);
      }
    }
  }
  for (const std::unique_ptr<GraphBuilder> &subgraph : builder_.subgraphs_) {
    if (optimized_subgraphs.insert(subgraph.get()).second) {
      optimize_subgraph(*subgraph, builder_.nodes_.size());
    }
  }

  std::vector<int> priorities;
  priorities.reserve(patterns_.size());
  for (const std::shared_ptr<PatternOptimization> &pattern : patterns_) {
    if (priorities.empty() || priorities.back() != pattern->priority) {
      priorities.push_back(pattern->priority);
    }
  }
  if (priorities.empty()) {
    priorities.push_back(0);
  }

  if (max_iter == -1) {
    max_iter =
        static_cast<int>(std::max<std::size_t>(builder_.nodes_.size(), 10) * priorities.size());
  }

  std::size_t priority_index = 0;
  for (int iteration = 0; iteration < max_iter; ++iteration) {
    if (report != nullptr) {
      ++report->iterations;
    }
    const int current_priority = priorities[priority_index];
    std::unordered_set<const NodeProto *> marked;
    std::vector<AcceptedMatch> matches;
    std::chrono::steady_clock::time_point matching_start;
    if (report != nullptr) {
      matching_start = std::chrono::steady_clock::now();
    }

    for (std::size_t pattern_index = 0; pattern_index < patterns_.size(); ++pattern_index) {
      const std::shared_ptr<PatternOptimization> &pattern = patterns_[pattern_index];
      if (pattern->priority > current_priority) {
        break;
      }
      const std::set<std::string> &fast_op_types = fast_op_types_by_pattern[pattern_index];
      for (const NodeProto &candidate : builder_.nodes_) {
        if (!fast_op_types.empty() &&
            fast_op_types.find(candidate.op_type().value()) == fast_op_types.end()) {
          continue;
        }

        const auto match_start = std::chrono::steady_clock::now();
        MatchResult match = pattern->Match(*this, candidate);
        const int64_t match_time_ns = ElapsedNanoseconds(match_start);
        if (report != nullptr) {
          PatternOptimizationStatistics &statistics = report->patterns[pattern_index];
          ++statistics.attempts;
          statistics.match_time_ns += match_time_ns;
        }
        if (match.pattern == nullptr) {
          if (report != nullptr && match.no_match.has_value()) {
            RecordNoMatch(report->patterns[pattern_index], match.no_match->source_file,
                          match.no_match->source_line, match.no_match->reason);
          }
          continue;
        }
        if (match.pattern != pattern.get()) {
          throw BuilderError("GraphGraph::Optimize: a match refers to another pattern.");
        }
        if (match.nodes.empty()) {
          throw BuilderError("GraphGraph::Optimize: a non-empty match is required.");
        }

        bool skip = false;
        bool has_node = false;
        for (const NodeProto *node : match.nodes) {
          if (node == nullptr) {
            continue;
          }
          has_node = true;
          Position(*node);
          if (marked.find(node) != marked.end() || (do_not_remove_ && do_not_remove_(*node))) {
            skip = true;
            break;
          }
        }
        if (!has_node) {
          throw BuilderError(
              "GraphGraph::Optimize: a match must contain at least one non-null node.");
        }
        if (skip) {
          continue;
        }
        if (match.insert_at != nullptr) {
          Position(*match.insert_at);
        }
        for (const NodeProto *node : match.nodes) {
          if (node != nullptr) {
            marked.insert(node);
          }
        }
        if (report != nullptr) {
          ++report->patterns[pattern_index].matches;
        }
        matches.push_back(AcceptedMatch{std::move(match), &candidate, pattern_index});
      }
    }
    if (report != nullptr) {
      report->matching_time_ns += ElapsedNanoseconds(matching_start);
    }

    std::chrono::steady_clock::time_point rewriting_start;
    if (report != nullptr) {
      rewriting_start = std::chrono::steady_clock::now();
    }
    std::unordered_set<const NodeProto *> removed;
    const std::size_t batch_node_count = builder_.nodes_.size();
    utils::RepeatedProtoField<NodeProto> batch_nodes;
    std::unordered_map<const NodeProto *, std::size_t> original_indices;
    std::unordered_map<const NodeProto *, std::vector<std::string>> original_references;
    for (std::size_t index = 0; index < batch_node_count; ++index) {
      batch_nodes.add();
      batch_nodes.get(index) = builder_.nodes_.shared_at(index);
      original_indices.emplace(&builder_.nodes_[index], index);
      builder_.CollectNodeReferences(builder_.nodes_[index],
                                     original_references[&builder_.nodes_[index]]);
    }
    const std::size_t batch_initializer_count = builder_.initializers_.size();
    std::vector<PendingReplacement> replacements;
    int64_t constant_folding_time_ns = 0;
    replacements.reserve(matches.size());
    for (const AcceptedMatch &accepted_match : matches) {
      GraphBuilder::AncestorRefreshGuard ancestor_refresh_guard(builder_);
      const MatchResult &match = accepted_match.match;
      const MatchResult verified = match.pattern->Match(*this, *accepted_match.candidate);
      if (verified.pattern != match.pattern || verified.nodes != match.nodes ||
          verified.insert_at != match.insert_at) {
        if (report != nullptr && verified.no_match.has_value()) {
          RecordNoMatch(report->patterns[accepted_match.pattern_index],
                        verified.no_match->source_file, verified.no_match->source_line,
                        verified.no_match->reason);
        }
        continue;
      }
      std::size_t position = builder_.nodes_.size();
      if (match.insert_at != nullptr) {
        position = Position(*match.insert_at);
      } else {
        for (const NodeProto *node : match.nodes) {
          if (node != nullptr) {
            position = std::min(position, Position(*node));
          }
        }
      }
      const std::size_t initializers_before = builder_.initializers_.size();
      const auto apply_start = std::chrono::steady_clock::now();
      utils::RepeatedProtoField<NodeProto> replacement_nodes =
          match.pattern->Apply(*this, match.nodes);
      const int64_t apply_time_ns = ElapsedNanoseconds(apply_start);
      LocalRewritingProto rewriting;
      rewriting.set_pattern_name(match.pattern->Name());
      for (const auto &name : graph_path) {
        rewriting.add_graph_path(name);
      }
      rewriting.ref_added_nodes() = replacement_nodes;
      rewriting.ref_matched_nodes().reserve(match.nodes.size());
      std::unordered_set<const NodeProto *> persisted_nodes;
      for (const NodeProto *node : match.nodes) {
        if (node != nullptr && persisted_nodes.insert(node).second) {
          rewriting.add_matched_nodes(Position(*node));
        }
      }
      for (std::size_t i = initializers_before; i < builder_.initializers_.size(); ++i) {
        rewriting.add_added_initializers(builder_.initializers_[i]);
        rewriting.add_added_initializer_positions(i);
      }
      replacements.push_back(
          PendingReplacement{position, std::move(replacement_nodes), std::move(rewriting)});
      for (const NodeProto *node : match.nodes) {
        if (node != nullptr) {
          removed.insert(node);
        }
      }
      if (report != nullptr) {
        report->patterns[accepted_match.pattern_index].apply_time_ns += apply_time_ns;
      }
    }

    if (!matches.empty()) {
      GraphBuilder::AncestorRefreshGuard ancestor_refresh_guard(builder_);
      utils::RepeatedProtoField<NodeProto> rebuilt;
      std::unordered_set<std::string> affected_values;
      for (const NodeProto *node : removed) {
        builder_.compute_.Shapes().EraseNodeConstraints(*node);
        for (const auto &output : node->output()) {
          if (!output.empty()) {
            affected_values.insert(output.value());
            builder_.compute_.Shapes().Erase(output.value());
          }
        }
      }
      std::size_t replacement_nodes = 0;
      for (const PendingReplacement &replacement : replacements) {
        replacement_nodes += replacement.nodes.size();
      }
      rebuilt.reserve(builder_.nodes_.size() - removed.size() + replacement_nodes);

      std::unordered_map<std::size_t, std::vector<std::size_t>> insertions;
      for (std::size_t i = 0; i < replacements.size(); ++i) {
        insertions[replacements[i].position].push_back(i);
      }
      for (std::size_t i = 0; i < builder_.nodes_.size(); ++i) {
        auto insertion = insertions.find(i);
        if (insertion != insertions.end()) {
          for (std::size_t replacement_index : insertion->second) {
            rebuilt.extend(std::move(replacements[replacement_index].nodes));
          }
        }
        const NodeProto *node = &builder_.nodes_[i];
        if (removed.find(node) == removed.end()) {
          rebuilt.add();
          rebuilt.get(rebuilt.size() - 1) = builder_.nodes_.shared_at(i);
        }
      }
      builder_.nodes_ = std::move(rebuilt);
      const auto metadata_count = builder_.compute_.Size();
      std::vector<std::size_t> old_indices;
      old_indices.reserve(builder_.nodes_.size());
      for (const auto &node : builder_.nodes_) {
        const auto found = original_indices.find(&node);
        old_indices.push_back(found == original_indices.end() || found->second >= metadata_count
                                  ? metadata_count
                                  : found->second);
        for (const auto &output : node.output())
          if (!output.empty())
            builder_.names_.insert(output.value());
      }
      builder_.RefreshAfterRewriting(old_indices, std::move(affected_values));

      std::unordered_map<std::string, std::size_t> output_owners;
      std::unordered_set<std::string> replacement_outputs;
      for (std::size_t i = 0; i < replacements.size(); ++i) {
        for (const NodeProto &node : replacements[i].rewriting.added_nodes()) {
          const bool can_prepare = IsPreparationPrimitive(node);
          for (std::size_t o = 0; o < node.output().size(); ++o) {
            const std::string output(node.output(static_cast<std::size_t>(o)));
            if (!output.empty()) {
              output_owners.emplace(output, i);
              if (can_prepare)
                replacement_outputs.insert(output);
            }
          }
        }
      }

      const std::size_t initializers_before_folding = builder_.initializers_.size();
      const auto constant_folding_start = std::chrono::steady_clock::now();
      builder_.ConstantFoldNodes(folding_options, replacement_outputs);
      constant_folding_time_ns = ElapsedNanoseconds(constant_folding_start);
      if (report != nullptr) {
        report->constant_folding_time_ns += constant_folding_time_ns;
      }

      std::unordered_set<std::string> folded_outputs;
      std::unordered_map<std::string, std::size_t> folded_positions;
      for (std::size_t i = initializers_before_folding; i < builder_.initializers_.size(); ++i) {
        const TensorProto &initializer = builder_.initializers_[i];
        const std::string name = initializer.name().value();
        const auto owner = output_owners.find(name);
        if (owner != output_owners.end()) {
          folded_outputs.insert(name);
          folded_positions.emplace(name, i);
        }
      }
      for (PendingReplacement &replacement : replacements) {
        utils::RepeatedProtoField<NodeProto> unfolded_nodes;
        for (const NodeProto &node : replacement.rewriting.added_nodes()) {
          bool folded = !node.output().empty();
          for (std::size_t o = 0; o < node.output().size(); ++o) {
            const std::string output(node.output(static_cast<std::size_t>(o)));
            if (!output.empty() && folded_outputs.find(output) == folded_outputs.end()) {
              folded = false;
              break;
            }
          }
          if (folded) {
            auto &rewrite = replacement.rewriting;
            rewrite.ref_initializer_calls().push_back(
                CaptureInitializerCall(node, builder_, collection));
            for (const auto &output : node.output()) {
              if (output.empty())
                continue;
              rewrite.add_computed_initializer_positions(folded_positions.at(output.value()));
            }
          } else {
            unfolded_nodes.push_back(node);
          }
        }
        replacement.rewriting.ref_added_nodes() = std::move(unfolded_nodes);
      }

      OrderPreparationDependencies(replacements);
      std::unordered_set<std::string> available = builder_.inherited_names_;
      for (const auto &input : builder_.inputs_)
        available.insert(input.name().value());
      for (const auto &initializer : builder_.initializers_)
        available.insert(initializer.name().value());
      for (const auto &initializer : builder_.encoded_initializers_)
        available.insert(initializer.name().value());
      for (const auto &initializer : builder_.paged_cache_initializers_)
        available.insert(initializer.name().value());
      for (const auto &initializer : builder_.graph_template_.sparse_initializer())
        available.insert(initializer.values().name().value());
      const auto normalized = NormalizeBatch(
          replacements, batch_nodes, batch_initializer_count, available,
          [&](const auto &node) {
            const auto original = original_references.find(&node);
            if (original != original_references.end())
              return original->second;
            std::vector<std::string> references;
            builder_.CollectNodeReferences(node, references);
            return references;
          },
          [](const auto &nodes, const auto &external, const auto &references, const auto &added) {
            return GraphBuilder::InsertNodesInDependencyOrder(nodes, external, references, added);
          });
      std::unordered_map<std::string, std::size_t> current_indices;
      for (std::size_t index = 0; index < builder_.nodes_.size(); ++index)
        for (const auto &output : builder_.nodes_[index].output())
          if (!output.empty())
            current_indices.emplace(output.value(), index);
      utils::RepeatedProtoField<NodeProto> ordered;
      old_indices.clear();
      for (const auto &node : normalized) {
        const auto output = std::find_if(node.output().begin(), node.output().end(),
                                         [](const auto &name) { return !name.empty(); });
        if (output == node.output().end())
          throw BuilderError("GraphGraph::Optimize: a retained node must produce a named value.");
        const auto index = current_indices.at(output->value());
        old_indices.push_back(index);
        ordered.add();
        ordered.get(ordered.size() - 1) = builder_.nodes_.shared_at(index);
      }
      bool reordered = false;
      for (std::size_t index = 0; index < old_indices.size(); ++index)
        reordered |= old_indices[index] != index;
      if (reordered) {
        builder_.nodes_ = std::move(ordered);
        std::vector<std::vector<std::string>> references(builder_.nodes_.size());
        for (std::size_t index = 0; index < builder_.nodes_.size(); ++index)
          builder_.CollectNodeReferences(builder_.nodes_[index], references[index]);
        builder_.compute_.UpdateNodes(builder_.nodes_, old_indices, references, {},
                                      builder_.device_);
        builder_.compute_.RefreshIncrementalMemory(builder_.nodes_, builder_.Shapes());
      }
      for (auto &replacement : replacements) {
        applied.push_back(std::move(replacement.rewriting));
      }
    }
    if (report != nullptr) {
      report->rewriting_time_ns += ElapsedNanoseconds(rewriting_start) - constant_folding_time_ns;
      report->rewrites = applied.size();
    }

    std::chrono::steady_clock::time_point cleanup_start;
    if (report != nullptr) {
      cleanup_start = std::chrono::steady_clock::now();
    }
    const std::size_t rewrites_before_cleanup = applied.size();
    const std::size_t cleaned = Cleanup(applied, remove_unused);
    for (std::size_t i = rewrites_before_cleanup; i < applied.size(); ++i) {
      for (const auto &name : graph_path) {
        applied[i].add_graph_path(name);
      }
    }
    if (report != nullptr) {
      report->cleanup_time_ns += ElapsedNanoseconds(cleanup_start);
      report->rewrites = applied.size();
    }

    if (!matches.empty() || cleaned != 0) {
      continue;
    }
    if (priority_index + 1 == priorities.size()) {
      break;
    }
    ++priority_index;
  }
  return result;
}

std::size_t GraphGraph::Cleanup(utils::RepeatedProtoField<LocalRewritingProto> &rewrites,
                                bool remove_unused) {
  const auto original_nodes = builder_.nodes_;
  const std::size_t first_rewrite = rewrites.size();
  const auto initializer_names = [](const auto &initializers) {
    std::vector<std::string> names;
    names.reserve(initializers.size());
    for (const auto &initializer : initializers)
      names.push_back(initializer.name().value());
    return names;
  };
  const auto all_initializer_names = [&]() {
    auto names = initializer_names(builder_.initializers_);
    for (const auto &initializer : builder_.encoded_initializers_)
      names.push_back(initializer.name().value());
    for (const auto &initializer : builder_.paged_cache_initializers_)
      names.push_back(initializer.name().value());
    return names;
  };
  const auto initializers_before = all_initializer_names();
  const auto record_removed_initializers = [](const std::vector<std::string> &original,
                                              const auto &initializers, auto &positions) {
    std::unordered_set<std::string> retained;
    for (const auto &initializer : initializers)
      retained.insert(initializer.name().value());
    for (std::size_t i = 0; i < original.size(); ++i)
      if (retained.count(original[i]) == 0)
        positions.push_back(i);
  };
  const auto record_nodes = [&](const std::string &name, const auto &cleanup) {
    const std::size_t node_count = builder_.nodes_.size();
    const auto original_initializers = initializer_names(builder_.initializers_);
    const auto original_encoded = initializer_names(builder_.encoded_initializers_);
    const auto original_paged = initializer_names(builder_.paged_cache_initializers_);
    std::unordered_map<std::string, std::string> applied_renames;
    const std::size_t removed = cleanup(applied_renames);
    const std::size_t removed_initializers =
        original_initializers.size() - builder_.initializers_.size() + original_encoded.size() -
        builder_.encoded_initializers_.size() + original_paged.size() -
        builder_.paged_cache_initializers_.size();
    if (removed == 0 && removed_initializers == 0) {
      return std::size_t{0};
    }
    LocalRewritingProto rewriting;
    rewriting.set_pattern_name(name);
    rewriting.ref_matched_nodes().reserve(node_count);
    for (std::size_t i = 0; i < node_count; ++i) {
      rewriting.add_matched_nodes(i);
    }
    rewriting.ref_added_nodes() = builder_.nodes_;
    rewriting.ref_added_nodes_positions().reserve(builder_.nodes_.size());
    for (std::size_t i = 0; i < builder_.nodes_.size(); ++i) {
      rewriting.add_added_nodes_positions(i);
    }
    if (removed_initializers != 0) {
      record_removed_initializers(original_initializers, builder_.initializers_,
                                  rewriting.ref_removed_initializers());
      record_removed_initializers(original_encoded, builder_.encoded_initializers_,
                                  rewriting.ref_removed_encoded_initializers());
      record_removed_initializers(original_paged, builder_.paged_cache_initializers_,
                                  rewriting.ref_removed_paged_cache_initializers());
    }
    RecordRenames(rewriting, applied_renames);
    rewrites.push_back(std::move(rewriting));
    return removed + removed_initializers;
  };

  std::size_t cleaned = record_nodes("RemoveDuplicateNodes", [&](auto &renames) {
    return builder_.RemoveDuplicateNodesImpl(false, &renames);
  });
  cleaned += record_nodes("RemoveIdentityNodes", [&](auto &renames) {
    return builder_.RemoveIdentityNodesImpl(false, &renames);
  });
  if (remove_unused) {
    cleaned += record_nodes("RemoveUnusedNodes",
                            [&](auto &) { return builder_.RemoveUnusedNodesImpl(false); });
  }

  const std::size_t node_count = builder_.nodes_.size();
  const auto original_initializers = initializer_names(builder_.initializers_);
  const auto original_encoded = initializer_names(builder_.encoded_initializers_);
  std::unordered_map<std::string, std::string> applied_renames;
  const std::size_t removed_initializers = builder_.DeduplicateInitializers(
      GraphBuilder::InitializerContentIndex{}, false, &applied_renames);
  if (removed_initializers != 0) {
    LocalRewritingProto rewriting;
    rewriting.set_pattern_name("RemoveDuplicateInitializers");
    rewriting.ref_matched_nodes().reserve(node_count);
    for (std::size_t i = 0; i < node_count; ++i) {
      rewriting.add_matched_nodes(i);
    }
    rewriting.ref_added_nodes() = builder_.nodes_;
    rewriting.ref_added_nodes_positions().reserve(builder_.nodes_.size());
    for (std::size_t i = 0; i < builder_.nodes_.size(); ++i) {
      rewriting.add_added_nodes_positions(i);
    }
    record_removed_initializers(original_initializers, builder_.initializers_,
                                rewriting.ref_removed_initializers());
    record_removed_initializers(original_encoded, builder_.encoded_initializers_,
                                rewriting.ref_removed_encoded_initializers());
    RecordRenames(rewriting, applied_renames);
    rewrites.push_back(std::move(rewriting));
    cleaned += removed_initializers;
  }
  const auto remaining_initializers = all_initializer_names();
  if (cleaned != 0 || initializers_before != remaining_initializers) {
    std::unordered_map<std::string, std::size_t> original_indices;
    for (std::size_t index = 0; index < original_nodes.size(); ++index)
      for (const auto &output : original_nodes[index].output())
        if (!output.empty())
          original_indices.emplace(output.value(), index);
    std::vector<std::size_t> old_indices;
    std::unordered_set<std::size_t> retained;
    std::unordered_set<std::string> affected_values;
    for (const auto &node : builder_.nodes_) {
      const auto output = std::find_if(node.output().begin(), node.output().end(),
                                       [](const auto &name) { return !name.empty(); });
      if (output == node.output().end())
        throw BuilderError("GraphGraph::Cleanup: a node must produce a named value.");
      const auto index = original_indices.at(output->value());
      old_indices.push_back(index);
      retained.insert(index);
      if (node.input().size() != original_nodes[index].input().size() ||
          !std::equal(node.input().begin(), node.input().end(),
                      original_nodes[index].input().begin()))
        for (const auto &name : node.output())
          if (!name.empty())
            affected_values.insert(name.value());
    }
    for (std::size_t index = 0; index < original_nodes.size(); ++index) {
      if (retained.count(index) == 0) {
        builder_.compute_.Shapes().EraseNodeConstraints(original_nodes[index]);
        for (const auto &output : original_nodes[index].output()) {
          if (!output.empty()) {
            affected_values.insert(output.value());
            builder_.compute_.Shapes().Erase(output.value());
          }
        }
      }
    }
    for (std::size_t index = first_rewrite; index < rewrites.size(); ++index)
      for (const auto &rename : rewrites[index].value_renames()) {
        affected_values.insert(rename.key().value());
        affected_values.insert(rename.value().value());
      }
    std::unordered_set<std::string> retained_initializers;
    retained_initializers.insert(remaining_initializers.begin(), remaining_initializers.end());
    for (const auto &name : initializers_before) {
      if (retained_initializers.count(name) == 0) {
        affected_values.insert(name);
        builder_.compute_.EraseValue(name);
        if (std::none_of(builder_.inputs_.begin(), builder_.inputs_.end(),
                         [&](const auto &input) { return input.name() == name; }))
          builder_.compute_.Shapes().Erase(name);
      }
    }
    builder_.RefreshAfterRewriting(old_indices, std::move(affected_values));
  } else {
    builder_.RefreshAncestors();
  }
  Rebuild();
  return cleaned;
}

void GraphBuilder::ApplyRewriting(const LocalRewritingProto &rewrite,
                                  const utils::RepeatedProtoField<FunctionProto> &functions) {
  FunctionIndex function_index;
  for (const auto &function : functions) {
    function_index.emplace(Identity(function), &function);
  }
  for (const auto &node : rewrite.added_nodes()) {
    ValidateFunctionCalls(node, function_index);
    const int opset = OpsetVersion(node.domain().value());
    if (opset <= 0) {
      throw BuilderError("ApplyRewriting: added node '" + node.op_type().value() +
                         "' has no explicit opset import.");
    }
    if (function_index.count(Identity(node)) == 0) {
      const LightOpSchema *schema = nullptr;
      for (const auto &candidate :
           DomainSchemas(node.op_type().value(), RewriteDomain(node.domain().value()))) {
        if (candidate.since_version() <= opset &&
            (schema == nullptr || candidate.since_version() > schema->since_version())) {
          schema = &candidate;
        }
      }
      if (schema != nullptr && (node.output_size() < schema->min_output() ||
                                node.output_size() > schema->max_output())) {
        throw BuilderError("ApplyRewriting: added node '" + node.op_type().value() +
                           "' has an incompatible output count.");
      }
    }
  }
  auto &nodes = nodes_;
  auto &initializers = initializers_;
  std::unordered_set<const NodeProto *> removed;
  std::unordered_set<std::size_t> removed_initializers;
  std::unordered_map<std::string, std::string> value_renames;
  struct AddedInitializer {
    std::size_t position;
    const TensorProto *initializer;
  };
  std::vector<AddedInitializer> added_initializers;
  struct AddedNode {
    std::size_t position;
    const NodeProto *node;
  };
  std::vector<AddedNode> added_nodes;
  if (rewrite.matched_nodes().empty() && rewrite.removed_initializers().empty() &&
      rewrite.removed_encoded_initializers().empty() &&
      rewrite.removed_paged_cache_initializers().empty() && rewrite.added_nodes().empty() &&
      rewrite.added_initializers().empty() && rewrite.value_renames().empty() &&
      rewrite.initializer_calls().empty()) {
    throw BuilderError("ApplyRewriting: an empty rewrite is not allowed.");
  }
  for (const uint64_t encoded_position : rewrite.matched_nodes()) {
    const std::size_t position = CheckedPosition(encoded_position);
    if (position >= nodes.size()) {
      throw BuilderError("ApplyRewriting: matched node position " + std::to_string(position) +
                         " is outside the graph.");
    }
    if (!removed.insert(&nodes[position]).second) {
      throw BuilderError("ApplyRewriting: duplicate matched node position " +
                         std::to_string(position) + ".");
    }
  }
  if (rewrite.added_nodes_positions().size() != rewrite.added_nodes().size()) {
    throw BuilderError("ApplyRewriting: added node positions must align with added nodes.");
  }
  for (std::size_t node_index = 0; node_index < rewrite.added_nodes().size(); ++node_index) {
    added_nodes.push_back(AddedNode{CheckedPosition(rewrite.added_nodes_positions(node_index)),
                                    &rewrite.added_nodes(node_index)});
  }
  for (const uint64_t encoded_position : rewrite.removed_initializers()) {
    const std::size_t position = CheckedPosition(encoded_position);
    if (position >= initializers.size()) {
      throw BuilderError("ApplyRewriting: removed initializer position " +
                         std::to_string(position) + " is outside the graph.");
    }
    if (!removed_initializers.insert(position).second) {
      throw BuilderError("ApplyRewriting: duplicate removed initializer position " +
                         std::to_string(position) + ".");
    }
  }
  const auto retained_structured_initializers = [&](const auto &initializers,
                                                    const auto &positions) {
    std::decay_t<decltype(initializers)> retained;
    if (positions.empty())
      return retained;
    std::unordered_set<std::size_t> removed_positions;
    for (const uint64_t encoded_position : positions) {
      const std::size_t position = CheckedPosition(encoded_position);
      if (position >= initializers.size())
        throw BuilderError("ApplyRewriting: removed structured initializer position is outside "
                           "the graph.");
      if (!removed_positions.insert(position).second)
        throw BuilderError("ApplyRewriting: duplicate removed structured initializer position.");
    }
    retained.reserve(initializers.size() - removed_positions.size());
    for (std::size_t position = 0; position < initializers.size(); ++position) {
      if (removed_positions.count(position) == 0) {
        retained.add();
        retained.get(retained.size() - 1) = initializers.shared_at(position);
      }
    }
    return retained;
  };
  auto rebuilt_encoded_initializers = retained_structured_initializers(
      encoded_initializers_, rewrite.removed_encoded_initializers());
  auto rebuilt_paged_cache_initializers = retained_structured_initializers(
      paged_cache_initializers_, rewrite.removed_paged_cache_initializers());
  for (const auto &rename : rewrite.value_renames()) {
    const std::string key = rename.key().value();
    const std::string value = rename.value().value();
    const auto inserted = value_renames.emplace(key, value);
    if (!inserted.second && inserted.first->second != value) {
      throw BuilderError("ApplyRewriting: conflicting replacements for value '" + key + "'.");
    }
  }
  if (rewrite.added_initializer_positions().size() != rewrite.added_initializers().size()) {
    throw BuilderError(
        "ApplyRewriting: added initializer positions must align with added initializers.");
  }
  for (std::size_t initializer_index = 0; initializer_index < rewrite.added_initializers().size();
       ++initializer_index) {
    const TensorProto &initializer = rewrite.added_initializers(initializer_index);
    const std::size_t position =
        CheckedPosition(rewrite.added_initializer_positions(initializer_index));
    added_initializers.push_back(AddedInitializer{position, &initializer});
  }
  std::size_t computed_count = 0;
  for (const auto &call : rewrite.initializer_calls()) {
    if (call.output().size() > std::numeric_limits<std::size_t>::max() - computed_count)
      throw BuilderError("ApplyRewriting: initializer output count overflow.");
    computed_count += call.output().size();
  }
  if (computed_count != rewrite.computed_initializer_positions().size())
    throw BuilderError(
        "ApplyRewriting: computed initializer positions must align with call outputs.");
  std::unordered_set<std::size_t> initializer_positions;
  const std::size_t retained_count = initializers.size() - removed_initializers.size();
  if (computed_count > std::numeric_limits<std::size_t>::max() - added_initializers.size() ||
      computed_count + added_initializers.size() >
          std::numeric_limits<std::size_t>::max() - retained_count)
    throw BuilderError("ApplyRewriting: initializer count overflow.");
  const std::size_t initializer_count = retained_count + added_initializers.size() + computed_count;
  for (const auto &addition : added_initializers) {
    if (addition.position >= initializer_count ||
        !initializer_positions.insert(addition.position).second)
      throw BuilderError("ApplyRewriting: invalid or overlapping initializer positions.");
  }
  for (uint64_t encoded : rewrite.computed_initializer_positions()) {
    const std::size_t position = CheckedPosition(encoded);
    if (position >= initializer_count || !initializer_positions.insert(position).second)
      throw BuilderError("ApplyRewriting: invalid or overlapping computed initializer positions.");
  }
  std::vector<TensorProto> computed;
  computed.reserve(computed_count);
  if (!rewrite.initializer_calls().empty()) {
    std::unordered_map<std::string, const TensorProto *> constants;
    std::unordered_set<std::string> graph_inputs;
    std::vector<const GraphBuilder *> scopes;
    for (const GraphBuilder *scope = this; scope != nullptr; scope = scope->parent_)
      scopes.push_back(scope);
    for (auto scope = scopes.rbegin(); scope != scopes.rend(); ++scope) {
      for (const auto &input : (*scope)->inputs_) {
        graph_inputs.insert(input.name().value());
        constants.erase(input.name().value());
      }
      for (const auto &node : (*scope)->nodes_)
        for (const auto &output : node.output())
          constants.erase(output.value());
      for (const auto &initializer : (*scope)->encoded_initializers_)
        constants.erase(initializer.name().value());
      for (const auto &initializer : (*scope)->paged_cache_initializers_)
        constants.erase(initializer.name().value());
      for (const auto &initializer : (*scope)->graph_template_.sparse_initializer())
        constants.erase(initializer.values().name().value());
      for (const auto &initializer : (*scope)->initializers_) {
        const auto &inputs = (*scope)->inputs_;
        if (std::none_of(inputs.begin(), inputs.end(),
                         [&](const auto &input) { return input.name() == initializer.name(); }))
          constants[initializer.name().value()] = &initializer;
      }
    }
    for (const auto &literal : rewrite.added_initializers()) {
      if (!constants.emplace(literal.name().value(), &literal).second)
        throw BuilderError("ApplyRewriting: a preparation literal shadows an existing constant.");
    }
    std::unordered_set<std::string> available;
    for (const auto &[name, value] : constants)
      available.insert(name);
    std::unordered_set<std::string> call_outputs;
    for (const auto &call : rewrite.initializer_calls()) {
      std::set<FunctionKey> active;
      ValidatePreparation(call, function_index, active);
      for (const auto &input : call.input()) {
        if (input.empty() || graph_inputs.count(input.value()) != 0 ||
            available.count(input.value()) == 0)
          throw BuilderError(
              "ApplyRewriting: initializer call input is not an available constant.");
      }
      for (const auto &output : call.output()) {
        if (output.empty() || !call_outputs.insert(output.value()).second)
          throw BuilderError(
              "ApplyRewriting: initializer calls must have unique, nonempty outputs.");
        if (std::any_of(rewrite.added_initializers().begin(), rewrite.added_initializers().end(),
                        [&](const auto &literal) { return literal.name() == output.value(); }))
          throw BuilderError(
              "ApplyRewriting: an initializer cannot have both a literal and a recipe.");
        available.insert(output.value());
      }
    }
    core::runtime::RuntimeContext runtime;
    ModelProto preparation;
    preparation.mutable_graph();
    for (std::size_t i = 0; i < functions.size(); ++i) {
      preparation.ref_functions().add();
      preparation.ref_functions().get(i) = functions.shared_at(i);
    }
    core::runtime::RegisterModelFunctions(preparation, runtime);
    std::unordered_set<std::string> seeded;
    for (const auto &call : rewrite.initializer_calls()) {
      for (const auto &input : call.input()) {
        if (seeded.insert(input.value()).second) {
          const auto source = constants.find(input.value());
          if (source != constants.end())
            runtime.Put(input.value(), core::runtime::TensorFromProto(*source->second));
        }
      }
      core::runtime::RunNode(call, runtime);
      for (const auto &output : call.output()) {
        computed.push_back(
            GraphBuilder::ProtoFromRuntimeTensor(runtime.Get(output.value()), output.value()));
        seeded.insert(output.value());
      }
    }
    for (std::size_t i = 0; i < computed.size(); ++i)
      added_initializers.push_back(
          {CheckedPosition(rewrite.computed_initializer_positions(i)), &computed[i]});
  }
  const std::size_t retained_nodes = nodes.size() - removed.size();
  const std::size_t retained_initializers = initializers.size() - removed_initializers.size();
  if (added_nodes.size() > std::numeric_limits<std::size_t>::max() - retained_nodes ||
      added_initializers.size() > std::numeric_limits<std::size_t>::max() - retained_initializers) {
    throw BuilderError("ApplyRewriting: rewritten graph size overflows size_t.");
  }

  const std::size_t final_node_count = retained_nodes + added_nodes.size();
  std::vector<const NodeProto *> nodes_by_position(final_node_count, nullptr);
  for (const AddedNode &addition : added_nodes) {
    if (addition.position >= final_node_count) {
      throw BuilderError("ApplyRewriting: added node position " +
                         std::to_string(addition.position) + " is outside the rewritten graph.");
    }
    if (nodes_by_position[addition.position] != nullptr) {
      throw BuilderError("ApplyRewriting: added node positions overlap at position " +
                         std::to_string(addition.position) + ".");
    }
    nodes_by_position[addition.position] = addition.node;
  }

  const std::size_t final_initializer_count = retained_initializers + added_initializers.size();
  std::vector<const TensorProto *> initializers_by_position(final_initializer_count, nullptr);
  for (const auto &addition : added_initializers) {
    if (addition.position >= final_initializer_count) {
      throw BuilderError(
          "ApplyRewriting: added initializer position is outside the rewritten graph.");
    }
    if (initializers_by_position[addition.position] != nullptr) {
      throw BuilderError("ApplyRewriting: duplicate added initializer position.");
    }
    initializers_by_position[addition.position] = addition.initializer;
  }
  utils::RepeatedProtoField<TensorProto> rebuilt_initializers;
  rebuilt_initializers.reserve(final_initializer_count);
  std::size_t original_initializer = 0;
  for (std::size_t position = 0; position < final_initializer_count; ++position) {
    if (initializers_by_position[position] != nullptr) {
      rebuilt_initializers.push_back(*initializers_by_position[position]);
      continue;
    }
    while (removed_initializers.count(original_initializer) != 0) {
      ++original_initializer;
    }
    if (original_initializer >= initializers.size()) {
      throw BuilderError("ApplyRewriting: invalid retained initializer positions.");
    }
    rebuilt_initializers.add();
    rebuilt_initializers.get(rebuilt_initializers.size() - 1) =
        initializers.shared_at(original_initializer++);
  }
  std::unordered_set<std::string> initializer_names;
  for (const auto &initializer : rebuilt_initializers) {
    if (!initializer.has_name() || initializer.name().empty() ||
        !initializer_names.insert(initializer.name().value()).second) {
      throw BuilderError("ApplyRewriting: initializers must have unique, nonempty names.");
    }
  }
  const auto validate_node_outputs = [&](const NodeProto &node) {
    for (const auto &output : node.output()) {
      if (initializer_names.count(output.value()) != 0) {
        throw BuilderError(
            "ApplyRewriting: an initializer name conflicts with a retained node output.");
      }
    }
  };
  for (const auto &node : nodes) {
    if (removed.count(&node) == 0) {
      validate_node_outputs(node);
    }
  }
  for (const auto &addition : added_nodes) {
    validate_node_outputs(*addition.node);
  }
  const auto &remaining_encoded_initializers = rewrite.removed_encoded_initializers().empty()
                                                   ? encoded_initializers_
                                                   : rebuilt_encoded_initializers;
  for (const auto &initializer : remaining_encoded_initializers) {
    if (initializer_names.count(initializer.name().value()) != 0) {
      throw BuilderError(
          "ApplyRewriting: an initializer name conflicts with an encoded initializer.");
    }
  }
  const auto &remaining_paged_cache_initializers =
      rewrite.removed_paged_cache_initializers().empty() ? paged_cache_initializers_
                                                         : rebuilt_paged_cache_initializers;
  for (const auto &initializer : remaining_paged_cache_initializers) {
    if (initializer_names.count(initializer.name().value()) != 0) {
      throw BuilderError(
          "ApplyRewriting: an initializer name conflicts with a paged-cache initializer.");
    }
  }

  RewriteInitializerReferences(value_renames);
  for (const auto &addition : added_initializers) {
    // Replacing an existing default must not bypass the public-input contract.
    ValidateInputDefault(inputs_, *addition.initializer);
    const auto &name = addition.initializer->name();
    bool replaces_removed_value = false;
    for (const NodeProto *node : removed) {
      for (const auto &output : node->output()) {
        replaces_removed_value = replaces_removed_value || output == name;
      }
    }
    for (std::size_t position : removed_initializers) {
      replaces_removed_value = replaces_removed_value || initializers[position].name() == name;
    }
    for (uint64_t position : rewrite.removed_encoded_initializers())
      replaces_removed_value =
          replaces_removed_value || encoded_initializers_[position].name() == name;
    for (uint64_t position : rewrite.removed_paged_cache_initializers())
      replaces_removed_value =
          replaces_removed_value || paged_cache_initializers_[position].name() == name;
    if (!replaces_removed_value) {
      const bool input_default = std::any_of(
          inputs_.begin(), inputs_.end(), [&](const auto &input) { return input.name() == name; });
      if (!input_default &&
          (inherited_names_.count(name.value()) != 0 || !names_.insert(name.value()).second))
        throw BuilderError("ApplyRewriting: initializer name is already reserved.");
    }
  }
  utils::RepeatedProtoField<NodeProto> rebuilt;
  rebuilt.reserve(final_node_count);
  std::vector<std::size_t> old_indices;
  old_indices.reserve(final_node_count);
  std::unordered_set<std::string> affected_values;
  for (const NodeProto *node : removed) {
    compute_.Shapes().EraseNodeConstraints(*node);
    for (const auto &output : node->output()) {
      if (!output.empty()) {
        affected_values.insert(output.value());
        compute_.Shapes().Erase(output.value());
      }
    }
  }
  const auto erase_removed_values = [&](const auto &initializers, const auto &positions) {
    for (const auto position : positions) {
      const std::string name = initializers[position].name().value();
      affected_values.insert(name);
      compute_.EraseValue(name);
      if (std::none_of(inputs_.begin(), inputs_.end(),
                       [&](const auto &input) { return input.name() == name; })) {
        compute_.Shapes().Erase(name);
      }
    }
  };
  erase_removed_values(initializers, removed_initializers);
  erase_removed_values(encoded_initializers_, rewrite.removed_encoded_initializers());
  erase_removed_values(paged_cache_initializers_, rewrite.removed_paged_cache_initializers());
  for (const auto &addition : added_initializers) {
    affected_values.insert(addition.initializer->name().value());
  }
  for (const auto &[source, target] : value_renames) {
    affected_values.insert(source);
    affected_values.insert(target);
  }
  std::size_t original_position = 0;
  for (std::size_t position = 0; position < final_node_count; ++position) {
    if (nodes_by_position[position] != nullptr) {
      rebuilt.push_back(*nodes_by_position[position]);
      NodeProto &node = rebuilt.back();
      const std::unordered_set<std::string> excluded(node.output().begin(), node.output().end());
      node.ref_attribute() = ImportAttributes(node, excluded);
      old_indices.push_back(nodes.size());
      for (const auto &output : node.output()) {
        if (!output.empty()) {
          names_.insert(output.value());
        }
      }
      continue;
    }
    while (original_position < nodes.size() &&
           removed.find(&nodes[original_position]) != removed.end()) {
      ++original_position;
    }
    if (original_position == nodes.size()) {
      throw BuilderError("ApplyRewriting: added node positions leave no slot for a retained node.");
    }
    rebuilt.add();
    old_indices.push_back(original_position);
    rebuilt.get(rebuilt.size() - 1) = nodes.shared_at(original_position++);
  }
  while (original_position < nodes.size() &&
         removed.find(&nodes[original_position]) != removed.end()) {
    ++original_position;
  }
  if (original_position != nodes.size()) {
    throw BuilderError(
        "ApplyRewriting: added node positions do not leave enough slots for retained nodes.");
  }
  nodes = std::move(rebuilt);
  initializers = std::move(rebuilt_initializers);
  if (!rewrite.removed_encoded_initializers().empty())
    encoded_initializers_ = std::move(rebuilt_encoded_initializers);
  if (!rewrite.removed_paged_cache_initializers().empty())
    paged_cache_initializers_ = std::move(rebuilt_paged_cache_initializers);
  PruneValueInfos();
  RefreshAfterRewriting(old_indices, std::move(affected_values));
}

GraphBuilder GraphBuilder::ApplyRewritings(const RewritingProto &rewrites) const {
  if (parent_ != nullptr || is_local_function_)
    throw BuilderError("ApplyRewritings requires a root model builder.");
  ModelProto model = model_template_;
  model.ref_struct_types() = Shapes().StructTypes();
  for (const auto &[domain, version] : opsets_)
    model.add_opset(domain, version);
  for (const auto &opset : rewrites.required_opset_imports())
    MergeImport(model.ref_opset_import(), opset);
  for (const auto &function : local_functions_)
    MergeFunction(model.ref_functions(), function->BuildFunction(function->function_domain_));
  for (const auto &function : rewrites.added_functions()) {
    const auto key = Identity(function);
    const auto identical =
        std::find_if(model.functions().begin(), model.functions().end(), [&](const auto &existing) {
          return Identity(existing) == key && existing.Equals(function);
        });
    if (identical != model.functions().end()) {
      continue;
    }
    const auto source =
        std::find_if(local_functions_.begin(), local_functions_.end(), [&](const auto &existing) {
          return FunctionKey(RewriteDomain(existing->function_domain_), existing->name_,
                             existing->function_template_.overload().value()) == key;
        });
    if (source == local_functions_.end()) {
      MergeFunction(model.ref_functions(), function);
      continue;
    }
    ModelProto comparison;
    comparison.mutable_graph();
    comparison.ref_struct_types() = Shapes().StructTypes();
    for (const auto &[domain, version] : (*source)->opsets_) {
      comparison.add_opset(domain, version);
    }
    comparison.ref_functions() = model.functions();
    const auto definition =
        std::find_if(comparison.ref_functions().begin(), comparison.ref_functions().end(),
                     [&](const auto &existing) { return Identity(existing) == key; });
    const std::size_t position =
        static_cast<std::size_t>(std::distance(comparison.ref_functions().begin(), definition));
    *definition = function;
    GraphBuilder normalized(comparison, schema_lookup_);
    const auto &candidate = *normalized.local_functions_[position];
    auto canonical = candidate.BuildFunction((*source)->function_domain_);
    if (candidate.opsets_ == (*source)->opsets_) {
      canonical.ref_opset_import() = model.functions(position).opset_import();
    }
    MergeFunction(model.ref_functions(), canonical);
  }
  *model.mutable_graph() =
      BuildGraphImpl(/*validate_persistent_bindings=*/true, /*share_initializers=*/true);
  FunctionIndex functions;
  for (const auto &function : model.functions()) {
    functions.emplace(Identity(function), &function);
  }
  for (const auto &node : model.graph().node()) {
    ValidateFunctionCalls(node, functions);
  }
  for (const auto &function : model.functions()) {
    for (const auto &node : function.node()) {
      ValidateFunctionCalls(node, functions);
    }
    for (const auto &attribute : function.attribute_proto()) {
      NodeProto defaults;
      defaults.ref_attribute().push_back(attribute);
      ValidateFunctionCalls(defaults, functions);
    }
  }
  GraphBuilder result(model, schema_lookup_);
  result.set_device(device_);
  std::size_t index = 0;
  while (index < rewrites.rewrites().size()) {
    const auto &rewrite = rewrites.rewrites(index);
    GraphBuilder *target = &result;
    for (const auto &path_component : rewrite.graph_path()) {
      const std::string name = path_component.value();
      GraphBuilder *child = FindNamedBuilder(target->subgraphs_, name);
      if (child == nullptr)
        throw BuilderError("ApplyRewriting: unknown subgraph '" + name + "'.");
      target = child;
    }
    std::size_t end = index;
    if (IsNodeOnlyRewrite(rewrite) && index + 1 < rewrites.rewrites().size() &&
        SameGraphPath(rewrite, rewrites.rewrites(index + 1)) &&
        IsNodeOnlyRewrite(rewrites.rewrites(index + 1))) {
      RewritingNodeBatch batch(target->nodes_);
      while (end < rewrites.rewrites().size() && SameGraphPath(rewrite, rewrites.rewrites(end)) &&
             batch.Append(rewrites.rewrites(end)))
        ++end;
      if (end > index + 1)
        target->ApplyRewriting(batch.Rewriting(), model.functions());
    }
    if (end <= index + 1) {
      target->ApplyRewriting(rewrite, model.functions());
      end = index + 1;
    }
    index = end;
  }
  result.BuildGraphImpl(/*validate_persistent_bindings=*/true, /*share_initializers=*/true);
  return result;
}

void GraphGraph::RebuildSuccessors() {
  const utils::RepeatedProtoField<NodeProto> &nodes = builder_.Nodes();
  for (const NodeProto &node : nodes) {
    for (std::size_t in = 0; in < node.input().size(); ++in) {
      std::string name(node.input(static_cast<std::size_t>(in)));
      if (name.empty()) {
        continue;
      }
      std::vector<const NodeProto *> &consumers = successors_[name];
      // Avoid recording the same consumer twice when a node reads the same
      // value from more than one input slot. Consumer lists are short, so a
      // linear membership scan is cheap.
      bool present = false;
      for (const NodeProto *consumer : consumers) {
        if (consumer == &node) {
          present = true;
          break;
        }
      }
      if (!present) {
        consumers.push_back(&node);
      }
    }
    // Mark every value a referenced subgraph reads from the enclosing scope so
    // a rewrite never deletes a producer the subgraph still relies on.
    for (const GraphBuilder *subgraph : builder_.ReferencedSubgraphs(node)) {
      std::unordered_set<std::string> captured;
      subgraph->CollectImplicitInputs(captured);
      for (const std::string &name : captured) {
        subgraph_captured_.insert(name);
      }
    }
  }
}

const NodeProto *GraphGraph::NodeBefore(const std::string &name) const {
  auto it = predecessors_.find(name);
  return it == predecessors_.end() ? nullptr : it->second;
}

const std::vector<const NodeProto *> &GraphGraph::NextNodes(const std::string &name) const {
  auto it = successors_.find(name);
  return it == successors_.end() ? EmptyNodeList() : it->second;
}

std::vector<const NodeProto *> GraphGraph::Predecessors(const NodeProto &node) const {
  std::vector<const NodeProto *> result;
  for (std::size_t in = 0; in < node.input().size(); ++in) {
    std::string name(node.input(static_cast<std::size_t>(in)));
    if (name.empty()) {
      continue;
    }
    const NodeProto *producer = NodeBefore(name);
    if (producer == nullptr) {
      continue;
    }
    if (std::find(result.begin(), result.end(), producer) == result.end()) {
      result.push_back(producer);
    }
  }
  return result;
}

std::vector<const NodeProto *> GraphGraph::Successors(const NodeProto &node) const {
  std::vector<const NodeProto *> result;
  for (std::size_t o = 0; o < node.output().size(); ++o) {
    std::string name(node.output(static_cast<std::size_t>(o)));
    if (name.empty()) {
      continue;
    }
    for (const NodeProto *consumer : NextNodes(name)) {
      if (std::find(result.begin(), result.end(), consumer) == result.end()) {
        result.push_back(consumer);
      }
    }
  }
  return result;
}

bool GraphGraph::IsOutput(const std::string &name) const { return output_names_.count(name) != 0; }

bool GraphGraph::IsUsedBySubgraph(const std::string &name) const {
  return subgraph_captured_.count(name) != 0;
}

bool GraphGraph::IsUsed(const std::string &name) const {
  return IsUsedBySubgraph(name) || successors_.count(name) != 0 || IsOutput(name);
}

bool GraphGraph::IsUsedMoreThanOnce(const std::string &name) const {
  if (IsUsedBySubgraph(name) || IsOutput(name)) {
    return true;
  }
  auto it = successors_.find(name);
  return it != successors_.end() && it->second.size() > 1;
}

std::size_t GraphGraph::Position(const NodeProto &node) const {
  auto it = positions_.find(&node);
  if (it == positions_.end()) {
    throw BuilderError("GraphGraph::Position: node is not part of the index.");
  }
  return it->second;
}

bool GraphGraph::HasShape(const std::string &name) const {
  return builder_.HasShape(name) || (parent_graph_ != nullptr &&
                                     parent_graph_->IsVisibleBefore(name, parent_position_limit_) &&
                                     parent_graph_->HasShape(name));
}

const SymTensor &GraphGraph::GetShape(const std::string &name) const {
  if (!builder_.HasShape(name) && parent_graph_ != nullptr &&
      parent_graph_->IsVisibleBefore(name, parent_position_limit_)) {
    return parent_graph_->GetShape(name);
  }
  return builder_.GetShape(name);
}

bool GraphGraph::HasType(const std::string &name) const {
  return HasShape(name) && GetShape(name).Dtype() != TensorType::kUndefined;
}

TensorType GraphGraph::GetType(const std::string &name) const { return GetShape(name).Dtype(); }

bool GraphGraph::IsConstant(const std::string &name) const {
  return initializers_.find(name) != initializers_.end() ||
         computed_constants_.find(name) != computed_constants_.end() ||
         builder_.Compute().IsConstantValue(name) ||
         (parent_graph_ != nullptr &&
          parent_graph_->IsVisibleBefore(name, parent_position_limit_) &&
          parent_graph_->IsConstant(name));
}

const TensorProto *GraphGraph::GetComputedConstant(const std::string &name) const {
  auto cached = computed_constants_.find(name);
  if (cached != computed_constants_.end()) {
    return &cached->second;
  }
  auto init = initializers_.find(name);
  if (init != initializers_.end()) {
    return init->second;
  }
  const NodeProto *node = NodeBefore(name);
  if (node != nullptr && node->domain().value().empty() && node->op_type().value() == "Constant") {
    const AttributeProto *value = FindAttribute(*node, "value");
    if (value != nullptr && value->has_t()) {
      return &value->t();
    }
  }
  return parent_graph_ == nullptr || !parent_graph_->IsVisibleBefore(name, parent_position_limit_)
             ? nullptr
             : parent_graph_->GetComputedConstant(name);
}

bool GraphGraph::IsVisibleBefore(const std::string &name, std::size_t position_limit) const {
  const NodeProto *producer = NodeBefore(name);
  return producer == nullptr || Position(*producer) < position_limit;
}

void GraphGraph::SetComputedConstant(const std::string &name, TensorProto value) {
  computed_constants_[name] = std::move(value);
}

bool GraphGraph::ConstantShape(const std::string &name, std::vector<int64_t> &dims) const {
  const TensorProto *tensor = GetComputedConstant(name);
  if (tensor != nullptr) {
    dims.clear();
    for (std::size_t i = 0; i < tensor->dims().size(); ++i) {
      dims.push_back(tensor->dims()[static_cast<std::size_t>(i)]);
    }
    return true;
  }
  // A ``Constant`` node using ``value_int`` / ``value_float`` is a scalar.
  const NodeProto *node = NodeBefore(name);
  if (node != nullptr && node->domain().value().empty() && node->op_type().value() == "Constant") {
    if (FindAttribute(*node, "value_int") != nullptr ||
        FindAttribute(*node, "value_float") != nullptr) {
      dims.clear();
      return true;
    }
  }
  return false;
}

bool GraphGraph::ConstantScalarValue(const std::string &name, double &out) const {
  const TensorProto *tensor = GetComputedConstant(name);
  if (tensor != nullptr) {
    return ReadScalarAsDouble(*tensor, out);
  }
  const NodeProto *node = NodeBefore(name);
  if (node != nullptr && node->domain().value().empty() && node->op_type().value() == "Constant") {
    const AttributeProto *value_int = FindAttribute(*node, "value_int");
    if (value_int != nullptr && value_int->has_i()) {
      out = static_cast<double>(value_int->i());
      return true;
    }
    const AttributeProto *value_float = FindAttribute(*node, "value_float");
    if (value_float != nullptr && value_float->has_f()) {
      out = static_cast<double>(value_float->f());
      return true;
    }
  }
  return false;
}

namespace {

// Returns ``true`` when the concrete shape ``dims`` is a scalar shape according
// to the broadcast rule: without broadcasting only ``()`` and ``(1,)`` qualify;
// with broadcasting every dimension must be ``1``.
bool IsScalarShape(const std::vector<int64_t> &dims, bool broadcast) {
  if (broadcast) {
    for (int64_t d : dims) {
      if (d != 1) {
        return false;
      }
    }
    return true;
  }
  return dims.empty() || (dims.size() == 1 && dims[0] == 1);
}

} // namespace

bool GraphGraph::IsConstantScalar(const std::string &name, bool broadcast) const {
  if (!IsConstant(name)) {
    return false;
  }
  std::vector<int64_t> dims;
  if (!ConstantShape(name, dims)) {
    return false;
  }
  return IsScalarShape(dims, broadcast);
}

bool GraphGraph::IsConstantScalar(const std::string &name, double value, bool broadcast) const {
  if (!IsConstantScalar(name, broadcast)) {
    return false;
  }
  double scalar = 0.0;
  if (!ConstantScalarValue(name, scalar)) {
    return false;
  }
  return scalar == value;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
