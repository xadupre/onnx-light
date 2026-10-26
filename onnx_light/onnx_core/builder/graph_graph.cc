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
#include "onnx_core/runtime/kernels/run_nodes_internal.h"
#include "onnx_core/shapes/dispatch_table.h"
#include "onnx_lib/common/scoped_resource.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

namespace detail {

struct RewritingUndoState {
  struct ScopeDevices {
    Device device = Device::kUndefined;
    core::compute::ComputeContext::TagState tags;
    std::vector<std::string> initializer_names;
    std::map<std::string, ScopeDevices> subgraphs;

    bool operator==(const ScopeDevices &) const = default;
  };

  struct Edit {
    Edit() = default;

    GraphProto before;
    GraphProto after;
    std::size_t node_count = 0;
    std::size_t initializer_count = 0;
    std::size_t extended_initializer_count = 0;
  };

  RewritingProto plan;
  std::vector<Edit> edits;
  GraphProto original_graph_headers;
  utils::RepeatedProtoField<OperatorSetIdProto> original_opsets;
  utils::RepeatedProtoField<FunctionProto> original_functions;
  utils::RepeatedProtoField<StructTypeProto> original_struct_types;
  ModelProto::rewritings_t original_rewritings;
  ScopeDevices devices;
  std::map<std::tuple<std::string, std::string, std::string>, ScopeDevices> function_devices;
};

} // namespace detail

namespace {

using UndoState = detail::RewritingUndoState;

struct PendingReplacement {
  std::size_t position;
  utils::RepeatedProtoField<NodeProto> nodes;
  LocalRewritingProto rewriting;
  UndoState::Edit undo;
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
  functions.push_back(utils::CopyWithSharedTensorStorage(function));
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
  NodeProto body = utils::CopyWithSharedTensorStorage(node);
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
    FunctionProto candidate = utils::CopyWithSharedTensorStorage(existing);
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

GraphProto GraphHeader(const GraphProto &graph) {
  GraphProto result;
  result.name_ = graph.name_;
  result.sparse_initializer_ = utils::CopyWithSharedTensorStorage(graph.sparse_initializer_);
  result.doc_string_ = graph.doc_string_;
  result.input_ = graph.input_;
  result.output_ = graph.output_;
  result.value_info_ = graph.value_info_;
  result.quantization_annotation_ = graph.quantization_annotation_;
  result.metadata_props_ = graph.metadata_props_;
  result.persistent_bindings_ = graph.persistent_bindings_;
  return result;
}

using PreparedInitializers = std::map<std::vector<std::string>, std::unordered_set<std::string>>;

PreparedInitializers PreparationOutputs(const RewritingProto &plan) {
  PreparedInitializers result;
  for (const auto &rewrite : plan.rewrites()) {
    std::vector<std::string> path;
    for (const auto &component : rewrite.graph_path())
      path.push_back(component.value());
    for (const auto &call : rewrite.initializer_calls())
      for (const auto &output : call.output())
        if (output.empty() || !result[path].insert(output.value()).second)
          throw BuilderError("Rewriting: ambiguous initializer preparation output.");
  }
  return result;
}

void ClearPreparedPayload(TensorProto &tensor) {
  tensor.clear_raw_data();
  tensor.clr_float_data();
  tensor.clr_int32_data();
  tensor.clr_string_data();
  tensor.clr_int64_data();
  tensor.clr_double_data();
  tensor.clr_uint64_data();
  tensor.clr_external_data();
}

void ProjectPreparedGraph(GraphProto &graph, const PreparedInitializers &prepared,
                          const std::vector<std::string> &path);

void ProjectPreparedNode(NodeProto &node, const PreparedInitializers &prepared,
                         const std::vector<std::string> &path) {
  for (auto &attribute : node.ref_attribute()) {
    const auto visit = [&](GraphProto &child) {
      auto child_path = path;
      child_path.push_back(child.name().value());
      ProjectPreparedGraph(child, prepared, child_path);
    };
    if (attribute.has_g())
      visit(*attribute.mutable_g());
    for (auto &child : attribute.ref_graphs())
      visit(child);
  }
}

void ProjectPreparedGraph(GraphProto &graph, const PreparedInitializers &prepared,
                          const std::vector<std::string> &path) {
  const auto found = prepared.find(path);
  if (found != prepared.end())
    for (auto &initializer : graph.ref_initializer())
      if (found->second.count(initializer.name().value()) != 0)
        ClearPreparedPayload(initializer);
  for (auto &node : graph.ref_node())
    ProjectPreparedNode(node, prepared, path);
}

GraphProto GraphHeaders(const GraphProto &graph) {
  GraphProto result = GraphHeader(graph);
  for (const auto &node : graph.node()) {
    auto &entry = *result.add_node();
    for (const auto &attribute : node.attribute()) {
      if (!attribute.has_g() && attribute.graphs().empty())
        continue;
      auto &copy = *entry.add_attribute();
      copy = utils::CopyWithSharedTensorStorage(attribute);
      if (attribute.has_g())
        *copy.mutable_g() = GraphHeaders(attribute.g());
      copy.clr_graphs();
      for (const auto &child : attribute.graphs())
        copy.ref_graphs().push_back(GraphHeaders(child));
    }
  }
  return result;
}

void RestoreGraphHeaders(GraphProto &graph, const GraphProto &headers) {
  if (graph.node().size() != headers.node().size() || !headers.initializer().empty() ||
      !headers.extended_initializer().empty())
    throw BuilderError("UndoRewritings: incompatible original graph structure.");
  for (std::size_t i = 0; i < graph.node().size(); ++i) {
    NodeProto shell = headers.node(i);
    shell.clr_attribute();
    if (!shell.Equals(NodeProto{}))
      throw BuilderError("UndoRewritings: malformed original graph headers.");
    auto &node = graph.ref_node()[i];
    std::size_t graph_attributes = 0;
    for (auto &attribute : node.ref_attribute()) {
      if (!attribute.has_g() && attribute.graphs().empty())
        continue;
      ++graph_attributes;
      const auto &expected = headers.node(i).attribute();
      const auto found = std::find_if(expected.begin(), expected.end(), [&](const auto &value) {
        return value.name() == attribute.name();
      });
      if (found == expected.end() || found->has_g() != attribute.has_g() ||
          found->graphs().size() != attribute.graphs().size())
        throw BuilderError("UndoRewritings: incompatible original subgraph structure.");
      if (attribute.has_g())
        RestoreGraphHeaders(*attribute.mutable_g(), found->g());
      for (std::size_t j = 0; j < attribute.graphs().size(); ++j)
        RestoreGraphHeaders(attribute.ref_graphs()[j], found->graphs(j));
    }
    if (graph_attributes != headers.node(i).attribute().size())
      throw BuilderError("UndoRewritings: missing original subgraph.");
  }
  GraphProto restored = GraphHeader(headers);
  restored.ref_node() = std::move(graph.ref_node());
  restored.ref_initializer() = std::move(graph.ref_initializer());
  restored.ref_extended_initializer() = std::move(graph.ref_extended_initializer());
  graph = std::move(restored);
}

template <typename Field, typename Positions>
Field SelectPositions(const Field &values, const Positions &positions) {
  Field result;
  std::unordered_set<std::size_t> seen;
  for (uint64_t encoded : positions) {
    const auto position = CheckedPosition(encoded);
    if (position >= values.size() || !seen.insert(position).second)
      throw BuilderError("Rewriting: invalid or duplicate inverse position.");
    result.add() = utils::CopyWithSharedTensorStorage(values[position]);
  }
  return result;
}

template <typename Field, typename Positions>
Field UndoEntries(const Field &current, const Positions &added_positions, const Field &added_values,
                  const Positions &removed_positions, const Field &removed_values,
                  uint64_t encoded_count, const PreparedInitializers *prepared = nullptr,
                  const std::vector<std::string> &path = {}) {
  const auto original_count = CheckedPosition(encoded_count);
  if (added_positions.size() != added_values.size() ||
      removed_positions.size() != removed_values.size() || removed_values.size() > original_count ||
      original_count - removed_values.size() >
          std::numeric_limits<std::size_t>::max() - added_values.size() ||
      current.size() != original_count - removed_values.size() + added_values.size())
    throw BuilderError("UndoRewritings: incomplete inverse payload or incompatible counts.");
  const auto expected = SelectPositions(current, added_positions);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    auto actual = utils::CopyWithSharedTensorStorage(expected[i]);
    if (prepared != nullptr) {
      if constexpr (std::is_same_v<Field, utils::RepeatedProtoField<TensorProto>>) {
        const auto found = prepared->find(path);
        if (found != prepared->end() && found->second.count(actual.name().value()) != 0)
          ClearPreparedPayload(actual);
      } else if constexpr (std::is_same_v<Field, utils::RepeatedProtoField<NodeProto>>) {
        ProjectPreparedNode(actual, *prepared, path);
      }
    }
    std::string difference;
    if (!actual.Equals(added_values[i], &difference))
      throw BuilderError("UndoRewritings: replacement definition does not match: " + difference);
  }
  std::unordered_set<std::size_t> added;
  for (uint64_t position : added_positions)
    added.insert(CheckedPosition(position));
  std::unordered_map<std::size_t, std::size_t> removed;
  for (std::size_t i = 0; i < removed_positions.size(); ++i) {
    const auto position = CheckedPosition(removed_positions[i]);
    if (position >= original_count || !removed.emplace(position, i).second)
      throw BuilderError("UndoRewritings: invalid original position.");
  }
  Field result;
  std::size_t index = 0;
  for (std::size_t position = 0; position < original_count; ++position) {
    const auto found = removed.find(position);
    if (found != removed.end()) {
      result.add() = utils::CopyWithSharedTensorStorage(removed_values[found->second]);
    } else {
      while (index < current.size() && added.count(index) != 0)
        ++index;
      if (index == current.size())
        throw BuilderError("UndoRewritings: incomplete retained definitions.");
      result.add();
      result.get(result.size() - 1) = current.shared_at(index++);
    }
  }
  return result;
}

GraphProto *FindGraph(GraphProto &root, const utils::RepeatedStringField &path) {
  GraphProto *graph = &root;
  for (const auto &component : path) {
    GraphProto *found = nullptr;
    const auto visit = [&](GraphProto &child) {
      if (child.name().value() == component.value()) {
        if (found != nullptr)
          throw BuilderError("UndoRewritings: ambiguous subgraph path.");
        found = &child;
      }
    };
    for (auto &node : graph->ref_node())
      for (auto &attribute : node.ref_attribute()) {
        if (attribute.has_g())
          visit(*attribute.mutable_g());
        for (auto &child : attribute.ref_graphs())
          visit(child);
      }
    if (found == nullptr)
      throw BuilderError("UndoRewritings: unknown subgraph path.");
    graph = found;
  }
  return graph;
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
  for (std::size_t index = 0; index <= node_count; ++index) {
    const auto additions = insertions.find(index);
    if (additions != insertions.end())
      slots.insert(slots.end(), additions->second.begin(), additions->second.end());
    if (index < node_count) {
      slots.push_back(original_nodes.shared_at(index));
      originals.insert(&original_nodes[index]);
      index_node(original_nodes[index]);
    }
  }
  LiveNodeOrder nodes(std::move(slots), originals);
  // Preparation order must not change the reserved insertion order at shared anchors.
  OrderPreparationDependencies(replacements);
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
         rewrite.removed_extended_initializers().empty() && rewrite.added_initializers().empty() &&
         rewrite.added_initializer_positions().empty() && rewrite.initializer_calls().empty() &&
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

// Replay batches preserve input/output dependencies. Undo capture also batches
// cardinality-changing edits already applied by the optimizer.
// Buckets precede original nodes; live counts translate sequential coordinates.
class RewritingNodeBatch {
public:
  explicit RewritingNodeBatch(const utils::RepeatedProtoField<NodeProto> &nodes)
      : nodes_(nodes), removed_(nodes.size(), false), added_(nodes.size() + 1),
        live_(nodes.size() + 1) {
    for (std::size_t i = 0; i < removed_.size(); ++i)
      live_.Update(i, true);
  }

  bool Append(const LocalRewritingProto &rewrite, UndoState::Edit *inverse = nullptr) {
    if (!IsNodeOnlyRewrite(rewrite) ||
        rewrite.added_nodes().size() != rewrite.added_nodes_positions().size())
      return false;
    const std::size_t node_count = live_.Before(added_.size());
    std::vector<std::size_t> originals;
    utils::RepeatedProtoField<NodeProto> removed_nodes;
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
      if (inverse != nullptr)
        removed_nodes.push_back(utils::CopyWithSharedTensorStorage(nodes_[original]));
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
    if (inverse == nullptr) {
      if (originals.size() != additions.size())
        return false;
      std::sort(originals.begin(), originals.end());
      for (std::size_t i = 0; i < originals.size(); ++i) {
        const auto &original = nodes_[originals[i]];
        const auto &replacement = *additions[i].second;
        if (!std::equal(original.input().begin(), original.input().end(),
                        replacement.input().begin(), replacement.input().end()) ||
            !std::equal(original.output().begin(), original.output().end(),
                        replacement.output().begin(), replacement.output().end()))
          return false;
      }
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
    if (inverse != nullptr) {
      inverse->node_count = node_count;
      inverse->before.ref_node() = std::move(removed_nodes);
    }
    committed = true;
    return true;
  }

  LocalRewritingProto Rewriting() const {
    LocalRewritingProto rewrite;
    std::size_t position = 0;
    for (std::size_t original = 0; original < added_.size(); ++original) {
      for (const NodeProto *node : added_[original]) {
        rewrite.ref_added_nodes().push_back(utils::CopyWithSharedTensorStorage(*node));
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

void CaptureInversePlan(const RewritingProto &plan, UndoState &undo, GraphProto &canonical) {
  if (undo.edits.size() != plan.rewrites().size())
    throw BuilderError("Optimize: incomplete in-memory undo capture.");
  std::size_t index = 0;
  while (index < plan.rewrites().size()) {
    const auto &rewrite = plan.rewrites(index);
    auto &inverse = undo.edits[index];
    GraphProto &graph = *FindGraph(canonical, rewrite.graph_path());
    const auto capture_counts = [&](UndoState::Edit &edit) {
      edit.initializer_count = graph.initializer().size();
      edit.extended_initializer_count = graph.extended_initializer().size();
    };
    RewritingNodeBatch batch(graph.node());
    std::size_t end = index;
    while (end < plan.rewrites().size()) {
      const auto &edit = plan.rewrites(end);
      auto &captured = undo.edits[end];
      if (!SameGraphPath(rewrite, edit) ||
          captured.after.node().size() != edit.added_nodes().size() ||
          !batch.Append(edit, &captured))
        break;
      capture_counts(captured);
      ++end;
    }
    if (end != index) {
      // Capture independent edits in live coordinates and materialize their combined edit once.
      const auto combined = batch.Rewriting();
      const auto removed = SelectPositions(graph.node(), combined.matched_nodes());
      graph.ref_node() =
          UndoEntries(graph.node(), combined.matched_nodes(), removed,
                      combined.added_nodes_positions(), combined.added_nodes(),
                      graph.node().size() - removed.size() + combined.added_nodes().size());
      index = end;
      continue;
    }
    inverse.node_count = graph.node().size();
    capture_counts(inverse);
    auto &before = inverse.before;
    before.ref_node() = SelectPositions(graph.node(), rewrite.matched_nodes());
    before.ref_initializer() = SelectPositions(graph.initializer(), rewrite.removed_initializers());
    before.ref_extended_initializer() =
        SelectPositions(graph.extended_initializer(), rewrite.removed_extended_initializers());
    auto &after = inverse.after;
    if (after.node().size() != rewrite.added_nodes().size())
      throw BuilderError("Optimize: incomplete materialized replacement definitions.");
    if (rewrite.initializer_calls().empty())
      after.ref_initializer() = utils::CopyWithSharedTensorStorage(rewrite.added_initializers());
    graph.ref_node() = UndoEntries(
        graph.node(), rewrite.matched_nodes(), before.node(), rewrite.added_nodes_positions(),
        after.node(), graph.node().size() - before.node().size() + after.node().size());
    if (!rewrite.removed_initializers().empty() || !after.initializer().empty()) {
      auto positions = rewrite.added_initializer_positions();
      for (auto position : rewrite.computed_initializer_positions())
        positions.push_back(position);
      graph.ref_initializer() = UndoEntries(
          graph.initializer(), rewrite.removed_initializers(), before.initializer(), positions,
          after.initializer(),
          graph.initializer().size() - before.initializer().size() + after.initializer().size());
    }
    if (!rewrite.removed_extended_initializers().empty()) {
      utils::RepeatedField<uint64_t> none;
      utils::RepeatedProtoField<ExtendedInitializerProto> no_values;
      graph.ref_extended_initializer() =
          UndoEntries(graph.extended_initializer(), rewrite.removed_extended_initializers(),
                      before.extended_initializer(), none, no_values,
                      graph.extended_initializer().size() - before.extended_initializer().size());
    }
    ++index;
  }
}

std::size_t UndoNodeBatch(GraphProto &graph, const RewritingProto &plan, const UndoState &undo,
                          std::size_t end) {
  const auto &last = plan.rewrites(end - 1);
  if (!IsNodeOnlyRewrite(last))
    return end;
  RewritingNodeBatch batch(graph.node());
  std::vector<LocalRewritingProto> reversed;
  std::size_t begin = end;
  while (begin != 0) {
    const auto &rewrite = plan.rewrites(begin - 1);
    const auto &inverse = undo.edits[begin - 1];
    if (!SameGraphPath(last, rewrite) || !IsNodeOnlyRewrite(rewrite))
      break;
    if (!GraphHeader(inverse.before).Equals(GraphProto{}) ||
        !GraphHeader(inverse.after).Equals(GraphProto{}) || !inverse.before.initializer().empty() ||
        !inverse.after.initializer().empty() || !inverse.before.extended_initializer().empty() ||
        !inverse.after.extended_initializer().empty() ||
        graph.initializer().size() != inverse.initializer_count ||
        graph.extended_initializer().size() != inverse.extended_initializer_count)
      throw BuilderError("UndoRewritings: incompatible node-only undo capture.");
    auto &reverse = reversed.emplace_back();
    reverse.ref_matched_nodes() = rewrite.added_nodes_positions();
    reverse.ref_added_nodes() = utils::CopyWithSharedTensorStorage(inverse.before.node());
    reverse.ref_added_nodes_positions() = rewrite.matched_nodes();
    UndoState::Edit displaced;
    if (!batch.Append(reverse, &displaced))
      break;
    if (displaced.before.node().size() != inverse.after.node().size() ||
        displaced.node_count - displaced.before.node().size() + inverse.before.node().size() !=
            inverse.node_count)
      throw BuilderError("UndoRewritings: incompatible node-only undo counts.");
    for (std::size_t i = 0; i < displaced.before.node().size(); ++i)
      if (!displaced.before.node(i).Equals(inverse.after.node(i)))
        throw BuilderError("UndoRewritings: replacement definition does not match.");
    --begin;
  }
  if (begin != end) {
    const auto combined = batch.Rewriting();
    const auto removed = SelectPositions(graph.node(), combined.matched_nodes());
    graph.ref_node() = UndoEntries(graph.node(), combined.matched_nodes(), removed,
                                   combined.added_nodes_positions(), combined.added_nodes(),
                                   undo.edits[begin].node_count);
  }
  return begin;
}

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

std::size_t GraphGraph::Canonicalize() {
  bool reference_changed = false;
  return CanonicalizeImpl(reference_changed);
}

std::size_t GraphGraph::CanonicalizeImpl(bool &reference_changed) {
  std::size_t removed = 0;
  const auto original_nodes = utils::CopyWithSharedTensorStorage(builder_.nodes_);
  GraphBuilder::InitializerContentIndex inherited;
  std::vector<const EncodedValueProto *> encoded_inherited;
  std::unordered_set<std::string> inherited_names;
  if (!builder_.is_local_function_)
    for (auto *parent = builder_.parent_; parent != nullptr; parent = parent->parent_) {
      const auto visible = [&](const std::string &name) {
        return builder_.inherited_names_.count(name) != 0 && inherited_names.insert(name).second &&
               std::none_of(parent->inputs_.begin(), parent->inputs_.end(),
                            [&](const auto &input) { return input.name() == name; });
      };
      for (const auto &initializer : parent->initializers_)
        if (visible(initializer.name().value()))
          inherited[initializer.ContentHash(false)].push_back(&initializer);
      for (const auto &initializer : parent->extended_initializers_)
        if (initializer.has_encoded_value() && visible(initializer.name().value()))
          encoded_inherited.push_back(&initializer.encoded_value());
      if (parent->is_local_function_)
        break;
    }
  std::vector<std::string> original_initializers;
  for (const auto &initializer : builder_.initializers_)
    original_initializers.push_back(initializer.name().value());
  for (const auto &initializer : builder_.extended_initializers_)
    original_initializers.push_back(initializer.name().value());
  std::unordered_set<std::string> affected;
  std::size_t iteration, local_removed = 0;
  const auto cleanup = [&](const auto &run) {
    std::unordered_map<std::string, std::string> renames;
    const auto count = run(renames);
    for (const auto &[source, target] : renames) {
      affected.insert(source);
      affected.insert(target);
    }
    return count;
  };
  do {
    iteration =
        cleanup([&](auto &renames) { return builder_.RemoveDuplicateNodesImpl(false, &renames); });
    iteration +=
        cleanup([&](auto &renames) { return builder_.RemoveIdentityNodesImpl(false, &renames); });
    iteration += cleanup([&](auto &renames) {
      return builder_.DeduplicateInitializers(inherited, false, &renames, encoded_inherited);
    });
    local_removed += iteration;
  } while (iteration != 0);
  const auto value_info_count = builder_.value_infos_.size();
  builder_.PruneValueInfos();
  reference_changed |= local_removed != 0 || builder_.value_infos_.size() != value_info_count;
  if (local_removed != 0) {
    std::unordered_map<std::string, std::size_t> originals;
    for (std::size_t i = 0; i < original_nodes.size(); ++i)
      for (const auto &output : original_nodes[i].output())
        if (!output.empty())
          originals.emplace(output.value(), i);
    std::vector<std::size_t> indices;
    std::unordered_set<std::size_t> retained;
    for (const auto &node : builder_.nodes_) {
      const auto output = std::find_if(node.output().begin(), node.output().end(),
                                       [](const auto &value) { return !value.empty(); });
      if (output == node.output().end())
        throw BuilderError("Canonicalize: a node must produce a named value.");
      const auto index = originals.at(output->value());
      indices.push_back(index);
      retained.insert(index);
      if (!node.Equals(original_nodes[index]))
        for (const auto &name : node.output())
          if (!name.empty())
            affected.insert(name.value());
    }
    for (std::size_t i = 0; i < original_nodes.size(); ++i)
      if (retained.count(i) == 0) {
        builder_.compute_.Shapes().EraseNodeConstraints(original_nodes[i]);
        for (const auto &output : original_nodes[i].output())
          if (!output.empty()) {
            affected.insert(output.value());
            builder_.compute_.Shapes().Erase(output.value());
          }
      }
    std::unordered_set<std::string> initializers;
    for (const auto &initializer : builder_.initializers_)
      initializers.insert(initializer.name().value());
    for (const auto &initializer : builder_.extended_initializers_)
      initializers.insert(initializer.name().value());
    for (const auto &name : original_initializers)
      if (initializers.count(name) == 0) {
        affected.insert(name);
        builder_.compute_.EraseValue(name);
        builder_.compute_.Shapes().Erase(name);
      }
    builder_.RefreshAfterRewriting(indices, std::move(affected));
  }
  for (const auto &function : builder_.local_functions_) {
    GraphGraph child(*function, patterns_);
    removed += child.CanonicalizeImpl(reference_changed);
  }
  for (const auto &subgraph : builder_.subgraphs_) {
    GraphGraph child(*subgraph, patterns_);
    removed += child.CanonicalizeImpl(reference_changed);
  }
  reference_changed |= builder_.MoveShapeAndSizeNodesImpl(false) != 0;
  computed_constants_.clear();
  Rebuild();
  return removed + local_removed;
}

RewritingProto GraphGraph::Optimize(int max_iter, OptimizationReport *report,
                                    const ConstantFoldingOptions &folding_options,
                                    bool remove_unused, const OptimizationCallback &callback) {
  if (max_iter < -1)
    throw BuilderError("GraphGraph::Optimize: max_iter must be at least -1.");
  bool reference_changed = false;
  CanonicalizeImpl(reference_changed);
  GraphBuilder *root = &builder_;
  while (root->parent_ != nullptr)
    root = root->parent_;
  GraphProto canonical;
  if (&builder_ == root)
    canonical = root->BuildGraph();
  const auto original_struct_types = root->Shapes().StructTypes();
  const auto original_opsets = root->opsets_;
  auto undo = std::make_shared<UndoState>();
  if (&builder_ == root && !reference_changed)
    undo->original_rewritings = utils::CopyWithSharedTensorStorage(root->rewritings_);
  if (&builder_ == root) {
    const auto capture_devices = [&](const auto &self,
                                     const GraphBuilder &scope) -> UndoState::ScopeDevices {
      UndoState::ScopeDevices devices;
      devices.device = scope.device_;
      devices.tags = scope.compute_.CaptureTagState();
      if (scope.is_local_function_)
        for (const auto &initializer : scope.initializers_)
          devices.initializer_names.push_back(initializer.name().value());
      for (const auto &subgraph : scope.subgraphs_)
        devices.subgraphs.emplace(subgraph->name_, self(self, *subgraph));
      for (const auto &function : scope.local_functions_) {
        auto targets = self(self, *function);
        const FunctionKey key{RewriteDomain(function->function_domain_), function->name_,
                              function->function_template_.overload().value()};
        const auto [found, inserted] = undo->function_devices.emplace(key, targets);
        if (!inserted && found->second != targets)
          throw BuilderError("Optimize: conflicting device targets for exported function '" +
                             function->name_ + "'.");
      }
      return devices;
    };
    undo->devices = capture_devices(capture_devices, *root);
  }
  RewritingProto collection;
  const auto capture_functions =
      [&](const auto &self, const GraphBuilder &scope,
          const utils::RepeatedProtoField<FunctionProto> *originals) -> void {
    for (const auto &function : scope.local_functions_) {
      FunctionProto definition = function->BuildFunction(function->function_domain_);
      auto existing =
          std::find_if(collection.ref_added_functions().begin(),
                       collection.ref_added_functions().end(), [&](const FunctionProto &candidate) {
                         return Identity(candidate) == Identity(definition);
                       });
      if (existing == collection.ref_added_functions().end()) {
        collection.ref_added_functions().push_back(std::move(definition));
      } else if (!existing->Equals(definition)) {
        if (originals == nullptr) {
          MergeFunction(collection.ref_added_functions(), definition);
        } else {
          const auto original =
              std::find_if(originals->begin(), originals->end(), [&](const auto &candidate) {
                return Identity(candidate) == Identity(definition);
              });
          if (original == originals->end()) {
            MergeFunction(collection.ref_added_functions(), definition);
          } else if (existing->Equals(*original)) {
            *existing = std::move(definition);
          } else {
            MergeFunction(collection.ref_added_functions(), definition);
          }
        }
      }
      self(self, *function, originals);
    }
    for (const auto &subgraph : scope.subgraphs_)
      self(self, *subgraph, originals);
  };
  capture_functions(capture_functions, *root, nullptr);
  auto original_definitions = GraphBuilder::DependencyOrderFunctions(
      utils::CopyWithSharedTensorStorage(collection.added_functions()));
  const std::size_t original_functions = collection.added_functions().size();
  auto result = OptimizeImpl(max_iter, report, {}, folding_options, remove_unused, collection,
                             *undo, callback);
  capture_functions(capture_functions, *root, &original_definitions);
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
  if (&builder_ == root) {
    undo->original_graph_headers = GraphHeaders(canonical);
    CaptureInversePlan(result, *undo, canonical);
    undo->original_functions = std::move(original_definitions);
    for (const auto &[domain, version] : original_opsets) {
      auto &opset = undo->original_opsets.add();
      opset.set_domain(domain);
      opset.set_version(version);
    }
    undo->original_struct_types = original_struct_types;
    const auto prepared = PreparationOutputs(result);
    for (std::size_t i = 0; i < result.rewrites().size(); ++i) {
      const auto &rewrite = result.rewrites(i);
      std::vector<std::string> path;
      for (const auto &component : rewrite.graph_path())
        path.push_back(component.value());
      ProjectPreparedGraph(undo->edits[i].before, prepared, path);
      ProjectPreparedGraph(undo->edits[i].after, prepared, path);
    }
    undo->plan = utils::CopyWithSharedTensorStorage(result);
    root->rewriting_undo_ = std::move(undo);
  }
  return result;
}

bool GraphGraph::IsInsideOperatorBody(const std::string &domain, const std::string &op_type) const {
  for (const GraphGraph *scope = this; scope->parent_graph_ != nullptr;
       scope = scope->parent_graph_) {
    const auto &parent = *scope->parent_graph_;
    if (scope->parent_position_limit_ >= parent.builder_.Nodes().size())
      continue;
    const auto &node = parent.builder_.Nodes()[scope->parent_position_limit_];
    if (node.domain().value() != domain || node.op_type().value() != op_type)
      continue;
    const auto children = parent.builder_.ReferencedSubgraphs(node);
    if (std::find(children.begin(), children.end(), &scope->builder_) != children.end())
      return true;
  }
  return false;
}

RewritingProto GraphGraph::OptimizeImpl(int max_iter, OptimizationReport *report,
                                        const std::vector<std::string> &graph_path,
                                        const ConstantFoldingOptions &folding_options,
                                        bool remove_unused, RewritingProto &collection,
                                        detail::RewritingUndoState &undo,
                                        const OptimizationCallback &callback) {
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
                                 folding_options, remove_unused, collection, undo, callback);

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
        if (callback && !callback(*this, match)) {
          continue;
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
      rewriting.ref_added_nodes() = utils::CopyWithSharedTensorStorage(replacement_nodes);
      rewriting.ref_matched_nodes().reserve(match.nodes.size());
      std::unordered_set<const NodeProto *> persisted_nodes;
      for (const NodeProto *node : match.nodes) {
        if (node != nullptr && persisted_nodes.insert(node).second) {
          rewriting.add_matched_nodes(Position(*node));
        }
      }
      for (std::size_t i = initializers_before; i < builder_.initializers_.size(); ++i) {
        rewriting.ref_added_initializers().push_back(
            utils::CopyWithSharedTensorStorage(builder_.initializers_[i]));
        rewriting.add_added_initializer_positions(i);
      }
      replacements.push_back(
          PendingReplacement{position, std::move(replacement_nodes), std::move(rewriting), {}});
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
      builder_.PruneValueInfos();
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
        replacement.undo.after.ref_initializer() =
            utils::CopyWithSharedTensorStorage(replacement.rewriting.added_initializers());
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
              replacement.undo.after.ref_initializer().push_back(utils::CopyWithSharedTensorStorage(
                  builder_.initializers_[folded_positions.at(output.value())]));
            }
          } else {
            unfolded_nodes.push_back(utils::CopyWithSharedTensorStorage(node));
          }
        }
        replacement.rewriting.ref_added_nodes() = std::move(unfolded_nodes);
      }

      std::unordered_set<std::string> available = builder_.inherited_names_;
      for (const auto &input : builder_.inputs_)
        available.insert(input.name().value());
      for (const auto &initializer : builder_.initializers_)
        available.insert(initializer.name().value());
      for (const auto &initializer : builder_.extended_initializers_)
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
                                      builder_.device_, builder_.MemoryNodeMaterializer());
        builder_.compute_.RefreshIncrementalMemory(builder_.nodes_, builder_.Shapes(),
                                                   builder_.device_,
                                                   builder_.MemoryNodeMaterializer());
      }
      for (auto &replacement : replacements) {
        auto &snapshot = replacement.undo.after;
        snapshot.ref_node() =
            utils::CopyWithSharedTensorStorage(replacement.rewriting.added_nodes());
        for (auto &node : snapshot.ref_node())
          builder_.MaterializeGraphReferences(node);
        applied.push_back(std::move(replacement.rewriting));
        undo.edits.push_back(std::move(replacement.undo));
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
    const std::size_t cleaned = Cleanup(applied, remove_unused, undo);
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
                                bool remove_unused, detail::RewritingUndoState &undo) {
  const auto original_nodes = utils::CopyWithSharedTensorStorage(builder_.nodes_);
  const auto initializer_names = [](const auto &initializers) {
    std::vector<std::string> names;
    names.reserve(initializers.size());
    for (const auto &initializer : initializers)
      names.push_back(initializer.name().value());
    return names;
  };
  const auto all_initializer_names = [&]() {
    auto names = initializer_names(builder_.initializers_);
    for (const auto &initializer : builder_.extended_initializers_)
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
    const auto original_extended = initializer_names(builder_.extended_initializers_);
    const std::size_t removed = cleanup();
    const std::size_t removed_initializers =
        original_initializers.size() - builder_.initializers_.size() + original_extended.size() -
        builder_.extended_initializers_.size();
    if (removed == 0 && removed_initializers == 0) {
      return std::size_t{0};
    }
    LocalRewritingProto rewriting;
    rewriting.set_pattern_name(name);
    rewriting.ref_matched_nodes().reserve(node_count);
    for (std::size_t i = 0; i < node_count; ++i) {
      rewriting.add_matched_nodes(i);
    }
    rewriting.ref_added_nodes() = utils::CopyWithSharedTensorStorage(builder_.nodes_);
    UndoState::Edit inverse;
    auto &snapshot = inverse.after;
    snapshot.ref_node() = utils::CopyWithSharedTensorStorage(rewriting.added_nodes());
    for (auto &node : snapshot.ref_node())
      builder_.MaterializeGraphReferences(node);
    rewriting.ref_added_nodes_positions().reserve(builder_.nodes_.size());
    for (std::size_t i = 0; i < builder_.nodes_.size(); ++i) {
      rewriting.add_added_nodes_positions(i);
    }
    if (removed_initializers != 0) {
      record_removed_initializers(original_initializers, builder_.initializers_,
                                  rewriting.ref_removed_initializers());
      record_removed_initializers(original_extended, builder_.extended_initializers_,
                                  rewriting.ref_removed_extended_initializers());
    }
    rewrites.push_back(std::move(rewriting));
    undo.edits.push_back(std::move(inverse));
    return removed + removed_initializers;
  };

  std::size_t cleaned = 0;
  if (remove_unused) {
    cleaned +=
        record_nodes("RemoveUnusedNodes", [&]() { return builder_.RemoveUnusedNodesImpl(false); });
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
      rewrite.removed_extended_initializers().empty() && rewrite.added_nodes().empty() &&
      rewrite.added_initializers().empty() && rewrite.initializer_calls().empty()) {
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
  std::unordered_set<std::size_t> removed_extended;
  for (uint64_t encoded_position : rewrite.removed_extended_initializers()) {
    const std::size_t position = CheckedPosition(encoded_position);
    if (position >= extended_initializers_.size())
      throw BuilderError("ApplyRewriting: removed extended initializer position is outside "
                         "the graph.");
    if (!removed_extended.insert(position).second)
      throw BuilderError("ApplyRewriting: duplicate removed extended initializer position.");
  }
  utils::RepeatedProtoField<ExtendedInitializerProto> rebuilt_extended_initializers;
  if (!removed_extended.empty()) {
    rebuilt_extended_initializers.reserve(extended_initializers_.size() - removed_extended.size());
    for (std::size_t position = 0; position < extended_initializers_.size(); ++position) {
      if (removed_extended.count(position) == 0) {
        rebuilt_extended_initializers.add();
        rebuilt_extended_initializers.get(rebuilt_extended_initializers.size() - 1) =
            extended_initializers_.shared_at(position);
      }
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
      for (const auto &initializer : (*scope)->extended_initializers_)
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
    for (std::size_t position : removed_initializers) {
      if (initializers[position].name() == addition.initializer->name()) {
        throw BuilderError("ApplyRewriting: initializer name is already reserved.");
      }
    }
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
      rebuilt_initializers.push_back(
          utils::CopyWithSharedTensorStorage(*initializers_by_position[position]));
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
  const auto &remaining_extended_initializers =
      removed_extended.empty() ? extended_initializers_ : rebuilt_extended_initializers;
  for (const auto &initializer : remaining_extended_initializers) {
    if (initializer_names.count(initializer.name().value()) != 0) {
      throw BuilderError(
          "ApplyRewriting: an initializer name conflicts with an extended initializer.");
    }
  }

  for (const auto &addition : added_initializers) {
    // First-time defaults must respect the public-input contract.
    ValidateInputDefault(inputs_, *addition.initializer);
    const auto &name = addition.initializer->name();
    bool replaces_removed_value = false;
    for (const NodeProto *node : removed) {
      for (const auto &output : node->output()) {
        replaces_removed_value = replaces_removed_value || output == name;
      }
    }
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
  erase_removed_values(extended_initializers_, removed_extended);
  for (const auto &addition : added_initializers) {
    affected_values.insert(addition.initializer->name().value());
  }
  std::unordered_set<std::string> retired_outputs;
  for (const NodeProto *node : removed)
    for (const auto &output : node->output())
      if (!output.empty())
        retired_outputs.insert(output.value());
  for (const NodeProto *node : nodes_by_position)
    if (node != nullptr)
      for (const auto &output : node->output())
        retired_outputs.erase(output.value());
  std::size_t original_position = 0;
  for (std::size_t position = 0; position < final_node_count; ++position) {
    if (nodes_by_position[position] != nullptr) {
      rebuilt.push_back(utils::CopyWithSharedTensorStorage(*nodes_by_position[position]));
      NodeProto &node = rebuilt.back();
      std::unordered_set<std::string> excluded(node.output().begin(), node.output().end());
      if (NodeCarriesSubgraph(node))
        excluded.insert(retired_outputs.begin(), retired_outputs.end());
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
  if (!removed_extended.empty())
    extended_initializers_ = std::move(rebuilt_extended_initializers);
  PruneValueInfos();
  RefreshAfterRewriting(old_indices, std::move(affected_values));
}

void GraphBuilder::AddRewriting(const std::string &name, const RewritingProto &rewrites) {
  if (parent_ != nullptr || is_local_function_)
    throw BuilderError("AddRewriting requires a root model builder.");
  if (name.empty() || rewritings_.count(name) != 0)
    throw BuilderError("AddRewriting: empty or duplicate variant name '" + name + "'.");
  rewritings_.emplace(name, utils::CopyWithSharedTensorStorage(rewrites));
}

GraphBuilder GraphBuilder::ApplyRewritings(const std::string &name) const {
  const auto found = rewritings_.find(name);
  if (found == rewritings_.end())
    throw BuilderError("ApplyRewritings: unknown variant '" + name + "'.");
  GraphBuilder result = ApplyRewritings(found->second);
  const auto device = device_ == Device::kUndefined ? Device::kCPU : device_;
  std::set<FunctionKey> active, checked;
  std::function<void(const utils::RepeatedProtoField<NodeProto> &, std::size_t,
                     const GraphBuilder *)>
      check = [&](const utils::RepeatedProtoField<NodeProto> &nodes, std::size_t depth,
                  const GraphBuilder *scope) {
        if (depth >= 64)
          throw BuilderError("ApplyRewritings: implementation dependency nesting exceeds 64.");
        for (const auto &node : nodes) {
          const auto *function = result.Shapes().GetLocalFunction(
              ShapesContext::LocalFunctionKey(node.domain(), node.op_type(), node.overload()));
          if (function != nullptr) {
            const auto key = Identity(*function);
            const bool cacheable =
                function->attribute().empty() && function->attribute_proto().empty();
            if (!cacheable || checked.count(key) == 0) {
              if (!active.insert(key).second)
                throw BuilderError("ApplyRewritings: recursive local function dependency.");
              if (cacheable) {
                check(function->node(), depth + 1, nullptr);
              } else {
                NodeProto call = node;
                if (scope != nullptr)
                  scope->MaterializeGraphReferences(call, /*share_initializers=*/true);
                const auto bound = core::runtime::detail::BindFunctionAttributes(*function, call);
                check(bound.node(), depth + 1, nullptr);
              }
              active.erase(key);
              if (cacheable)
                checked.insert(key);
            }
            continue;
          } else {
            const std::string key =
                RewriteDomain(node.domain().value()) + ":" + node.op_type().value();
            // ResolveNodeKernelDefault handles control flow outside the dispatch table.
            const bool control_flow = RewriteDomain(node.domain().value()) == kDefaultOnnxDomain &&
                                      (node.op_type() == "If" || node.op_type() == "Loop" ||
                                       node.op_type() == "Scan" || node.op_type() == "SequenceMap");
            if (control_flow && node.op_type() == "SequenceMap" &&
                !core::runtime::GetSequenceMapPackFn()) {
              throw BuilderError("ApplyRewritings '" + name +
                                 "': no SequenceMap packing function has been registered.");
            }
            if (!control_flow &&
                core::runtime::KernelDispatchTable().count(
                    key + core::symbolic::DeviceKeySuffix(device)) == 0 &&
                core::runtime::GlobalCustomKernels().count(key) == 0) {
              throw BuilderError("ApplyRewritings '" + name + "': no implementation for '" + key +
                                 "'.");
            }
          }
          if (scope != nullptr) {
            for (const auto *child : scope->ReferencedSubgraphs(node))
              check(child->Nodes(), depth + 1, child);
          } else {
            for (const auto &attribute : node.attribute()) {
              if (attribute.has_g())
                check(attribute.g().node(), depth + 1, nullptr);
              for (const auto &graph : attribute.graphs())
                check(graph.node(), depth + 1, nullptr);
            }
          }
        }
      };
  check(result.Nodes(), 0, &result);
  result.set_device(device_);
  return result;
}

GraphBuilder GraphBuilder::ApplyRewritings(const RewritingProto &rewrites) const {
  if (parent_ != nullptr || is_local_function_)
    throw BuilderError("ApplyRewritings requires a root model builder.");
  ModelProto model = model_template_;
  model.ref_struct_types() = Shapes().StructTypes();
  for (const auto &[domain, version] : opsets_)
    model.add_opset(domain, version);
  for (const auto &function : local_functions_)
    MergeFunction(model.ref_functions(), function->BuildFunction(function->function_domain_));
  for (const auto &opset : rewrites.required_opset_imports())
    MergeImport(model.ref_opset_import(), opset);
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
      defaults.ref_attribute().push_back(utils::CopyWithSharedTensorStorage(attribute));
      ValidateFunctionCalls(defaults, functions);
    }
  }
  GraphBuilder result(model, schema_lookup_);
  const auto validate_tags = [&](const auto &self, const GraphBuilder &source,
                                 const GraphBuilder &reconstructed,
                                 const std::string &path) -> void {
    const auto &tags = reconstructed.Compute().ValueTags();
    for (const auto &[name, tag] : source.Compute().ValueTags()) {
      const auto found = tags.find(name);
      if (found == tags.end() || found->second != tag) {
        throw BuilderError("ApplyRewritings: value '" + name + "' in '" + path +
                           "' has existing tag '" + tag + "' but reconstructed tag " +
                           (found == tags.end() ? "<absent>" : "'" + found->second + "'") + ".");
      }
    }
    if (source.nodes_.size() != reconstructed.nodes_.size()) {
      throw BuilderError("ApplyRewritings: incompatible reconstructed node count in '" + path +
                         "'.");
    }
    for (std::size_t index = 0; index < source.nodes_.size(); ++index) {
      const auto originals = source.ReferencedSubgraphs(source.nodes_[index]);
      const auto children = reconstructed.ReferencedSubgraphs(reconstructed.nodes_[index]);
      for (std::size_t child = 0; child < originals.size(); ++child)
        self(self, *originals[child], *children.at(child),
             path + "/subgraph:" + originals[child]->name_);
    }
  };
  validate_tags(validate_tags, *this, result, name_);
  using ScopedFunction = std::pair<const GraphBuilder *, std::string>;
  std::map<FunctionIdentifier, std::vector<ScopedFunction>> source_functions;
  const auto collect_functions = [&](const auto &self, const GraphBuilder &scope,
                                     const std::string &path) -> void {
    for (const auto &function : scope.local_functions_) {
      const auto identity = MakeFunctionIdentifier(function->function_domain_, function->name_,
                                                   function->function_template_.overload());
      const auto function_path = path + "/function:" + function->function_domain_ +
                                 "::" + function->name_ + ":" +
                                 function->function_template_.overload().value();
      source_functions[identity].emplace_back(function.get(), function_path);
      self(self, *function, function_path);
    }
    for (const auto &subgraph : scope.subgraphs_)
      self(self, *subgraph, path + "/subgraph:" + subgraph->name_);
  };
  collect_functions(collect_functions, *this, name_);
  const auto validate_functions = [&](const auto &self, const GraphBuilder &scope) -> void {
    for (const auto &function : scope.local_functions_) {
      const auto identity = MakeFunctionIdentifier(function->function_domain_, function->name_,
                                                   function->function_template_.overload());
      const auto originals = source_functions.find(identity);
      if (originals != source_functions.end()) {
        for (const auto &[original, path] : originals->second)
          validate_tags(validate_tags, *original, *function, path);
      }
      self(self, *function);
    }
    for (const auto &subgraph : scope.subgraphs_)
      self(self, *subgraph);
  };
  validate_functions(validate_functions, result);
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

GraphBuilder GraphBuilder::UndoRewritings(const RewritingProto &rewrites) const {
  if (parent_ != nullptr || is_local_function_ || rewriting_undo_ == nullptr)
    throw BuilderError("UndoRewritings: requires an in-memory root-model capture.");
  const auto &undo = *rewriting_undo_;
  if (!rewrites.Equals(undo.plan))
    throw BuilderError("UndoRewritings: plan does not match the in-memory capture.");
  ModelProto model = model_template_;
  *model.mutable_graph() = BuildGraph();
  const auto prepared = PreparationOutputs(rewrites);
  std::size_t index = rewrites.rewrites().size();
  while (index != 0) {
    const auto &rewrite = rewrites.rewrites(index - 1);
    const auto &inverse = undo.edits[index - 1];
    GraphProto &graph = *FindGraph(*model.mutable_graph(), rewrite.graph_path());
    const auto begin = UndoNodeBatch(graph, rewrites, undo, index);
    if (begin != index) {
      index = begin;
      continue;
    }
    const auto &before = inverse.before;
    const auto &after = inverse.after;
    std::vector<std::string> path;
    for (const auto &component : rewrite.graph_path())
      path.push_back(component.value());
    if (!GraphHeader(before).Equals(GraphProto{}) || !GraphHeader(after).Equals(GraphProto{}) ||
        rewrite.added_nodes().size() != rewrite.added_nodes_positions().size() ||
        rewrite.added_initializers().size() != rewrite.added_initializer_positions().size())
      throw BuilderError("UndoRewritings: graph header or edit positions do not match.");
    auto initializer_positions = rewrite.added_initializer_positions();
    for (auto position : rewrite.computed_initializer_positions())
      initializer_positions.push_back(position);
    std::size_t prepared_count = 0;
    for (const auto &call : rewrite.initializer_calls()) {
      if (call.output().size() > std::numeric_limits<std::size_t>::max() - prepared_count)
        throw BuilderError("UndoRewritings: initializer count overflow.");
      prepared_count += call.output().size();
    }
    if (prepared_count != rewrite.computed_initializer_positions().size() ||
        prepared_count >
            std::numeric_limits<std::size_t>::max() - rewrite.added_initializers().size() ||
        after.initializer().size() != rewrite.added_initializers().size() + prepared_count ||
        !after.extended_initializer().empty())
      throw BuilderError("UndoRewritings: incomplete initializer preparation payload.");
    if (!rewrite.matched_nodes().empty() || !rewrite.added_nodes().empty())
      graph.ref_node() =
          UndoEntries(graph.node(), rewrite.added_nodes_positions(), after.node(),
                      rewrite.matched_nodes(), before.node(), inverse.node_count, &prepared, path);
    else if (graph.node().size() != inverse.node_count)
      throw BuilderError("UndoRewritings: incompatible retained node count.");
    if (!initializer_positions.empty() || !rewrite.removed_initializers().empty())
      graph.ref_initializer() =
          UndoEntries(graph.initializer(), initializer_positions, after.initializer(),
                      rewrite.removed_initializers(), before.initializer(),
                      inverse.initializer_count, &prepared, path);
    else if (graph.initializer().size() != inverse.initializer_count)
      throw BuilderError("UndoRewritings: incompatible retained initializer count.");
    if (!rewrite.removed_extended_initializers().empty()) {
      utils::RepeatedField<uint64_t> none;
      utils::RepeatedProtoField<ExtendedInitializerProto> no_values;
      graph.ref_extended_initializer() = UndoEntries(
          graph.extended_initializer(), none, no_values, rewrite.removed_extended_initializers(),
          before.extended_initializer(), inverse.extended_initializer_count);
    } else if (graph.extended_initializer().size() != inverse.extended_initializer_count) {
      throw BuilderError("UndoRewritings: incompatible retained extended initializer count.");
    }
    --index;
  }
  model.ref_opset_import() = undo.original_opsets;
  model.ref_functions() = utils::CopyWithSharedTensorStorage(undo.original_functions);
  model.ref_struct_types() = undo.original_struct_types;
  RestoreGraphHeaders(*model.mutable_graph(), undo.original_graph_headers);
  model.clr_rewritings();
  GraphBuilder result(model, schema_lookup_);
  const auto restore_devices = [&](const auto &self, GraphBuilder &scope,
                                   const UndoState::ScopeDevices &devices) -> void {
    if (scope.subgraphs_.size() != devices.subgraphs.size())
      throw BuilderError("UndoRewritings: incompatible restored device scopes.");
    std::size_t node_offset = 0;
    if (scope.is_local_function_ && scope.nodes_.size() != devices.tags.nodes.size()) {
      node_offset = devices.initializer_names.size();
      if (node_offset > scope.nodes_.size())
        throw BuilderError("UndoRewritings: incompatible restored function initializers.");
      for (std::size_t index = 0; index < node_offset; ++index) {
        const auto &node = scope.nodes_[index];
        if (node.op_type() != "Constant" || RewriteDomain(node.domain().value()) != "ai.onnx" ||
            node.output().size() != 1 || node.output(0) != devices.initializer_names[index])
          throw BuilderError("UndoRewritings: incompatible restored function initializer.");
      }
    }
    scope.compute_.RestoreTagState(devices.tags, scope.nodes_, node_offset,
                                   scope.MemoryNodeMaterializer());
    scope.set_device(devices.device);
    for (const auto &subgraph : scope.subgraphs_) {
      const auto found = devices.subgraphs.find(subgraph->name_);
      if (found == devices.subgraphs.end())
        throw BuilderError("UndoRewritings: missing captured subgraph device.");
      self(self, *subgraph, found->second);
    }
    for (const auto &function : scope.local_functions_) {
      const FunctionKey key{RewriteDomain(function->function_domain_), function->name_,
                            function->function_template_.overload().value()};
      const auto found = undo.function_devices.find(key);
      if (found == undo.function_devices.end())
        throw BuilderError("UndoRewritings: missing captured function device.");
      self(self, *function, found->second);
    }
  };
  restore_devices(restore_devices, result, undo.devices);
  result.rewritings_ = utils::CopyWithSharedTensorStorage(undo.original_rewritings);
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
