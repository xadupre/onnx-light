// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "onnx_core/builder/pattern_registry.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

namespace {

struct PendingReplacement {
  std::size_t position;
  utils::RepeatedProtoField<NodeProto> nodes;
};

} // namespace

GraphGraph::GraphGraph(GraphBuilder &builder) : GraphGraph(builder, CreateRegisteredPatterns()) {}

GraphGraph::GraphGraph(GraphBuilder &builder,
                       std::vector<std::unique_ptr<PatternOptimization>> patterns,
                       DoNotRemovePredicate do_not_remove)
    : builder_(builder), patterns_(std::move(patterns)), do_not_remove_(std::move(do_not_remove)) {
  for (const std::unique_ptr<PatternOptimization> &pattern : patterns_) {
    if (pattern == nullptr) {
      throw BuilderError("GraphGraph: a pattern must not be null.");
    }
  }
  std::stable_sort(patterns_.begin(), patterns_.end(), [](const auto &left, const auto &right) {
    return left->priority < right->priority;
  });
  Rebuild();
}

void GraphGraph::Rebuild() {
  predecessors_.clear();
  successors_.clear();
  positions_.clear();
  output_names_.clear();
  subgraph_captured_.clear();
  initializers_.clear();
  computed_constants_.clear();

  for (const TensorProto &initializer : builder_.Initializers()) {
    initializers_.emplace(initializer.name().value(), &initializer);
  }
  for (const ValueInfoProto &output : builder_.Outputs()) {
    output_names_.insert(output.name().value());
  }

  const utils::RepeatedProtoField<NodeProto> &nodes = builder_.Nodes();
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const NodeProto &node = nodes[i];
    positions_.emplace(&node, i);
    for (int o = 0; o < node.output().size(); ++o) {
      std::string name(node.output(static_cast<std::size_t>(o)));
      if (!name.empty()) {
        predecessors_[std::move(name)] = &node;
      }
    }
  }
  RebuildSuccessors();
}

std::size_t GraphGraph::Optimize(int max_iter) {
  if (max_iter < -1) {
    throw BuilderError("GraphGraph::Optimize: max_iter must be at least -1.");
  }

  std::vector<int> priorities;
  priorities.reserve(patterns_.size());
  for (const std::unique_ptr<PatternOptimization> &pattern : patterns_) {
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

  std::size_t applied = 0;
  std::size_t priority_index = 0;
  for (int iteration = 0; iteration < max_iter; ++iteration) {
    const int current_priority = priorities[priority_index];
    std::unordered_set<const NodeProto *> marked;
    std::vector<MatchResult> matches;

    for (const std::unique_ptr<PatternOptimization> &pattern : patterns_) {
      if (pattern->priority > current_priority) {
        break;
      }
      const std::set<std::string> fast_op_types = pattern->FastOpType();
      for (const NodeProto &candidate : builder_.nodes_) {
        if (!fast_op_types.empty() &&
            fast_op_types.find(candidate.op_type().value()) == fast_op_types.end()) {
          continue;
        }

        MatchResult match = pattern->Match(*this, candidate);
        if (match.pattern == nullptr) {
          continue;
        }
        if (match.pattern != pattern.get()) {
          throw BuilderError("GraphGraph::Optimize: a match refers to another pattern.");
        }
        if (match.nodes.empty()) {
          throw BuilderError("GraphGraph::Optimize: a non-empty match is required.");
        }

        bool skip = false;
        for (const NodeProto *node : match.nodes) {
          if (node == nullptr) {
            throw BuilderError("GraphGraph::Optimize: a matched node must not be null.");
          }
          Position(*node);
          if (marked.find(node) != marked.end() || (do_not_remove_ && do_not_remove_(*node))) {
            skip = true;
            break;
          }
        }
        if (skip) {
          continue;
        }
        if (match.insert_at != nullptr) {
          Position(*match.insert_at);
        }
        marked.insert(match.nodes.begin(), match.nodes.end());
        matches.push_back(std::move(match));
      }
    }

    std::unordered_set<const NodeProto *> removed;
    std::vector<PendingReplacement> replacements;
    replacements.reserve(matches.size());
    for (const MatchResult &match : matches) {
      std::size_t position = builder_.nodes_.size();
      if (match.insert_at != nullptr) {
        position = Position(*match.insert_at);
      } else {
        for (const NodeProto *node : match.nodes) {
          position = std::min(position, Position(*node));
        }
      }
      replacements.push_back(
          PendingReplacement{position, match.pattern->Apply(*this, match.nodes)});
      removed.insert(match.nodes.begin(), match.nodes.end());
    }

    if (!matches.empty()) {
      utils::RepeatedProtoField<NodeProto> rebuilt;
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
      applied += matches.size();
    }

    std::size_t cleaned = builder_.RemoveDuplicateNodes();
    cleaned += builder_.RemoveIdentityNodes();
    cleaned += builder_.RemoveUnusedNodes();
    Rebuild();

    if (!matches.empty() || cleaned != 0) {
      continue;
    }
    if (priority_index + 1 == priorities.size()) {
      break;
    }
    ++priority_index;
  }
  return applied;
}

void GraphGraph::RebuildSuccessors() {
  const utils::RepeatedProtoField<NodeProto> &nodes = builder_.Nodes();
  for (const NodeProto &node : nodes) {
    for (int in = 0; in < node.input().size(); ++in) {
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
  for (int in = 0; in < node.input().size(); ++in) {
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
  for (int o = 0; o < node.output().size(); ++o) {
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

bool GraphGraph::HasShape(const std::string &name) const { return builder_.HasShape(name); }

const SymTensor &GraphGraph::GetShape(const std::string &name) const {
  return builder_.GetShape(name);
}

bool GraphGraph::HasType(const std::string &name) const {
  return builder_.HasShape(name) && builder_.GetShape(name).Dtype() != TensorType::kUndefined;
}

TensorType GraphGraph::GetType(const std::string &name) const {
  return builder_.GetShape(name).Dtype();
}

bool GraphGraph::IsConstant(const std::string &name) const {
  return builder_.Compute().IsConstantValue(name);
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
  return nullptr;
}

void GraphGraph::SetComputedConstant(const std::string &name, TensorProto value) {
  computed_constants_[name] = std::move(value);
}

bool GraphGraph::ConstantShape(const std::string &name, std::vector<int64_t> &dims) const {
  const TensorProto *tensor = GetComputedConstant(name);
  if (tensor != nullptr) {
    dims.clear();
    for (int i = 0; i < tensor->dims().size(); ++i) {
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
