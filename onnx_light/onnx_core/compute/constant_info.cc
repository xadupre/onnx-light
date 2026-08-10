// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/constant_info.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx_core/graph/graph_manipulations.h"

namespace ONNX_LIGHT_NAMESPACE::core::compute {

namespace {

// Sets ``obj.metadata_props[key] = value`` (updating an existing entry rather
// than appending a duplicate), mirroring the helper used by value_tags.cc so
// the write is idempotent.
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

// Returns whether ``graph`` (or any nested subgraph) contains a
// non-deterministic op, in which case a control-flow node embedding it cannot
// be treated as constant.
bool GraphHasNonDeterministicOp(const GraphProto &graph) {
  for (int i = 0; i < graph.node().size(); ++i) {
    const NodeProto &node = graph.node()[i];
    if (IsNonDeterministicOp(node.domain(), node.op_type())) {
      return true;
    }
    for (int a = 0; a < node.attribute().size(); ++a) {
      const AttributeProto &attr = node.attribute()[a];
      if (attr.has_g() && GraphHasNonDeterministicOp(attr.g())) {
        return true;
      }
      for (int g = 0; g < attr.graphs().size(); ++g) {
        if (GraphHasNonDeterministicOp(attr.graphs()[g])) {
          return true;
        }
      }
    }
  }
  return false;
}

bool NodeSubgraphsHaveNonDeterministicOp(const NodeProto &node) {
  for (int a = 0; a < node.attribute().size(); ++a) {
    const AttributeProto &attr = node.attribute()[a];
    if (attr.has_g() && GraphHasNonDeterministicOp(attr.g())) {
      return true;
    }
    for (int g = 0; g < attr.graphs().size(); ++g) {
      if (GraphHasNonDeterministicOp(attr.graphs()[g])) {
        return true;
      }
    }
  }
  return false;
}

// Shared driver: iterates ``nodes`` to a fixed point (robust to any
// topological ordering), starting from the already-seeded ``constants`` set.
template <typename Nodes>
std::pair<std::unordered_set<std::string>, std::vector<ConstantInfo>>
InferConstantsFromNodes(const Nodes &nodes, std::unordered_set<std::string> constants) {
  std::vector<ConstantInfo> node_constant(static_cast<std::size_t>(nodes.size()),
                                          ConstantInfo::kNotConstant);
  bool changed = true;
  while (changed) {
    changed = false;
    for (int i = 0; i < nodes.size(); ++i) {
      if (node_constant[static_cast<std::size_t>(i)] == ConstantInfo::kConstant) {
        continue;
      }
      const NodeProto &node = nodes[i];
      if (!IsNodeConstant(node, constants)) {
        continue;
      }
      node_constant[static_cast<std::size_t>(i)] = ConstantInfo::kConstant;
      changed = true;
      for (int o = 0; o < node.output().size(); ++o) {
        const std::string out = node.output()[o];
        if (!out.empty()) {
          constants.insert(out);
        }
      }
    }
  }
  return {std::move(constants), std::move(node_constant)};
}

// Builds the constant seed set for a graph: outer captures plus initializers.
std::unordered_set<std::string>
GraphConstantSeed(const GraphProto &graph, const std::unordered_set<std::string> &outer_constants) {
  std::unordered_set<std::string> constants = outer_constants;
  for (int i = 0; i < graph.initializer().size(); ++i) {
    constants.insert(graph.initializer()[i].name());
  }
  return constants;
}

void WriteConstantInfoToGraphImpl(GraphProto &graph,
                                  const std::unordered_set<std::string> &outer_constants);

// Recurses into a node's subgraph attributes to write constant metadata,
// seeding each subgraph with the constant values visible from the enclosing
// scope.
void RecurseSubgraphs(NodeProto &node, const std::unordered_set<std::string> &constants) {
  for (int a = 0; a < node.attribute().size(); ++a) {
    AttributeProto *attr = node.mutable_attribute(static_cast<std::size_t>(a));
    if (attr->has_g()) {
      WriteConstantInfoToGraphImpl(*attr->mutable_g(), constants);
    }
    for (int g = 0; g < attr->graphs().size(); ++g) {
      WriteConstantInfoToGraphImpl(*attr->mutable_graphs(static_cast<std::size_t>(g)), constants);
    }
  }
}

void WriteConstantInfoToGraphImpl(GraphProto &graph,
                                  const std::unordered_set<std::string> &outer_constants) {
  const auto inferred =
      InferConstantsFromNodes(graph.node(), GraphConstantSeed(graph, outer_constants));
  const std::unordered_set<std::string> &constants = inferred.first;
  const std::vector<ConstantInfo> &node_constant = inferred.second;

  const std::size_t node_limit =
      std::min(node_constant.size(), static_cast<std::size_t>(graph.node().size()));
  for (std::size_t i = 0; i < node_limit; ++i) {
    if (node_constant[i] == ConstantInfo::kConstant) {
      SetMetadataValue(*graph.mutable_node(i), kConstantMetadataKey, "1");
    }
  }

  const auto mark_values = [&](auto &entries) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(entries.size()); ++i) {
      if (constants.count(entries[i].name()) != 0) {
        SetMetadataValue(entries[i], kConstantMetadataKey, "1");
      }
    }
  };
  mark_values(*graph.mutable_input());
  mark_values(*graph.mutable_value_info());
  mark_values(*graph.mutable_output());
  mark_values(*graph.mutable_initializer());

  for (int i = 0; i < graph.node().size(); ++i) {
    RecurseSubgraphs(*graph.mutable_node(static_cast<std::size_t>(i)), constants);
  }
}

} // namespace

bool IsNonDeterministicOp(const std::string &domain, const std::string &op_type) {
  if (!domain.empty() && domain != "ai.onnx" && domain != "onnx.ai") {
    // Only the default ONNX domain is classified; custom-domain ops are treated
    // conservatively as non-deterministic (never constant-folded here).
    return true;
  }
  static const std::unordered_set<std::string> kNonDeterministic = {
      "RandomNormal",      "RandomUniform", "RandomNormalLike",
      "RandomUniformLike", "Multinomial",   "Bernoulli"};
  return kNonDeterministic.count(op_type) != 0;
}

bool IsNodeConstant(const NodeProto &node, const std::unordered_set<std::string> &constants) {
  if (node.output().empty()) {
    return false;
  }
  if (IsNonDeterministicOp(node.domain(), node.op_type())) {
    return false;
  }
  if (NodeSubgraphsHaveNonDeterministicOp(node)) {
    return false;
  }
  const std::vector<std::string> referenced =
      ::ONNX_LIGHT_NAMESPACE::core::graph::CollectNodeInputs(node);
  for (const std::string &name : referenced) {
    if (constants.count(name) == 0) {
      return false;
    }
  }
  return true;
}

std::pair<std::unordered_set<std::string>, std::vector<ConstantInfo>>
InferConstants(const GraphProto &graph, const std::unordered_set<std::string> &outer_constants) {
  return InferConstantsFromNodes(graph.node(), GraphConstantSeed(graph, outer_constants));
}

std::pair<std::unordered_set<std::string>, std::vector<ConstantInfo>>
InferConstants(const FunctionProto &function,
               const std::unordered_set<std::string> &outer_constants) {
  return InferConstantsFromNodes(function.node(), outer_constants);
}

void WriteConstantInfoToMetadata(GraphProto &graph) { WriteConstantInfoToGraphImpl(graph, {}); }

void WriteConstantInfoToMetadata(FunctionProto &function) {
  const auto inferred = InferConstantsFromNodes(function.node(), {});
  const std::unordered_set<std::string> &constants = inferred.first;
  const std::vector<ConstantInfo> &node_constant = inferred.second;

  const std::size_t node_limit =
      std::min(node_constant.size(), static_cast<std::size_t>(function.node().size()));
  for (std::size_t i = 0; i < node_limit; ++i) {
    if (node_constant[i] == ConstantInfo::kConstant) {
      SetMetadataValue(*function.mutable_node(i), kConstantMetadataKey, "1");
    }
  }
  for (int i = 0; i < function.value_info().size(); ++i) {
    if (constants.count(function.value_info()[i].name()) != 0) {
      SetMetadataValue(*function.mutable_value_info(static_cast<std::size_t>(i)),
                       kConstantMetadataKey, "1");
    }
  }
  for (int i = 0; i < function.node().size(); ++i) {
    RecurseSubgraphs(*function.mutable_node(static_cast<std::size_t>(i)), constants);
  }
}

void WriteConstantInfoToMetadata(ModelProto &model) {
  if (model.has_graph()) {
    WriteConstantInfoToMetadata(*model.mutable_graph());
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
