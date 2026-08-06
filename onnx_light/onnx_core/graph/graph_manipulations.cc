// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/graph/graph_manipulations.h"

#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE::core::graph {

namespace {

void CollectGraphExternalInputs(const GraphProto &graph, std::vector<std::string> &out,
                                std::unordered_set<std::string> &seen,
                                const std::unordered_set<std::string> &outer_produced) {
  std::unordered_set<std::string> local;
  for (size_t i = 0; i < graph.input().size(); ++i) {
    local.insert(graph.input()[i].name());
  }
  for (size_t i = 0; i < graph.initializer().size(); ++i) {
    local.insert(graph.initializer()[i].name());
  }
  for (size_t i = 0; i < graph.node().size(); ++i) {
    const NodeProto &nd = graph.node()[i];
    for (size_t j = 0; j < nd.output().size(); ++j) {
      const std::string &name = nd.output()[j];
      if (!name.empty()) {
        local.insert(name);
      }
    }
  }

  std::unordered_set<std::string> inner_outer = outer_produced;
  inner_outer.insert(local.begin(), local.end());

  for (size_t i = 0; i < graph.node().size(); ++i) {
    const NodeProto &nd = graph.node()[i];
    for (size_t j = 0; j < nd.input().size(); ++j) {
      const std::string &name = nd.input()[j];
      if (name.empty() || local.count(name) || outer_produced.count(name)) {
        continue;
      }
      if (seen.insert(name).second) {
        out.push_back(name);
      }
    }
    for (size_t a = 0; a < nd.attribute().size(); ++a) {
      const AttributeProto &attr = nd.attribute()[a];
      if (attr.type() == AttributeProto::AttributeType::GRAPH && attr.has_g()) {
        CollectGraphExternalInputs(attr.g(), out, seen, inner_outer);
      } else if (attr.type() == AttributeProto::AttributeType::GRAPHS) {
        for (size_t k = 0; k < attr.graphs().size(); ++k) {
          CollectGraphExternalInputs(attr.graphs()[k], out, seen, inner_outer);
        }
      }
    }
  }
}

template <class NodeRange>
std::vector<std::string> CollectExternalInputsImpl(const NodeRange &nodes) {
  std::unordered_set<std::string> produced;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const NodeProto &nd = nodes[i];
    for (size_t j = 0; j < nd.output().size(); ++j) {
      const std::string &name = nd.output()[j];
      if (!name.empty()) {
        produced.insert(name);
      }
    }
  }
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const NodeProto &nd = nodes[i];
    for (size_t j = 0; j < nd.input().size(); ++j) {
      const std::string &name = nd.input()[j];
      if (name.empty() || produced.count(name)) {
        continue;
      }
      if (seen.insert(name).second) {
        out.push_back(name);
      }
    }
    for (size_t a = 0; a < nd.attribute().size(); ++a) {
      const AttributeProto &attr = nd.attribute()[a];
      if (attr.type() == AttributeProto::AttributeType::GRAPH && attr.has_g()) {
        CollectGraphExternalInputs(attr.g(), out, seen, produced);
      } else if (attr.type() == AttributeProto::AttributeType::GRAPHS) {
        for (size_t k = 0; k < attr.graphs().size(); ++k) {
          CollectGraphExternalInputs(attr.graphs()[k], out, seen, produced);
        }
      }
    }
  }
  return out;
}

template <class NodeRange>
std::vector<std::vector<std::string>>
CollectRemainingInputsImpl(const NodeRange &nodes, const std::vector<std::string> &outputs) {
  const size_t n = nodes.size();

  // Pre-compute, for every node, the names it reads (direct inputs plus the
  // external inputs captured by its subgraph attributes) and the names it
  // produces. These are independent of the starting index, so they are
  // computed only once.
  std::vector<std::vector<std::string>> deps(n);
  std::vector<std::vector<std::string>> produced(n);
  for (size_t k = 0; k < n; ++k) {
    deps[k] = CollectNodeInputs(nodes[k]);
    const NodeProto &nd = nodes[k];
    for (size_t j = 0; j < nd.output().size(); ++j) {
      const std::string &name = nd.output()[j];
      if (!name.empty()) {
        produced[k].push_back(name);
      }
    }
  }

  std::vector<std::vector<std::string>> result;
  result.reserve(n);
  for (size_t start = 0; start < n; ++start) {
    // Backward reachability over the suffix ``nodes[start..]``: starting from
    // the requested ``outputs``, a node is relevant when it produces a needed
    // name; its own inputs then become needed as well. Because ONNX graphs are
    // topologically sorted, a single backward pass discovers every ancestor.
    std::unordered_set<std::string> needed(outputs.begin(), outputs.end());
    std::vector<char> relevant(n, 0);
    for (size_t k = n; k-- > start;) {
      bool is_relevant = false;
      for (const std::string &name : produced[k]) {
        if (needed.count(name)) {
          is_relevant = true;
          break;
        }
      }
      if (!is_relevant) {
        continue;
      }
      relevant[k] = 1;
      for (const std::string &name : deps[k]) {
        needed.insert(name);
      }
    }

    // Names produced by the relevant nodes of the suffix are computed while
    // running it; everything else a relevant node reads must already be
    // available before node ``start`` runs. Collect those external ancestors in
    // first-seen order without duplicates.
    std::unordered_set<std::string> produced_in_suffix;
    for (size_t k = start; k < n; ++k) {
      if (relevant[k]) {
        for (const std::string &name : produced[k]) {
          produced_in_suffix.insert(name);
        }
      }
    }
    std::vector<std::string> remaining;
    std::unordered_set<std::string> seen;
    for (size_t k = start; k < n; ++k) {
      if (!relevant[k]) {
        continue;
      }
      for (const std::string &name : deps[k]) {
        if (produced_in_suffix.count(name)) {
          continue;
        }
        if (seen.insert(name).second) {
          remaining.push_back(name);
        }
      }
    }
    result.push_back(std::move(remaining));
  }
  return result;
}

// Adapts a vector of node pointers to the ``size()`` / ``operator[] -> const
// NodeProto&`` interface expected by the ``*Impl`` helpers above, so the
// pointer-range overloads can reuse the exact same traversal logic.
struct NodePointerRange {
  const std::vector<const NodeProto *> &nodes;
  size_t size() const { return nodes.size(); }
  const NodeProto &operator[](size_t i) const { return *nodes[i]; }
};

} // namespace

std::vector<std::string> CollectExternalInputs(const utils::RepeatedProtoField<NodeProto> &nodes) {
  return CollectExternalInputsImpl(nodes);
}

std::vector<std::string> CollectExternalInputs(const std::vector<const NodeProto *> &nodes) {
  return CollectExternalInputsImpl(NodePointerRange{nodes});
}

std::vector<std::vector<std::string>>
CollectRemainingInputs(const utils::RepeatedProtoField<NodeProto> &nodes,
                       const std::vector<std::string> &outputs) {
  return CollectRemainingInputsImpl(nodes, outputs);
}

std::vector<std::string> CollectNodeInputs(const NodeProto &node) {
  std::vector<std::string> out;
  out.reserve(node.input().size());
  std::unordered_set<std::string> seen;
  for (size_t i = 0; i < node.input().size(); ++i) {
    const std::string &name = node.input()[i];
    if (name.empty()) {
      continue;
    }
    if (seen.insert(name).second) {
      out.push_back(name);
    }
  }

  std::unordered_set<std::string> empty_outer;
  for (size_t a = 0; a < node.attribute().size(); ++a) {
    const AttributeProto &attr = node.attribute()[a];
    if (attr.type() == AttributeProto::AttributeType::GRAPH && attr.has_g()) {
      CollectGraphExternalInputs(attr.g(), out, seen, empty_outer);
    } else if (attr.type() == AttributeProto::AttributeType::GRAPHS) {
      for (size_t k = 0; k < attr.graphs().size(); ++k) {
        CollectGraphExternalInputs(attr.graphs()[k], out, seen, empty_outer);
      }
    }
  }
  return out;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
