// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/result_lifetime.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace compute {

namespace {

// Recursively collects the names a subgraph reads from an enclosing scope: a
// node input that is neither produced within ``graph`` (its own inputs,
// initializers or node outputs) nor already produced by an ancestor subgraph
// is a capture of the enclosing scope. ``ancestor_locals`` accumulates names
// already produced by ancestor subgraphs tracked in ``ancestor_locals``.
void CollectGraphExternalInputs(const GraphProto &graph, std::vector<std::string> &out,
                                std::unordered_set<std::string> &seen,
                                const std::unordered_set<std::string> &ancestor_locals) {
  std::unordered_set<std::string> local;
  for (int i = 0; i < graph.input().size(); ++i) {
    local.insert(graph.input()[i].name());
  }
  for (int i = 0; i < graph.initializer().size(); ++i) {
    local.insert(graph.initializer()[i].name());
  }
  for (int i = 0; i < graph.node().size(); ++i) {
    const NodeProto &nd = graph.node()[i];
    for (int j = 0; j < nd.output().size(); ++j) {
      const std::string name = nd.output()[j];
      if (!name.empty()) {
        local.insert(name);
      }
    }
  }

  std::unordered_set<std::string> visible_locals = ancestor_locals;
  visible_locals.insert(local.begin(), local.end());

  for (int i = 0; i < graph.node().size(); ++i) {
    const NodeProto &nd = graph.node()[i];
    for (int j = 0; j < nd.input().size(); ++j) {
      const std::string name = nd.input()[j];
      // Keep only true captures from an outer scope: skip empty names,
      // values local to this subgraph, and names produced by ancestor
      // subgraphs. Such names are already available within the enclosing
      // control-flow body and are not captures of the top-level node itself.
      if (name.empty() || local.count(name) || ancestor_locals.count(name)) {
        continue;
      }
      if (seen.insert(name).second) {
        out.push_back(name);
      }
    }
    for (int a = 0; a < nd.attribute().size(); ++a) {
      const AttributeProto &attr = nd.attribute()[a];
      if (attr.type() == AttributeProto::AttributeType::GRAPH && attr.has_g()) {
        CollectGraphExternalInputs(attr.g(), out, seen, visible_locals);
      } else if (attr.type() == AttributeProto::AttributeType::GRAPHS) {
        for (int k = 0; k < attr.graphs().size(); ++k) {
          CollectGraphExternalInputs(attr.graphs()[k], out, seen, visible_locals);
        }
      }
    }
  }
}

} // namespace

std::vector<std::string> CollectNodeInputs(const NodeProto &node) {
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  for (int i = 0; i < node.input().size(); ++i) {
    const std::string name = node.input()[i];
    if (name.empty()) {
      continue;
    }
    if (seen.insert(name).second) {
      out.push_back(name);
    }
  }

  std::unordered_set<std::string> empty_outer;
  for (int a = 0; a < node.attribute().size(); ++a) {
    const AttributeProto &attr = node.attribute()[a];
    if (attr.type() == AttributeProto::AttributeType::GRAPH && attr.has_g()) {
      CollectGraphExternalInputs(attr.g(), out, seen, empty_outer);
    } else if (attr.type() == AttributeProto::AttributeType::GRAPHS) {
      for (int k = 0; k < attr.graphs().size(); ++k) {
        CollectGraphExternalInputs(attr.graphs()[k], out, seen, empty_outer);
      }
    }
  }
  return out;
}

ResultLifetimeInfo ComputeResultLifetimeInfo(const GraphProto &graph, bool allow_input_overwrite) {
  const int num_nodes = graph.node().size();
  ResultLifetimeInfo info;
  info.resize(static_cast<std::size_t>(num_nodes));

  // Names whose buffers must never be overwritten in place: declared graph
  // inputs, initializers and declared graph outputs are owned by the caller
  // or must outlive the run. Declared graph inputs are protected unless
  // ``allow_input_overwrite`` explicitly opts into reusing them, in which case
  // only initializers and outputs stay protected.
  for (int i = 0; i < graph.input().size(); ++i) {
    const std::string name = graph.input()[i].name();
    info.graph_inputs.insert(name);
    if (!allow_input_overwrite) {
      info.keep.insert(name);
    }
  }
  for (int i = 0; i < graph.initializer().size(); ++i) {
    const std::string name = graph.initializer()[i].name();
    info.graph_initializers.insert(name);
    info.keep.insert(name);
  }
  for (int i = 0; i < graph.output().size(); ++i) {
    info.keep.insert(graph.output()[i].name());
    info.graph_outputs.insert(graph.output()[i].name());
  }

  // Producer node index for every top-level intermediate, and the index of
  // the last node that references each name (directly or via a subgraph).
  // When input overwrite is allowed, declared graph inputs are treated as
  // available before the first node (producer index ``-1``) so they can be
  // reused once they reach their last use.
  std::vector<std::vector<std::string>> referenced_per_node(static_cast<std::size_t>(num_nodes));
  if (allow_input_overwrite) {
    for (const std::string &name : info.graph_inputs) {
      if (!name.empty()) {
        info.producer[name] = -1;
      }
    }
  }
  for (int i = 0; i < num_nodes; ++i) {
    const NodeProto &node = graph.node()[i];
    std::vector<std::string> &referenced = referenced_per_node[static_cast<std::size_t>(i)];
    referenced = CollectNodeInputs(node);
    for (const std::string &name : referenced) {
      info.last_use[name] = i;
    }
    for (int o = 0; o < node.output_size(); ++o) {
      const std::string name = node.output(o);
      if (!name.empty() && info.producer.find(name) == info.producer.end()) {
        info.producer[name] = i;
      }
    }
  }

  for (int i = 0; i < num_nodes; ++i) {
    const std::vector<std::string> &referenced = referenced_per_node[static_cast<std::size_t>(i)];
    for (const std::string &name : referenced) {
      auto use_it = info.last_use.find(name);
      if (use_it == info.last_use.end() || use_it->second != i) {
        continue;
      }
      if (info.graph_inputs.count(name) || info.graph_initializers.count(name)) {
        info[static_cast<std::size_t>(i)].not_used_after.push_back(name);
      }
      if (info.keep.count(name)) {
        continue;
      }
      auto prod_it = info.producer.find(name);
      // Releasable values here are top-level intermediates produced by an
      // earlier node. ``-1`` marks declared graph inputs when input overwrite
      // is enabled; they are not considered "results to release" metadata.
      // Values produced at this same node are not eligible.
      if (prod_it == info.producer.end() || prod_it->second < 0 || prod_it->second >= i) {
        continue;
      }
      info[static_cast<std::size_t>(i)].release_after.push_back(name);
    }
  }

  return info;
}

} // namespace compute
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
