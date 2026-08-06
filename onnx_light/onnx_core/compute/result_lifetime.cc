// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/result_lifetime.h"

#include "onnx_core/graph/graph_manipulations.h"

namespace ONNX_LIGHT_NAMESPACE::core::compute {

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
    referenced = ::ONNX_LIGHT_NAMESPACE::core::graph::CollectNodeInputs(node);
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

} // namespace ONNX_LIGHT_NAMESPACE::core::compute
