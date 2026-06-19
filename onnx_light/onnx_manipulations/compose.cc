// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// source: https://github.com/onnx/onnx/blob/main/onnx/compose.py

#include "compose.h"

#include "attr_proto_util.h"

#include <algorithm>
#include <deque>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE {

namespace {

// ---------------------------------------------------------------------------
// String-field update helper
// ---------------------------------------------------------------------------

/// Updates every element of a ``RepeatedField<utils::String>`` that appears
/// as a key in *name_map* with the corresponding mapped value.
void UpdateRepeatedStr(utils::RepeatedField<utils::String> &field,
                       const std::unordered_map<std::string, std::string> &name_map) {
  for (size_t i = 0; i < field.size(); ++i) {
    const std::string s = field[i].as_string();
    auto it = name_map.find(s);
    if (it != name_map.end()) {
      field[i] = it->second;
    }
  }
}

// ---------------------------------------------------------------------------
// Sub-graph extraction helpers
// ---------------------------------------------------------------------------

/// Returns a map from each node-output name to the index of the node that
/// produces it inside *graph*.  Empty output names are skipped.
std::unordered_map<std::string, size_t> BuildOutputDict(const GraphProto &graph) {
  std::unordered_map<std::string, size_t> outmap;
  for (size_t i = 0; i < graph.node().size(); ++i) {
    const NodeProto &nd = graph.node()[i];
    for (size_t j = 0; j < nd.output().size(); ++j) {
      const std::string name = nd.output()[j].as_string();
      if (!name.empty()) {
        outmap[name] = i;
      }
    }
  }
  return outmap;
}

/// Backward DFS from *node_output_name* that populates *reachable* with the
/// indices of nodes (in *graph*) whose outputs are transitively required to
/// compute *node_output_name*, stopping at any name listed in
/// *graph_input_names*.
void DfsSearchReachableNodes(const std::string &node_output_name,
                             const std::unordered_set<std::string> &graph_input_names,
                             const GraphProto &graph,
                             const std::unordered_map<std::string, size_t> &outmap,
                             std::unordered_set<size_t> &reachable) {
  std::vector<std::string> stack = {node_output_name};
  while (!stack.empty()) {
    const std::string current = std::move(stack.back());
    stack.pop_back();
    if (graph_input_names.count(current)) {
      continue;
    }
    auto it = outmap.find(current);
    if (it != outmap.end()) {
      const size_t idx = it->second;
      if (reachable.insert(idx).second) {
        const NodeProto &nd = graph.node()[idx];
        for (size_t j = 0; j < nd.input().size(); ++j) {
          const std::string inp = nd.input()[j].as_string();
          if (!inp.empty()) {
            stack.push_back(inp);
          }
        }
      }
    }
  }
}

/// Returns, in topological order, the nodes of *graph* that are needed to
/// compute *output_names* given that *input_names* are already available.
std::vector<NodeProto> CollectReachableNodes(const GraphProto &graph,
                                             const std::vector<std::string> &input_names,
                                             const std::vector<std::string> &output_names) {
  const std::unordered_map<std::string, size_t> outmap = BuildOutputDict(graph);
  const std::unordered_set<std::string> input_set(input_names.begin(), input_names.end());

  std::unordered_set<size_t> reachable;
  for (const std::string &name : output_names) {
    DfsSearchReachableNodes(name, input_set, graph, outmap, reachable);
  }

  // Return nodes in original (topological) order.
  std::vector<size_t> sorted_indices(reachable.begin(), reachable.end());
  std::sort(sorted_indices.begin(), sorted_indices.end());

  std::vector<NodeProto> result;
  result.reserve(sorted_indices.size());
  for (size_t idx : sorted_indices) {
    result.push_back(graph.node()[idx]);
  }
  return result;
}

/// Collects the initializers and value-infos referenced by *nodes*.
void CollectReachableTensors(const GraphProto &graph, const std::vector<NodeProto> &nodes,
                             std::vector<TensorProto> &initializers,
                             std::vector<ValueInfoProto> &value_infos) {
  // Build name→TensorProto / name→ValueInfoProto maps from the graph.
  std::unordered_map<std::string, const TensorProto *> init_map;
  for (size_t i = 0; i < graph.initializer().size(); ++i) {
    init_map[graph.initializer()[i].name().as_string()] = &graph.initializer()[i];
  }

  std::unordered_map<std::string, const ValueInfoProto *> vi_map;
  for (size_t i = 0; i < graph.value_info().size(); ++i) {
    vi_map[graph.value_info()[i].name().as_string()] = &graph.value_info()[i];
  }
  for (size_t i = 0; i < graph.input().size(); ++i) {
    vi_map[graph.input()[i].name().as_string()] = &graph.input()[i];
  }
  for (size_t i = 0; i < graph.output().size(); ++i) {
    vi_map[graph.output()[i].name().as_string()] = &graph.output()[i];
  }

  std::unordered_set<std::string> all_names;
  for (const NodeProto &nd : nodes) {
    for (size_t j = 0; j < nd.input().size(); ++j) {
      all_names.insert(nd.input()[j].as_string());
    }
    for (size_t j = 0; j < nd.output().size(); ++j) {
      all_names.insert(nd.output()[j].as_string());
    }
  }

  for (const std::string &name : all_names) {
    auto it_init = init_map.find(name);
    if (it_init != init_map.end()) {
      initializers.push_back(*it_init->second);
    }
    auto it_vi = vi_map.find(name);
    if (it_vi != vi_map.end()) {
      value_infos.push_back(*it_vi->second);
    }
  }
}

/// Builds a name→ValueInfoProto map from the graph inputs, outputs and
/// value_info entries.
std::unordered_map<std::string, ValueInfoProto> BuildValueInfoMap(const GraphProto &graph) {
  std::unordered_map<std::string, ValueInfoProto> vi_map;
  for (size_t i = 0; i < graph.value_info().size(); ++i) {
    vi_map[graph.value_info()[i].name().as_string()] = graph.value_info()[i];
  }
  for (size_t i = 0; i < graph.input().size(); ++i) {
    vi_map[graph.input()[i].name().as_string()] = graph.input()[i];
  }
  for (size_t i = 0; i < graph.output().size(); ++i) {
    vi_map[graph.output()[i].name().as_string()] = graph.output()[i];
  }
  return vi_map;
}

/// Extracts a sub-graph from *graph* that computes *output_names* starting
/// from *input_names*.  Returns a new GraphProto for that sub-graph.
/// The caller is responsible for attaching the correct opset imports to the
/// containing model.
GraphProto ExtractGraph(const GraphProto &graph, const std::vector<std::string> &input_names,
                        const std::vector<std::string> &output_names) {
  const std::unordered_map<std::string, ValueInfoProto> vi_map = BuildValueInfoMap(graph);

  // Validate requested names.
  for (const std::string &n : input_names) {
    EXT_ENFORCE_INVALID(vi_map.count(n), "The following name was not found in value_infos: ", n);
  }
  for (const std::string &n : output_names) {
    EXT_ENFORCE_INVALID(vi_map.count(n), "The following name was not found in value_infos: ", n);
  }

  std::vector<NodeProto> nodes = CollectReachableNodes(graph, input_names, output_names);

  std::vector<TensorProto> initializers;
  std::vector<ValueInfoProto> value_infos_collected;
  CollectReachableTensors(graph, nodes, initializers, value_infos_collected);

  GraphProto g;
  const std::string graph_name =
      "Extracted from {" + (graph.has_name() ? graph.name().as_string() : "") + "}";
  g.set_name(graph_name);

  for (const NodeProto &nd : nodes) {
    g.add_node(nd);
  }

  for (const std::string &n : input_names) {
    auto it = vi_map.find(n);
    if (it != vi_map.end()) {
      g.add_input(it->second);
    }
  }
  for (const std::string &n : output_names) {
    auto it = vi_map.find(n);
    if (it != vi_map.end()) {
      g.add_output(it->second);
    }
  }

  for (const TensorProto &t : initializers) {
    g.add_initializer(t);
  }

  // Add value_infos that are not graph inputs/outputs.
  const std::unordered_set<std::string> input_set(input_names.begin(), input_names.end());
  const std::unordered_set<std::string> output_set(output_names.begin(), output_names.end());
  for (const ValueInfoProto &vi : value_infos_collected) {
    const std::string name = vi.name().as_string();
    if (!input_set.count(name) && !output_set.count(name)) {
      g.add_value_info(vi);
    }
  }

  return g;
}

// ---------------------------------------------------------------------------
// connect_io – renames g2 node inputs according to reversed_io_map
// ---------------------------------------------------------------------------

void ConnectIO(GraphProto &graph, size_t start, size_t end,
               const std::unordered_map<std::string, std::string> &reversed_io_map) {
  for (size_t idx = start; idx < end; ++idx) {
    NodeProto &nd = (*graph.mutable_node())[idx];
    // Recurse into subgraph attributes.
    for (size_t a = 0; a < nd.attribute().size(); ++a) {
      AttributeProto &attr = (*nd.mutable_attribute())[a];
      if (attr.type() == AttributeProto::GRAPH && attr.has_g()) {
        ConnectIO(*attr.mutable_g(), 0, attr.g().node().size(), reversed_io_map);
      } else if (attr.type() == AttributeProto::GRAPHS) {
        for (size_t k = 0; k < attr.graphs().size(); ++k) {
          GraphProto &sg = (*attr.mutable_graphs())[k];
          ConnectIO(sg, 0, sg.node().size(), reversed_io_map);
        }
      }
    }
    UpdateRepeatedStr(*nd.mutable_input(), reversed_io_map);
  }
}

// ---------------------------------------------------------------------------
// AddPrefixGraphInPlace – core in-place implementation
// ---------------------------------------------------------------------------

void AddPrefixGraphInPlace(GraphProto &g, const std::string &prefix, bool rename_nodes,
                           bool rename_edges, bool rename_inputs, bool rename_outputs,
                           bool rename_initializers, bool rename_value_infos,
                           std::unordered_map<std::string, std::string> &name_map) {
  auto pfx = [&](const std::string &s) -> std::string { return s.empty() ? s : prefix + s; };

  // Collect edge renames: all node outputs that are not graph outputs.
  if (rename_edges) {
    std::unordered_set<std::string> graph_output_names;
    for (size_t i = 0; i < g.output().size(); ++i) {
      graph_output_names.insert(g.output()[i].name().as_string());
    }
    for (size_t i = 0; i < g.node().size(); ++i) {
      const NodeProto &nd = g.node()[i];
      for (size_t j = 0; j < nd.output().size(); ++j) {
        const std::string e = nd.output()[j].as_string();
        if (!e.empty() && !graph_output_names.count(e)) {
          name_map[e] = pfx(e);
        }
      }
    }
  }

  if (rename_inputs) {
    for (size_t i = 0; i < g.input().size(); ++i) {
      const std::string n = g.input()[i].name().as_string();
      name_map[n] = pfx(n);
    }
  }
  if (rename_outputs) {
    for (size_t i = 0; i < g.output().size(); ++i) {
      const std::string n = g.output()[i].name().as_string();
      name_map[n] = pfx(n);
    }
  }

  if (rename_nodes) {
    for (size_t i = 0; i < g.node().size(); ++i) {
      NodeProto &nd = (*g.mutable_node())[i];
      if (nd.has_name()) {
        nd.set_name(pfx(nd.name().as_string()));
      }
      // Recurse into subgraph attributes.
      for (size_t a = 0; a < nd.attribute().size(); ++a) {
        AttributeProto &attr = (*nd.mutable_attribute())[a];
        if (attr.type() == AttributeProto::GRAPH && attr.has_g()) {
          AddPrefixGraphInPlace(*attr.mutable_g(), prefix, rename_nodes, rename_edges,
                                rename_inputs, rename_outputs, rename_initializers,
                                rename_value_infos, name_map);
        }
        if (attr.type() == AttributeProto::GRAPHS) {
          for (size_t k = 0; k < attr.graphs().size(); ++k) {
            AddPrefixGraphInPlace((*attr.mutable_graphs())[k], prefix, rename_nodes, rename_edges,
                                  rename_inputs, rename_outputs, rename_initializers,
                                  rename_value_infos, name_map);
          }
        }
      }
    }
  }

  if (rename_initializers) {
    for (size_t i = 0; i < g.initializer().size(); ++i) {
      const std::string n = g.initializer()[i].name().as_string();
      name_map[n] = pfx(n);
    }
    for (size_t i = 0; i < g.sparse_initializer().size(); ++i) {
      {
        const std::string n = g.sparse_initializer()[i].values().name().as_string();
        name_map[n] = pfx(n);
      }
      {
        const std::string n = g.sparse_initializer()[i].indices().name().as_string();
        name_map[n] = pfx(n);
      }
    }
  }

  if (rename_value_infos) {
    for (size_t i = 0; i < g.value_info().size(); ++i) {
      const std::string n = g.value_info()[i].name().as_string();
      name_map[n] = pfx(n);
    }
  }

  // Apply name_map to all node inputs/outputs.
  for (size_t i = 0; i < g.node().size(); ++i) {
    NodeProto &nd = (*g.mutable_node())[i];
    UpdateRepeatedStr(*nd.mutable_output(), name_map);
    UpdateRepeatedStr(*nd.mutable_input(), name_map);
  }

  // Apply to graph inputs.
  for (size_t i = 0; i < g.input().size(); ++i) {
    ValueInfoProto &vi = (*g.mutable_input())[i];
    const std::string n = vi.name().as_string();
    auto it = name_map.find(n);
    if (it != name_map.end()) {
      vi.set_name(it->second);
    }
  }

  // Apply to graph outputs.
  for (size_t i = 0; i < g.output().size(); ++i) {
    ValueInfoProto &vi = (*g.mutable_output())[i];
    const std::string n = vi.name().as_string();
    auto it = name_map.find(n);
    if (it != name_map.end()) {
      vi.set_name(it->second);
    }
  }

  // Apply to initializers.
  for (size_t i = 0; i < g.initializer().size(); ++i) {
    TensorProto &init = (*g.mutable_initializer())[i];
    const std::string n = init.name().as_string();
    auto it = name_map.find(n);
    if (it != name_map.end()) {
      init.set_name(it->second);
    }
  }

  // Apply to sparse initializers.
  for (size_t i = 0; i < g.sparse_initializer().size(); ++i) {
    SparseTensorProto &si = (*g.mutable_sparse_initializer())[i];
    {
      const std::string n = si.values().name().as_string();
      auto it = name_map.find(n);
      if (it != name_map.end()) {
        si.values().set_name(it->second);
      }
    }
    {
      const std::string n = si.indices().name().as_string();
      auto it = name_map.find(n);
      if (it != name_map.end()) {
        si.indices().set_name(it->second);
      }
    }
  }

  // Apply to value_infos.
  for (size_t i = 0; i < g.value_info().size(); ++i) {
    ValueInfoProto &vi = (*g.mutable_value_info())[i];
    const std::string n = vi.name().as_string();
    auto it = name_map.find(n);
    if (it != name_map.end()) {
      vi.set_name(it->second);
    }
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Public API implementations
// ---------------------------------------------------------------------------

std::vector<std::pair<std::string, std::vector<std::string>>>
CheckOverlappingNames(const GraphProto &g1, const GraphProto &g2,
                      const std::vector<std::pair<std::string, std::string>> &io_map) {
  std::unordered_set<std::string> io_map_inputs;
  for (const auto &p : io_map) {
    io_map_inputs.insert(p.second);
  }

  // Collect edge names from a graph (node inputs and outputs), optionally
  // excluding a set of names from the result.
  auto edge_names = [](const GraphProto &g,
                       const std::unordered_set<std::string> &exclude) -> std::vector<std::string> {
    std::vector<std::string> result;
    for (size_t i = 0; i < g.node().size(); ++i) {
      const NodeProto &nd = g.node()[i];
      for (size_t j = 0; j < nd.input().size(); ++j) {
        const std::string s = nd.input()[j].as_string();
        if (!s.empty() && !exclude.count(s)) {
          result.push_back(s);
        }
      }
      for (size_t j = 0; j < nd.output().size(); ++j) {
        const std::string s = nd.output()[j].as_string();
        if (!s.empty() && !exclude.count(s)) {
          result.push_back(s);
        }
      }
    }
    return result;
  };

  auto overlapping = [](const std::vector<std::string> &a,
                        const std::vector<std::string> &b) -> std::vector<std::string> {
    const std::unordered_set<std::string> sb(b.begin(), b.end());
    std::unordered_set<std::string> seen;
    std::vector<std::string> result;
    for (const std::string &s : a) {
      if (sb.count(s) && seen.insert(s).second) {
        result.push_back(s);
      }
    }
    return result;
  };

  std::vector<std::pair<std::string, std::vector<std::string>>> result;

  {
    const std::unordered_set<std::string> no_exclude;
    auto ov = overlapping(edge_names(g1, no_exclude), edge_names(g2, io_map_inputs));
    if (!ov.empty()) {
      result.emplace_back("edge", std::move(ov));
    }
  }

  {
    std::vector<std::string> vi1, vi2;
    for (size_t i = 0; i < g1.value_info().size(); ++i) {
      vi1.push_back(g1.value_info()[i].name().as_string());
    }
    for (size_t i = 0; i < g2.value_info().size(); ++i) {
      vi2.push_back(g2.value_info()[i].name().as_string());
    }
    auto ov = overlapping(vi1, vi2);
    if (!ov.empty()) {
      result.emplace_back("value_info", std::move(ov));
    }
  }

  {
    std::vector<std::string> in1, in2;
    for (size_t i = 0; i < g1.initializer().size(); ++i) {
      in1.push_back(g1.initializer()[i].name().as_string());
    }
    for (size_t i = 0; i < g2.initializer().size(); ++i) {
      in2.push_back(g2.initializer()[i].name().as_string());
    }
    auto ov = overlapping(in1, in2);
    if (!ov.empty()) {
      result.emplace_back("initializer", std::move(ov));
    }
  }

  {
    std::vector<std::string> si1, si2;
    for (size_t i = 0; i < g1.sparse_initializer().size(); ++i) {
      si1.push_back(g1.sparse_initializer()[i].values().name().as_string());
      si1.push_back(g1.sparse_initializer()[i].indices().name().as_string());
    }
    for (size_t i = 0; i < g2.sparse_initializer().size(); ++i) {
      si2.push_back(g2.sparse_initializer()[i].values().name().as_string());
      si2.push_back(g2.sparse_initializer()[i].indices().name().as_string());
    }
    auto ov = overlapping(si1, si2);
    if (!ov.empty()) {
      result.emplace_back("sparse_initializer", std::move(ov));
    }
  }

  return result;
}

GraphProto AddPrefixGraph(const GraphProto &graph, const std::string &prefix, bool rename_nodes,
                          bool rename_edges, bool rename_inputs, bool rename_outputs,
                          bool rename_initializers, bool rename_value_infos, bool inplace,
                          std::unordered_map<std::string, std::string> *name_map) {
  GraphProto g;
  if (!inplace) {
    g.CopyFrom(graph);
  } else {
    g.CopyFrom(graph); // We always operate on a local copy for simplicity.
  }

  std::unordered_map<std::string, std::string> local_map;
  std::unordered_map<std::string, std::string> &used_map = name_map ? *name_map : local_map;

  AddPrefixGraphInPlace(g, prefix, rename_nodes, rename_edges, rename_inputs, rename_outputs,
                        rename_initializers, rename_value_infos, used_map);
  return g;
}

ModelProto AddPrefix(const ModelProto &model, const std::string &prefix, bool rename_nodes,
                     bool rename_edges, bool rename_inputs, bool rename_outputs,
                     bool rename_initializers, bool rename_value_infos, bool rename_functions,
                     bool /*inplace*/) {
  ModelProto m;
  m.CopyFrom(model);

  std::unordered_map<std::string, std::string> name_map;
  AddPrefixGraphInPlace(*m.mutable_graph(), prefix, rename_nodes, rename_edges, rename_inputs,
                        rename_outputs, rename_initializers, rename_value_infos, name_map);

  if (rename_functions) {
    // Build function rename map.
    std::unordered_map<std::string, std::string> f_name_map;
    for (size_t i = 0; i < m.functions().size(); ++i) {
      const std::string old_name = m.functions()[i].name().as_string();
      const std::string new_name = prefix + old_name;
      f_name_map[old_name] = new_name;
    }

    // Rename functions.
    for (size_t i = 0; i < m.functions().size(); ++i) {
      FunctionProto &fn = (*m.mutable_functions())[i];
      const std::string old_name = fn.name().as_string();
      auto it = f_name_map.find(old_name);
      if (it != f_name_map.end()) {
        fn.set_name(it->second);
      }
    }

    // Rename function call sites inside function bodies.
    for (size_t i = 0; i < m.functions().size(); ++i) {
      FunctionProto &fn = (*m.mutable_functions())[i];
      for (size_t j = 0; j < fn.node().size(); ++j) {
        NodeProto &nd = (*fn.mutable_node())[j];
        auto it = f_name_map.find(nd.op_type().as_string());
        if (it != f_name_map.end()) {
          nd.set_op_type(it->second);
        }
      }
    }

    // Rename function call sites in the main graph.
    for (size_t i = 0; i < m.graph().node().size(); ++i) {
      NodeProto &nd = (*m.mutable_graph()->mutable_node())[i];
      auto it = f_name_map.find(nd.op_type().as_string());
      if (it != f_name_map.end()) {
        nd.set_op_type(it->second);
      }
    }
  }

  return m;
}

GraphProto MergeGraphs(const GraphProto &g1_in, const GraphProto &g2_in,
                       const std::vector<std::pair<std::string, std::string>> &io_map,
                       const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs, const std::string &prefix1,
                       const std::string &prefix2, const std::string &name,
                       const std::string &doc_string) {
  // Apply prefixes if requested.
  GraphProto g1, g2;
  std::vector<std::pair<std::string, std::string>> effective_io_map = io_map;

  if (!prefix1.empty() || !prefix2.empty()) {
    if (!prefix1.empty()) {
      g1 = AddPrefixGraph(g1_in, prefix1);
    } else {
      g1.CopyFrom(g1_in);
    }
    if (!prefix2.empty()) {
      g2 = AddPrefixGraph(g2_in, prefix2);
    } else {
      g2.CopyFrom(g2_in);
    }
    // Update io_map to reflect new names.
    for (auto &p : effective_io_map) {
      if (!prefix1.empty()) {
        p.first = prefix1 + p.first;
      }
      if (!prefix2.empty()) {
        p.second = prefix2 + p.second;
      }
    }
  } else {
    g1.CopyFrom(g1_in);
    g2.CopyFrom(g2_in);
  }

  // Build lookup sets for io_map.
  std::unordered_set<std::string> io_map_g1_outs;
  std::unordered_set<std::string> io_map_g2_ins;
  std::unordered_map<std::string, std::string> reversed_io_map;
  for (const auto &p : effective_io_map) {
    io_map_g1_outs.insert(p.first);
    io_map_g2_ins.insert(p.second);
    reversed_io_map[p.second] = p.first;
  }

  std::unordered_set<std::string> g1_outs;
  for (size_t i = 0; i < g1.output().size(); ++i) {
    g1_outs.insert(g1.output()[i].name().as_string());
  }
  std::unordered_set<std::string> g2_ins;
  for (size_t i = 0; i < g2.input().size(); ++i) {
    g2_ins.insert(g2.input()[i].name().as_string());
  }

  // Optional subgraph extraction.
  if (!inputs.empty() || !outputs.empty()) {
    std::vector<std::string> g1_inputs_filt, g1_outputs_filt;
    std::vector<std::string> g2_inputs_filt, g2_outputs_filt;

    if (inputs.empty()) {
      for (size_t i = 0; i < g1.input().size(); ++i) {
        g1_inputs_filt.push_back(g1.input()[i].name().as_string());
      }
      for (size_t i = 0; i < g2.input().size(); ++i) {
        g2_inputs_filt.push_back(g2.input()[i].name().as_string());
      }
    } else {
      const std::unordered_set<std::string> input_set(inputs.begin(), inputs.end());
      for (size_t i = 0; i < g1.input().size(); ++i) {
        const std::string n = g1.input()[i].name().as_string();
        if (input_set.count(n)) {
          g1_inputs_filt.push_back(n);
        }
      }
      for (size_t i = 0; i < g2.input().size(); ++i) {
        const std::string n = g2.input()[i].name().as_string();
        if (input_set.count(n) || io_map_g2_ins.count(n)) {
          g2_inputs_filt.push_back(n);
        }
      }
    }

    if (outputs.empty()) {
      for (size_t i = 0; i < g1.output().size(); ++i) {
        g1_outputs_filt.push_back(g1.output()[i].name().as_string());
      }
      for (size_t i = 0; i < g2.output().size(); ++i) {
        g2_outputs_filt.push_back(g2.output()[i].name().as_string());
      }
    } else {
      const std::unordered_set<std::string> output_set(outputs.begin(), outputs.end());
      for (size_t i = 0; i < g1.output().size(); ++i) {
        const std::string n = g1.output()[i].name().as_string();
        if (output_set.count(n) || io_map_g1_outs.count(n)) {
          g1_outputs_filt.push_back(n);
        }
      }
      for (size_t i = 0; i < g2.output().size(); ++i) {
        const std::string n = g2.output()[i].name().as_string();
        if (output_set.count(n)) {
          g2_outputs_filt.push_back(n);
        }
      }
    }

    if (g1_inputs_filt.size() < g1.input().size() || g1_outputs_filt.size() < g1.output().size()) {
      g1 = ExtractGraph(g1, g1_inputs_filt, g1_outputs_filt);
    }
    if (g2_inputs_filt.size() < g2.input().size() || g2_outputs_filt.size() < g2.output().size()) {
      g2 = ExtractGraph(g2, g2_inputs_filt, g2_outputs_filt);
    }
  }

  // Validate io_map.
  for (const auto &p : effective_io_map) {
    EXT_ENFORCE_INVALID(g1_outs.count(p.first), "Output ", p.first, " is not present in g1");
    EXT_ENFORCE_INVALID(g2_ins.count(p.second), "Input ", p.second, " is not present in g2");
  }

  // Check for overlapping names.
  auto overlapping_names = CheckOverlappingNames(g1, g2, effective_io_map);
  if (!overlapping_names.empty()) {
    const auto &first = overlapping_names[0];
    std::string names_str;
    for (size_t i = 0; i < first.second.size(); ++i) {
      if (i > 0) {
        names_str += ", ";
      }
      names_str += first.second[i];
    }
    EXT_THROW_INVALID("Can't merge two graphs with overlapping names. Found repeated ", first.first,
                      " names: ", names_str,
                      "\nConsider using onnx_light.onnx.compose.add_prefix to add a prefix "
                      "to names in one of the graphs.");
  }

  GraphProto g;

  const size_t g2_nodes_begin = g1.node().size();
  g.mutable_node()->extend(g1.node());
  g.mutable_node()->extend(g2.node());
  const size_t g2_nodes_end = g.node().size();

  ConnectIO(g, g2_nodes_begin, g2_nodes_end, reversed_io_map);

  // Inputs.
  if (!inputs.empty()) {
    const std::unordered_set<std::string> input_set(inputs.begin(), inputs.end());
    for (size_t i = 0; i < g1.input().size(); ++i) {
      if (input_set.count(g1.input()[i].name().as_string())) {
        g.add_input(g1.input()[i]);
      }
    }
    for (size_t i = 0; i < g2.input().size(); ++i) {
      if (input_set.count(g2.input()[i].name().as_string())) {
        g.add_input(g2.input()[i]);
      }
    }
  } else {
    g.mutable_input()->extend(g1.input());
    for (size_t i = 0; i < g2.input().size(); ++i) {
      if (!io_map_g2_ins.count(g2.input()[i].name().as_string())) {
        g.add_input(g2.input()[i]);
      }
    }
  }

  // Outputs.
  if (!outputs.empty()) {
    const std::unordered_set<std::string> output_set(outputs.begin(), outputs.end());
    for (size_t i = 0; i < g1.output().size(); ++i) {
      if (output_set.count(g1.output()[i].name().as_string())) {
        g.add_output(g1.output()[i]);
      }
    }
    for (size_t i = 0; i < g2.output().size(); ++i) {
      if (output_set.count(g2.output()[i].name().as_string())) {
        g.add_output(g2.output()[i]);
      }
    }
  } else {
    for (size_t i = 0; i < g1.output().size(); ++i) {
      if (!io_map_g1_outs.count(g1.output()[i].name().as_string())) {
        g.add_output(g1.output()[i]);
      }
    }
    g.mutable_output()->extend(g2.output());
  }

  // Initializers.
  g.mutable_initializer()->extend(g1.initializer());
  for (size_t i = 0; i < g2.initializer().size(); ++i) {
    if (!io_map_g2_ins.count(g2.initializer()[i].name().as_string())) {
      g.add_initializer(g2.initializer()[i]);
    }
  }

  // Sparse initializers.
  g.mutable_sparse_initializer()->extend(g1.sparse_initializer());
  for (size_t i = 0; i < g2.sparse_initializer().size(); ++i) {
    if (!io_map_g2_ins.count(g2.sparse_initializer()[i].values().name().as_string())) {
      g.add_sparse_initializer(g2.sparse_initializer()[i]);
    }
  }

  // Value infos.
  g.mutable_value_info()->extend(g1.value_info());
  for (size_t i = 0; i < g2.value_info().size(); ++i) {
    if (!io_map_g2_ins.count(g2.value_info()[i].name().as_string())) {
      g.add_value_info(g2.value_info()[i]);
    }
  }

  // Promote connected g1 outputs to value_info (if not already there or
  // in the final outputs).
  {
    std::unordered_set<std::string> vi_names;
    for (size_t i = 0; i < g.value_info().size(); ++i) {
      vi_names.insert(g.value_info()[i].name().as_string());
    }
    std::unordered_set<std::string> out_names;
    for (size_t i = 0; i < g.output().size(); ++i) {
      out_names.insert(g.output()[i].name().as_string());
    }
    for (size_t i = 0; i < g1.output().size(); ++i) {
      const std::string n = g1.output()[i].name().as_string();
      if (io_map_g1_outs.count(n) && !vi_names.count(n) && !out_names.count(n)) {
        g.add_value_info(g1.output()[i]);
      }
    }
  }

  // Graph name and doc_string.
  if (!name.empty()) {
    g.set_name(name);
  } else {
    const std::string n1 = g1.has_name() ? g1.name().as_string() : "";
    const std::string n2 = g2.has_name() ? g2.name().as_string() : "";
    g.set_name(n1 + "_" + n2);
  }

  if (!doc_string.empty()) {
    g.set_doc_string(doc_string);
  } else {
    const std::string n1 = g1.has_name() ? g1.name().as_string() : "";
    const std::string n2 = g2.has_name() ? g2.name().as_string() : "";
    const std::string ds1 = g1.has_doc_string() ? g1.doc_string().as_string() : "";
    const std::string ds2 = g2.has_doc_string() ? g2.doc_string().as_string() : "";
    g.set_doc_string("Graph combining " + n1 + " and " + n2 + "\n" + n1 + "\n\n" + ds1 + "\n\n" +
                     n2 + "\n\n" + ds2);
  }

  return g;
}

ModelProto MergeModels(const ModelProto &m1_in, const ModelProto &m2_in,
                       const std::vector<std::pair<std::string, std::string>> &io_map,
                       const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs, const std::string &prefix1,
                       const std::string &prefix2, const std::string &name,
                       const std::string &doc_string, const std::string &producer_name,
                       const std::string &producer_version, const std::string &domain,
                       int64_t model_version) {
  const int64_t ir1 = m1_in.has_ir_version() ? m1_in.ir_version() : int64_t{0};
  const int64_t ir2 = m2_in.has_ir_version() ? m2_in.ir_version() : int64_t{0};
  EXT_ENFORCE_INVALID(ir1 == ir2, "IR version mismatch ", std::to_string(ir1),
                      " != ", std::to_string(ir2), ". Both models should have the same IR version");
  const int64_t ir_version = ir1;

  // Merge opset imports (must be compatible).
  std::unordered_map<std::string, int64_t> opset_import_map;
  auto merge_opset = [&](const utils::RepeatedProtoField<OperatorSetIdProto> &ops,
                         const ModelProto &src) {
    for (size_t i = 0; i < ops.size(); ++i) {
      const std::string dom = ops[i].domain().as_string();
      const int64_t ver = ops[i].version();
      auto it = opset_import_map.find(dom);
      if (it != opset_import_map.end()) {
        EXT_ENFORCE_INVALID(
            !(it->second != ver),
            "Can't merge two models with different operator set ids for a given domain. "
            "Got conflicting versions for domain '",
            dom, "'");
      } else {
        opset_import_map[dom] = ver;
      }
    }
    (void)src;
  };
  merge_opset(m1_in.opset_import(), m1_in);
  merge_opset(m2_in.opset_import(), m2_in);

  // Apply prefix if requested.
  ModelProto m1, m2;
  std::vector<std::pair<std::string, std::string>> effective_io_map = io_map;
  if (!prefix1.empty() || !prefix2.empty()) {
    if (!prefix1.empty()) {
      m1 = AddPrefix(m1_in, prefix1);
    } else {
      m1.CopyFrom(m1_in);
    }
    if (!prefix2.empty()) {
      m2 = AddPrefix(m2_in, prefix2);
    } else {
      m2.CopyFrom(m2_in);
    }
    for (auto &p : effective_io_map) {
      if (!prefix1.empty()) {
        p.first = prefix1 + p.first;
      }
      if (!prefix2.empty()) {
        p.second = prefix2 + p.second;
      }
    }
  } else {
    m1.CopyFrom(m1_in);
    m2.CopyFrom(m2_in);
  }

  GraphProto graph = MergeGraphs(m1.graph(), m2.graph(), effective_io_map, inputs, outputs, "", "",
                                 name, doc_string);

  ModelProto model;
  model.set_ir_version(ir_version);
  model.set_graph(graph);
  model.set_producer_name(producer_name);
  model.set_producer_version(producer_version);
  if (!domain.empty()) {
    model.set_domain(domain);
  }
  if (model_version >= 0) {
    model.set_model_version(model_version);
  }

  // Combine opset imports.
  for (const auto &kv : opset_import_map) {
    model.add_opset(kv.first, kv.second);
  }

  // Merge metadata_props.
  std::unordered_map<std::string, std::string> model_props;
  for (size_t i = 0; i < m1.metadata_props().size(); ++i) {
    model_props[m1.metadata_props()[i].key().as_string()] =
        m1.metadata_props()[i].value().as_string();
  }
  for (size_t i = 0; i < m2.metadata_props().size(); ++i) {
    const std::string key = m2.metadata_props()[i].key().as_string();
    const std::string val = m2.metadata_props()[i].value().as_string();
    auto it = model_props.find(key);
    if (it != model_props.end()) {
      EXT_ENFORCE_INVALID(
          !(it->second != val),
          "Can't merge models with different values for the same model metadata property. "
          "Found: property = ",
          key, ", with values ", it->second, " and ", val, ".");
    } else {
      model_props[key] = val;
    }
  }
  for (const auto &kv : model_props) {
    model.add_metadata(kv.first, kv.second);
  }

  // Check for overlapping local function names.
  std::unordered_set<std::string> fn_names_1;
  for (size_t i = 0; i < m1.functions().size(); ++i) {
    fn_names_1.insert(m1.functions()[i].name().as_string());
  }
  std::vector<std::string> fn_overlap;
  for (size_t i = 0; i < m2.functions().size(); ++i) {
    const std::string fn_name = m2.functions()[i].name().as_string();
    if (fn_names_1.count(fn_name)) {
      fn_overlap.push_back(fn_name);
    }
  }
  if (!fn_overlap.empty()) {
    std::string names_str;
    for (size_t i = 0; i < fn_overlap.size(); ++i) {
      if (i > 0) {
        names_str += ", ";
      }
      names_str += fn_overlap[i];
    }
    EXT_THROW_INVALID(
        "Can't merge models with overlapping local function names. Found in both graphs: ",
        names_str);
  }

  model.mutable_functions()->extend(m1.functions());
  model.mutable_functions()->extend(m2.functions());

  return model;
}

GraphProto ExpandOutDimGraph(const GraphProto &graph, int64_t dim_idx, bool /*inplace*/) {
  GraphProto g;
  g.CopyFrom(graph);

  // Build rename map: each original output name → collapsed name.
  std::unordered_map<std::string, std::string> collapsed_map;
  for (size_t i = 0; i < g.output().size(); ++i) {
    const std::string name = g.output()[i].name().as_string();
    collapsed_map[name] = name + "_collapsed_dim_" + std::to_string(dim_idx);
  }

  // Rename edges in all existing nodes.
  for (size_t i = 0; i < g.node().size(); ++i) {
    NodeProto &nd = (*g.mutable_node())[i];
    UpdateRepeatedStr(*nd.mutable_output(), collapsed_map);
    UpdateRepeatedStr(*nd.mutable_input(), collapsed_map);
  }

  // Build unique name for the dim-index constant.
  const std::string expand_dim_k =
      (g.has_name() ? g.name().as_string() : "") + "_expand_out_dim_idx";

  // Constant node holding the axis index.
  {
    TensorProto t;
    t.set_name(expand_dim_k + "-value");
    t.set_data_type(TensorProto::DataType::INT64);
    t.add_dims(1);
    t.add_int64_data(dim_idx);

    AttributeProto attr = MakeAttribute("value", std::move(t));
    NodeProto &cnode = g.add_node("Constant", {}, {expand_dim_k}, "", expand_dim_k + "-constant");
    cnode.add_attribute(attr);
  }

  // Collect original outputs before clearing.
  std::vector<ValueInfoProto> orig_outputs;
  orig_outputs.reserve(g.output().size());
  for (size_t i = 0; i < g.output().size(); ++i) {
    orig_outputs.push_back(g.output()[i]);
  }

  // Clear existing outputs.
  g.mutable_output()->clear();

  for (const ValueInfoProto &o : orig_outputs) {
    const std::string orig_name = o.name().as_string();
    const std::string prev_name = orig_name + "_collapsed_dim_" + std::to_string(dim_idx);

    // Unsqueeze node.
    NodeProto &unode = g.add_node("Unsqueeze", {prev_name, expand_dim_k}, {orig_name}, "",
                                  "unsqueeze-" + orig_name);
    (void)unode;

    // Build new output ValueInfoProto with expanded shape.
    ValueInfoProto new_vi;
    new_vi.set_name(orig_name);
    if (o.has_type() && o.type().has_tensor_type()) {
      TypeProto::Tensor *tt = new_vi.mutable_type()->mutable_tensor_type();
      if (o.type().tensor_type().has_elem_type()) {
        tt->set_elem_type(o.type().tensor_type().elem_type());
      }
      if (o.type().tensor_type().has_shape()) {
        const TensorShapeProto &old_shape = o.type().tensor_type().shape();
        TensorShapeProto *new_shape = tt->mutable_shape();
        // Insert dim at dim_idx.
        int64_t rank = static_cast<int64_t>(old_shape.dim().size());
        int64_t insert_at;
        if (dim_idx < 0) {
          insert_at = std::max<int64_t>(0, rank + dim_idx);
        } else {
          insert_at = std::min(dim_idx, rank);
        }
        for (int64_t d = 0; d < rank + 1; ++d) {
          TensorShapeProto::Dimension *dim = new_shape->add_dim();
          if (d == insert_at) {
            dim->set_dim_value(1);
          } else {
            const int64_t src = d < insert_at ? d : d - 1;
            const TensorShapeProto::Dimension &src_dim = old_shape.dim()[static_cast<size_t>(src)];
            if (src_dim.has_dim_value()) {
              dim->set_dim_value(src_dim.dim_value());
            } else if (src_dim.has_dim_param()) {
              dim->set_dim_param(src_dim.dim_param().as_string());
            }
          }
        }
      }
    }
    g.add_output(new_vi);
  }

  return g;
}

ModelProto ExpandOutDim(const ModelProto &model, int64_t dim_idx, bool /*inplace*/) {
  ModelProto m;
  m.CopyFrom(model);
  GraphProto new_graph = ExpandOutDimGraph(m.graph(), dim_idx, /*inplace=*/true);
  m.set_graph(new_graph);
  return m;
}

} // namespace ONNX_LIGHT_NAMESPACE
