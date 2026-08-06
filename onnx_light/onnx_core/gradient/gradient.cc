// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/gradient/gradient.h"
#include "onnx_core/gradient/grad_dispatcher.h"
#include "onnx_light_helpers.h"
#include "onnx_proto/onnx_helper.h"
#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::gradient {

namespace {

// Builds a mapping from each output name to the index of the node that
// produces it.  Iterates over [nodes_begin, nodes_end).
template <typename NodeIt>
std::unordered_map<std::string, size_t> BuildOutputToNodeIndex(NodeIt nodes_begin,
                                                               NodeIt nodes_end) {
  std::unordered_map<std::string, size_t> output_map;
  size_t i = 0;
  for (auto it = nodes_begin; it != nodes_end; ++it, ++i) {
    for (const auto &output : it->output()) {
      output_map[output] = i;
    }
  }
  return output_map;
}

// Returns the indices of nodes that transitively contribute to @p target in
// topological order (producers before consumers).
template <typename NodeIt>
std::vector<size_t> TopologicalOrder(NodeIt nodes_begin, NodeIt nodes_end,
                                     const std::string &target,
                                     const std::unordered_map<std::string, size_t> &output_map) {
  std::unordered_set<size_t> visited;
  std::vector<size_t> order;
  // Pre-size to avoid repeated allocations.
  order.reserve(static_cast<size_t>(std::distance(nodes_begin, nodes_end)));

  std::function<void(const std::string &)> visit = [&](const std::string &name) {
    auto it = output_map.find(name);
    if (it == output_map.end())
      return; // leaf input
    size_t idx = it->second;
    if (!visited.insert(idx).second)
      return; // already processed
    auto node_it = nodes_begin;
    std::advance(node_it, static_cast<std::ptrdiff_t>(idx));
    for (const auto &inp : node_it->input()) {
      if (!inp.empty())
        visit(inp);
    }
    order.push_back(idx);
  };

  visit(target);
  return order;
}

// Core algorithm shared by GradientOfNodes and GradientOfFunction.
template <typename NodeIt, typename StrIt>
FunctionProto ComputeGradient(NodeIt nodes_begin, NodeIt nodes_end, StrIt xs_begin, StrIt xs_end,
                              const std::string &y, StrIt zs_begin, StrIt zs_end,
                              const std::string &func_domain, const std::string &func_name,
                              const GradRegistry &registry) {
  EXT_ENFORCE_INVALID(xs_begin != xs_end, "onnx_gradient: xs must not be empty");
  EXT_ENFORCE_INVALID(!y.empty(), "onnx_gradient: y must not be empty");

  auto output_map = BuildOutputToNodeIndex(nodes_begin, nodes_end);

  EXT_ENFORCE_INVALID(output_map.find(y) != output_map.end(), "onnx_gradient: output '", y,
                      "' is not produced by any node");

  auto topo_order = TopologicalOrder(nodes_begin, nodes_end, y, output_map);

  // grad_table maps variable name → current gradient tensor name (in the
  // backward graph).  grad_accum holds the accumulator for each variable.
  std::unordered_map<std::string, std::string> grad_table;
  std::unordered_map<std::string, std::string> grad_accum;
  int counter = 0;

  // Seed: the gradient of y w.r.t. itself is the "dy" function input.
  grad_table[y] = "dy";

  // Build the FunctionProto.
  FunctionProto func;
  func.set_domain(func_domain);
  func.set_name(func_name);

  // Inputs: xs + zs + "dy"
  for (auto it = xs_begin; it != xs_end; ++it)
    func.add_input(*it);
  for (auto it = zs_begin; it != zs_end; ++it)
    func.add_input(*it);
  func.add_input("dy");

  // Outputs: one per xs element.
  for (auto it = xs_begin; it != xs_end; ++it)
    func.add_output("grad_" + *it);

  // Traverse nodes in reverse topological order.
  for (int i = static_cast<int>(topo_order.size()) - 1; i >= 0; --i) {
    auto node_it = nodes_begin;
    std::advance(node_it, static_cast<std::ptrdiff_t>(topo_order[static_cast<size_t>(i)]));
    ApplyBackward(*node_it, grad_table, grad_accum, counter, func, registry);
    // Promote newly accumulated gradients into grad_table so downstream
    // backward passes can use them.
    for (const auto &[var, acc] : grad_accum) {
      grad_table[var] = acc;
    }
  }

  // Emit Identity nodes to rename accumulators to canonical output names.
  for (auto it = xs_begin; it != xs_end; ++it) {
    const std::string &x = *it;
    auto git = grad_table.find(x);
    EXT_ENFORCE(git != grad_table.end(), "onnx_gradient: no gradient was computed for '", x, "'");
    const std::string out_name = "grad_" + x;
    if (git->second != out_name) {
      func.add_node("Identity", {git->second}, {out_name});
    }
  }

  // Required opset for the ops we emit.
  func.add_opset("", 21);

  return func;
}

} // namespace

FunctionProto GradientOfNodes(std::span<const NodeProto> nodes, std::span<const std::string> inputs,
                              std::span<const TensorProto> initializers,
                              std::span<const std::string> xs, const std::string &y,
                              std::span<const std::string> zs, const GradRegistry &registry) {
  (void)inputs;
  (void)initializers;
  return ComputeGradient(nodes.begin(), nodes.end(), xs.begin(), xs.end(), y, zs.begin(), zs.end(),
                         "", "gradient", registry);
}

FunctionProto GradientOfFunction(const FunctionProto &function, std::span<const std::string> xs,
                                 const std::string &y, std::span<const std::string> zs,
                                 const GradRegistry &registry) {
  std::vector<NodeProto> nodes;
  nodes.reserve(static_cast<size_t>(function.node_size()));
  for (int i = 0; i < function.node_size(); ++i) {
    nodes.push_back(function.node(i));
  }
  const std::string domain = function.has_domain() ? function.domain().value() : "";
  const std::string name = function.has_name() ? function.name().value() + "_grad" : "gradient";
  return ComputeGradient(nodes.begin(), nodes.end(), xs.begin(), xs.end(), y, zs.begin(), zs.end(),
                         domain, name, registry);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::gradient
