// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/graph/layer.h"

#include <string>
#include <unordered_set>

#include "onnx_light_helpers.h"

namespace ONNX_LIGHT_NAMESPACE::core::graph {

namespace {

void ValidateLayerGraph(const GraphProto &graph,
                        const std::unordered_set<std::string> &outer = {}) {
  std::unordered_set<std::string> local;
  auto define = [&](const std::string &name) {
    EXT_ENFORCE_INVALID(!name.empty() && local.insert(name).second,
                        "Layer: empty or duplicate body value '", name, "'.");
  };
  for (const auto &input : graph.input())
    define(input.name());
  std::unordered_set<std::string> initializers;
  auto define_initializer = [&](const std::string &name) {
    EXT_ENFORCE_INVALID(!name.empty() && initializers.insert(name).second,
                        "Layer: empty or duplicate body initializer '", name, "'.");
    local.insert(name);
  };
  for (const auto &initializer : graph.initializer())
    define_initializer(initializer.name());
  for (const auto &initializer : graph.sparse_initializer())
    define_initializer(initializer.values().name());
  for (const auto &initializer : graph.extended_initializer()) {
    EXT_ENFORCE_INVALID(initializer.has_value(), "Layer: initializer has no payload.");
    if (initializer.has_encoded_value())
      define_initializer(initializer.encoded_value().name());
    else if (initializer.has_paged_cache())
      define_initializer(initializer.paged_cache().name());
    else
      EXT_THROW_INVALID("Layer: unsupported initializer payload.");
  }
  auto require = [&](const std::string &name) {
    EXT_ENFORCE_INVALID(local.contains(name) || outer.contains(name),
                        "Layer: undeclared body dependency '", name,
                        "'; outer-scope captures must be explicit inputs.");
  };
  for (const auto &node : graph.node()) {
    for (const auto &input : node.input())
      if (!input.empty())
        require(input);
    if (node.domain() == "ai.rt" && node.op_type() == "Layer") {
      ValidateLayer(node);
    } else {
      for (const auto &attr : node.attribute()) {
        if (!attr.has_g() && attr.graphs().empty())
          continue;
        std::unordered_set<std::string> visible = outer;
        visible.insert(local.begin(), local.end());
        if (attr.has_g())
          ValidateLayerGraph(attr.g(), visible);
        for (const auto &nested : attr.graphs())
          ValidateLayerGraph(nested, visible);
      }
    }
    for (const auto &output : node.output())
      if (!output.empty())
        define(output);
  }
  std::unordered_set<std::string> outputs;
  for (const auto &output : graph.output()) {
    EXT_ENFORCE_INVALID(!output.name().empty() && outputs.insert(output.name()).second,
                        "Layer: empty or duplicate body output.");
    EXT_ENFORCE_INVALID(local.contains(output.name()),
                        "Layer: graph output must be defined locally; captured outputs require "
                        "an Identity node.");
  }
}

} // namespace

const GraphProto &ValidateLayer(const NodeProto &node) {
  const GraphProto *body = nullptr;
  for (const auto &attr : node.attribute()) {
    EXT_ENFORCE_INVALID(attr.name() == "body" && body == nullptr &&
                            attr.type() == AttributeProto::GRAPH && attr.has_g() &&
                            attr.ref_attr_name().empty(),
                        "Layer: expects exactly one GRAPH attribute named 'body'.");
    body = &attr.g();
  }
  EXT_ENFORCE_INVALID(body != nullptr, "Layer: missing required GRAPH attribute 'body'.");
  EXT_ENFORCE_INVALID(node.input_size() == body->input_size(),
                      "Layer: input arity does not match body graph inputs.");
  EXT_ENFORCE_INVALID(node.output_size() == body->output_size() && node.output_size() > 0,
                      "Layer: output arity does not match body graph outputs.");
  for (const auto &input : node.input())
    EXT_ENFORCE_INVALID(!input.empty(), "Layer: positional inputs must not be omitted.");
  std::unordered_set<std::string> outputs;
  for (const auto &output : node.output())
    EXT_ENFORCE_INVALID(!output.empty() && outputs.insert(output).second,
                        "Layer: positional outputs must not be omitted or duplicated.");
  ValidateLayerGraph(*body);
  return *body;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::graph
