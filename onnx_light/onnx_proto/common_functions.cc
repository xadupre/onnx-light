// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "common_functions.h"

#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE {

namespace {

void CollectGraphExternalInputs(const GraphProto &graph, std::vector<std::string> &out,
                                std::unordered_set<std::string> &seen,
                                const std::unordered_set<std::string> &outer_produced) {
  std::unordered_set<std::string> local;
  for (size_t i = 0; i < graph.input().size(); ++i) {
    local.insert(graph.input()[i].name().as_string());
  }
  for (size_t i = 0; i < graph.initializer().size(); ++i) {
    local.insert(graph.initializer()[i].name().as_string());
  }
  for (size_t i = 0; i < graph.node().size(); ++i) {
    const NodeProto &nd = graph.node()[i];
    for (size_t j = 0; j < nd.output().size(); ++j) {
      const std::string name = nd.output()[j].as_string();
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
      const std::string name = nd.input()[j].as_string();
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
std::vector<std::string> CollectExternalInputsImpl(const NodeRange &nodes, size_t start = 0) {
  std::unordered_set<std::string> produced;
  for (size_t i = start; i < nodes.size(); ++i) {
    const NodeProto &nd = nodes[i];
    for (size_t j = 0; j < nd.output().size(); ++j) {
      const std::string name = nd.output()[j].as_string();
      if (!name.empty()) {
        produced.insert(name);
      }
    }
  }
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  for (size_t i = start; i < nodes.size(); ++i) {
    const NodeProto &nd = nodes[i];
    for (size_t j = 0; j < nd.input().size(); ++j) {
      const std::string name = nd.input()[j].as_string();
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
std::vector<std::vector<std::string>> CollectRemainingInputsImpl(const NodeRange &nodes) {
  std::vector<std::vector<std::string>> out;
  out.reserve(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    out.push_back(CollectExternalInputsImpl(nodes, i));
  }
  return out;
}

} // namespace

std::vector<std::string> CollectExternalInputs(const utils::RepeatedProtoField<NodeProto> &nodes) {
  return CollectExternalInputsImpl(nodes);
}

std::vector<std::string> CollectExternalInputs(const std::vector<NodeProto> &nodes) {
  return CollectExternalInputsImpl(nodes);
}

std::vector<std::vector<std::string>>
CollectRemainingInputs(const utils::RepeatedProtoField<NodeProto> &nodes) {
  return CollectRemainingInputsImpl(nodes);
}

std::vector<std::vector<std::string>> CollectRemainingInputs(const std::vector<NodeProto> &nodes) {
  return CollectRemainingInputsImpl(nodes);
}

std::vector<std::string> CollectNodeInputs(const NodeProto &node) {
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  for (size_t i = 0; i < node.input().size(); ++i) {
    const std::string name = node.input()[i].as_string();
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

} // namespace ONNX_LIGHT_NAMESPACE
