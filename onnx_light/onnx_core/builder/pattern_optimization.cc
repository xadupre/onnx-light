// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/pattern_optimization.h"

#include <sstream>

namespace ONNX_LIGHT_NAMESPACE::core::builder {

namespace {

std::string NodeSummary(const NodeProto *node) {
  if (node == nullptr) {
    return "<null>";
  }
  std::ostringstream os;
  os << node->op_type().value() << "(outputs=[";
  bool first = true;
  for (int i = 0; i < node->output_size(); ++i) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << node->output(static_cast<std::size_t>(i)).value();
  }
  os << "])";
  return os.str();
}

void AppendNodes(std::ostringstream &os, const utils::RepeatedProtoField<NodeProto> &nodes) {
  os << "[";
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << NodeSummary(&nodes[i]);
  }
  os << "]";
}

} // namespace

std::string LocalRewriting::ToString() const {
  std::ostringstream os;
  os << "LocalRewriting(pattern=" << (pattern == nullptr ? "<null>" : pattern->Name())
     << ", matched_nodes=[";
  for (std::size_t i = 0; i < matched_nodes.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << matched_nodes[i];
  }
  os << "], added_nodes=";
  AppendNodes(os, added_nodes);
  os << ", added_initializers=[";
  for (std::size_t i = 0; i < added_initializers.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << added_initializers[i].name().value();
  }
  os << "], insert_at=" << insert_at << ", iteration=" << iteration << ")";
  return os.str();
}

std::string MatchResult::ToString() const {
  std::ostringstream os;
  os << "MatchResult(pattern=" << (pattern == nullptr ? "<null>" : pattern->Name()) << ", nodes=[";
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << NodeSummary(nodes[i]);
  }
  os << "], insert_at=" << NodeSummary(insert_at) << ")";
  return os.str();
}

std::string PatternOptimization::ToString() const {
  std::ostringstream os;
  os << "PatternOptimization(name=" << (Name().empty() ? "<unnamed>" : Name())
     << ", priority=" << priority << ", fast_op_types=[";
  const std::set<std::string> fast_op_types = FastOpType();
  bool first = true;
  for (const std::string &op_type : fast_op_types) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << op_type;
  }
  os << "])";
  return os.str();
}

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
