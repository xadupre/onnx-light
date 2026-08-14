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

std::string PatternNoMatchStatistics::ToString() const {
  std::ostringstream os;
  os << source_file << ":" << source_line << "(occurrences=" << occurrences;
  if (!reason.empty()) {
    os << ", reason=" << reason;
  }
  os << ")";
  return os.str();
}

std::string PatternOptimizationStatistics::ToString() const {
  std::ostringstream os;
  os << pattern_name << "(attempts=" << attempts << ", matches=" << matches
     << ", match_time_ns=" << match_time_ns << ", apply_time_ns=" << apply_time_ns;
  if (!no_matches.empty()) {
    os << ", no_matches=[";
    for (std::size_t i = 0; i < no_matches.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << no_matches[i].ToString();
    }
    os << "]";
  }
  os << ")";
  return os.str();
}

std::string SubgraphOptimizationStatistics::ToString() const {
  std::ostringstream os;
  os << "Subgraph(path=";
  if (graph_path.empty()) {
    os << "<root>";
  } else {
    for (std::size_t i = 0; i < graph_path.size(); ++i) {
      if (i != 0) {
        os << "/";
      }
      os << graph_path[i];
    }
  }
  os << ", iterations=" << iterations << ", rewrites=" << rewrites
     << ", elapsed_time_ns=" << elapsed_time_ns << ")";
  return os.str();
}

int64_t OptimizationReport::TotalTimeNs() const noexcept {
  return matching_time_ns + rewriting_time_ns + cleanup_time_ns + constant_folding_time_ns +
         subgraph_optimization_time_ns;
}

std::string OptimizationReport::ToString() const {
  std::ostringstream os;
  os << "OptimizationReport(iterations=" << iterations << ", rewrites=" << rewrites
     << ", total_time_ns=" << TotalTimeNs() << ", phases={matching: " << matching_time_ns
     << ", rewriting: " << rewriting_time_ns << ", cleanup: " << cleanup_time_ns
     << ", constant_folding: " << constant_folding_time_ns
     << ", subgraph_optimization: " << subgraph_optimization_time_ns << "}, patterns=[";
  for (std::size_t i = 0; i < patterns.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << patterns[i].ToString();
  }
  os << "], subgraphs=[";
  for (std::size_t i = 0; i < subgraphs.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << subgraphs[i].ToString();
  }
  os << "])";
  return os.str();
}

std::string LocalRewriting::ToString() const {
  std::ostringstream os;
  os << "LocalRewriting(pattern=" << (pattern == nullptr ? "<null>" : pattern->Name())
     << ", graph_path=[";
  for (std::size_t i = 0; i < graph_path.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << graph_path[i];
  }
  os << "], matched_nodes=[";
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
  os << "], added_initializer_positions=[";
  for (std::size_t i = 0; i < added_initializer_positions.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << added_initializer_positions[i];
  }
  os << "], removed_initializers=[";
  for (std::size_t i = 0; i < removed_initializers.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << removed_initializers[i];
  }
  os << "], value_renames=[";
  for (std::size_t i = 0; i < value_renames.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << value_renames[i].first << "->" << value_renames[i].second;
  }
  os << "], insert_at=" << insert_at << ", iteration=" << iteration
     << ", match_time_ns=" << match_time_ns << ", apply_time_ns=" << apply_time_ns << ")";
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
  os << "], insert_at=" << NodeSummary(insert_at);
  if (no_match.has_value()) {
    os << ", no_match=" << no_match->ToString();
  }
  os << ")";
  return os.str();
}

std::string PatternNoMatch::ToString() const {
  std::ostringstream os;
  os << source_file << ":" << source_line << "(candidate=" << NodeSummary(candidate);
  if (!reason.empty()) {
    os << ", reason=" << reason;
  }
  os << ")";
  return os.str();
}

MatchResult PatternOptimization::NoMatchImpl(const NodeProto &candidate, std::string_view reason,
                                             const std::source_location location) const {
  MatchResult result;
  result.no_match = PatternNoMatch{&candidate, location.file_name(), location.line(), reason};
  return result;
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
