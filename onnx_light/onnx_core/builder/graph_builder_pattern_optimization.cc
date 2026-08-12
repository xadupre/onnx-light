// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_builder_pattern_optimization.h"

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "onnx_core/builder/pattern_registry.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

namespace {

struct PendingReplacement {
  std::size_t position;
  utils::RepeatedProtoField<NodeProto> nodes;
};

} // namespace

GraphBuilderPatternOptimization::GraphBuilderPatternOptimization(GraphBuilder &builder)
    : GraphBuilderPatternOptimization(builder, CreateRegisteredPatterns()) {}

GraphBuilderPatternOptimization::GraphBuilderPatternOptimization(
    GraphBuilder &builder, std::vector<std::unique_ptr<PatternOptimization>> patterns,
    DoNotRemovePredicate do_not_remove)
    : builder_(builder), graph_(std::make_unique<GraphGraph>(builder)),
      patterns_(std::move(patterns)), do_not_remove_(std::move(do_not_remove)) {
  for (const std::unique_ptr<PatternOptimization> &pattern : patterns_) {
    if (pattern == nullptr) {
      throw BuilderError("GraphBuilderPatternOptimization: a pattern must not be null.");
    }
  }
  std::stable_sort(patterns_.begin(), patterns_.end(), [](const auto &left, const auto &right) {
    return left->priority < right->priority;
  });
}

void GraphBuilderPatternOptimization::RebuildGraph() {
  graph_ = std::make_unique<GraphGraph>(builder_);
}

std::size_t GraphBuilderPatternOptimization::Optimize(int max_iter) {
  if (max_iter < -1) {
    throw BuilderError("GraphBuilderPatternOptimization::Optimize: max_iter must be at least -1.");
  }

  std::vector<int> priorities;
  priorities.reserve(patterns_.size());
  for (const std::unique_ptr<PatternOptimization> &pattern : patterns_) {
    if (priorities.empty() || priorities.back() != pattern->priority) {
      priorities.push_back(pattern->priority);
    }
  }
  if (priorities.empty()) {
    priorities.push_back(0);
  }

  if (max_iter == -1) {
    max_iter =
        static_cast<int>(std::max<std::size_t>(builder_.nodes_.size(), 10) * priorities.size());
  }

  std::size_t applied = 0;
  std::size_t priority_index = 0;
  for (int iteration = 0; iteration < max_iter; ++iteration) {
    const int current_priority = priorities[priority_index];
    std::unordered_set<const NodeProto *> marked;
    std::vector<MatchResult> matches;

    for (const std::unique_ptr<PatternOptimization> &pattern : patterns_) {
      if (pattern->priority > current_priority) {
        break;
      }
      const std::set<std::string> fast_op_types = pattern->FastOpType();
      for (const NodeProto &candidate : builder_.nodes_) {
        if (!fast_op_types.empty() &&
            fast_op_types.find(candidate.op_type().value()) == fast_op_types.end()) {
          continue;
        }

        MatchResult match = pattern->Match(*this, candidate);
        if (match.pattern == nullptr) {
          continue;
        }
        if (match.pattern != pattern.get()) {
          throw BuilderError(
              "GraphBuilderPatternOptimization::Optimize: a match refers to another pattern.");
        }
        if (match.nodes.empty()) {
          throw BuilderError(
              "GraphBuilderPatternOptimization::Optimize: a non-empty match is required.");
        }

        bool skip = false;
        for (const NodeProto *node : match.nodes) {
          if (node == nullptr) {
            throw BuilderError(
                "GraphBuilderPatternOptimization::Optimize: a matched node must not be null.");
          }
          graph_->Position(*node);
          if (marked.find(node) != marked.end() || (do_not_remove_ && do_not_remove_(*node))) {
            skip = true;
            break;
          }
        }
        if (skip) {
          continue;
        }
        if (match.insert_at != nullptr) {
          graph_->Position(*match.insert_at);
        }
        marked.insert(match.nodes.begin(), match.nodes.end());
        matches.push_back(std::move(match));
      }
    }

    std::unordered_set<const NodeProto *> removed;
    std::vector<PendingReplacement> replacements;
    replacements.reserve(matches.size());
    for (const MatchResult &match : matches) {
      std::size_t position = builder_.nodes_.size();
      if (match.insert_at != nullptr) {
        position = graph_->Position(*match.insert_at);
      } else {
        for (const NodeProto *node : match.nodes) {
          position = std::min(position, graph_->Position(*node));
        }
      }
      replacements.push_back(
          PendingReplacement{position, match.pattern->Apply(*this, match.nodes)});
      removed.insert(match.nodes.begin(), match.nodes.end());
    }

    if (!matches.empty()) {
      utils::RepeatedProtoField<NodeProto> rebuilt;
      std::size_t replacement_nodes = 0;
      for (const PendingReplacement &replacement : replacements) {
        replacement_nodes += replacement.nodes.size();
      }
      rebuilt.reserve(builder_.nodes_.size() - removed.size() + replacement_nodes);

      std::unordered_map<std::size_t, std::vector<std::size_t>> insertions;
      for (std::size_t i = 0; i < replacements.size(); ++i) {
        insertions[replacements[i].position].push_back(i);
      }
      for (std::size_t i = 0; i < builder_.nodes_.size(); ++i) {
        auto insertion = insertions.find(i);
        if (insertion != insertions.end()) {
          for (std::size_t replacement_index : insertion->second) {
            rebuilt.extend(std::move(replacements[replacement_index].nodes));
          }
        }
        const NodeProto *node = &builder_.nodes_[i];
        if (removed.find(node) == removed.end()) {
          rebuilt.add();
          rebuilt.get(rebuilt.size() - 1) = builder_.nodes_.shared_at(i);
        }
      }
      builder_.nodes_ = std::move(rebuilt);
      applied += matches.size();
    }

    std::size_t cleaned = builder_.RemoveDuplicateNodes();
    cleaned += builder_.RemoveIdentityNodes();
    cleaned += builder_.RemoveUnusedNodes();
    RebuildGraph();

    if (!matches.empty() || cleaned != 0) {
      continue;
    }
    if (priority_index + 1 == priorities.size()) {
      break;
    }
    ++priority_index;
  }
  return applied;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
