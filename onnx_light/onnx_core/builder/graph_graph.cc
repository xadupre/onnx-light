// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/builder/graph_graph.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "onnx_core/builder/pattern_registry.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::builder {

namespace {

struct PendingReplacement {
  std::size_t position;
  utils::RepeatedProtoField<NodeProto> nodes;
  LocalRewriting rewriting;
};

struct TimedMatch {
  MatchResult match;
  std::size_t pattern_index;
  int64_t match_time_ns;
};

class CleanupPattern final : public PatternOptimization {
public:
  explicit CleanupPattern(std::string name) : PatternOptimization(0, std::move(name)) {}

  MatchResult Match(GraphGraph &, const NodeProto &) const override { return {}; }

  utils::RepeatedProtoField<NodeProto>
  Apply(GraphGraph &, const std::vector<const NodeProto *> &) const override {
    return utils::RepeatedProtoField<NodeProto>();
  }
};

std::shared_ptr<const PatternOptimization> CleanupPatternOwner(const std::string &name) {
  static const std::shared_ptr<const PatternOptimization> duplicate_nodes =
      std::make_shared<CleanupPattern>("RemoveDuplicateNodes");
  static const std::shared_ptr<const PatternOptimization> identities =
      std::make_shared<CleanupPattern>("RemoveIdentityNodes");
  static const std::shared_ptr<const PatternOptimization> dead_ends =
      std::make_shared<CleanupPattern>("RemoveUnusedNodes");
  static const std::shared_ptr<const PatternOptimization> duplicate_initializers =
      std::make_shared<CleanupPattern>("RemoveDuplicateInitializers");
  if (name == duplicate_nodes->Name()) {
    return duplicate_nodes;
  }
  if (name == identities->Name()) {
    return identities;
  }
  if (name == dead_ends->Name()) {
    return dead_ends;
  }
  if (name == duplicate_initializers->Name()) {
    return duplicate_initializers;
  }
  throw BuilderError("GraphGraph::Cleanup: unknown cleanup operation '" + name + "'.");
}

int64_t ElapsedNanoseconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                              start)
      .count();
}

void RecordNoMatch(PatternOptimizationStatistics &statistics, std::string_view source_file,
                   std::uint_least32_t source_line, std::string_view reason,
                   std::size_t occurrences = 1) {
  const auto existing = std::find_if(statistics.no_matches.begin(), statistics.no_matches.end(),
                                     [&](const PatternNoMatchStatistics &candidate) {
                                       return candidate.source_file == source_file &&
                                              candidate.source_line == source_line &&
                                              candidate.reason == reason;
                                     });
  if (existing != statistics.no_matches.end()) {
    existing->occurrences += occurrences;
    return;
  }
  statistics.no_matches.push_back(
      {std::string(source_file), source_line, std::string(reason), occurrences});
}

} // namespace

GraphGraph::GraphGraph(GraphBuilder &builder) : GraphGraph(builder, CreateRegisteredPatterns()) {}

GraphGraph::GraphGraph(GraphBuilder &builder,
                       std::vector<std::unique_ptr<PatternOptimization>> patterns,
                       DoNotRemovePredicate do_not_remove)
    : builder_(builder), do_not_remove_(std::move(do_not_remove)) {
  patterns_.reserve(patterns.size());
  for (std::unique_ptr<PatternOptimization> &pattern : patterns) {
    if (pattern == nullptr) {
      throw BuilderError("GraphGraph: a pattern must not be null.");
    }
    if (pattern->Name().empty()) {
      throw BuilderError("GraphGraph: a pattern must have a stable diagnostic name.");
    }
    patterns_.push_back(std::move(pattern));
  }
  std::stable_sort(patterns_.begin(), patterns_.end(), [](const auto &left, const auto &right) {
    return left->priority < right->priority;
  });
  Rebuild();
}

GraphGraph::GraphGraph(GraphBuilder &builder,
                       std::vector<std::shared_ptr<PatternOptimization>> patterns)
    : builder_(builder), patterns_(std::move(patterns)) {
  for (const std::shared_ptr<PatternOptimization> &pattern : patterns_) {
    if (pattern == nullptr) {
      throw BuilderError("GraphGraph: a pattern must not be null.");
    }
    if (pattern->Name().empty()) {
      throw BuilderError("GraphGraph: a pattern must have a stable diagnostic name.");
    }
  }
  std::stable_sort(patterns_.begin(), patterns_.end(), [](const auto &left, const auto &right) {
    return left->priority < right->priority;
  });
  Rebuild();
}

GraphGraph::GraphGraph(GraphBuilder &builder,
                       const std::vector<std::shared_ptr<PatternOptimization>> &patterns,
                       DoNotRemovePredicate do_not_remove, const GraphGraph *parent_graph,
                       std::size_t parent_position_limit)
    : builder_(builder), patterns_(patterns), do_not_remove_(std::move(do_not_remove)),
      parent_graph_(parent_graph), parent_position_limit_(parent_position_limit) {
  Rebuild();
}

void GraphGraph::Rebuild() {
  predecessors_.clear();
  successors_.clear();
  positions_.clear();
  output_names_.clear();
  subgraph_captured_.clear();
  initializers_.clear();
  computed_constants_.clear();

  for (const TensorProto &initializer : builder_.Initializers()) {
    initializers_.emplace(initializer.name().value(), &initializer);
  }
  for (const ValueInfoProto &output : builder_.Outputs()) {
    output_names_.insert(output.name().value());
  }

  const utils::RepeatedProtoField<NodeProto> &nodes = builder_.Nodes();
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const NodeProto &node = nodes[i];
    positions_.emplace(&node, i);
    for (std::size_t o = 0; o < node.output().size(); ++o) {
      std::string name(node.output(static_cast<std::size_t>(o)));
      if (!name.empty()) {
        predecessors_[std::move(name)] = &node;
      }
    }
  }
  RebuildSuccessors();
}

std::vector<LocalRewriting> GraphGraph::Optimize(int max_iter, OptimizationReport *report) {
  return OptimizeImpl(max_iter, report, {});
}

std::vector<LocalRewriting> GraphGraph::OptimizeImpl(int max_iter, OptimizationReport *report,
                                                     const std::vector<std::string> &graph_path) {
  if (max_iter < -1) {
    throw BuilderError("GraphGraph::Optimize: max_iter must be at least -1.");
  }
  if (report != nullptr) {
    *report = OptimizationReport{};
    report->patterns.reserve(patterns_.size());
    for (const std::shared_ptr<PatternOptimization> &pattern : patterns_) {
      report->patterns.push_back(PatternOptimizationStatistics{.pattern_name = pattern->Name()});
    }
  }

  std::vector<std::set<std::string>> fast_op_types_by_pattern;
  fast_op_types_by_pattern.reserve(patterns_.size());
  for (const std::shared_ptr<PatternOptimization> &pattern : patterns_) {
    fast_op_types_by_pattern.push_back(pattern->FastOpType());
  }

  std::vector<LocalRewriting> applied;
  std::size_t rewrite_batch = 0;
  std::unordered_set<GraphBuilder *> optimized_subgraphs;
  const auto optimize_subgraph = [&](GraphBuilder &subgraph, std::size_t position_limit) {
    std::vector<std::string> child_path = graph_path;
    child_path.push_back(subgraph.name());
    OptimizationReport child_report;
    std::chrono::steady_clock::time_point subgraph_start;
    if (report != nullptr) {
      subgraph_start = std::chrono::steady_clock::now();
    }
    GraphGraph child_graph(subgraph, patterns_, do_not_remove_, this, position_limit);
    std::vector<LocalRewriting> child_rewrites =
        child_graph.OptimizeImpl(max_iter, report == nullptr ? nullptr : &child_report, child_path);

    for (LocalRewriting &rewriting : child_rewrites) {
      rewriting.iteration += rewrite_batch;
    }
    if (!child_rewrites.empty()) {
      rewrite_batch = child_rewrites.back().iteration + 1;
    }
    applied.insert(applied.end(), std::make_move_iterator(child_rewrites.begin()),
                   std::make_move_iterator(child_rewrites.end()));

    if (report != nullptr) {
      const int64_t elapsed_time_ns = ElapsedNanoseconds(subgraph_start);
      report->iterations += child_report.iterations;
      report->rewrites += child_report.rewrites;
      report->subgraph_optimization_time_ns += elapsed_time_ns;
      for (std::size_t i = 0; i < report->patterns.size(); ++i) {
        report->patterns[i].attempts += child_report.patterns[i].attempts;
        report->patterns[i].matches += child_report.patterns[i].matches;
        report->patterns[i].match_time_ns += child_report.patterns[i].match_time_ns;
        report->patterns[i].apply_time_ns += child_report.patterns[i].apply_time_ns;
        for (const PatternNoMatchStatistics &no_match : child_report.patterns[i].no_matches) {
          RecordNoMatch(report->patterns[i], no_match.source_file, no_match.source_line,
                        no_match.reason, no_match.occurrences);
        }
      }
      report->subgraphs.push_back(
          {child_path, child_report.iterations, child_report.rewrites, elapsed_time_ns});
      report->subgraphs.insert(report->subgraphs.end(), child_report.subgraphs.begin(),
                               child_report.subgraphs.end());
    }
  };
  for (std::size_t position = 0; position < builder_.nodes_.size(); ++position) {
    for (GraphBuilder *subgraph : builder_.ReferencedSubgraphs(builder_.nodes_[position])) {
      if (optimized_subgraphs.insert(subgraph).second) {
        optimize_subgraph(*subgraph, position);
      }
    }
  }
  for (const std::unique_ptr<GraphBuilder> &subgraph : builder_.subgraphs_) {
    if (optimized_subgraphs.insert(subgraph.get()).second) {
      optimize_subgraph(*subgraph, builder_.nodes_.size());
    }
  }

  std::vector<int> priorities;
  priorities.reserve(patterns_.size());
  for (const std::shared_ptr<PatternOptimization> &pattern : patterns_) {
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

  std::size_t priority_index = 0;
  for (int iteration = 0; iteration < max_iter; ++iteration) {
    if (report != nullptr) {
      ++report->iterations;
    }
    const int current_priority = priorities[priority_index];
    std::unordered_set<const NodeProto *> marked;
    std::vector<TimedMatch> matches;
    std::chrono::steady_clock::time_point matching_start;
    if (report != nullptr) {
      matching_start = std::chrono::steady_clock::now();
    }

    for (std::size_t pattern_index = 0; pattern_index < patterns_.size(); ++pattern_index) {
      const std::shared_ptr<PatternOptimization> &pattern = patterns_[pattern_index];
      if (pattern->priority > current_priority) {
        break;
      }
      const std::set<std::string> &fast_op_types = fast_op_types_by_pattern[pattern_index];
      for (const NodeProto &candidate : builder_.nodes_) {
        if (!fast_op_types.empty() &&
            fast_op_types.find(candidate.op_type().value()) == fast_op_types.end()) {
          continue;
        }

        const auto match_start = std::chrono::steady_clock::now();
        MatchResult match = pattern->Match(*this, candidate);
        const int64_t match_time_ns = ElapsedNanoseconds(match_start);
        if (report != nullptr) {
          PatternOptimizationStatistics &statistics = report->patterns[pattern_index];
          ++statistics.attempts;
          statistics.match_time_ns += match_time_ns;
        }
        if (match.pattern == nullptr) {
          if (report != nullptr && match.no_match.has_value()) {
            RecordNoMatch(report->patterns[pattern_index], match.no_match->source_file,
                          match.no_match->source_line, match.no_match->reason);
          }
          continue;
        }
        if (match.pattern != pattern.get()) {
          throw BuilderError("GraphGraph::Optimize: a match refers to another pattern.");
        }
        if (match.nodes.empty()) {
          throw BuilderError("GraphGraph::Optimize: a non-empty match is required.");
        }

        bool skip = false;
        bool has_node = false;
        for (const NodeProto *node : match.nodes) {
          if (node == nullptr) {
            continue;
          }
          has_node = true;
          Position(*node);
          if (marked.find(node) != marked.end() || (do_not_remove_ && do_not_remove_(*node))) {
            skip = true;
            break;
          }
        }
        if (!has_node) {
          throw BuilderError(
              "GraphGraph::Optimize: a match must contain at least one non-null node.");
        }
        if (skip) {
          continue;
        }
        if (match.insert_at != nullptr) {
          Position(*match.insert_at);
        }
        for (const NodeProto *node : match.nodes) {
          if (node != nullptr) {
            marked.insert(node);
          }
        }
        if (report != nullptr) {
          ++report->patterns[pattern_index].matches;
        }
        matches.push_back(TimedMatch{std::move(match), pattern_index, match_time_ns});
      }
    }
    if (report != nullptr) {
      report->matching_time_ns += ElapsedNanoseconds(matching_start);
    }

    std::chrono::steady_clock::time_point rewriting_start;
    if (report != nullptr) {
      rewriting_start = std::chrono::steady_clock::now();
    }
    std::unordered_set<const NodeProto *> removed;
    std::vector<PendingReplacement> replacements;
    int64_t constant_folding_time_ns = 0;
    replacements.reserve(matches.size());
    for (const TimedMatch &timed_match : matches) {
      const MatchResult &match = timed_match.match;
      std::size_t position = builder_.nodes_.size();
      if (match.insert_at != nullptr) {
        position = Position(*match.insert_at);
      } else {
        for (const NodeProto *node : match.nodes) {
          if (node != nullptr) {
            position = std::min(position, Position(*node));
          }
        }
      }
      const std::size_t initializers_before = builder_.initializers_.size();
      const auto apply_start = std::chrono::steady_clock::now();
      utils::RepeatedProtoField<NodeProto> replacement_nodes =
          match.pattern->Apply(*this, match.nodes);
      const int64_t apply_time_ns = ElapsedNanoseconds(apply_start);
      LocalRewriting rewriting;
      const auto pattern_owner =
          std::find_if(patterns_.begin(), patterns_.end(),
                       [&match](const std::shared_ptr<PatternOptimization> &candidate) {
                         return candidate.get() == match.pattern;
                       });
      if (pattern_owner == patterns_.end()) {
        throw BuilderError("GraphGraph::Optimize: matched pattern has no owner.");
      }
      rewriting.pattern = *pattern_owner;
      rewriting.graph_path = graph_path;
      rewriting.iteration = rewrite_batch;
      rewriting.match_time_ns = timed_match.match_time_ns;
      rewriting.apply_time_ns = apply_time_ns;
      rewriting.added_nodes = replacement_nodes;
      rewriting.matched_nodes.reserve(match.nodes.size());
      std::unordered_set<const NodeProto *> persisted_nodes;
      for (const NodeProto *node : match.nodes) {
        if (node != nullptr && persisted_nodes.insert(node).second) {
          rewriting.matched_nodes.push_back(Position(*node));
        }
      }
      for (std::size_t i = initializers_before; i < builder_.initializers_.size(); ++i) {
        rewriting.added_initializers.push_back(builder_.initializers_[i]);
        rewriting.added_initializer_positions.push_back(i);
      }
      replacements.push_back(
          PendingReplacement{position, std::move(replacement_nodes), std::move(rewriting)});
      for (const NodeProto *node : match.nodes) {
        if (node != nullptr) {
          removed.insert(node);
        }
      }
      if (report != nullptr) {
        report->patterns[timed_match.pattern_index].apply_time_ns += apply_time_ns;
      }
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

      std::unordered_map<std::string, std::size_t> output_owners;
      std::unordered_set<std::string> replacement_outputs;
      for (std::size_t i = 0; i < replacements.size(); ++i) {
        for (const NodeProto &node : replacements[i].rewriting.added_nodes) {
          for (std::size_t o = 0; o < node.output().size(); ++o) {
            const std::string output(node.output(static_cast<std::size_t>(o)));
            if (!output.empty()) {
              output_owners.emplace(output, i);
              replacement_outputs.insert(output);
            }
          }
        }
      }

      const std::size_t initializers_before_folding = builder_.initializers_.size();
      const auto constant_folding_start = std::chrono::steady_clock::now();
      builder_.ConstantFoldNodes(ConstantFoldingOptions{}, replacement_outputs);
      constant_folding_time_ns = ElapsedNanoseconds(constant_folding_start);
      if (report != nullptr) {
        report->constant_folding_time_ns += constant_folding_time_ns;
      }

      std::unordered_set<std::string> folded_outputs;
      for (std::size_t i = initializers_before_folding; i < builder_.initializers_.size(); ++i) {
        const TensorProto &initializer = builder_.initializers_[i];
        const std::string name = initializer.name().value();
        const auto owner = output_owners.find(name);
        if (owner != output_owners.end()) {
          replacements[owner->second].rewriting.added_initializers.push_back(initializer);
          replacements[owner->second].rewriting.added_initializer_positions.push_back(i);
          folded_outputs.insert(name);
        }
      }
      for (PendingReplacement &replacement : replacements) {
        utils::RepeatedProtoField<NodeProto> unfolded_nodes;
        for (const NodeProto &node : replacement.rewriting.added_nodes) {
          bool folded = !node.output().empty();
          for (std::size_t o = 0; o < node.output().size(); ++o) {
            const std::string output(node.output(static_cast<std::size_t>(o)));
            if (!output.empty() && folded_outputs.find(output) == folded_outputs.end()) {
              folded = false;
              break;
            }
          }
          if (!folded) {
            unfolded_nodes.push_back(node);
          }
        }
        replacement.rewriting.added_nodes = std::move(unfolded_nodes);
      }

      std::unordered_map<std::string, std::size_t> output_positions;
      for (std::size_t position = 0; position < builder_.nodes_.size(); ++position) {
        const NodeProto &node = builder_.nodes_[position];
        for (std::size_t output_index = 0; output_index < node.output().size(); ++output_index) {
          const std::string output(node.output(output_index));
          if (!output.empty()) {
            output_positions.emplace(output, position);
          }
        }
      }
      for (PendingReplacement &replacement : replacements) {
        replacement.rewriting.added_nodes_positions.reserve(
            replacement.rewriting.added_nodes.size());
        for (const NodeProto &node : replacement.rewriting.added_nodes) {
          std::size_t position = builder_.nodes_.size();
          for (std::size_t output_index = 0; output_index < node.output().size(); ++output_index) {
            const std::string output(node.output(output_index));
            if (output.empty()) {
              continue;
            }
            const auto found = output_positions.find(output);
            if (found == output_positions.end()) {
              throw BuilderError("GraphGraph::Optimize: added node output '" + output +
                                 "' is missing after applying the rewrite batch.");
            }
            if (position != builder_.nodes_.size() && position != found->second) {
              throw BuilderError(
                  "GraphGraph::Optimize: outputs of one added node have different positions.");
            }
            position = found->second;
          }
          if (position == builder_.nodes_.size()) {
            throw BuilderError("GraphGraph::Optimize: an added node must produce a named value.");
          }
          replacement.rewriting.added_nodes_positions.push_back(position);
        }
        applied.push_back(std::move(replacement.rewriting));
      }
      ++rewrite_batch;
    }
    if (report != nullptr) {
      report->rewriting_time_ns += ElapsedNanoseconds(rewriting_start) - constant_folding_time_ns;
      report->rewrites = applied.size();
    }

    std::chrono::steady_clock::time_point cleanup_start;
    if (report != nullptr) {
      cleanup_start = std::chrono::steady_clock::now();
    }
    const std::size_t rewrites_before_cleanup = applied.size();
    const std::size_t cleaned = Cleanup(applied, rewrite_batch);
    for (std::size_t i = rewrites_before_cleanup; i < applied.size(); ++i) {
      applied[i].graph_path = graph_path;
    }
    if (report != nullptr) {
      report->cleanup_time_ns += ElapsedNanoseconds(cleanup_start);
      report->rewrites = applied.size();
    }

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

std::size_t GraphGraph::Cleanup(std::vector<LocalRewriting> &rewrites, std::size_t &rewrite_batch) {
  const auto record_nodes = [&](const std::string &name, const auto &cleanup) {
    const std::size_t node_count = builder_.nodes_.size();
    std::unordered_map<std::string, std::string> applied_renames;
    const std::size_t removed = cleanup(applied_renames);
    if (removed == 0) {
      return std::size_t{0};
    }
    LocalRewriting rewriting;
    rewriting.pattern = CleanupPatternOwner(name);
    rewriting.matched_nodes.reserve(node_count);
    for (std::size_t i = 0; i < node_count; ++i) {
      rewriting.matched_nodes.push_back(i);
    }
    rewriting.added_nodes = builder_.nodes_;
    rewriting.added_nodes_positions.reserve(builder_.nodes_.size());
    for (std::size_t i = 0; i < builder_.nodes_.size(); ++i) {
      rewriting.added_nodes_positions.push_back(i);
    }
    rewriting.value_renames.assign(applied_renames.begin(), applied_renames.end());
    std::sort(rewriting.value_renames.begin(), rewriting.value_renames.end());
    rewriting.iteration = rewrite_batch++;
    rewrites.push_back(std::move(rewriting));
    return removed;
  };

  std::size_t cleaned = record_nodes("RemoveDuplicateNodes", [&](auto &renames) {
    return builder_.RemoveDuplicateNodesImpl(false, &renames);
  });
  cleaned += record_nodes("RemoveIdentityNodes", [&](auto &renames) {
    return builder_.RemoveIdentityNodesImpl(false, &renames);
  });
  cleaned += record_nodes("RemoveUnusedNodes",
                          [&](auto &) { return builder_.RemoveUnusedNodesImpl(false); });

  const std::size_t node_count = builder_.nodes_.size();
  const std::size_t initializer_count = builder_.initializers_.size();
  std::unordered_map<std::string, std::string> applied_renames;
  const std::size_t removed_initializers = builder_.DeduplicateInitializers(
      GraphBuilder::InitializerContentIndex{}, false, &applied_renames);
  if (removed_initializers != 0) {
    LocalRewriting rewriting;
    rewriting.pattern = CleanupPatternOwner("RemoveDuplicateInitializers");
    rewriting.matched_nodes.reserve(node_count);
    for (std::size_t i = 0; i < node_count; ++i) {
      rewriting.matched_nodes.push_back(i);
    }
    rewriting.added_nodes = builder_.nodes_;
    rewriting.added_nodes_positions.reserve(builder_.nodes_.size());
    for (std::size_t i = 0; i < builder_.nodes_.size(); ++i) {
      rewriting.added_nodes_positions.push_back(i);
    }
    rewriting.removed_initializers.reserve(initializer_count);
    for (std::size_t i = 0; i < initializer_count; ++i) {
      rewriting.removed_initializers.push_back(i);
    }
    rewriting.added_initializers = builder_.initializers_;
    rewriting.added_initializer_positions.reserve(builder_.initializers_.size());
    for (std::size_t i = 0; i < builder_.initializers_.size(); ++i) {
      rewriting.added_initializer_positions.push_back(i);
    }
    rewriting.value_renames.assign(applied_renames.begin(), applied_renames.end());
    std::sort(rewriting.value_renames.begin(), rewriting.value_renames.end());
    rewriting.iteration = rewrite_batch++;
    rewrites.push_back(std::move(rewriting));
    cleaned += removed_initializers;
  }
  Rebuild();
  return cleaned;
}

void GraphGraph::ApplyRewritingBatch(const std::vector<LocalRewriting> &rewrites, std::size_t begin,
                                     std::size_t end) {
  std::unordered_set<const NodeProto *> removed;
  std::unordered_set<std::size_t> removed_initializers;
  std::unordered_map<std::string, std::string> value_renames;
  struct AddedInitializer {
    std::size_t position;
    const TensorProto *initializer;
    bool replaces_matched_output;
  };
  std::vector<AddedInitializer> added_initializers;
  struct AddedNode {
    std::size_t position;
    const NodeProto *node;
  };
  std::vector<AddedNode> added_nodes;
  std::size_t replacement_nodes = 0;
  for (std::size_t i = begin; i < end; ++i) {
    const LocalRewriting &rewrite = rewrites[i];
    if (rewrite.pattern == nullptr) {
      throw BuilderError("Replay: a rewrite must link to its pattern.");
    }
    if (rewrite.matched_nodes.empty() && rewrite.removed_initializers.empty() &&
        rewrite.added_nodes.empty() && rewrite.added_initializers.empty() &&
        rewrite.value_renames.empty()) {
      throw BuilderError("Replay: an empty rewrite is not allowed.");
    }
    for (const std::size_t position : rewrite.matched_nodes) {
      if (position >= builder_.nodes_.size()) {
        throw BuilderError("Replay: matched node position " + std::to_string(position) +
                           " is outside the graph.");
      }
      if (!removed.insert(&builder_.nodes_[position]).second) {
        throw BuilderError("Replay: rewrites in one iteration overlap at node position " +
                           std::to_string(position) + ".");
      }
    }
    if (rewrite.added_nodes_positions.size() != rewrite.added_nodes.size()) {
      throw BuilderError("Replay: added node positions must align with added nodes.");
    }
    for (std::size_t node_index = 0; node_index < rewrite.added_nodes.size(); ++node_index) {
      added_nodes.push_back(
          AddedNode{rewrite.added_nodes_positions[node_index], &rewrite.added_nodes[node_index]});
    }
    for (const std::size_t position : rewrite.removed_initializers) {
      if (position >= builder_.initializers_.size()) {
        throw BuilderError("Replay: removed initializer position " + std::to_string(position) +
                           " is outside the graph.");
      }
      if (!removed_initializers.insert(position).second) {
        throw BuilderError("Replay: rewrites in one iteration overlap at initializer position " +
                           std::to_string(position) + ".");
      }
    }
    for (const auto &rename : rewrite.value_renames) {
      const auto inserted = value_renames.emplace(rename);
      if (!inserted.second && inserted.first->second != rename.second) {
        throw BuilderError("Replay: conflicting replacements for value '" + rename.first + "'.");
      }
    }
    if (!rewrite.added_initializer_positions.empty() &&
        rewrite.added_initializer_positions.size() != rewrite.added_initializers.size()) {
      throw BuilderError("Replay: added initializer positions must align with added initializers.");
    }
    for (std::size_t initializer_index = 0; initializer_index < rewrite.added_initializers.size();
         ++initializer_index) {
      const TensorProto &initializer = rewrite.added_initializers[initializer_index];
      if (!rewrite.removed_initializers.empty()) {
        continue;
      }
      bool replaces_matched_output = false;
      for (const std::size_t position : rewrite.matched_nodes) {
        const NodeProto &matched = builder_.nodes_[position];
        for (std::size_t o = 0; o < matched.output().size(); ++o) {
          if (matched.output(static_cast<std::size_t>(o)) == initializer.name()) {
            replaces_matched_output = true;
            break;
          }
        }
        if (replaces_matched_output) {
          break;
        }
      }
      const std::size_t position = rewrite.added_initializer_positions.empty()
                                       ? builder_.initializers_.size() + added_initializers.size()
                                       : rewrite.added_initializer_positions[initializer_index];
      added_initializers.push_back(
          AddedInitializer{position, &initializer, replaces_matched_output});
    }
    replacement_nodes += rewrite.added_nodes.size();
  }

  const std::size_t final_node_count = builder_.nodes_.size() - removed.size() + replacement_nodes;
  std::vector<const NodeProto *> nodes_by_position(final_node_count, nullptr);
  for (const AddedNode &addition : added_nodes) {
    if (addition.position >= final_node_count) {
      throw BuilderError("Replay: added node position " + std::to_string(addition.position) +
                         " is outside the rewritten graph.");
    }
    if (nodes_by_position[addition.position] != nullptr) {
      throw BuilderError("Replay: added node positions overlap at position " +
                         std::to_string(addition.position) + ".");
    }
    nodes_by_position[addition.position] = addition.node;
  }

  builder_.RewriteInitializerReferences(value_renames);

  if (!removed_initializers.empty()) {
    utils::RepeatedProtoField<TensorProto> rebuilt_initializers;
    rebuilt_initializers.reserve(builder_.initializers_.size() - removed_initializers.size());
    for (std::size_t i = 0; i < builder_.initializers_.size(); ++i) {
      if (removed_initializers.find(i) == removed_initializers.end()) {
        rebuilt_initializers.push_back(builder_.initializers_[i]);
      }
    }
    builder_.initializers_ = std::move(rebuilt_initializers);
    for (std::size_t i = begin; i < end; ++i) {
      const LocalRewriting &rewrite = rewrites[i];
      for (std::size_t j = 0; j < rewrite.added_initializers.size(); ++j) {
        builder_.initializers_.push_back(rewrite.added_initializers[j]);
      }
    }
  } else {
    std::stable_sort(added_initializers.begin(), added_initializers.end(),
                     [](const AddedInitializer &left, const AddedInitializer &right) {
                       return left.position < right.position;
                     });
    for (const AddedInitializer &addition : added_initializers) {
      if (addition.position != builder_.initializers_.size()) {
        throw BuilderError("Replay: added initializer position " +
                           std::to_string(addition.position) + " is not the next position.");
      }
      if (addition.replaces_matched_output) {
        builder_.initializers_.push_back(*addition.initializer);
      } else {
        builder_.MakeInitializer(*addition.initializer);
      }
    }
  }

  utils::RepeatedProtoField<NodeProto> rebuilt;
  rebuilt.reserve(final_node_count);
  std::size_t original_position = 0;
  for (std::size_t position = 0; position < final_node_count; ++position) {
    if (nodes_by_position[position] != nullptr) {
      rebuilt.push_back(*nodes_by_position[position]);
      continue;
    }
    while (original_position < builder_.nodes_.size() &&
           removed.find(&builder_.nodes_[original_position]) != removed.end()) {
      ++original_position;
    }
    if (original_position == builder_.nodes_.size()) {
      throw BuilderError("Replay: added node positions leave no slot for a retained node.");
    }
    rebuilt.add();
    rebuilt.get(rebuilt.size() - 1) = builder_.nodes_.shared_at(original_position++);
  }
  while (original_position < builder_.nodes_.size() &&
         removed.find(&builder_.nodes_[original_position]) != removed.end()) {
    ++original_position;
  }
  if (original_position != builder_.nodes_.size()) {
    throw BuilderError(
        "Replay: added node positions do not leave enough slots for retained nodes.");
  }
  builder_.nodes_ = std::move(rebuilt);
  Rebuild();
}

GraphProto Replay(const ModelProto &model, const std::vector<LocalRewriting> &rewrites,
                  GraphBuilder::SchemaLookupFn schema_lookup) {
  GraphBuilder builder(model, std::move(schema_lookup));

  std::size_t next_iteration = 0;
  std::size_t begin = 0;
  while (begin < rewrites.size()) {
    const std::size_t iteration = rewrites[begin].iteration;
    if (iteration < next_iteration) {
      throw BuilderError("Replay: rewrites must be ordered by optimization iteration.");
    }
    std::size_t end = begin + 1;
    while (end < rewrites.size() && rewrites[end].iteration == iteration) {
      if (rewrites[end].graph_path != rewrites[begin].graph_path) {
        throw BuilderError("Replay: one optimization iteration cannot span multiple graphs.");
      }
      ++end;
    }
    GraphBuilder *target = &builder;
    for (const std::string &name : rewrites[begin].graph_path) {
      if (!target->HasSubgraph(name)) {
        throw BuilderError("Replay: unknown subgraph '" + name + "'.");
      }
      target = &target->Subgraph(name);
    }
    GraphGraph graph(*target, std::vector<std::unique_ptr<PatternOptimization>>{});
    graph.ApplyRewritingBatch(rewrites, begin, end);
    next_iteration = iteration + 1;
    begin = end;
  }
  return builder.BuildGraph();
}

void GraphGraph::RebuildSuccessors() {
  const utils::RepeatedProtoField<NodeProto> &nodes = builder_.Nodes();
  for (const NodeProto &node : nodes) {
    for (std::size_t in = 0; in < node.input().size(); ++in) {
      std::string name(node.input(static_cast<std::size_t>(in)));
      if (name.empty()) {
        continue;
      }
      std::vector<const NodeProto *> &consumers = successors_[name];
      // Avoid recording the same consumer twice when a node reads the same
      // value from more than one input slot. Consumer lists are short, so a
      // linear membership scan is cheap.
      bool present = false;
      for (const NodeProto *consumer : consumers) {
        if (consumer == &node) {
          present = true;
          break;
        }
      }
      if (!present) {
        consumers.push_back(&node);
      }
    }
    // Mark every value a referenced subgraph reads from the enclosing scope so
    // a rewrite never deletes a producer the subgraph still relies on.
    for (const GraphBuilder *subgraph : builder_.ReferencedSubgraphs(node)) {
      std::unordered_set<std::string> captured;
      subgraph->CollectImplicitInputs(captured);
      for (const std::string &name : captured) {
        subgraph_captured_.insert(name);
      }
    }
  }
}

const NodeProto *GraphGraph::NodeBefore(const std::string &name) const {
  auto it = predecessors_.find(name);
  return it == predecessors_.end() ? nullptr : it->second;
}

const std::vector<const NodeProto *> &GraphGraph::NextNodes(const std::string &name) const {
  auto it = successors_.find(name);
  return it == successors_.end() ? EmptyNodeList() : it->second;
}

std::vector<const NodeProto *> GraphGraph::Predecessors(const NodeProto &node) const {
  std::vector<const NodeProto *> result;
  for (std::size_t in = 0; in < node.input().size(); ++in) {
    std::string name(node.input(static_cast<std::size_t>(in)));
    if (name.empty()) {
      continue;
    }
    const NodeProto *producer = NodeBefore(name);
    if (producer == nullptr) {
      continue;
    }
    if (std::find(result.begin(), result.end(), producer) == result.end()) {
      result.push_back(producer);
    }
  }
  return result;
}

std::vector<const NodeProto *> GraphGraph::Successors(const NodeProto &node) const {
  std::vector<const NodeProto *> result;
  for (std::size_t o = 0; o < node.output().size(); ++o) {
    std::string name(node.output(static_cast<std::size_t>(o)));
    if (name.empty()) {
      continue;
    }
    for (const NodeProto *consumer : NextNodes(name)) {
      if (std::find(result.begin(), result.end(), consumer) == result.end()) {
        result.push_back(consumer);
      }
    }
  }
  return result;
}

bool GraphGraph::IsOutput(const std::string &name) const { return output_names_.count(name) != 0; }

bool GraphGraph::IsUsedBySubgraph(const std::string &name) const {
  return subgraph_captured_.count(name) != 0;
}

bool GraphGraph::IsUsed(const std::string &name) const {
  return IsUsedBySubgraph(name) || successors_.count(name) != 0 || IsOutput(name);
}

bool GraphGraph::IsUsedMoreThanOnce(const std::string &name) const {
  if (IsUsedBySubgraph(name) || IsOutput(name)) {
    return true;
  }
  auto it = successors_.find(name);
  return it != successors_.end() && it->second.size() > 1;
}

std::size_t GraphGraph::Position(const NodeProto &node) const {
  auto it = positions_.find(&node);
  if (it == positions_.end()) {
    throw BuilderError("GraphGraph::Position: node is not part of the index.");
  }
  return it->second;
}

bool GraphGraph::HasShape(const std::string &name) const {
  return builder_.HasShape(name) || (parent_graph_ != nullptr &&
                                     parent_graph_->IsVisibleBefore(name, parent_position_limit_) &&
                                     parent_graph_->HasShape(name));
}

const SymTensor &GraphGraph::GetShape(const std::string &name) const {
  if (!builder_.HasShape(name) && parent_graph_ != nullptr &&
      parent_graph_->IsVisibleBefore(name, parent_position_limit_)) {
    return parent_graph_->GetShape(name);
  }
  return builder_.GetShape(name);
}

bool GraphGraph::HasType(const std::string &name) const {
  return HasShape(name) && GetShape(name).Dtype() != TensorType::kUndefined;
}

TensorType GraphGraph::GetType(const std::string &name) const { return GetShape(name).Dtype(); }

bool GraphGraph::IsConstant(const std::string &name) const {
  return initializers_.find(name) != initializers_.end() ||
         computed_constants_.find(name) != computed_constants_.end() ||
         builder_.Compute().IsConstantValue(name) ||
         (parent_graph_ != nullptr &&
          parent_graph_->IsVisibleBefore(name, parent_position_limit_) &&
          parent_graph_->IsConstant(name));
}

const TensorProto *GraphGraph::GetComputedConstant(const std::string &name) const {
  auto cached = computed_constants_.find(name);
  if (cached != computed_constants_.end()) {
    return &cached->second;
  }
  auto init = initializers_.find(name);
  if (init != initializers_.end()) {
    return init->second;
  }
  const NodeProto *node = NodeBefore(name);
  if (node != nullptr && node->domain().value().empty() && node->op_type().value() == "Constant") {
    const AttributeProto *value = FindAttribute(*node, "value");
    if (value != nullptr && value->has_t()) {
      return &value->t();
    }
  }
  return parent_graph_ == nullptr || !parent_graph_->IsVisibleBefore(name, parent_position_limit_)
             ? nullptr
             : parent_graph_->GetComputedConstant(name);
}

bool GraphGraph::IsVisibleBefore(const std::string &name, std::size_t position_limit) const {
  const NodeProto *producer = NodeBefore(name);
  return producer == nullptr || Position(*producer) < position_limit;
}

void GraphGraph::SetComputedConstant(const std::string &name, TensorProto value) {
  computed_constants_[name] = std::move(value);
}

bool GraphGraph::ConstantShape(const std::string &name, std::vector<int64_t> &dims) const {
  const TensorProto *tensor = GetComputedConstant(name);
  if (tensor != nullptr) {
    dims.clear();
    for (std::size_t i = 0; i < tensor->dims().size(); ++i) {
      dims.push_back(tensor->dims()[static_cast<std::size_t>(i)]);
    }
    return true;
  }
  // A ``Constant`` node using ``value_int`` / ``value_float`` is a scalar.
  const NodeProto *node = NodeBefore(name);
  if (node != nullptr && node->domain().value().empty() && node->op_type().value() == "Constant") {
    if (FindAttribute(*node, "value_int") != nullptr ||
        FindAttribute(*node, "value_float") != nullptr) {
      dims.clear();
      return true;
    }
  }
  return false;
}

bool GraphGraph::ConstantScalarValue(const std::string &name, double &out) const {
  const TensorProto *tensor = GetComputedConstant(name);
  if (tensor != nullptr) {
    return ReadScalarAsDouble(*tensor, out);
  }
  const NodeProto *node = NodeBefore(name);
  if (node != nullptr && node->domain().value().empty() && node->op_type().value() == "Constant") {
    const AttributeProto *value_int = FindAttribute(*node, "value_int");
    if (value_int != nullptr && value_int->has_i()) {
      out = static_cast<double>(value_int->i());
      return true;
    }
    const AttributeProto *value_float = FindAttribute(*node, "value_float");
    if (value_float != nullptr && value_float->has_f()) {
      out = static_cast<double>(value_float->f());
      return true;
    }
  }
  return false;
}

namespace {

// Returns ``true`` when the concrete shape ``dims`` is a scalar shape according
// to the broadcast rule: without broadcasting only ``()`` and ``(1,)`` qualify;
// with broadcasting every dimension must be ``1``.
bool IsScalarShape(const std::vector<int64_t> &dims, bool broadcast) {
  if (broadcast) {
    for (int64_t d : dims) {
      if (d != 1) {
        return false;
      }
    }
    return true;
  }
  return dims.empty() || (dims.size() == 1 && dims[0] == 1);
}

} // namespace

bool GraphGraph::IsConstantScalar(const std::string &name, bool broadcast) const {
  if (!IsConstant(name)) {
    return false;
  }
  std::vector<int64_t> dims;
  if (!ConstantShape(name, dims)) {
    return false;
  }
  return IsScalarShape(dims, broadcast);
}

bool GraphGraph::IsConstantScalar(const std::string &name, double value, bool broadcast) const {
  if (!IsConstantScalar(name, broadcast)) {
    return false;
  }
  double scalar = 0.0;
  if (!ConstantScalarValue(name, scalar)) {
    return false;
  }
  return scalar == value;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::builder
