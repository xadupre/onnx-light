// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/execution_plan.h"
#include "onnx_core/annotations/inplace_reuse.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

namespace {

/// Returns the value of the ``metadata_props`` entry keyed by ``key`` on
/// ``node``, or an empty string when the key is absent.
std::string ReadNodeMetadata(const NodeProto &node, const char *key) {
  for (int i = 0; i < node.metadata_props().size(); ++i) {
    if (node.metadata_props()[i].key() == key) {
      return node.metadata_props()[i].value();
    }
  }
  return std::string();
}

/// Splits a ``;``-separated metadata value into its individual names, dropping
/// empty entries.
std::vector<std::string> SplitNames(const std::string &value) {
  std::vector<std::string> names;
  size_t start = 0;
  while (start <= value.size()) {
    const size_t end = value.find(';', start);
    const size_t stop = end == std::string::npos ? value.size() : end;
    if (stop > start) {
      names.emplace_back(value.substr(start, stop - start));
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return names;
}

/// Parses the ``kInPlaceReuseMetadataKey`` value (``;``-separated
/// ``output_index:input_index:kind`` triplets, ``kind`` being ``equal`` or
/// ``greater``) into a list of :cpp:class:`InPlaceReuse` decisions. Malformed
/// triplets are skipped.
std::vector<annotations::InPlaceReuse> ParseInPlaceReuse(const std::string &value) {
  std::vector<annotations::InPlaceReuse> reuse;
  for (const std::string &triplet : SplitNames(value)) {
    const size_t first = triplet.find(':');
    if (first == std::string::npos) {
      continue;
    }
    const size_t second = triplet.find(':', first + 1);
    if (second == std::string::npos) {
      continue;
    }
    annotations::InPlaceReuse r;
    r.output_index = std::strtoll(triplet.substr(0, first).c_str(), nullptr, 10);
    r.input_index =
        std::strtoll(triplet.substr(first + 1, second - first - 1).c_str(), nullptr, 10);
    r.kind = triplet.substr(second + 1) == "greater" ? annotations::InPlaceReuseKind::kGreater
                                                     : annotations::InPlaceReuseKind::kEqual;
    reuse.push_back(r);
  }
  return reuse;
}

} // namespace

ExecutionPlan::ExecutionPlan(RawBufferAllocator *allocator) : allocator_(allocator) {
  BuildActions();
}

ExecutionPlan::ExecutionPlan(const utils::RepeatedProtoField<NodeProto> &nodes,
                             std::unordered_set<std::string> keep, RawBufferAllocator *allocator)
    : allocator_(allocator), keep_(std::move(keep)),
      releasable_(RuntimeContext::ComputeReleasableInputs(nodes, keep_)) {
  nodes_.reserve(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    nodes_.push_back(&nodes[i]);
    node_index_.emplace(&nodes[i], i);
  }
  BuildActions();
}

ExecutionPlan::ExecutionPlan(const GraphProto &graph, RawBufferAllocator *allocator)
    : allocator_(allocator) {
  for (size_t i = 0; i < graph.input().size(); ++i) {
    const std::string name = graph.input()[i].name();
    if (!name.empty()) {
      keep_.insert(name);
      inputs_.push_back(name);
    }
  }
  for (size_t i = 0; i < graph.initializer().size(); ++i) {
    const std::string name = graph.initializer()[i].name();
    if (!name.empty()) {
      keep_.insert(name);
      initializers_.push_back(name);
    }
  }
  for (size_t i = 0; i < graph.output().size(); ++i) {
    const std::string name = graph.output()[i].name();
    if (!name.empty()) {
      keep_.insert(name);
    }
  }
  releasable_ = RuntimeContext::ComputeReleasableInputs(graph.node(), keep_);
  nodes_.reserve(graph.node().size());
  for (size_t i = 0; i < graph.node().size(); ++i) {
    nodes_.push_back(&graph.node()[i]);
    node_index_.emplace(&graph.node()[i], i);
  }
  BuildActions();
}

ExecutionPlan::ExecutionPlan(const FunctionProto &func, RawBufferAllocator *allocator)
    : allocator_(allocator) {
  for (size_t i = 0; i < func.input_size(); ++i) {
    const std::string name = func.input(i);
    if (!name.empty()) {
      keep_.insert(name);
      inputs_.push_back(name);
    }
  }
  for (size_t i = 0; i < func.output_size(); ++i) {
    const std::string name = func.output(i);
    if (!name.empty()) {
      keep_.insert(name);
    }
  }
  releasable_ = RuntimeContext::ComputeReleasableInputs(func.node(), keep_);
  nodes_.reserve(func.node().size());
  for (size_t i = 0; i < func.node().size(); ++i) {
    nodes_.push_back(&func.node()[i]);
    node_index_.emplace(&func.node()[i], i);
  }
  BuildActions();
}

void ExecutionPlan::BuildActions() {
  actions_.clear();

  const std::unordered_set<std::string> input_set(inputs_.begin(), inputs_.end());
  const std::unordered_set<std::string> initializer_set(initializers_.begin(), initializers_.end());

  // Collect the names classified as "shape" by value tagging across the whole
  // node range (via the per-node release_after_shape_tag annotation). A value in
  // this set is a shape (its shape is created / destroyed); any other value is a
  // result (its buffer is allocated / freed).
  std::unordered_set<std::string> shape_tagged;
  for (const NodeProto *node_ptr : nodes_) {
    for (const std::string &name :
         SplitNames(ReadNodeMetadata(*node_ptr, annotations::kReleaseAfterShapeTagMetadataKey))) {
      shape_tagged.insert(name);
    }
  }

  // Track inputs / initializers that are currently locked so each is locked
  // exactly once (on first use) and unlocked exactly once (on last use).
  std::unordered_set<std::string> locked;
  auto lock_if_needed = [&](const std::string &name) {
    if (locked.count(name) != 0) {
      return;
    }
    if (initializer_set.count(name) != 0) {
      actions_.emplace_back(ExecuteActionKind::kLockInitializer, name);
      locked.insert(name);
    } else if (input_set.count(name) != 0) {
      actions_.emplace_back(ExecuteActionKind::kLockInput, name);
      locked.insert(name);
    }
  };

  for (size_t i = 0; i < nodes_.size(); ++i) {
    const NodeProto &node = *nodes_[i];

    // Lock an input / initializer the first time it is referenced.
    for (int in = 0; in < node.input_size(); ++in) {
      const std::string name = node.input(in);
      if (!name.empty()) {
        lock_if_needed(name);
      }
    }

    // Read the in-place reuse decisions attached to this node so that an output
    // covered by a reuse opportunity reuses an input buffer instead of being
    // freshly allocated.
    const std::vector<annotations::InPlaceReuse> reuse =
        ParseInPlaceReuse(ReadNodeMetadata(node, annotations::kInPlaceReuseMetadataKey));

    // Allocate a result (or create a shape) for each named output.
    for (int o = 0; o < node.output_size(); ++o) {
      const std::string out = node.output(o);
      if (out.empty()) {
        continue;
      }
      if (shape_tagged.count(out) != 0) {
        // A shape is created (metadata only, no data buffer).
        actions_.emplace_back(ExecuteActionKind::kCreateShape, out);
        continue;
      }
      const annotations::InPlaceReuse *match = nullptr;
      for (const annotations::InPlaceReuse &r : reuse) {
        if (r.output_index == o) {
          match = &r;
          break;
        }
      }
      if (match != nullptr) {
        // The result reuses an input buffer in place: no fresh allocation, the
        // action carries the in-place decision and the reused input name.
        std::string target;
        if (match->input_index >= 0 &&
            match->input_index < static_cast<int64_t>(node.input_size())) {
          target = node.input(static_cast<int>(match->input_index));
        }
        actions_.emplace_back(ExecuteActionKind::kAllocateBuffer, out, nullptr, 0, 0,
                              std::move(target), *match);
      } else {
        // A result is allocated from the plan's allocator.
        actions_.emplace_back(ExecuteActionKind::kAllocateBuffer, out, allocator_);
      }
    }

    // A node execution does not target a named result, so its name is empty;
    // the node it runs is identified by ``node_index``.
    actions_.emplace_back(ExecuteActionKind::kExecuteNode, std::string(), nullptr, i);

    // Free the intermediates whose last use falls at this node (release_after)
    // and unlock the inputs / initializers reaching their last use here
    // (not_used_after) — both are read from the node metadata.
    for (const std::string &name :
         SplitNames(ReadNodeMetadata(node, annotations::kReleaseAfterMetadataKey))) {
      if (shape_tagged.count(name) != 0) {
        actions_.emplace_back(ExecuteActionKind::kDeleteShape, name);
      } else {
        actions_.emplace_back(ExecuteActionKind::kDeleteBuffer, name, allocator_);
      }
    }
    for (const std::string &name :
         SplitNames(ReadNodeMetadata(node, annotations::kNotUsedAfterMetadataKey))) {
      if (initializer_set.count(name) != 0) {
        actions_.emplace_back(ExecuteActionKind::kUnlockInitializer, name);
        locked.erase(name);
      } else if (input_set.count(name) != 0) {
        actions_.emplace_back(ExecuteActionKind::kUnlockInput, name);
        locked.erase(name);
      }
    }
  }

  // Unlock anything still locked once the whole sequence has run (e.g. inputs
  // that are also graph outputs, or when last-use metadata is unavailable).
  // Iterate in declared order for a deterministic schedule.
  for (const std::string &name : inputs_) {
    if (locked.erase(name) != 0) {
      actions_.emplace_back(ExecuteActionKind::kUnlockInput, name);
    }
  }
  for (const std::string &name : initializers_) {
    if (locked.erase(name) != 0) {
      actions_.emplace_back(ExecuteActionKind::kUnlockInitializer, name);
    }
  }
}

void ExecutionPlan::ReleaseAfter(const NodeProto &node, RuntimeContext &rt) const {
  auto it = node_index_.find(&node);
  if (it == node_index_.end()) {
    return;
  }
  for (const auto &name : releasable_[it->second]) {
    rt.Remove(name);
    rt.RemoveSequence(name);
  }
}

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
