// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/execution_plan.h"
#include "onnx_core/compute/inplace_reuse.h"
#include "onnx_core/compute/peak_memory.h"
#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstddef>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

namespace {

/// Returns the value of the ``metadata_props`` entry keyed by ``key`` on
/// ``node``, or an empty string when the key is absent.
std::string ReadNodeMetadata(const NodeProto &node, const char *key) {
  for (std::size_t i = 0; i < node.metadata_props().size(); ++i) {
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
std::vector<compute::InPlaceReuse> ParseInPlaceReuse(const std::string &value) {
  std::vector<compute::InPlaceReuse> reuse;
  for (const std::string &triplet : SplitNames(value)) {
    const size_t first = triplet.find(':');
    if (first == std::string::npos) {
      continue;
    }
    const size_t second = triplet.find(':', first + 1);
    if (second == std::string::npos) {
      continue;
    }
    compute::InPlaceReuse r;
    r.output_index = std::strtoll(triplet.substr(0, first).c_str(), nullptr, 10);
    r.input_index =
        std::strtoll(triplet.substr(first + 1, second - first - 1).c_str(), nullptr, 10);
    r.kind = triplet.substr(second + 1) == "greater" ? compute::InPlaceReuseKind::kGreater
                                                     : compute::InPlaceReuseKind::kEqual;
    reuse.push_back(r);
  }
  return reuse;
}

/// Parses the ``kNodePeakMemoryMetadataKey`` value (the decimal string
/// representation of an ``int64_t`` byte count written by
/// :cpp:func:`compute::WritePeakMemoryToMetadata`) into a size in bytes.
/// Returns ``0`` when the metadata is absent, empty or non-positive, i.e. when
/// the node needs no scratch/temporary buffer.
size_t ParsePeakMemory(const std::string &value) {
  if (value.empty()) {
    return 0;
  }
  const long long bytes = std::strtoll(value.c_str(), nullptr, 10);
  return bytes > 0 ? static_cast<size_t>(bytes) : 0;
}

} // namespace

ExecutionPlan::ExecutionPlan(const utils::RepeatedProtoField<NodeProto> &nodes,
                             std::unordered_set<std::string> keep)
    : keep_(std::move(keep)) {
  nodes_.reserve(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    nodes_.push_back(&nodes[i]);
    node_index_.emplace(&nodes[i], i);
  }
  BuildActions();
}

ExecutionPlan::ExecutionPlan(const GraphProto &graph) {
  inputs_.reserve(graph.input().size());
  for (size_t i = 0; i < graph.input().size(); ++i) {
    const std::string name = graph.input()[i].name();
    if (!name.empty()) {
      keep_.insert(name);
      inputs_.push_back(name);
    }
  }
  initializers_.reserve(graph.initializer().size());
  for (size_t i = 0; i < graph.initializer().size(); ++i) {
    const std::string name = graph.initializer()[i].name();
    if (!name.empty()) {
      keep_.insert(name);
      initializers_.push_back(name);
    }
  }
  outputs_.reserve(graph.output().size());
  for (size_t i = 0; i < graph.output().size(); ++i) {
    const std::string name = graph.output()[i].name();
    if (!name.empty()) {
      keep_.insert(name);
      outputs_.push_back(name);
    }
  }
  nodes_.reserve(graph.node().size());
  for (size_t i = 0; i < graph.node().size(); ++i) {
    nodes_.push_back(&graph.node()[i]);
    node_index_.emplace(&graph.node()[i], i);
  }
  BuildActions();
}

ExecutionPlan::ExecutionPlan(const FunctionProto &func) {
  inputs_.reserve(func.input_size());
  for (size_t i = 0; i < static_cast<std::size_t>(func.input_size()); ++i) {
    const std::string name = func.input(i);
    if (!name.empty()) {
      keep_.insert(name);
      inputs_.push_back(name);
    }
  }
  outputs_.reserve(func.output_size());
  for (size_t i = 0; i < static_cast<std::size_t>(func.output_size()); ++i) {
    const std::string name = func.output(i);
    if (!name.empty()) {
      keep_.insert(name);
      outputs_.push_back(name);
    }
  }
  nodes_.reserve(func.node().size());
  for (size_t i = 0; i < func.node().size(); ++i) {
    nodes_.push_back(&func.node()[i]);
    node_index_.emplace(&func.node()[i], i);
  }
  BuildActions();
}

void ExecutionPlan::BuildActions() {
  actions_.clear();
  if (nodes_.empty()) {
    return;
  }
  // Each node typically emits lock-input, allocate-output, execute, and
  // release actions; 4× the node count is a reasonable lower bound.
  actions_.reserve(4 * nodes_.size());

  const std::unordered_set<std::string> input_set(inputs_.begin(), inputs_.end());
  const std::unordered_set<std::string> initializer_set(initializers_.begin(), initializers_.end());
  const std::unordered_set<std::string> output_set(outputs_.begin(), outputs_.end());

  // The memory-management schedule is metadata-driven. Two independent signals
  // are read from the node metadata written by :cpp:class:`compute::
  // ComputeContext`:
  //
  //   * ``annotated`` (any ``release_after`` entry) selects the source of the
  //     per-node release schedule: the ``release_after`` metadata when present,
  //     or a topology fallback (each intermediate freed after its last use) for
  //     un-annotated node ranges. Either way the schedule is materialised as
  //     :cpp:enumerator:`ExecuteActionKind::kDeleteBuffer` /
  //     :cpp:enumerator:`ExecuteActionKind::kDeleteShape` actions tagged with
  //     the releasing node index, so :cpp:func:`ReleaseAfter` reads it back
  //     from :cpp:func:`actions` alone.
  //   * ``strict`` (any ``not_used_after`` entry) means the node range carries
  //     explicit lock-lifetime information; the plan then enforces that the
  //     metadata fully covers the schedule (every consumed result is released,
  //     every input / initializer is unlocked at its last use, every released
  //     shape was created) and unlocks at last use instead of at the very end.
  //     Node ranges without lifetime metadata (e.g. a model run without the
  //     in-place reuse pass, or annotated only for memory profiling) are built
  //     best-effort and the completeness checks are skipped.
  bool annotated = false;
  bool strict = false;
  for (const NodeProto *node_ptr : nodes_) {
    if (!ReadNodeMetadata(*node_ptr, compute::kReleaseAfterMetadataKey).empty()) {
      annotated = true;
    }
    if (!ReadNodeMetadata(*node_ptr, compute::kNotUsedAfterMetadataKey).empty()) {
      strict = true;
    }
  }

  // When the node range is not annotated with ``release_after`` metadata, the
  // per-node release schedule is derived from graph topology: each intermediate
  // is freed after its last use, excluding the structural ``keep`` names.
  std::vector<std::vector<std::string>> topology_releases;
  if (!annotated) {
    const size_t n = nodes_.size();
    std::vector<std::vector<std::string>> per_node_inputs;
    per_node_inputs.reserve(n);
    std::unordered_map<std::string, size_t> last_use;
    for (size_t i = 0; i < n; ++i) {
      std::vector<std::string> node_inputs =
          ::ONNX_LIGHT_NAMESPACE::core::graph::CollectNodeInputs(*nodes_[i]);
      for (const std::string &name : node_inputs) {
        last_use[name] = i;
      }
      per_node_inputs.push_back(std::move(node_inputs));
    }
    topology_releases.assign(n, {});
    for (size_t i = 0; i < n; ++i) {
      for (const std::string &name : per_node_inputs[i]) {
        if (keep_.count(name) != 0) {
          continue;
        }
        auto it = last_use.find(name);
        if (it != last_use.end() && it->second == i) {
          topology_releases[i].push_back(name);
        }
      }
    }
  }

  // Collect the names classified as "shape" by value tagging across the whole
  // node range (via the per-node release_after_shape_tag annotation). A value in
  // this set is a shape (its shape is created / destroyed); any other value is a
  // result (its buffer is allocated / freed).
  std::unordered_set<std::string> shape_tagged;
  for (const NodeProto *node_ptr : nodes_) {
    for (const std::string &name :
         SplitNames(ReadNodeMetadata(*node_ptr, compute::kReleaseAfterShapeTagMetadataKey))) {
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

  // Names for which a shape was created / a buffer was released, used by the
  // completeness checks in annotated mode.
  std::unordered_set<std::string> created_shapes;
  std::unordered_set<std::string> released;

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
    const std::vector<compute::InPlaceReuse> reuse =
        ParseInPlaceReuse(ReadNodeMetadata(node, compute::kInPlaceReuseMetadataKey));

    // Allocate a result (or create a shape) for each named output.
    for (int o = 0; o < node.output_size(); ++o) {
      const std::string out = node.output(o);
      if (out.empty()) {
        continue;
      }
      if (shape_tagged.count(out) != 0) {
        // A shape is created (metadata only, no data buffer).
        actions_.emplace_back(ExecuteActionKind::kCreateShape, out);
        created_shapes.insert(out);
        continue;
      }
      const compute::InPlaceReuse *match = nullptr;
      for (const compute::InPlaceReuse &r : reuse) {
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
        actions_.emplace_back(ExecuteActionKind::kAllocateBuffer, out, 0, 0, std::move(target),
                              *match);
      } else {
        // A result is freshly allocated (no in-place reuse for this output).
        actions_.emplace_back(ExecuteActionKind::kAllocateBuffer, out);
      }
    }

    // A temporary/scratch buffer required by the node's kernel to handle a
    // memory peak is allocated right before the node runs and freed right
    // after. The size is read from the peak-memory metadata written by
    // :cpp:func:`compute::WritePeakMemoryToMetadata`; nodes without it (or
    // with a non-positive estimate) need no scratch buffer.
    const size_t peak_memory =
        ParsePeakMemory(ReadNodeMetadata(node, compute::kNodePeakMemoryMetadataKey));
    if (peak_memory != 0) {
      actions_.emplace_back(ExecuteActionKind::kAllocateTemporaryBuffer, std::string(), i,
                            peak_memory);
    }

    // A node execution does not target a named result, so its name is empty;
    // the node it runs is identified by ``node_index``.
    actions_.emplace_back(ExecuteActionKind::kExecuteNode, std::string(), i);

    if (peak_memory != 0) {
      actions_.emplace_back(ExecuteActionKind::kDeleteTemporaryBuffer, std::string(), i,
                            peak_memory);
    }

    // Free the intermediates whose last use falls at this node and unlock the
    // inputs / initializers reaching their last use here (not_used_after). When
    // the node range is annotated, the releases come from the per-node
    // ``release_after`` metadata; otherwise they come from the topology
    // fallback computed above. Both are emitted as delete actions tagged with
    // the releasing node index ``i`` so :cpp:func:`ReleaseAfter` can read them
    // back from :cpp:func:`actions`.
    if (annotated) {
      for (const std::string &name :
           SplitNames(ReadNodeMetadata(node, compute::kReleaseAfterMetadataKey))) {
        if (shape_tagged.count(name) != 0) {
          // A shape is destroyed: it must have been created earlier.
          EXT_ENFORCE(!strict || created_shapes.count(name) != 0, "ExecutionPlan: shape '", name,
                      "' is released but was never created (missing shape metadata).");
          actions_.emplace_back(ExecuteActionKind::kDeleteShape, name, i);
        } else {
          actions_.emplace_back(ExecuteActionKind::kDeleteBuffer, name, i);
        }
        released.insert(name);
      }
    } else {
      for (const std::string &name : topology_releases[i]) {
        actions_.emplace_back(ExecuteActionKind::kDeleteBuffer, name, i);
        released.insert(name);
      }
    }
    for (const std::string &name :
         SplitNames(ReadNodeMetadata(node, compute::kNotUsedAfterMetadataKey))) {
      if (initializer_set.count(name) != 0) {
        actions_.emplace_back(ExecuteActionKind::kUnlockInitializer, name);
        locked.erase(name);
      } else if (input_set.count(name) != 0) {
        actions_.emplace_back(ExecuteActionKind::kUnlockInput, name);
        locked.erase(name);
      }
    }
  }

  if (strict) {
    // The node range carries explicit lock-lifetime metadata, so the metadata
    // must fully cover the schedule: every intermediate result that is consumed
    // by a later node must be released, and every input / initializer that
    // reaches its last use must be unlocked. Anything left unresolved means the
    // ComputeContext annotation is incomplete.
    std::unordered_set<std::string> consumed;
    for (const NodeProto *node_ptr : nodes_) {
      for (int in = 0; in < node_ptr->input_size(); ++in) {
        const std::string name = node_ptr->input(in);
        if (!name.empty()) {
          consumed.insert(name);
        }
      }
    }
    for (const NodeProto *node_ptr : nodes_) {
      for (int o = 0; o < node_ptr->output_size(); ++o) {
        const std::string out = node_ptr->output(o);
        if (out.empty() || output_set.count(out) != 0) {
          // Empty or declared graph / function output: kept, never released.
          continue;
        }
        if (consumed.count(out) == 0) {
          // Dead result never referenced again: the ComputeContext does not
          // schedule a release for it, so neither does the plan.
          continue;
        }
        EXT_ENFORCE(released.count(out) != 0, "ExecutionPlan: result '", out,
                    "' is never released or unlocked (missing release metadata).");
      }
    }
    for (const std::string &name : locked) {
      // An input / initializer that is also a declared output stays locked for
      // the caller; anything else must have been unlocked at its last use.
      EXT_ENFORCE(output_set.count(name) != 0, "ExecutionPlan: input or initializer '", name,
                  "' is never unlocked (missing not_used_after metadata).");
    }
  } else {
    // No lifetime metadata: unlock anything still locked once the whole
    // sequence has run. Iterate in declared order for determinism.
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
}

void ExecutionPlan::ReleaseAfter(const NodeProto &node, RuntimeContext &rt) const {
  auto it = node_index_.find(&node);
  if (it == node_index_.end()) {
    return;
  }
  // The release schedule lives entirely in :cpp:func:`actions`: every
  // intermediate whose last use falls at ``node`` is materialised as a
  // kDeleteBuffer / kDeleteShape / kDeleteSequence / kDeleteMap action tagged
  // with this node's index.
  const size_t index = it->second;
  for (const ExecuteAction &action : actions_) {
    if (action.node_index() != index) {
      continue;
    }
    // Each delete kind targets a single map, so exactly one remover is called
    // depending on the action's kind.
    switch (action.kind()) {
    case ExecuteActionKind::kDeleteBuffer:
      rt.Remove(action.name());
      break;
    case ExecuteActionKind::kDeleteShape:
      rt.RemoveShape(action.name());
      break;
    case ExecuteActionKind::kDeleteSequence:
      rt.RemoveSequence(action.name());
      break;
    case ExecuteActionKind::kDeleteMap:
      rt.RemoveMap(action.name());
      break;
    default:
      break;
    }
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
