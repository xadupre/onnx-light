// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/execution_plan.h"
#include "onnx_core/runtime/runtime_context.h"

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

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
  // Lock initializers and inputs before any node runs.
  for (const std::string &name : initializers_) {
    actions_.emplace_back(ExecuteActionKind::kLockInitializer, name, allocator_);
  }
  for (const std::string &name : inputs_) {
    actions_.emplace_back(ExecuteActionKind::kLockInput, name, allocator_);
  }
  // For every node, allocate a buffer and create the shape for each named
  // output, execute the node, then delete the shape and free the buffer of
  // every intermediate whose last reference falls at this node.
  for (size_t i = 0; i < nodes_.size(); ++i) {
    const NodeProto &node = *nodes_[i];
    for (int o = 0; o < node.output_size(); ++o) {
      const std::string out = node.output(o);
      if (out.empty()) {
        continue;
      }
      actions_.emplace_back(ExecuteActionKind::kAllocateBuffer, out, allocator_);
      actions_.emplace_back(ExecuteActionKind::kCreateShape, out);
    }
    actions_.emplace_back(ExecuteActionKind::kExecuteNode, std::string(), nullptr, i);
    if (i < releasable_.size()) {
      for (const std::string &name : releasable_[i]) {
        actions_.emplace_back(ExecuteActionKind::kDeleteShape, name);
        actions_.emplace_back(ExecuteActionKind::kDeleteBuffer, name, allocator_);
      }
    }
  }
  // Unlock inputs and initializers once the whole sequence has run.
  for (const std::string &name : inputs_) {
    actions_.emplace_back(ExecuteActionKind::kUnlockInput, name);
  }
  for (const std::string &name : initializers_) {
    actions_.emplace_back(ExecuteActionKind::kUnlockInitializer, name);
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
