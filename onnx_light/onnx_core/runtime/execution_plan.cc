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

ExecutionPlan::ExecutionPlan(const utils::RepeatedProtoField<NodeProto> &nodes,
                             std::unordered_set<std::string> keep)
    : keep_(std::move(keep)), releasable_(RuntimeContext::ComputeReleasableInputs(nodes, keep_)) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    node_index_.emplace(&nodes[i], i);
  }
}

ExecutionPlan::ExecutionPlan(const GraphProto &graph) {
  for (size_t i = 0; i < graph.input().size(); ++i) {
    const std::string name = graph.input()[i].name();
    if (!name.empty()) {
      keep_.insert(name);
    }
  }
  for (size_t i = 0; i < graph.initializer().size(); ++i) {
    const std::string name = graph.initializer()[i].name();
    if (!name.empty()) {
      keep_.insert(name);
    }
  }
  for (size_t i = 0; i < graph.output().size(); ++i) {
    const std::string name = graph.output()[i].name();
    if (!name.empty()) {
      keep_.insert(name);
    }
  }
  releasable_ = RuntimeContext::ComputeReleasableInputs(graph.node(), keep_);
  for (size_t i = 0; i < graph.node().size(); ++i) {
    node_index_.emplace(&graph.node()[i], i);
  }
}

ExecutionPlan::ExecutionPlan(const FunctionProto &func) {
  for (size_t i = 0; i < func.input_size(); ++i) {
    const std::string name = func.input(i);
    if (!name.empty()) {
      keep_.insert(name);
    }
  }
  for (size_t i = 0; i < func.output_size(); ++i) {
    const std::string name = func.output(i);
    if (!name.empty()) {
      keep_.insert(name);
    }
  }
  releasable_ = RuntimeContext::ComputeReleasableInputs(func.node(), keep_);
  for (size_t i = 0; i < func.node().size(); ++i) {
    node_index_.emplace(&func.node()[i], i);
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
