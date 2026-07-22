// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_session.h"

#include <cstddef>
#include <cstdint>

#include "onnx_core/runtime/run_nodes_internal.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

RuntimeSession::RuntimeSession(const utils::RepeatedProtoField<NodeProto> &nodes,
                               RuntimeContext &rt, const ExecutionPlan &plan)
    : nodes_(nodes), rt_(rt), plan_(plan) {
  InitializeKernels();
}

void RuntimeSession::InitializeKernels() {
  // Resolve the kernel for every node the plan will execute, once and up
  // front. Node indices come from the plan's kExecuteNode actions so nodes
  // the plan never runs (if any) are not resolved. Each entry caches the
  // normalised (domain, op_type) alongside the resolved trampoline so
  // :cpp:func:`Run` never has to redo the dispatch lookup.
  kernels_.assign(nodes_.size(), PreparedKernel{});
  for (const ExecuteAction &action : plan_.actions()) {
    if (action.kind() != ExecuteActionKind::kExecuteNode) {
      continue;
    }
    const size_t index = action.node_index();
    EXT_ENFORCE_INVALID(index < nodes_.size(), "RuntimeSession: plan references node index ", index,
                        " but only ", nodes_.size(), " node(s) are available.");
    const NodeProto &node = nodes_[index];
    PreparedKernel &prepared = kernels_[index];
    prepared.domain = ONNX_LIGHT_NAMESPACE::NormaliseDispatchDomain(node);
    prepared.op_type = node.op_type().value();
    prepared.kernel = detail::ResolveNodeKernel(node, rt_, prepared.domain, prepared.op_type);
  }
}

void RuntimeSession::Run() {
  // Replay the plan's ordered action list. Each kExecuteNode action invokes
  // the kernel prepared during construction; each kDeleteBuffer / kDeleteShape
  // action frees the intermediate the plan scheduled for release. Remaining
  // (lock / allocate / …) actions are informational and skipped here, since
  // the kernels manage their own allocations.
  for (const ExecuteAction &action : plan_.actions()) {
    switch (action.kind()) {
    case ExecuteActionKind::kExecuteNode: {
      const size_t index = action.node_index();
      const PreparedKernel &prepared = kernels_[index];
      // Every kExecuteNode index is resolved by InitializeKernels (which
      // walks the same plan), so a missing kernel here means the plan was
      // mutated after construction — surface it as a clear error instead of
      // invoking an empty std::function.
      EXT_ENFORCE_INVALID(static_cast<bool>(prepared.kernel),
                          "RuntimeSession: kernel for node index ", index,
                          " was not initialized before Run().");
      rt_.set_current_node_index(static_cast<int64_t>(index));
      detail::InvokeResolvedKernel(nodes_[index], rt_, prepared.domain, prepared.op_type,
                                   prepared.kernel);
      break;
    }
    case ExecuteActionKind::kDeleteBuffer:
    case ExecuteActionKind::kDeleteShape:
      rt_.Remove(action.name());
      rt_.RemoveSequence(action.name());
      break;
    default:
      break;
    }
  }
  rt_.set_current_node_index(-1);
}

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
