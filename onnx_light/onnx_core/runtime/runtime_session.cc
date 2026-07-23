// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_session.h"

#include <cstddef>
#include <cstdint>
#include <iostream>

#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_core/runtime/run_nodes_internal.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

RuntimeSession::RuntimeSession(const ExecutionPlan &plan) : plan_(plan) {}

std::vector<std::string>
RuntimeSession::CollectExternalInputs(const utils::RepeatedProtoField<NodeProto> &nodes) {
  return ::ONNX_LIGHT_NAMESPACE::core::graph::CollectExternalInputs(nodes);
}

std::vector<std::string>
RuntimeSession::CollectExternalInputs(const std::vector<NodeProto> &nodes) {
  return ::ONNX_LIGHT_NAMESPACE::core::graph::CollectExternalInputs(nodes);
}

std::vector<std::string> RuntimeSession::CollectNodeInputs(const NodeProto &node) {
  return ::ONNX_LIGHT_NAMESPACE::core::graph::CollectNodeInputs(node);
}

void RuntimeSession::InitializeKernels(RuntimeContext &rt) {
  // Resolve the kernel for every node the plan will execute, once and up
  // front. Node indices come from the plan's kExecuteNode actions so nodes
  // the plan never runs (if any) are not resolved. Each entry caches the
  // normalised (domain, op_type) alongside the resolved trampoline so
  // :cpp:func:`Run` never has to redo the dispatch lookup.
  const std::vector<const NodeProto *> &nodes = plan_.nodes();
  kernels_.assign(nodes.size(), PreparedKernel{});
  for (const ExecuteAction &action : plan_.actions()) {
    if (action.kind() != ExecuteActionKind::kExecuteNode) {
      continue;
    }
    const size_t index = action.node_index();
    EXT_ENFORCE_INVALID(index < nodes.size(), "RuntimeSession: plan references node index ", index,
                        " but only ", nodes.size(), " node(s) are available.");
    const NodeProto &node = *nodes[index];
    PreparedKernel &prepared = kernels_[index];
    prepared.domain = ONNX_LIGHT_NAMESPACE::NormaliseDispatchDomain(node);
    prepared.op_type = node.op_type().value();
    prepared.kernel = detail::ResolveNodeKernel(node, rt, prepared.domain, prepared.op_type);
  }
  // Record the external inputs the scheduled nodes read (names not produced by
  // any node in the plan, including values captured by subgraph attributes) so
  // :cpp:func:`Run` can verify the RuntimeContext supplies them before it
  // starts executing kernels.
  required_inputs_ = ::ONNX_LIGHT_NAMESPACE::core::graph::CollectExternalInputs(nodes);
  kernels_initialized_ = true;
}

void RuntimeSession::Run(RuntimeContext &rt) {
  // Kernels are resolved against ``rt`` on the first run and cached; later
  // runs reuse them without redoing the per-node dispatch lookup.
  if (!kernels_initialized_) {
    InitializeKernels(rt);
  }
  // Every external input the scheduled nodes read must already be available in
  // ``rt`` (as a tensor, sequence or map) — graph initializers are seeded
  // before Run and graph inputs are supplied by the caller. Surface a missing
  // one as a clear error before executing any kernel rather than failing
  // partway through the plan.
  for (const std::string &name : required_inputs_) {
    EXT_ENFORCE_INVALID(rt.Has(name) || rt.HasSequence(name) || rt.HasMap(name),
                        "RuntimeSession: required input '", name,
                        "' is not defined in the RuntimeContext before Run().");
  }
  const std::vector<const NodeProto *> &nodes = plan_.nodes();
  // Replay the plan's ordered action list. Each kExecuteNode action invokes
  // the kernel prepared during initialization; each kDeleteBuffer /
  // kDeleteShape action frees the intermediate the plan scheduled for release.
  // Remaining (lock / allocate / …) actions are informational and skipped
  // here, since the kernels manage their own allocations.
  for (const ExecuteAction &action : plan_.actions()) {
    switch (action.kind()) {
    case ExecuteActionKind::kExecuteNode: {
      const size_t index = action.node_index();
      const PreparedKernel &prepared = kernels_[index];
      // Every kExecuteNode index is resolved by InitializeKernels (which
      // walks the same plan), so a missing kernel here means the plan was
      // mutated after initialization — surface it as a clear error instead of
      // invoking an empty std::function.
      EXT_ENFORCE_INVALID(static_cast<bool>(prepared.kernel),
                          "RuntimeSession: kernel for node index ", index,
                          " was not initialized before Run().");
      rt.set_current_node_index(static_cast<int64_t>(index));
      detail::InvokeResolvedKernel(*nodes[index], rt, prepared.domain, prepared.op_type,
                                   prepared.kernel);
      break;
    }
    case ExecuteActionKind::kDeleteBuffer:
      if (rt.verbose() > 1) {
        std::cout << "[RuntimeSession] " << action.kind_name() << " name='" << action.name() << "'"
                  << std::endl;
      }
      rt.Remove(action.name());
      break;
    case ExecuteActionKind::kDeleteShape:
      if (rt.verbose() > 1) {
        std::cout << "[RuntimeSession] " << action.kind_name() << " name='" << action.name() << "'"
                  << std::endl;
      }
      rt.RemoveShape(action.name());
      break;
    case ExecuteActionKind::kDeleteSequence:
      if (rt.verbose() > 1) {
        std::cout << "[RuntimeSession] " << action.kind_name() << " name='" << action.name() << "'"
                  << std::endl;
      }
      rt.RemoveSequence(action.name());
      break;
    case ExecuteActionKind::kDeleteMap:
      if (rt.verbose() > 1) {
        std::cout << "[RuntimeSession] " << action.kind_name() << " name='" << action.name() << "'"
                  << std::endl;
      }
      rt.RemoveMap(action.name());
      break;
    default:
      break;
    }
  }
  rt.set_current_node_index(-1);
}

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
