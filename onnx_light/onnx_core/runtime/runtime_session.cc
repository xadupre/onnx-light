// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_session.h"

#include <cstddef>
#include <cstdint>

#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_core/runtime/run_nodes_internal.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

RuntimeSession::RuntimeSession(const ModelProto &model)
    : default_plan_(model.graph()), plan_(default_plan_) {
  SetDeclaredShapes(model.graph());
}

RuntimeSession::RuntimeSession(const ExecutionPlan &plan) : plan_(plan) {}

void RuntimeSession::SetDeclaredShapes(const GraphProto &graph) {
  // Records the declared shape of every tensor-typed value the graph exposes
  // (inputs, outputs and value_info). A value contributes an entry only when
  // its type carries a shape; a missing shape means the rank is unknown and
  // nothing can be validated for it.
  auto record = [this](const ValueInfoProto &vi) {
    if (vi.name().empty() || !vi.has_type() || !vi.type().has_tensor_type()) {
      return;
    }
    const TypeProto::Tensor &tt = vi.type().tensor_type();
    if (!tt.has_shape()) {
      return;
    }
    const TensorShapeProto &shape = tt.shape();
    std::vector<DeclaredDim> dims;
    dims.reserve(static_cast<size_t>(shape.dim().size()));
    for (int i = 0; i < shape.dim().size(); ++i) {
      const TensorShapeProto::Dimension &d = shape.dim()[i];
      DeclaredDim dim;
      if (d.has_dim_value()) {
        dim.has_value = true;
        dim.value = static_cast<int64_t>(d.dim_value());
      } else if (d.has_dim_param()) {
        dim.has_param = true;
        dim.param = d.dim_param().value();
      }
      dims.push_back(std::move(dim));
    }
    declared_shapes_[vi.name().value()] = std::move(dims);
  };
  for (int i = 0; i < graph.input().size(); ++i) {
    record(graph.input()[i]);
  }
  for (int i = 0; i < graph.output().size(); ++i) {
    record(graph.output()[i]);
  }
  for (int i = 0; i < graph.value_info().size(); ++i) {
    record(graph.value_info()[i]);
  }
}

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
  // Resolve and build the kernel instance for every node the plan will
  // execute, once and up front. Node indices come from the plan's
  // kExecuteNode actions so nodes the plan never runs (if any) are not
  // resolved. Each entry caches the normalised (domain, op_type) fused into
  // a single key alongside the resolved instance so :cpp:func:`Run` never
  // has to redo the dispatch lookup or reconstruct the concrete kernel.
  const std::vector<const NodeProto *> &nodes = plan_.nodes();
  kernels_.clear();
  kernels_.resize(nodes.size());
  for (const ExecuteAction &action : plan_.actions()) {
    if (action.kind() != ExecuteActionKind::kExecuteNode) {
      continue;
    }
    const size_t index = action.node_index();
    EXT_ENFORCE_INVALID(index < nodes.size(), "RuntimeSession: plan references node index ", index,
                        " but only ", nodes.size(), " node(s) are available.");
    const NodeProto &node = *nodes[index];
    const std::string &domain = ONNX_LIGHT_NAMESPACE::NormaliseDispatchDomain(node);
    const std::string op_type = node.op_type().value();
    PreparedKernel &prepared = kernels_[index];
    prepared.key = domain + ":" + op_type;
    NodeKernelFn factory = detail::ResolveNodeKernel(node, rt, domain, op_type);
    prepared.instance = factory(node, rt);
  }
  // Record the external inputs the scheduled nodes read (names not produced by
  // any node in the plan, including values captured by subgraph attributes) so
  // :cpp:func:`Run` can verify the RuntimeContext supplies them before it
  // starts executing kernels.
  required_inputs_ = ::ONNX_LIGHT_NAMESPACE::core::graph::CollectExternalInputs(nodes);
  kernels_initialized_ = true;
}

void RuntimeSession::VerifyOutputAllocators(const NodeProto &node, RuntimeContext &rt) const {
  for (int i = 0; i < node.output_size(); ++i) {
    const std::string &name = node.output(i);
    if (name.empty() || !rt.Has(name)) {
      continue;
    }
    const Tensor &output = rt.Get(name);
    if (!output.has_allocation()) {
      continue;
    }
    EXT_ENFORCE_INVALID(output.allocation_owner() == session_allocator_, "RuntimeSession: op '",
                        node.op_type(), "' produced output '", name,
                        "' backed by an allocator other than the session's unique allocator.");
  }
}

void RuntimeSession::VerifyDeclaredShape(const std::string &name, const RuntimeContext &rt,
                                         std::unordered_map<std::string, int64_t> &bindings) const {
  auto it = declared_shapes_.find(name);
  if (it == declared_shapes_.end() || !rt.Has(name)) {
    return;
  }
  const std::vector<DeclaredDim> &declared = it->second;
  const Tensor &tensor = rt.Get(name);
  EXT_ENFORCE_INVALID(tensor.shape.size() == declared.size(), "RuntimeSession: tensor '", name,
                      "' has concrete rank ", tensor.shape.size(),
                      " but its declared (symbolic) shape has rank ", declared.size(), ".");
  for (size_t i = 0; i < declared.size(); ++i) {
    const DeclaredDim &dim = declared[i];
    const int64_t concrete = tensor.shape[i];
    if (dim.has_value) {
      EXT_ENFORCE_INVALID(concrete == dim.value, "RuntimeSession: tensor '", name, "' dimension ",
                          i, " is ", concrete, " but the declared shape requires ", dim.value, ".");
    } else if (dim.has_param) {
      auto bound = bindings.find(dim.param);
      if (bound == bindings.end()) {
        bindings.emplace(dim.param, concrete);
      } else {
        EXT_ENFORCE_INVALID(bound->second == concrete, "RuntimeSession: tensor '", name,
                            "' dimension ", i, " is ", concrete, " but the symbolic dimension '",
                            dim.param, "' was already resolved to ", bound->second, ".");
      }
    }
    // A dimension with neither a value nor a param is fully unknown and
    // imposes no constraint on the concrete dimension.
  }
}

void RuntimeSession::Run(RuntimeContext &rt) {
  // Kernels are resolved against ``rt`` on the first run and cached; later
  // runs reuse the same built instances without redoing the per-node
  // dispatch lookup or re-constructing concrete kernels.
  if (!kernels_initialized_) {
    InitializeKernels(rt);
  }
  // Capture the allocator attached to ``rt`` once, on the first Run, as the
  // session's unique allocator; every output tensor produced from here on is
  // verified against this same allocator (see VerifyOutputAllocators).
  if (!session_allocator_captured_) {
    session_allocator_ = rt.allocator();
    session_allocator_captured_ = true;
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
  // When shape validation is enabled, resolve the declared (possibly symbolic)
  // shapes against the concrete tensors. ``shape_bindings`` maps each symbolic
  // ``dim_param`` to the first concrete value it resolves to during this run so
  // that every later occurrence of the same symbol is checked for consistency.
  // The graph inputs / initializers already present are validated up front;
  // each node's outputs are validated right after it runs.
  const bool check_shapes = check_shapes_ && !declared_shapes_.empty();
  std::unordered_map<std::string, int64_t> shape_bindings;
  if (check_shapes) {
    for (const std::string &name : required_inputs_) {
      VerifyDeclaredShape(name, rt, shape_bindings);
    }
  }
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
      // walks the same plan), so a missing kernel instance here means the
      // plan was mutated after initialization — surface it as a clear error
      // instead of dereferencing an empty pointer.
      EXT_ENFORCE_INVALID(static_cast<bool>(prepared.instance),
                          "RuntimeSession: kernel for node index ", index,
                          " was not initialized before Run().");
      rt.set_current_node_index(static_cast<int64_t>(index));
      // Split the fused "<domain>:<op_type>" key back into its two parts;
      // the domain itself never contains ':', so the first occurrence is
      // always the separator.
      const size_t sep = prepared.key.find(':');
      const std::string domain = prepared.key.substr(0, sep);
      const std::string op_type = prepared.key.substr(sep + 1);
      detail::InvokeKernel(*nodes[index], rt, domain, op_type, *prepared.instance);
      VerifyOutputAllocators(*nodes[index], rt);
      if (check_shapes) {
        const NodeProto &executed = *nodes[index];
        for (int i = 0; i < executed.output_size(); ++i) {
          VerifyDeclaredShape(executed.output(i), rt, shape_bindings);
        }
      }
      break;
    }
    case ExecuteActionKind::kDeleteBuffer:
      // Releasing intermediates is opt-in (RuntimeContext::release_intermediates);
      // when it is disabled every intermediate stays observable in ``rt`` after
      // Run() returns, so the scheduled delete actions are skipped entirely.
      // The removal itself is recorded in the RuntimeContext event log (a
      // ``kRemove`` RuntimeEvent carrying the allocator's live / peak memory)
      // when event logging is enabled.
      if (!rt.release_intermediates()) {
        break;
      }
      rt.Remove(action.name());
      break;
    case ExecuteActionKind::kDeleteShape:
      if (!rt.release_intermediates()) {
        break;
      }
      rt.RemoveShape(action.name());
      break;
    case ExecuteActionKind::kDeleteSequence:
      if (!rt.release_intermediates()) {
        break;
      }
      rt.RemoveSequence(action.name());
      break;
    case ExecuteActionKind::kDeleteMap:
      if (!rt.release_intermediates()) {
        break;
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
