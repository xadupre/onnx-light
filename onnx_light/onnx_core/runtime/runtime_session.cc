// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_session.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_core/runtime/kernels/run_nodes_internal.h"
#include "onnx_core/shapes/shapes_context.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

RuntimeSession::RuntimeSession(const ModelProto &model, int verbose)
    : RuntimeSession(model, RuntimeSessionOptions{
                                .parameters = RuntimeParameters(),
                                .verbose = verbose,
                                .check_shapes = false,
                            }) {}

RuntimeSession::RuntimeSession(const ModelProto &model, RuntimeSessionOptions options)
    : default_plan_(model.graph()), plan_(default_plan_), check_shapes_(options.check_shapes),
      allow_external_output_allocators_(options.allow_external_output_allocators),
      parameters_(std::move(options.parameters)), verbose_(options.verbose) {
  SetDeclaredShapes(model.graph());
}

RuntimeSession::RuntimeSession(const GraphProto &graph, int verbose)
    : default_plan_(graph), plan_(default_plan_), verbose_(verbose) {
  SetDeclaredShapes(graph);
}

RuntimeSession::RuntimeSession(const ExecutionPlan &plan, int verbose)
    : RuntimeSession(plan, RuntimeSessionOptions{
                               .parameters = RuntimeParameters(),
                               .verbose = verbose,
                               .check_shapes = false,
                           }) {}

RuntimeSession::RuntimeSession(const ExecutionPlan &plan, RuntimeSessionOptions options)
    : plan_(plan), check_shapes_(options.check_shapes),
      allow_external_output_allocators_(options.allow_external_output_allocators),
      parameters_(std::move(options.parameters)), verbose_(options.verbose) {}

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
    core::symbolic::SymShape dims;
    for (std::size_t i = 0; i < shape.dim().size(); ++i) {
      const TensorShapeProto::Dimension &d = shape.dim()[i];
      if (d.has_dim_value()) {
        dims.PushBack(core::symbolic::SymDim(static_cast<int64_t>(d.dim_value())));
      } else if (d.has_dim_param()) {
        dims.PushBack(core::symbolic::SymDim(d.dim_param().value()));
      } else {
        dims.PushBack(core::symbolic::SymDim(""));
      }
    }
    declared_shapes_[vi.name().value()] = std::move(dims);
  };
  for (std::size_t i = 0; i < graph.input().size(); ++i) {
    record(graph.input()[i]);
  }
  for (std::size_t i = 0; i < graph.output().size(); ++i) {
    record(graph.output()[i]);
  }
  // Record the graph's declared output names so Run() can detach any borrowed
  // output from the model before returning (see MaterializeBorrowedOutputs).
  output_names_.clear();
  output_names_set_.clear();
  for (std::size_t i = 0; i < graph.output().size(); ++i) {
    const std::string &name = graph.output()[i].name();
    if (!name.empty()) {
      output_names_.push_back(name);
      output_names_set_.insert(name);
    }
  }
  for (std::size_t i = 0; i < graph.value_info().size(); ++i) {
    record(graph.value_info()[i]);
  }
}

std::vector<std::string>
RuntimeSession::CollectExternalInputs(const utils::RepeatedProtoField<NodeProto> &nodes) {
  return ::ONNX_LIGHT_NAMESPACE::core::graph::CollectExternalInputs(nodes);
}

std::vector<std::string> RuntimeSession::CollectNodeInputs(const NodeProto &node) {
  return ::ONNX_LIGHT_NAMESPACE::core::graph::CollectNodeInputs(node);
}

std::unique_ptr<KernelBase> RuntimeSession::ResolveNodeKernel(const NodeProto &node,
                                                              RuntimeContext &rt,
                                                              const std::string &domain,
                                                              const std::string &op_type) const {
  return detail::ResolveNodeKernelDefault(node, rt, domain, op_type);
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
    prepared.instance = ResolveNodeKernel(node, rt, domain, op_type);
  }

  // Capture exactly one immutable registry generation after all factories have
  // run, then configure every tunable kernel from that generation.
  tuning_resolution_statistics_ = {};
  const auto snapshot_start = std::chrono::steady_clock::now();
  tuning_snapshot_ = GetKernelTuningRegistry().Snapshot();
  tuning_resolution_statistics_.snapshot_duration_ns =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - snapshot_start)
                                .count());
  const CpuExecutionDescriptor tuning_execution{
      platform::GetCpuDescriptor(), static_cast<uint32_t>(parameters_.EffectiveNumThreads())};
  struct PendingTuning {
    PreparedKernel *kernel;
    KernelTuningKey key;
  };
  std::vector<PendingTuning> pending_tuning;
  pending_tuning.reserve(kernels_.size());
  for (const ExecuteAction &action : plan_.actions()) {
    if (action.kind() != ExecuteActionKind::kExecuteNode) {
      continue;
    }
    const size_t index = action.node_index();
    const NodeProto &node = *nodes[index];
    int32_t element_type = static_cast<int32_t>(DataType::UNDEFINED);
    for (int input_index = 0; input_index < node.input_size(); ++input_index) {
      const std::string &input = node.input(input_index);
      if (!input.empty() && rt.Has(input)) {
        element_type = rt.Get(input).data_type;
        break;
      }
    }

    PreparedKernel &prepared = kernels_[index];
    const KernelTuningKey tuning_key = prepared.instance->TuningKey(element_type);
    if (tuning_key.device == Device::kUndefined) {
      continue;
    }
    pending_tuning.push_back({&prepared, tuning_key});
  }
  tuning_resolution_statistics_.tunable_kernels = pending_tuning.size();
  std::vector<const KernelTuningParameters *> resolved_parameters;
  resolved_parameters.reserve(pending_tuning.size());
  if (!pending_tuning.empty()) {
    const auto resolution_start = std::chrono::steady_clock::now();
    for (const PendingTuning &pending : pending_tuning) {
      resolved_parameters.push_back(tuning_snapshot_->Resolve(pending.key, tuning_execution));
    }
    tuning_resolution_statistics_.resolution_duration_ns =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - resolution_start)
                                  .count());
  }
  for (size_t i = 0; i < pending_tuning.size(); ++i) {
    const KernelTuningParameters *parameters = resolved_parameters[i];
    if (parameters != nullptr) {
      ++tuning_resolution_statistics_.resolved_profiles;
      pending_tuning[i].kernel->instance->Configure(*parameters);
    }
  }

  // Record the external inputs the scheduled nodes read (names not produced by
  // any node in the plan, including values captured by subgraph attributes) so
  // :cpp:func:`Run` can verify the RuntimeContext supplies them before it
  // starts executing kernels.
  required_inputs_ = ::ONNX_LIGHT_NAMESPACE::core::graph::CollectExternalInputs(nodes);
  kernels_initialized_ = true;
}

std::vector<std::string> RuntimeSession::used_kernels() const {
  std::vector<std::string> result;
  if (!kernels_initialized_) {
    return result;
  }
  result.reserve(kernels_.size());
  for (const ExecuteAction &action : plan_.actions()) {
    if (action.kind() == ExecuteActionKind::kExecuteNode) {
      result.push_back(kernels_[action.node_index()].key);
    }
  }
  return result;
}

bool RuntimeSession::ProducesDeclaredOutput(const NodeProto &node) const {
  if (output_names_set_.empty()) {
    return false;
  }
  for (int i = 0; i < node.output_size(); ++i) {
    const std::string &name = node.output(i);
    if (!name.empty() && output_names_set_.find(name) != output_names_set_.end()) {
      return true;
    }
  }
  return false;
}

void RuntimeSession::VerifyOutputAllocators(const NodeProto &node, RuntimeContext &rt) const {
  const bool routed_to_io = session_io_allocator_ != nullptr && ProducesDeclaredOutput(node);
  const RawBufferAllocator *expected = routed_to_io ? session_io_allocator_ : session_allocator_;
  const char *expected_name =
      routed_to_io ? "the session's I/O allocator" : "the session's execution allocator";
  for (int i = 0; i < node.output_size(); ++i) {
    const std::string &name = node.output(i);
    if (name.empty() || !rt.Has(name)) {
      continue;
    }
    const Tensor &output = rt.Get(name);
    if (!output.has_allocation()) {
      continue;
    }
    EXT_ENFORCE_INVALID(output.allocation_owner() == expected, "RuntimeSession: op '",
                        node.op_type(), "' produced output '", name,
                        "' backed by an allocator other than ", expected_name, ".");
  }
}

void RuntimeSession::VerifyDeclaredShape(const std::string &name, const RuntimeContext &rt,
                                         core::shapes::ShapesContext &bindings) const {
  auto it = declared_shapes_.find(name);
  if (it == declared_shapes_.end() || !rt.Has(name)) {
    return;
  }
  const core::symbolic::SymShape &declared = it->second;
  const Tensor &tensor = rt.Get(name);
  if (declared.FitsConcreteShape(tensor.shape.size(), tensor.shape.data(), bindings)) {
    return;
  }
  std::string concrete = "[";
  for (size_t i = 0; i < tensor.shape.size(); ++i) {
    if (i > 0) {
      concrete.push_back(',');
    }
    concrete += std::to_string(tensor.shape[i]);
  }
  concrete.push_back(']');
  EXT_ENFORCE_INVALID(false, "RuntimeSession: tensor '", name, "' has concrete shape ", concrete,
                      " which is incompatible with its declared (possibly symbolic) shape ",
                      declared.ToString(), ".");
}

void RuntimeSession::Run(RuntimeContext &rt) {
  // Kernels are resolved against ``rt`` on the first run and cached; later
  // runs reuse the same built instances without redoing the per-node
  // dispatch lookup or re-constructing concrete kernels.
  if (!kernels_initialized_) {
    InitializeKernels(rt);
  }
  // Use the session's construction-time verbosity when it is non-zero;
  // otherwise fall back to the RuntimeContext's own verbosity. The context
  // itself is left untouched.
  const int effective_verbose = verbose_ != 0 ? verbose_ : rt.verbose();
  // Capture the allocator attached to ``rt`` once, on the first Run, as the
  // session's unique allocator; every output tensor produced from here on is
  // verified against this same allocator (see VerifyOutputAllocators). The
  // I/O allocator (if any) is captured alongside it so declared graph
  // outputs can be routed to it (see the kExecuteNode case below).
  if (!session_allocator_captured_) {
    session_allocator_ = rt.execution_allocator();
    session_io_allocator_ = rt.io_allocator();
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
  // shapes against the concrete tensors. ``shape_bindings`` binds each symbolic
  // ``dim_param`` to the first concrete value it resolves to during this run so
  // that every later occurrence of the same symbol is checked for consistency.
  // The graph inputs / initializers already present are validated up front;
  // each node's outputs are validated right after it runs.
  const bool check_shapes = check_shapes_ && !declared_shapes_.empty();
  core::shapes::ShapesContext shape_bindings;
  if (check_shapes) {
    for (const std::string &name : required_inputs_) {
      VerifyDeclaredShape(name, rt, shape_bindings);
    }
  }
  // Replay the plan's ordered action list. Each kExecuteNode action invokes
  // the kernel prepared during initialization; each kDeleteBuffer /
  // kDeleteShape action frees the intermediate the plan scheduled for release.
  // Every other action (lock / unlock / allocate / transfer / temporary-buffer
  // bookkeeping) is informational for this session — the kernels manage their
  // own allocations — but is still matched by an explicit case so no scheduled
  // event is silently ignored.
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
      const NodeProto &node = *nodes[index];
      const std::string &domain = ONNX_LIGHT_NAMESPACE::NormaliseDispatchDomain(node);
      const std::string &op_type = node.op_type().value();
      detail::PrintNodeProgress(rt, node, domain, op_type, effective_verbose);

      // Route this node's kernel invocation through the I/O allocator instead
      // of the execution allocator when it produces at least one declared
      // graph output and an I/O allocator is attached to ``rt``. This lets a
      // kernel's zero-copy MakeOutputTensor call allocate the final result
      // directly from the I/O arena, with no promotion copy afterwards.
      const bool routed_to_io = session_io_allocator_ != nullptr && ProducesDeclaredOutput(node);
      rt.SetActiveAllocator(routed_to_io ? session_io_allocator_ : session_allocator_);

      // Only capture timing when event logging is active.
      const bool logging = rt.events_enabled();
      int64_t start_time_ns = 0;
      std::chrono::steady_clock::time_point t0;
      if (logging) {
        start_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
        t0 = std::chrono::steady_clock::now();
      }

      prepared.instance->Run(rt);

      if (logging) {
        const int64_t duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - t0)
                                        .count();
        rt.RecordRunNodeEvent(node, domain, op_type, start_time_ns, duration_ns);
      }
      if (!allow_external_output_allocators_) {
        VerifyOutputAllocators(*nodes[index], rt);
      }
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
    // Actions this session performs no explicit work for: the kernels manage
    // their own buffers, so locks/unlocks, (temporary-)buffer allocations,
    // transfers and shape creation require nothing here — they are matched by
    // an explicit case only so no scheduled event is silently ignored.
    case ExecuteActionKind::kLockInitializer:
    case ExecuteActionKind::kUnlockInitializer:
    case ExecuteActionKind::kLockInput:
    case ExecuteActionKind::kUnlockInput:
    case ExecuteActionKind::kAllocateBuffer:
    case ExecuteActionKind::kTransfer:
    case ExecuteActionKind::kCreateShape:
    case ExecuteActionKind::kAllocateTemporaryBuffer:
    case ExecuteActionKind::kDeleteTemporaryBuffer:
      break;
    }
  }
  rt.set_current_node_index(-1);
  // Detach any graph output that borrows into the model (e.g. a Constant's
  // raw_data value or a pass-through initializer) so the returned outputs own
  // their bytes and stay valid once the model is released.
  MaterializeBorrowedOutputs(rt);
}

void RuntimeSession::MaterializeBorrowedOutputs(RuntimeContext &rt) const {
  for (const std::string &name : output_names_) {
    if (!rt.Has(name)) {
      continue;
    }
    Tensor &output = rt.Get(name);
    if (output.is_borrowed()) {
      output = output.ToOwned();
    }
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
