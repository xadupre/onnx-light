// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_context.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

namespace {

void EnsureAllocatorBacked(Tensor &tensor, RawBufferAllocator *allocator, RuntimeEventKind kind) {
  // STRING tensors store their payload in string_data instead of raw bytes.
  if (allocator == nullptr || static_cast<DataType>(tensor.data_type) == DataType::STRING) {
    return;
  }
  // External inputs may borrow caller-owned storage for the duration of a run.
  // Copying them into the execution arena adds a full memory-bandwidth pass
  // before inference and defeats the Python runner's zero-copy NumPy adapter.
  if (kind == RuntimeEventKind::kInput && tensor.is_borrowed()) {
    return;
  }
  // Tensor was already allocated by a kernel that received the allocator
  // directly — nothing to migrate.
  if (tensor.has_allocation()) {
    return;
  }
  const size_t n_bytes = tensor.size_bytes();
  // Keep truly empty inline tensors as-is (no bytes and no allocator binding).
  if (n_bytes == 0) {
    return;
  }
  const uint8_t *src = tensor.bytes();
  RawBuffer *allocated = allocator->Allocate(n_bytes);
  EXT_ENFORCE(allocated != nullptr,
              "RuntimeContext: allocator returned a null RawBuffer allocation.");
  EXT_ENFORCE(src != nullptr, "RuntimeContext: tensor has non-zero size with a null data pointer.");
  std::memcpy(allocated->data(), src, n_bytes);
  tensor.SetAllocation(allocator, allocated);
}

int64_t NowNanos() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Decodes up to ``capacity`` values of dtype ``dtype`` starting at ``ptr``
// into a fixed-size ``double`` array. Returns the number of values
// written. Returns 0 for dtypes that are not representable as ``double``
// here.
int32_t DecodeNumericValues(int32_t dtype, const uint8_t *ptr, int64_t element_count,
                            int32_t capacity, std::array<double, kRuntimeEventValueLimit> &out) {
  const int32_t n =
      static_cast<int32_t>(std::min<int64_t>(static_cast<int64_t>(capacity), element_count));
  out.fill(0.0);
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT: {
    const float *p = reinterpret_cast<const float *>(ptr);
    for (int32_t i = 0; i < n; ++i)
      out[i] = static_cast<double>(p[i]);
    return n;
  }
  case DataType::DOUBLE: {
    const double *p = reinterpret_cast<const double *>(ptr);
    for (int32_t i = 0; i < n; ++i)
      out[i] = p[i];
    return n;
  }
  case DataType::INT8: {
    const int8_t *p = reinterpret_cast<const int8_t *>(ptr);
    for (int32_t i = 0; i < n; ++i)
      out[i] = static_cast<double>(p[i]);
    return n;
  }
  case DataType::INT16: {
    const int16_t *p = reinterpret_cast<const int16_t *>(ptr);
    for (int32_t i = 0; i < n; ++i)
      out[i] = static_cast<double>(p[i]);
    return n;
  }
  case DataType::INT32: {
    const int32_t *p = reinterpret_cast<const int32_t *>(ptr);
    for (int32_t i = 0; i < n; ++i)
      out[i] = static_cast<double>(p[i]);
    return n;
  }
  case DataType::INT64: {
    const int64_t *p = reinterpret_cast<const int64_t *>(ptr);
    for (int32_t i = 0; i < n; ++i)
      out[i] = static_cast<double>(p[i]);
    return n;
  }
  case DataType::UINT8:
  case DataType::BOOL: {
    for (int32_t i = 0; i < n; ++i)
      out[i] = static_cast<double>(ptr[i]);
    return n;
  }
  case DataType::UINT16: {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(ptr);
    for (int32_t i = 0; i < n; ++i)
      out[i] = static_cast<double>(p[i]);
    return n;
  }
  case DataType::UINT32: {
    const uint32_t *p = reinterpret_cast<const uint32_t *>(ptr);
    for (int32_t i = 0; i < n; ++i)
      out[i] = static_cast<double>(p[i]);
    return n;
  }
  case DataType::UINT64: {
    const uint64_t *p = reinterpret_cast<const uint64_t *>(ptr);
    for (int32_t i = 0; i < n; ++i)
      out[i] = static_cast<double>(p[i]);
    return n;
  }
  default:
    out.fill(0.0);
    return 0;
  }
}

// Resolves the node_index recorded on an event from its kind and the index
// of the node currently executing: ``-1`` for graph inputs, ``-2`` for
// initializers, and the producing node index for intermediate / output
// tensors.
int64_t ResolveNodeIndex(RuntimeEventKind kind, int64_t current_node_index) noexcept {
  switch (kind) {
  case RuntimeEventKind::kInput:
    return -1;
  case RuntimeEventKind::kInitializer:
    return -2;
  default:
    return current_node_index;
  }
}

RuntimeEvent MakeAddOrReplaceEvent(RuntimeEventAction action, RuntimeEventKind kind,
                                   const std::string &name, const Tensor &tensor,
                                   int64_t current_node_index, int64_t subgraph_node_index,
                                   const std::string &subgraph_attr_name) {
  RuntimeEvent ev;
  ev.action = action;
  ev.kind = kind;
  ev.timestamp_ns = NowNanos();
  ev.name = name;
  ev.node_index = ResolveNodeIndex(kind, current_node_index);
  ev.subgraph_node_index = subgraph_node_index;
  ev.subgraph_attr_name = subgraph_attr_name;
  const int64_t count = tensor.element_count();
  const int32_t capacity = static_cast<int32_t>(kRuntimeEventValueLimit);
  const int32_t truncated_count = static_cast<int32_t>(std::min<int64_t>(count, capacity));
  // Always populate the fixed-size value buffer with the first
  // min(element_count, kRuntimeEventValueLimit) entries; truncate the
  // remainder.
  if (static_cast<DataType>(tensor.data_type) == DataType::STRING) {
    for (int32_t i = 0; i < truncated_count && static_cast<size_t>(i) < tensor.string_data.size();
         ++i) {
      ev.string_values[i] = tensor.string_data[i];
    }
    ev.value_count = truncated_count;
  } else if (tensor.bytes() != nullptr && tensor.size_bytes() > 0) {
    ev.value_count =
        DecodeNumericValues(tensor.data_type, tensor.bytes(), count, capacity, ev.values);
  } else {
    ev.value_count = 0;
  }
  if (count > capacity) {
    // Signal the truncated payload with data_type = -1 and an empty
    // shape so the bounded event log clearly flags large tensors.
    ev.data_type = -1;
  } else {
    ev.data_type = tensor.data_type;
    ev.shape = tensor.shape;
  }
  return ev;
}

RuntimeEvent MakeRemoveEvent(RuntimeEventKind kind, const std::string &name,
                             int64_t subgraph_node_index, const std::string &subgraph_attr_name) {
  RuntimeEvent ev;
  ev.action = RuntimeEventAction::kRemove;
  ev.kind = kind;
  ev.timestamp_ns = NowNanos();
  ev.name = name;
  ev.data_type = static_cast<int32_t>(DataType::UNDEFINED);
  ev.value_count = 0;
  ev.subgraph_node_index = subgraph_node_index;
  ev.subgraph_attr_name = subgraph_attr_name;
  return ev;
}

} // namespace

std::string RuntimeEvent::summary() const {
  std::ostringstream oss;
  oss << "[" << RuntimeEventActionName(action) << "/" << RuntimeEventKindName(kind) << "] ";
  if (action == RuntimeEventAction::kRunNode) {
    if (!op_domain.empty()) {
      oss << op_domain << "::";
    }
    oss << op_type << "(";
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << inputs[i];
    }
    oss << ")";
  } else {
    oss << "'" << name << "'";
  }
  if (node_index >= 0) {
    oss << " node#" << node_index;
  }
  if (action == RuntimeEventAction::kRunNode) {
    oss << " took " << duration_ns << "ns";
  }
  oss << " mem=" << allocated_bytes << "B peak=" << peak_bytes << "B";
  return oss.str();
}

void RuntimeContext::StampAllocatorMemory(RuntimeEvent &ev) const noexcept {
  if (allocator_ == nullptr) {
    return;
  }
  ev.allocated_bytes = static_cast<int64_t>(allocator_->TotalAllocatedSize());
  ev.peak_bytes = static_cast<int64_t>(allocator_->PeakAllocatedSize());
}

void RuntimeContext::RecordRunNodeEvent(const NodeProto &node, const std::string &domain,
                                        const std::string &op_type, int64_t start_time_ns,
                                        int64_t duration_ns) noexcept {
  if (!events_enabled_) {
    return;
  }
  // Record the dispatch as a kRunNode RuntimeEvent so callers can profile
  // per-node execution from the event log alongside the tensor
  // add/replace/remove records.
  RuntimeEvent ev;
  ev.action = RuntimeEventAction::kRunNode;
  ev.kind = RuntimeEventKind::kUnknown;
  ev.timestamp_ns = start_time_ns;
  ev.data_type = static_cast<int32_t>(DataType::UNDEFINED);
  ev.value_count = 0;
  ev.node_index = current_node_index_;
  ev.op_domain = domain;
  ev.op_type = op_type;
  const size_t input_count = static_cast<size_t>(node.input_size());
  ev.inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    ev.inputs.push_back(node.input(i));
  }
  ev.duration_ns = duration_ns;
  ev.subgraph_node_index = current_subgraph_node_index_;
  ev.subgraph_attr_name = current_subgraph_attr_name_;
  StampAllocatorMemory(ev);
  events_.push_back(std::move(ev));
}

RuntimeContext::~RuntimeContext() = default;

void RuntimeContext::Set(const std::string &name, Tensor tensor, RuntimeEventKind kind) {
  EXT_ENFORCE(!Has(name), "RuntimeContext::Set: a tensor named '", name, "' already exists.");
  EnsureAllocatorBacked(tensor, allocator_, kind);
  if (events_enabled_) {
    RuntimeEvent ev =
        MakeAddOrReplaceEvent(RuntimeEventAction::kAdd, kind, name, tensor, current_node_index_,
                              current_subgraph_node_index_, current_subgraph_attr_name_);
    StampAllocatorMemory(ev);
    events_.push_back(std::move(ev));
  }
  tensors_[name] = std::move(tensor);
}

void RuntimeContext::Put(const std::string &name, Tensor tensor, RuntimeEventKind kind) {
  EnsureAllocatorBacked(tensor, allocator_, kind);
  if (events_enabled_) {
    const RuntimeEventAction action =
        Has(name) ? RuntimeEventAction::kReplace : RuntimeEventAction::kAdd;
    RuntimeEvent ev =
        MakeAddOrReplaceEvent(action, kind, name, tensor, current_node_index_,
                              current_subgraph_node_index_, current_subgraph_attr_name_);
    StampAllocatorMemory(ev);
    events_.push_back(std::move(ev));
  }
  tensors_[name] = std::move(tensor);
}

bool RuntimeContext::Remove(const std::string &name) {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    return false;
  }
  tensors_.erase(it);
  if (events_enabled_) {
    RuntimeEvent ev = MakeRemoveEvent(RuntimeEventKind::kUnknown, name,
                                      current_subgraph_node_index_, current_subgraph_attr_name_);
    StampAllocatorMemory(ev);
    events_.push_back(std::move(ev));
  }
  return true;
}

const Tensor &RuntimeContext::Get(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::out_of_range("RuntimeContext::Get: no tensor named '" + name + "'.");
  }
  return it->second;
}

Tensor &RuntimeContext::Get(const std::string &name) {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::out_of_range("RuntimeContext::Get: no tensor named '" + name + "'.");
  }
  return it->second;
}

const ExecutionPlan &RuntimeContext::GetExecutionPlan(const GraphProto &graph) {
  const void *key = static_cast<const void *>(&graph);
  auto it = execution_plans_.find(key);
  if (it == execution_plans_.end()) {
    it = execution_plans_.emplace(key, ExecutionPlan(graph)).first;
  }
  return it->second;
}

const ExecutionPlan &RuntimeContext::GetExecutionPlan(const FunctionProto &func) {
  const void *key = static_cast<const void *>(&func);
  auto it = execution_plans_.find(key);
  if (it == execution_plans_.end()) {
    it = execution_plans_.emplace(key, ExecutionPlan(func)).first;
  }
  return it->second;
}

void RuntimeContext::ClearExecutionPlans() noexcept { execution_plans_.clear(); }

RuntimeContext RuntimeContext::MakeSubgraphContext(const std::string &attr_name) const {
  RuntimeContext child(kernel_ctx_, RuntimeContextOptions{
                                        .allocator = nullptr,
                                        .events_enabled = events_enabled_,
                                        .verbose = verbose_,
                                        .release_intermediates = release_intermediates_,
                                        .device = device_,
                                    });
  // Subgraph contexts do not inherit the parent allocator. Body kernels use
  // inline tensor storage, and the parent's EnsureAllocatorBacked (called in
  // Put/Set) migrates final outputs to the parent allocator when results are
  // propagated back. This avoids double-free: if the child inherited the
  // allocator, tensors produced by the body would be freed when the child
  // context is destroyed, leaving any copies held by the caller with stale
  // allocation pointers.
  child.functions() = functions_;
  child.tensors() = tensors_;
  child.sequences() = sequences_;
  child.set_cpu_executor(cpu_executor_);
  child.set_current_subgraph(current_node_index_, attr_name);
  return child;
}

RuntimeContext RuntimeContext::MakeFunctionContext() const {
  RuntimeContext child(kernel_ctx_, RuntimeContextOptions{
                                        .allocator = allocator_,
                                        .io_allocator = io_allocator_,
                                        .events_enabled = false,
                                        .verbose = verbose_,
                                        .release_intermediates = release_intermediates_,
                                        .device = device_,
                                    });
  child.functions() = functions_;
  child.set_cpu_executor(cpu_executor_);
  return child;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
