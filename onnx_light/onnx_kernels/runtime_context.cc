// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/runtime_context.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

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
                            int32_t capacity, std::array<double, kTensorEventValueLimit> &out) {
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

TensorEvent MakeAddOrReplaceEvent(TensorEventAction action, TensorEventKind kind,
                                  const std::string &name, const Tensor &tensor) {
  TensorEvent ev;
  ev.action = action;
  ev.kind = kind;
  ev.timestamp_ns = NowNanos();
  ev.name = name;
  const int64_t count = tensor.element_count();
  const int32_t capacity = static_cast<int32_t>(kTensorEventValueLimit);
  const int32_t truncated_count = static_cast<int32_t>(std::min<int64_t>(count, capacity));
  // Always populate the fixed-size value buffer with the first
  // min(element_count, kTensorEventValueLimit) entries; truncate the
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

TensorEvent MakeRemoveEvent(TensorEventKind kind, const std::string &name) {
  TensorEvent ev;
  ev.action = TensorEventAction::kRemove;
  ev.kind = kind;
  ev.timestamp_ns = NowNanos();
  ev.name = name;
  ev.data_type = static_cast<int32_t>(DataType::UNDEFINED);
  ev.value_count = 0;
  return ev;
}

} // namespace

const char *TensorEventActionName(TensorEventAction action) noexcept {
  switch (action) {
  case TensorEventAction::kAdd:
    return "add";
  case TensorEventAction::kReplace:
    return "replace";
  case TensorEventAction::kRemove:
    return "remove";
  case TensorEventAction::kRunNode:
    return "run_node";
  }
  return "unknown";
}

const char *TensorEventKindName(TensorEventKind kind) noexcept {
  switch (kind) {
  case TensorEventKind::kUnknown:
    return "unknown";
  case TensorEventKind::kInitializer:
    return "initializer";
  case TensorEventKind::kInput:
    return "input";
  case TensorEventKind::kIntermediate:
    return "intermediate";
  case TensorEventKind::kOutput:
    return "output";
  }
  return "unknown";
}

void RuntimeContext::Set(const std::string &name, Tensor tensor, TensorEventKind kind) {
  EXT_ENFORCE(!Has(name), "RuntimeContext::Set: a tensor named '", name, "' already exists.");
  if (events_enabled_) {
    events_.push_back(MakeAddOrReplaceEvent(TensorEventAction::kAdd, kind, name, tensor));
  }
  tensors_[name] = std::move(tensor);
}

void RuntimeContext::Put(const std::string &name, Tensor tensor, TensorEventKind kind) {
  if (events_enabled_) {
    const TensorEventAction action =
        Has(name) ? TensorEventAction::kReplace : TensorEventAction::kAdd;
    events_.push_back(MakeAddOrReplaceEvent(action, kind, name, tensor));
  }
  tensors_[name] = std::move(tensor);
}

bool RuntimeContext::Remove(const std::string &name) {
  const bool erased = tensors_.erase(name) > 0;
  if (erased && events_enabled_) {
    events_.push_back(MakeRemoveEvent(TensorEventKind::kUnknown, name));
  }
  return erased;
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

void RuntimeContext::AppendRunNodeEvent(const std::string &op_domain, const std::string &op_type,
                                        std::vector<std::string> inputs, int64_t start_time_ns,
                                        int64_t duration_ns) {
  TensorEvent ev;
  ev.action = TensorEventAction::kRunNode;
  ev.kind = TensorEventKind::kUnknown;
  ev.timestamp_ns = start_time_ns;
  ev.data_type = static_cast<int32_t>(DataType::UNDEFINED);
  ev.value_count = 0;
  ev.op_domain = op_domain;
  ev.op_type = op_type;
  ev.inputs = std::move(inputs);
  ev.duration_ns = duration_ns;
  events_.push_back(std::move(ev));
}

namespace {

// Recursively walks ``graph``'s nodes (including their subgraph
// attributes) and appends to ``out`` every input name that is not
// produced inside the subgraph scope (``local``) and not produced
// by an outer scope (``outer_produced``). ``seen`` dedupes against
// names already appended by the outermost caller.
void CollectGraphExternalInputs(const GraphProto &graph, std::vector<std::string> &out,
                                std::unordered_set<std::string> &seen,
                                const std::unordered_set<std::string> &outer_produced) {
  std::unordered_set<std::string> local;
  for (size_t i = 0; i < graph.input().size(); ++i) {
    local.insert(graph.input()[i].name().as_string());
  }
  for (size_t i = 0; i < graph.initializer().size(); ++i) {
    local.insert(graph.initializer()[i].name().as_string());
  }
  for (size_t i = 0; i < graph.node().size(); ++i) {
    const NodeProto &nd = graph.node()[i];
    for (size_t j = 0; j < nd.output().size(); ++j) {
      const std::string name = nd.output()[j].as_string();
      if (!name.empty()) {
        local.insert(name);
      }
    }
  }

  // The combined "in-scope" set for any nested subgraph is the union
  // of the outer scope and this subgraph's local names.
  std::unordered_set<std::string> inner_outer = outer_produced;
  inner_outer.insert(local.begin(), local.end());

  for (size_t i = 0; i < graph.node().size(); ++i) {
    const NodeProto &nd = graph.node()[i];
    for (size_t j = 0; j < nd.input().size(); ++j) {
      const std::string name = nd.input()[j].as_string();
      if (name.empty() || local.count(name) || outer_produced.count(name)) {
        continue;
      }
      if (seen.insert(name).second) {
        out.push_back(name);
      }
    }
    for (size_t a = 0; a < nd.attribute().size(); ++a) {
      const AttributeProto &attr = nd.attribute()[a];
      if (attr.type() == AttributeProto::AttributeType::GRAPH && attr.has_g()) {
        CollectGraphExternalInputs(attr.g(), out, seen, inner_outer);
      } else if (attr.type() == AttributeProto::AttributeType::GRAPHS) {
        for (size_t k = 0; k < attr.graphs().size(); ++k) {
          CollectGraphExternalInputs(attr.graphs()[k], out, seen, inner_outer);
        }
      }
    }
  }
}

template <class NodeRange>
std::vector<std::string> CollectExternalInputsImpl(const NodeRange &nodes) {
  std::unordered_set<std::string> produced;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const NodeProto &nd = nodes[i];
    for (size_t j = 0; j < nd.output().size(); ++j) {
      const std::string name = nd.output()[j].as_string();
      if (!name.empty()) {
        produced.insert(name);
      }
    }
  }
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const NodeProto &nd = nodes[i];
    for (size_t j = 0; j < nd.input().size(); ++j) {
      const std::string name = nd.input()[j].as_string();
      if (name.empty() || produced.count(name)) {
        continue;
      }
      if (seen.insert(name).second) {
        out.push_back(name);
      }
    }
    for (size_t a = 0; a < nd.attribute().size(); ++a) {
      const AttributeProto &attr = nd.attribute()[a];
      if (attr.type() == AttributeProto::AttributeType::GRAPH && attr.has_g()) {
        CollectGraphExternalInputs(attr.g(), out, seen, produced);
      } else if (attr.type() == AttributeProto::AttributeType::GRAPHS) {
        for (size_t k = 0; k < attr.graphs().size(); ++k) {
          CollectGraphExternalInputs(attr.graphs()[k], out, seen, produced);
        }
      }
    }
  }
  return out;
}

} // namespace

std::vector<std::string>
RuntimeContext::CollectExternalInputs(const utils::RepeatedProtoField<NodeProto> &nodes) {
  return CollectExternalInputsImpl(nodes);
}

std::vector<std::string>
RuntimeContext::CollectExternalInputs(const std::vector<NodeProto> &nodes) {
  return CollectExternalInputsImpl(nodes);
}

std::vector<std::string> RuntimeContext::CollectNodeInputs(const NodeProto &node) {
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  for (size_t i = 0; i < node.input().size(); ++i) {
    const std::string name = node.input()[i].as_string();
    if (name.empty()) {
      continue;
    }
    if (seen.insert(name).second) {
      out.push_back(name);
    }
  }
  // Recursively collect every captured input of subgraph attributes
  // (``GRAPH`` / ``GRAPHS``). Outer-produced names are intentionally
  // empty: from the perspective of a single ``node``, only the names
  // referenced by its subgraphs are part of its data dependencies.
  std::unordered_set<std::string> empty_outer;
  for (size_t a = 0; a < node.attribute().size(); ++a) {
    const AttributeProto &attr = node.attribute()[a];
    if (attr.type() == AttributeProto::AttributeType::GRAPH && attr.has_g()) {
      CollectGraphExternalInputs(attr.g(), out, seen, empty_outer);
    } else if (attr.type() == AttributeProto::AttributeType::GRAPHS) {
      for (size_t k = 0; k < attr.graphs().size(); ++k) {
        CollectGraphExternalInputs(attr.graphs()[k], out, seen, empty_outer);
      }
    }
  }
  return out;
}

namespace {

template <class NodeRange>
std::vector<std::vector<std::string>>
ComputeReleasableInputsImpl(const NodeRange &nodes, const std::unordered_set<std::string> &keep) {
  const size_t n = nodes.size();
  std::vector<std::vector<std::string>> per_node_inputs;
  per_node_inputs.reserve(n);
  std::unordered_map<std::string, size_t> last_use;
  for (size_t i = 0; i < n; ++i) {
    std::vector<std::string> inputs = RuntimeContext::CollectNodeInputs(nodes[i]);
    for (const auto &name : inputs) {
      last_use[name] = i;
    }
    per_node_inputs.push_back(std::move(inputs));
  }
  std::vector<std::vector<std::string>> out(n);
  for (size_t i = 0; i < n; ++i) {
    for (const auto &name : per_node_inputs[i]) {
      if (keep.count(name)) {
        continue;
      }
      auto it = last_use.find(name);
      if (it != last_use.end() && it->second == i) {
        out[i].push_back(name);
      }
    }
  }
  return out;
}

} // namespace

std::vector<std::vector<std::string>>
RuntimeContext::ComputeReleasableInputs(const utils::RepeatedProtoField<NodeProto> &nodes,
                                        const std::unordered_set<std::string> &keep) {
  return ComputeReleasableInputsImpl(nodes, keep);
}

std::vector<std::vector<std::string>>
RuntimeContext::ComputeReleasableInputs(const std::vector<NodeProto> &nodes,
                                        const std::unordered_set<std::string> &keep) {
  return ComputeReleasableInputsImpl(nodes, keep);
}

ExecutionPlan::ExecutionPlan(const utils::RepeatedProtoField<NodeProto> &nodes,
                             std::unordered_set<std::string> keep)
    : keep_(std::move(keep)), releasable_(RuntimeContext::ComputeReleasableInputs(nodes, keep_)) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    node_index_.emplace(&nodes[i], i);
  }
}

ExecutionPlan::ExecutionPlan(const GraphProto &graph) {
  for (size_t i = 0; i < graph.input().size(); ++i) {
    const std::string name = graph.input()[i].name().as_string();
    if (!name.empty()) {
      keep_.insert(name);
    }
  }
  for (size_t i = 0; i < graph.initializer().size(); ++i) {
    const std::string name = graph.initializer()[i].name().as_string();
    if (!name.empty()) {
      keep_.insert(name);
    }
  }
  for (size_t i = 0; i < graph.output().size(); ++i) {
    const std::string name = graph.output()[i].name().as_string();
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
    const std::string name = func.input(i).as_string();
    if (!name.empty()) {
      keep_.insert(name);
    }
  }
  for (size_t i = 0; i < func.output_size(); ++i) {
    const std::string name = func.output(i).as_string();
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

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
