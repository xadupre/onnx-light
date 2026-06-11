// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/runtime_context.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

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
  events_.push_back(MakeAddOrReplaceEvent(TensorEventAction::kAdd, kind, name, tensor));
  tensors_[name] = std::move(tensor);
}

void RuntimeContext::Put(const std::string &name, Tensor tensor, TensorEventKind kind) {
  const TensorEventAction action =
      Has(name) ? TensorEventAction::kReplace : TensorEventAction::kAdd;
  events_.push_back(MakeAddOrReplaceEvent(action, kind, name, tensor));
  tensors_[name] = std::move(tensor);
}

bool RuntimeContext::Remove(const std::string &name) {
  const bool erased = tensors_.erase(name) > 0;
  if (erased) {
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

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
