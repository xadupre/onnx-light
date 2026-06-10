// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/runtime_context.h"

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

// Decodes ``element_count`` values of dtype ``dtype`` starting at ``ptr``
// into a ``double`` vector. Returns ``false`` for dtypes that are not
// representable as ``double`` here (the caller leaves ``out`` empty).
bool DecodeNumericValues(int32_t dtype, const uint8_t *ptr, int64_t element_count,
                         std::vector<double> &out) {
  const size_t n = static_cast<size_t>(element_count);
  out.clear();
  out.reserve(n);
  switch (static_cast<DataType>(dtype)) {
  case DataType::FLOAT: {
    const float *p = reinterpret_cast<const float *>(ptr);
    for (size_t i = 0; i < n; ++i)
      out.push_back(static_cast<double>(p[i]));
    return true;
  }
  case DataType::DOUBLE: {
    const double *p = reinterpret_cast<const double *>(ptr);
    for (size_t i = 0; i < n; ++i)
      out.push_back(p[i]);
    return true;
  }
  case DataType::INT8: {
    const int8_t *p = reinterpret_cast<const int8_t *>(ptr);
    for (size_t i = 0; i < n; ++i)
      out.push_back(static_cast<double>(p[i]));
    return true;
  }
  case DataType::INT16: {
    const int16_t *p = reinterpret_cast<const int16_t *>(ptr);
    for (size_t i = 0; i < n; ++i)
      out.push_back(static_cast<double>(p[i]));
    return true;
  }
  case DataType::INT32: {
    const int32_t *p = reinterpret_cast<const int32_t *>(ptr);
    for (size_t i = 0; i < n; ++i)
      out.push_back(static_cast<double>(p[i]));
    return true;
  }
  case DataType::INT64: {
    const int64_t *p = reinterpret_cast<const int64_t *>(ptr);
    for (size_t i = 0; i < n; ++i)
      out.push_back(static_cast<double>(p[i]));
    return true;
  }
  case DataType::UINT8:
  case DataType::BOOL: {
    for (size_t i = 0; i < n; ++i)
      out.push_back(static_cast<double>(ptr[i]));
    return true;
  }
  case DataType::UINT16: {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(ptr);
    for (size_t i = 0; i < n; ++i)
      out.push_back(static_cast<double>(p[i]));
    return true;
  }
  case DataType::UINT32: {
    const uint32_t *p = reinterpret_cast<const uint32_t *>(ptr);
    for (size_t i = 0; i < n; ++i)
      out.push_back(static_cast<double>(p[i]));
    return true;
  }
  case DataType::UINT64: {
    const uint64_t *p = reinterpret_cast<const uint64_t *>(ptr);
    for (size_t i = 0; i < n; ++i)
      out.push_back(static_cast<double>(p[i]));
    return true;
  }
  default:
    out.clear();
    return false;
  }
}

TensorEvent MakeAddOrReplaceEvent(TensorEventAction action, const std::string &name,
                                  const Tensor &tensor) {
  TensorEvent ev;
  ev.action = action;
  ev.timestamp_ns = NowNanos();
  ev.name = name;
  const int64_t count = tensor.element_count();
  if (count > kTensorEventValueLimit) {
    // Large tensors: keep the log bounded by recording only the action,
    // name and timestamp. Signal the elided payload with data_type = -1
    // and leave shape / values / string_values empty.
    ev.data_type = -1;
    return ev;
  }
  ev.data_type = tensor.data_type;
  ev.shape = tensor.shape;
  if (static_cast<DataType>(tensor.data_type) == DataType::STRING) {
    ev.string_values = tensor.string_data;
  } else if (tensor.bytes() != nullptr && tensor.size_bytes() > 0) {
    DecodeNumericValues(tensor.data_type, tensor.bytes(), count, ev.values);
  }
  return ev;
}

TensorEvent MakeRemoveEvent(const std::string &name) {
  TensorEvent ev;
  ev.action = TensorEventAction::kRemove;
  ev.timestamp_ns = NowNanos();
  ev.name = name;
  ev.data_type = static_cast<int32_t>(DataType::UNDEFINED);
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
  }
  return "unknown";
}

void RuntimeContext::Set(const std::string &name, Tensor tensor) {
  EXT_ENFORCE(!Has(name), "RuntimeContext::Set: a tensor named '", name, "' already exists.");
  events_.push_back(MakeAddOrReplaceEvent(TensorEventAction::kAdd, name, tensor));
  tensors_[name] = std::move(tensor);
}

void RuntimeContext::Put(const std::string &name, Tensor tensor) {
  const TensorEventAction action =
      Has(name) ? TensorEventAction::kReplace : TensorEventAction::kAdd;
  events_.push_back(MakeAddOrReplaceEvent(action, name, tensor));
  tensors_[name] = std::move(tensor);
}

bool RuntimeContext::Remove(const std::string &name) {
  const bool erased = tensors_.erase(name) > 0;
  if (erased) {
    events_.push_back(MakeRemoveEvent(name));
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

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
