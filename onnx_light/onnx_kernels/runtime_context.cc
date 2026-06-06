// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/runtime_context.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

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
