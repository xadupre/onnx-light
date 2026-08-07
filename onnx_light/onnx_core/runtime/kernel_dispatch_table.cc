// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernel_dispatch_table.h"

#include <string>
#include <unordered_map>
#include <utility>

#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

namespace {

// Returns the ``"<domain>:<op_type>"`` dispatch key, normalising an empty
// domain to :cpp:var:`kDefaultOnnxDomain`.
std::string DispatchKey(const std::string &domain, const std::string &op_type) {
  return (domain.empty() ? std::string(kDefaultOnnxDomain) : domain) + ":" + op_type;
}

// Returns the device-qualified dispatch key. The identifier of a kernel is
// ``(domain, op_type, device)``; :cpp:func:`symbolic::DeviceKeySuffix` keeps
// the default host devices (:cpp:enumerator:`symbolic::Device::kCPU` and
// :cpp:enumerator:`symbolic::Device::kUndefined`) in the plain
// ``"<domain>:<op_type>"`` form so existing keys are unchanged, and appends
// ``":<device>"`` for any other device to disambiguate.
std::string DispatchKey(const std::string &domain, const std::string &op_type,
                        symbolic::Device device) {
  return DispatchKey(domain, op_type) + symbolic::DeviceKeySuffix(device);
}

// Returns the mutable dispatch table singleton. Only
// :cpp:func:`RegisterKernelFn` writes to it; :cpp:func:`KernelDispatchTable`
// exposes a read-only view for lookups.
std::unordered_map<std::string, NodeKernelFn> &MutableKernelDispatchTable() {
  static std::unordered_map<std::string, NodeKernelFn> table;
  return table;
}

// Returns the mutable ``SequenceMap`` output-packing callback singleton.
SequenceMapPackFn &MutableSequenceMapPackFn() {
  static SequenceMapPackFn fn;
  return fn;
}

// Returns the process-wide (global) custom-kernel registry singleton. Only the
// Register/Unregister/Clear global helpers mutate it; :cpp:func:`GlobalCustomKernels`
// exposes a read-only view consulted during kernel resolution.
CustomKernelMap &MutableGlobalCustomKernels() {
  static CustomKernelMap kernels;
  return kernels;
}

// Returns the canonical ``"<domain>:<op_type>"`` custom-kernel key, normalising
// an empty domain to :cpp:var:`kDefaultOnnxDomain` (matching the key form used
// by :cpp:func:`RuntimeContext::RegisterCustomKernel` and the resolution logic
// in ``run_nodes.cc``).
std::string CustomKernelKey(const std::string &domain, const std::string &op_type) {
  return (domain.empty() ? kDefaultOnnxDomain : domain) + ":" + op_type;
}

} // namespace

const std::unordered_map<std::string, NodeKernelFn> &KernelDispatchTable() {
  return MutableKernelDispatchTable();
}

void RegisterKernelFn(const std::string &domain, const std::string &op_type,
                      symbolic::Device device, NodeKernelFn fn) {
  MutableKernelDispatchTable()[DispatchKey(domain, op_type, device)] = std::move(fn);
}

const CustomKernelMap &GlobalCustomKernels() { return MutableGlobalCustomKernels(); }

void RegisterGlobalCustomKernel(const std::string &domain, const std::string &op_type,
                                CustomKernelFn fn) {
  MutableGlobalCustomKernels()[CustomKernelKey(domain, op_type)] = std::move(fn);
}

bool UnregisterGlobalCustomKernel(const std::string &domain, const std::string &op_type) {
  return MutableGlobalCustomKernels().erase(CustomKernelKey(domain, op_type)) > 0;
}

void ClearGlobalCustomKernels() { MutableGlobalCustomKernels().clear(); }

const SequenceMapPackFn &GetSequenceMapPackFn() { return MutableSequenceMapPackFn(); }

void RegisterSequenceMapPackFn(SequenceMapPackFn fn) { MutableSequenceMapPackFn() = std::move(fn); }

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
