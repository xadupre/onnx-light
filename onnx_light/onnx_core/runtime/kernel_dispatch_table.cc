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

// Returns the mutable ``(domain, op_type, device) -> unique name`` table
// singleton, populated alongside the dispatch table by
// :cpp:func:`RegisterKernelFn`.
std::unordered_map<std::string, std::string> &MutableKernelNameTable() {
  static std::unordered_map<std::string, std::string> table;
  return table;
}

} // namespace

const std::unordered_map<std::string, NodeKernelFn> &KernelDispatchTable() {
  return MutableKernelDispatchTable();
}

void RegisterKernelFn(const std::string &domain, const std::string &op_type,
                      symbolic::Device device, NodeKernelFn fn, const char *name) {
  const std::string key = DispatchKey(domain, op_type, device);
  MutableKernelDispatchTable()[key] = std::move(fn);
  MutableKernelNameTable()[key] = name;
}

const std::string &KernelDispatchName(const std::string &domain, const std::string &op_type,
                                      symbolic::Device device) {
  static const std::string kEmpty;
  const auto &table = MutableKernelNameTable();
  auto it = table.find(DispatchKey(domain, op_type, device));
  return it == table.end() ? kEmpty : it->second;
}

const SequenceMapPackFn &GetSequenceMapPackFn() { return MutableSequenceMapPackFn(); }

void RegisterSequenceMapPackFn(SequenceMapPackFn fn) { MutableSequenceMapPackFn() = std::move(fn); }

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
