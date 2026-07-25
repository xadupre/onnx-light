// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernel_dispatch_table.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

namespace {

// Returns the ``"<domain>:<op_type>"`` dispatch key, normalising an empty
// domain to :cpp:var:`kDefaultOnnxDomain`.
std::string DispatchKey(const std::string &domain, const std::string &op_type) {
  return (domain.empty() ? std::string(kDefaultOnnxDomain) : domain) + ":" + op_type;
}

// Returns the device-qualified dispatch key. The identifier of a kernel is
// ``(domain, op_type, device)``; the default host devices
// (:cpp:enumerator:`symbolic::Device::kCPU` and
// :cpp:enumerator:`symbolic::Device::kUndefined`) keep the plain
// ``"<domain>:<op_type>"`` form so existing keys are unchanged, while any
// other device appends ``":<device>"`` (the integer value of the
// :cpp:enum:`symbolic::Device` enumerator) to disambiguate.
std::string DispatchKey(const std::string &domain, const std::string &op_type,
                        symbolic::Device device) {
  std::string key = DispatchKey(domain, op_type);
  if (device != symbolic::Device::kUndefined && device != symbolic::Device::kCPU) {
    key += ":" + std::to_string(static_cast<int32_t>(device));
  }
  return key;
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

} // namespace

const std::unordered_map<std::string, NodeKernelFn> &KernelDispatchTable() {
  return MutableKernelDispatchTable();
}

void RegisterKernelFn(const std::string &domain, const std::string &op_type,
                      symbolic::Device device, NodeKernelFn fn) {
  MutableKernelDispatchTable()[DispatchKey(domain, op_type, device)] = std::move(fn);
}

const SequenceMapPackFn &GetSequenceMapPackFn() { return MutableSequenceMapPackFn(); }

void RegisterSequenceMapPackFn(SequenceMapPackFn fn) { MutableSequenceMapPackFn() = std::move(fn); }

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
