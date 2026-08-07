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

} // namespace

const std::unordered_map<std::string, NodeKernelFn> &KernelDispatchTable() {
  return MutableKernelDispatchTable();
}

std::string KernelUniqueName(const std::string &library, symbolic::Device device,
                             const std::string &domain, const std::string &op_type) {
  const std::string normalised_domain = domain.empty() ? std::string(kDefaultOnnxDomain) : domain;
  return library + ":" + symbolic::DeviceName(device) + ":" + normalised_domain + ":" + op_type;
}

void RegisterKernelFn(const std::string &domain, const std::string &op_type,
                      symbolic::Device device, NodeKernelFn fn, const std::string &library) {
  // Wrap the factory so every kernel it produces carries its unique name
  // (library + device + domain + op_type). The name is assigned right after
  // construction so a resolved kernel always reports which implementation
  // (device + library) the runtime selected for it.
  std::string name = KernelUniqueName(library, device, domain, op_type);
  MutableKernelDispatchTable()[DispatchKey(domain, op_type, device)] =
      [fn = std::move(fn), name = std::move(name)](
          const NodeProto &node, RuntimeContext &rt) -> std::unique_ptr<KernelBase> {
    std::unique_ptr<KernelBase> kernel = fn(node, rt);
    if (kernel) {
      kernel->set_name(name);
    }
    return kernel;
  };
}

const SequenceMapPackFn &GetSequenceMapPackFn() { return MutableSequenceMapPackFn(); }

void RegisterSequenceMapPackFn(SequenceMapPackFn fn) { MutableSequenceMapPackFn() = std::move(fn); }

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
