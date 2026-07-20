// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernel_dispatch_table.h"

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

void RegisterKernelFn(const std::string &domain, const std::string &op_type, NodeKernelFn fn) {
  MutableKernelDispatchTable()[DispatchKey(domain, op_type)] = std::move(fn);
}

const SequenceMapPackFn &GetSequenceMapPackFn() { return MutableSequenceMapPackFn(); }

void RegisterSequenceMapPackFn(SequenceMapPackFn fn) { MutableSequenceMapPackFn() = std::move(fn); }

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
