// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/dispatch_table.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace shapes {

namespace {

// Returns the ``"<domain>:<op_type>"`` dispatch key, normalising an empty
// domain to :cpp:var:`kOnnxDomain`.
std::string DispatchKey(const std::string &domain, const std::string &op_type) {
  return (domain.empty() ? std::string(kOnnxDomain) : domain) + ":" + op_type;
}

// Returns the device-qualified dispatch key. The identifier of a peak-memory
// function is ``(domain, op_type, device)``; the default host devices
// (:cpp:enumerator:`Device::kCPU` and :cpp:enumerator:`Device::kUndefined`)
// keep the plain ``"<domain>:<op_type>"`` form so existing keys are
// unchanged, while any other device appends ``":<device>"`` (the integer
// value of the :cpp:enum:`Device` enumerator) to disambiguate.
std::string DispatchKey(const std::string &domain, const std::string &op_type, Device device) {
  std::string key = DispatchKey(domain, op_type);
  if (device != Device::kUndefined && device != Device::kCPU) {
    key += ":" + std::to_string(static_cast<int32_t>(device));
  }
  return key;
}

// Returns the mutable dispatch table singleton. Only
// :cpp:func:`RegisterComputeShapeFn` writes to it; :cpp:func:`DispatchTable`
// exposes a read-only view for lookups.
std::unordered_map<std::string, ComputeShapeFn> &MutableDispatchTable() {
  static std::unordered_map<std::string, ComputeShapeFn> table;
  return table;
}

// Returns the mutable peak-memory dispatch table singleton. Only
// :cpp:func:`RegisterComputePeakMemoryFn` writes to it;
// :cpp:func:`PeakMemoryDispatchTable` exposes a read-only view for lookups.
std::unordered_map<std::string, ComputePeakMemoryFn> &MutablePeakMemoryDispatchTable() {
  static std::unordered_map<std::string, ComputePeakMemoryFn> table;
  return table;
}

} // namespace

const std::unordered_map<std::string, ComputeShapeFn> &DispatchTable() {
  return MutableDispatchTable();
}

void RegisterComputeShapeFn(const std::string &domain, const std::string &op_type,
                            ComputeShapeFn fn) {
  MutableDispatchTable()[DispatchKey(domain, op_type)] = std::move(fn);
}

const std::unordered_map<std::string, ComputePeakMemoryFn> &PeakMemoryDispatchTable() {
  return MutablePeakMemoryDispatchTable();
}

void RegisterComputePeakMemoryFn(const std::string &domain, const std::string &op_type,
                                 Device device, ComputePeakMemoryFn fn) {
  MutablePeakMemoryDispatchTable()[DispatchKey(domain, op_type, device)] = std::move(fn);
}

int64_t ComputePeakMemory(const std::string &domain, const std::string &op_type, Device device,
                          const std::vector<SymShape> &input_shapes) {
  const auto &table = MutablePeakMemoryDispatchTable();
  auto it = table.find(DispatchKey(domain, op_type, device));
  if (it == table.end()) {
    return 0;
  }
  return it->second(device, input_shapes);
}

} // namespace shapes
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
