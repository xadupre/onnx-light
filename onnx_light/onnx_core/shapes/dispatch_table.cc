// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/shapes/dispatch_table.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx_core/graph/graph_manipulations.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::core::shapes {

namespace {

// Returns the ``"<domain>:<op_type>"`` dispatch key, normalising an empty
// domain to :cpp:var:`kOnnxDomain`.
std::string DispatchKey(const std::string &domain, const std::string &op_type) {
  return (domain.empty() ? std::string(kOnnxDomain) : domain) + ":" + op_type;
}

// Returns the device-qualified dispatch key. The identifier of a peak-memory
// function is ``(domain, op_type, device)``;
// :cpp:func:`symbolic::DeviceKeySuffix` keeps the default host devices
// (:cpp:enumerator:`Device::kCPU` and :cpp:enumerator:`Device::kUndefined`)
// in the plain ``"<domain>:<op_type>"`` form so existing keys are unchanged,
// and appends ``":<device>"`` for any other device to disambiguate.
std::string DispatchKey(const std::string &domain, const std::string &op_type, Device device) {
  return DispatchKey(domain, op_type) + symbolic::DeviceKeySuffix(device);
}

// Returns the mutable dispatch table singleton. Only
// :cpp:func:`RegisterComputeShapeFn` writes to it; :cpp:func:`DispatchTable`
// exposes a read-only view for lookups.
std::unordered_map<std::string, ComputeShapeFn> &MutableDispatchTable() {
  static std::unordered_map<std::string, ComputeShapeFn> table;
  return table;
}

std::unordered_set<std::string> &StructuredShapeInputs() {
  static std::unordered_set<std::string> keys;
  return keys;
}

std::unordered_set<std::string> &TypeOnlyTensorShapeInputs() {
  static std::unordered_set<std::string> keys;
  return keys;
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
                            ComputeShapeFn fn, bool accepts_structured_inputs,
                            bool accepts_type_only_tensor_inputs) {
  const auto key = DispatchKey(domain, op_type);
  MutableDispatchTable()[key] = std::move(fn);
  if (accepts_structured_inputs)
    StructuredShapeInputs().insert(key);
  else
    StructuredShapeInputs().erase(key);
  if (accepts_type_only_tensor_inputs)
    TypeOnlyTensorShapeInputs().insert(key);
  else
    TypeOnlyTensorShapeInputs().erase(key);
}

bool AcceptsStructuredShapeInputs(const std::string &domain, const std::string &op_type) {
  return StructuredShapeInputs().count(DispatchKey(domain, op_type)) != 0;
}

bool AcceptsTypeOnlyTensorShapeInputs(const std::string &domain, const std::string &op_type) {
  return TypeOnlyTensorShapeInputs().count(DispatchKey(domain, op_type)) != 0;
}

bool ShapeInputsAvailable(const ShapesContext &shapes, const NodeProto &node) {
  for (const std::string &input : core::graph::CollectNodeInputs(node)) {
    if (input.empty() || shapes.Has(input) || shapes.HasSequence(input)) {
      continue;
    }
    if (!shapes.HasType(input) ||
        (shapes.GetType(input).has_tensor_type() &&
         !AcceptsTypeOnlyTensorShapeInputs(node.domain(), node.op_type()))) {
      return false;
    }
  }
  return true;
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
  const int64_t bytes = it->second(device, input_shapes);
  EXT_ENFORCE_INVALID(bytes >= 0, "Negative peak-memory estimate for ", domain, ":", op_type, ".");
  return bytes;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::shapes
