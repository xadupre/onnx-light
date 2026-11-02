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
  return NormaliseDomain(domain) + ":" + op_type;
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

std::unordered_map<std::string, ComputePeakMemoryNodeFn> &NodePeakMemoryDispatchTable() {
  static std::unordered_map<std::string, ComputePeakMemoryNodeFn> table;
  return table;
}

void ValidatePeakMemoryEstimate(const PeakMemoryEstimate &estimate, const std::string &domain,
                                const std::string &op_type) {
  const std::string identifier = domain.empty() ? op_type : domain + ":" + op_type;
  for (const symbolic::SymDim *dim :
       {&estimate.worst_case, estimate.average_case ? &*estimate.average_case : nullptr}) {
    if (dim == nullptr)
      continue;
    EXT_ENFORCE_INVALID(!dim->IsInt() || dim->AsInt() >= 0, "Negative peak-memory estimate for ",
                        identifier, ".");
    EXT_ENFORCE_INVALID(!dim->IsExpr() || !dim->AsExpr().empty(),
                        "Empty peak-memory expression for ", identifier, ".");
  }
  EXT_ENFORCE_INVALID(!estimate.average_case || !estimate.worst_case.IsInt() ||
                          !estimate.average_case->IsInt() ||
                          estimate.average_case->AsInt() <= estimate.worst_case.AsInt(),
                      "Average peak-memory estimate exceeds worst case for ", identifier, ".");
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

bool HasShapeInference(const ShapesContext &shapes, const NodeProto &node) {
  return shapes.GetLocalFunction(ShapesContext::LocalFunctionKey(node.domain(), node.op_type(),
                                                                 node.overload())) != nullptr ||
         shapes.GetCustomShapeInferenceFunction(node.domain(), node.op_type()) != nullptr ||
         DispatchTable().count(DispatchKey(node.domain(), node.op_type())) != 0;
}

const std::unordered_map<std::string, ComputePeakMemoryFn> &PeakMemoryDispatchTable() {
  return MutablePeakMemoryDispatchTable();
}

void RegisterComputePeakMemoryFn(const std::string &domain, const std::string &op_type,
                                 Device device, ComputePeakMemoryFn fn) {
  const auto key = DispatchKey(domain, op_type, device);
  NodePeakMemoryDispatchTable().erase(key);
  MutablePeakMemoryDispatchTable()[key] = std::move(fn);
}

void RegisterComputePeakMemoryFn(const std::string &domain, const std::string &op_type,
                                 Device device, ComputePeakMemoryNodeFn fn) {
  const auto key = DispatchKey(domain, op_type, device);
  NodePeakMemoryDispatchTable()[key] = fn;
  MutablePeakMemoryDispatchTable()[key] =
      [domain, op_type, fn = std::move(fn)](Device target, const std::vector<SymShape> &inputs) {
        NodeProto node;
        node.set_domain(domain);
        node.set_op_type(op_type);
        return fn(target, node, inputs);
      };
}

PeakMemoryEstimate UnknownPeakMemory(const std::string &domain, const std::string &op_type) {
  const std::string key = DispatchKey(domain, op_type, Device::kUndefined);
  std::string symbol = "peak_memory_max_";
  constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : key) {
    symbol.push_back(hex[c >> 4]);
    symbol.push_back(hex[c & 15]);
  }
  return {symbolic::SymDim(std::move(symbol)), std::nullopt};
}

PeakMemoryEstimate ComputePeakMemory(const std::string &domain, const std::string &op_type,
                                     Device device, const std::vector<SymShape> &input_shapes) {
  const auto &table = MutablePeakMemoryDispatchTable();
  auto it = table.find(DispatchKey(domain, op_type, device));
  if (it == table.end()) {
    return UnknownPeakMemory(domain, op_type);
  }
  const PeakMemoryEstimate estimate = it->second(device, input_shapes);
  ValidatePeakMemoryEstimate(estimate, domain, op_type);
  return estimate;
}

PeakMemoryEstimate ComputePeakMemory(Device device, const NodeProto &node,
                                     const std::vector<SymShape> &input_shapes) {
  const auto &table = NodePeakMemoryDispatchTable();
  auto it = table.find(DispatchKey(node.domain(), node.op_type(), device));
  if (it == table.end())
    return ComputePeakMemory(node.domain(), node.op_type(), device, input_shapes);
  const PeakMemoryEstimate estimate = it->second(device, node, input_shapes);
  ValidatePeakMemoryEstimate(estimate, node.domain(), node.op_type());
  return estimate;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::shapes
