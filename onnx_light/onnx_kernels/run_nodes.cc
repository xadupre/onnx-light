// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/run_nodes.h"

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "onnx_kernels/kernels/math/include_math_kernels.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

// Canonical name of the default ONNX domain. Kept locally so this
// translation unit does not depend on onnx_op or onnx_optim just to
// reach the same constant.
constexpr const char *kDefaultOnnxDomain = "ai.onnx";

// Normalises the empty default ONNX domain to ``ai.onnx`` so that
// dispatch-table lookups always use a canonical key.
std::string NormaliseDispatchDomain(const NodeProto &node) {
  const std::string domain = node.domain().as_string();
  return domain.empty() ? std::string(kDefaultOnnxDomain) : domain;
}

// Looks up an input by name in the tensor map; throws a descriptive
// std::invalid_argument when the entry is missing so the caller sees
// the offending op_type/input name pair.
const Tensor &GetInput(const NodeProto &node, int index, const TensorMap &tensors) {
  const std::string name = node.input(index).as_string();
  if (name.empty()) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' input #" +
                                std::to_string(index) + " is unset (empty name).");
  }
  auto it = tensors.find(name);
  if (it == tensors.end()) {
    throw std::invalid_argument("RunNode: input '" + name + "' of op '" +
                                node.op_type().as_string() + "' is missing from the tensor map.");
  }
  return it->second;
}

// Inserts an output tensor in the map under the name declared by
// ``node.output(index)`` and tags it so downstream nodes can find it
// by name.
void SetOutput(const NodeProto &node, int index, Tensor result, TensorMap &tensors) {
  const std::string name = node.output(index).as_string();
  if (name.empty()) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' output #" +
                                std::to_string(index) + " is unset (empty name).");
  }
  result.name = name;
  tensors[name] = std::move(result);
}

// Validates the node declares exactly ``expected`` inputs (matching
// the per-operator class signature) and rejects unsupported variadic
// shapes early.
void RequireInputCount(const NodeProto &node, int expected) {
  if (static_cast<int>(node.input_size()) != expected) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' expects " +
                                std::to_string(expected) + " input(s), got " +
                                std::to_string(node.input_size()) + ".");
  }
}

// Validates the node declares exactly ``expected`` outputs.
void RequireOutputCount(const NodeProto &node, int expected) {
  if (static_cast<int>(node.output_size()) != expected) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' expects " +
                                std::to_string(expected) + " output(s), got " +
                                std::to_string(node.output_size()) + ".");
  }
}

// ---------------------------------------------------------------------------
// Trampoline factories. Each helper returns a NodeKernelFn that:
//   * validates the node's input/output count,
//   * reads the typed inputs from ``rt.tensors`` by name,
//   * constructs the kernel with ``rt.kernel_ctx``,
//   * stores the produced output back in ``rt.tensors`` by name.
// Centralising the boilerplate keeps the dispatch table compact.
// ---------------------------------------------------------------------------

template <class KernelT> NodeKernelFn MakeUnaryTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 1);
    RequireOutputCount(node, 1);
    const Tensor &x = GetInput(node, 0, rt.tensors());
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(x), rt.tensors());
  };
}

template <class KernelT> NodeKernelFn MakeBinaryTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 2);
    RequireOutputCount(node, 1);
    const Tensor &x = GetInput(node, 0, rt.tensors());
    const Tensor &y = GetInput(node, 1, rt.tensors());
    KernelT kernel(rt.kernel_ctx());
    SetOutput(node, 0, kernel(x, y), rt.tensors());
  };
}

} // namespace

const std::unordered_map<std::string, NodeKernelFn> &KernelDispatchTable() {
  static const std::unordered_map<std::string, NodeKernelFn> table = {
      // Element-wise unary math.
      {"ai.onnx:Abs", MakeUnaryTrampoline<kernel::Abs>()},
      {"ai.onnx:Neg", MakeUnaryTrampoline<kernel::Neg>()},
      // Element-wise binary math with NumPy-style broadcasting.
      {"ai.onnx:Add", MakeBinaryTrampoline<kernel::Add>()},
      {"ai.onnx:Sub", MakeBinaryTrampoline<kernel::Sub>()},
      {"ai.onnx:Mul", MakeBinaryTrampoline<kernel::Mul>()},
      {"ai.onnx:Div", MakeBinaryTrampoline<kernel::Div>()},
  };
  return table;
}

namespace {

// Builds the canonical "<domain>:<op_type>:<overload>" key used to
// look up model-local FunctionProto definitions in
// ``RuntimeContext::functions``. The default ONNX domain (empty
// ``NodeProto::domain``) is normalised to ``ai.onnx``.
std::string FunctionLookupKey(const std::string &domain, const std::string &op_type,
                              const std::string &overload) {
  const std::string d = domain.empty() ? std::string(kDefaultOnnxDomain) : domain;
  return d + ":" + op_type + ":" + overload;
}

// Invokes a model-local FunctionProto in response to a call site
// ``node``. The function is executed in a child RuntimeContext so its
// local names cannot collide with the caller's tensor map; only the
// formal outputs are propagated back to the caller under the names
// declared by ``node.output(i)``.
void CallModelLocalFunction(const NodeProto &node, const FunctionProto &func, RuntimeContext &rt) {
  const std::string op_type = node.op_type().as_string();
  if (static_cast<int>(node.input_size()) != static_cast<int>(func.input_size())) {
    throw std::invalid_argument("RunNode: call to model-local function '" + op_type + "' expects " +
                                std::to_string(func.input_size()) + " input(s), got " +
                                std::to_string(node.input_size()) + ".");
  }
  if (static_cast<int>(node.output_size()) != static_cast<int>(func.output_size())) {
    throw std::invalid_argument("RunNode: call to model-local function '" + op_type + "' expects " +
                                std::to_string(func.output_size()) + " output(s), got " +
                                std::to_string(node.output_size()) + ".");
  }

  // Build a child runtime context that shares the kernel construction
  // context and the function registry (so nested function calls work)
  // but starts with a fresh, isolated tensor map.
  RuntimeContext child(rt.kernel_ctx());
  child.functions() = rt.functions();

  // Bind formal function inputs to the caller's actuals.
  for (size_t i = 0; i < func.input_size(); ++i) {
    const std::string caller_name = node.input(i).as_string();
    const std::string param_name = func.input(i).as_string();
    // Optional/unused function inputs (empty actual or formal name) are skipped.
    if (caller_name.empty() || param_name.empty()) {
      continue;
    }
    auto it = rt.tensors().find(caller_name);
    if (it == rt.tensors().end()) {
      throw std::invalid_argument("RunNode: input '" + caller_name +
                                  "' of call to model-local "
                                  "function '" +
                                  op_type + "' is missing from the tensor map.");
    }
    Tensor bound = it->second;
    bound.name = param_name;
    child.tensors()[param_name] = std::move(bound);
  }

  RunFunction(func, child);

  // Copy the function's formal outputs back into the caller's tensor
  // map under the names declared by the node's output list.
  for (size_t i = 0; i < func.output_size(); ++i) {
    const std::string caller_name = node.output(i).as_string();
    const std::string param_name = func.output(i).as_string();
    if (caller_name.empty()) {
      // The caller does not want this output; skip it.
      continue;
    }
    auto it = child.tensors().find(param_name);
    if (it == child.tensors().end()) {
      throw std::invalid_argument("RunNode: output '" + param_name + "' of model-local function '" +
                                  op_type + "' was not produced by the function body.");
    }
    Tensor result = std::move(it->second);
    result.name = caller_name;
    rt.tensors()[caller_name] = std::move(result);
  }
}

} // namespace

void RunNode(const NodeProto &node, RuntimeContext &rt) {
  const std::string op_type = node.op_type().as_string();
  const std::string domain = NormaliseDispatchDomain(node);

  // A node referring to a model-local FunctionProto (registered by
  // ``RunModel`` from ``ModelProto::functions()``) takes priority over
  // the built-in kernel dispatch table so that user-defined functions
  // override same-named built-ins, matching the ONNX runtime semantics
  // for model-local functions.
  const std::string fkey = FunctionLookupKey(domain, op_type, node.overload().as_string());
  auto fit = rt.functions().find(fkey);
  if (fit != rt.functions().end()) {
    CallModelLocalFunction(node, *fit->second, rt);
    return;
  }

  const std::string key = domain + ":" + op_type;
  const auto &table = KernelDispatchTable();
  auto it = table.find(key);
  if (it == table.end()) {
    throw std::invalid_argument("RunNode: unsupported op_type '" + op_type + "' in domain '" +
                                domain + "'.");
  }
  it->second(node, rt);
}

void RunNodes(const utils::RepeatedProtoField<NodeProto> &nodes, RuntimeContext &rt) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    RunNode(nodes[i], rt);
  }
}

void RunGraph(const GraphProto &graph, RuntimeContext &rt) {
  // Seed the tensor map with all graph initializers.
  const auto &inits = graph.initializer();
  for (size_t i = 0; i < inits.size(); ++i) {
    const TensorProto &tp = inits[i];
    const std::string init_name = tp.name().as_string();
    // Only insert if the caller has not already provided a value for this
    // name (i.e. runtime overrides of initializers are respected).
    if (!rt.Has(init_name)) {
      rt.Set(init_name, TensorFromProto(tp));
    }
  }
  RunNodes(graph.node(), rt);
}

void RunFunction(const FunctionProto &func, RuntimeContext &rt) { RunNodes(func.node(), rt); }

void RunModel(const ModelProto &model, RuntimeContext &rt) {
  if (!model.has_graph()) {
    throw std::invalid_argument("RunModel: the ModelProto does not contain a graph.");
  }
  // Register every model-local function so that nodes referring to
  // them by (domain, op_type, overload) are dispatched to
  // :cpp:func:`RunFunction` rather than rejected as unsupported ops.
  const auto &fns = model.functions();
  for (size_t i = 0; i < fns.size(); ++i) {
    const FunctionProto &f = fns[i];
    const std::string key =
        FunctionLookupKey(f.domain().as_string(), f.name().as_string(), f.overload().as_string());
    rt.functions()[key] = &f;
  }
  RunGraph(model.ref_graph(), rt);
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
