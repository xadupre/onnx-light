// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/run_nodes.h"

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "onnx_backend_test/kernels/math/include_math_kernels.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

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
    const Tensor &x = GetInput(node, 0, rt.tensors);
    KernelT kernel(rt.kernel_ctx);
    SetOutput(node, 0, kernel(x), rt.tensors);
  };
}

template <class KernelT> NodeKernelFn MakeBinaryTrampoline() {
  return [](const NodeProto &node, RuntimeContext &rt) {
    RequireInputCount(node, 2);
    RequireOutputCount(node, 1);
    const Tensor &x = GetInput(node, 0, rt.tensors);
    const Tensor &y = GetInput(node, 1, rt.tensors);
    KernelT kernel(rt.kernel_ctx);
    SetOutput(node, 0, kernel(x, y), rt.tensors);
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

void RunNode(const NodeProto &node, RuntimeContext &rt) {
  const std::string op_type = node.op_type().as_string();
  const std::string key = NormaliseDispatchDomain(node) + ":" + op_type;
  const auto &table = KernelDispatchTable();
  auto it = table.find(key);
  if (it == table.end()) {
    throw std::invalid_argument("RunNode: unsupported op_type '" + op_type + "' in domain '" +
                                NormaliseDispatchDomain(node) + "'.");
  }
  it->second(node, rt);
}

void RunNodes(const utils::RepeatedProtoField<NodeProto> &nodes, RuntimeContext &rt) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    RunNode(nodes[i], rt);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
