// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/run_nodes.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_kernels/kernels/controlflow/include_controlflow_kernels.h"
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

const AttributeProto *FindAttribute(const NodeProto &node, const std::string &name) {
  for (size_t i = 0; i < node.attribute().size(); ++i) {
    const AttributeProto &attr = node.attribute()[i];
    if (attr.name().as_string() == name) {
      return &attr;
    }
  }
  return nullptr;
}

const GraphProto &GetRequiredGraphAttribute(const NodeProto &node, const std::string &name) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' is missing '" +
                                name + "' graph attribute.");
  }
  if (attr->type() != AttributeProto::AttributeType::GRAPH) {
    throw std::invalid_argument("RunNode: attribute '" + name + "' of op '" +
                                node.op_type().as_string() + "' must be a GRAPH.");
  }
  return attr->ref_g();
}

int64_t GetAttributeIntOrDefault(const NodeProto &node, const std::string &name, int64_t fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  if (attr->type() != AttributeProto::AttributeType::INT) {
    throw std::invalid_argument("RunNode: attribute '" + name + "' of op '" +
                                node.op_type().as_string() + "' must be an INT.");
  }
  return attr->i();
}

std::vector<int64_t> GetAttributeIntsOrDefault(const NodeProto &node, const std::string &name,
                                               const std::vector<int64_t> &fallback) {
  const AttributeProto *attr = FindAttribute(node, name);
  if (attr == nullptr) {
    return fallback;
  }
  if (attr->type() != AttributeProto::AttributeType::INTS) {
    throw std::invalid_argument("RunNode: attribute '" + name + "' of op '" +
                                node.op_type().as_string() + "' must be INTS.");
  }
  std::vector<int64_t> values;
  values.reserve(attr->ints().size());
  for (size_t i = 0; i < attr->ints().size(); ++i) {
    values.push_back(attr->ints()[i]);
  }
  return values;
}

int64_t ParseInt64Scalar(const Tensor &t, const std::string &where) {
  if (t.data_type != DataType::INT64 || t.element_count() != 1) {
    throw std::invalid_argument("RunNode: " + where + " must be an INT64 scalar.");
  }
  return t.AsInt64()[0];
}

bool ParseBoolScalar(const Tensor &t, const std::string &where) {
  if (t.data_type != DataType::BOOL || t.element_count() != 1) {
    throw std::invalid_argument("RunNode: " + where + " must be a BOOL scalar.");
  }
  return t.AsBool()[0] != 0;
}

Tensor MakeInt64Scalar(const std::string &name, int64_t v) {
  return Tensor::FromInt64(name, {}, {v});
}

Tensor MakeBoolScalar(const std::string &name, bool v) {
  return Tensor::FromBool(name, {}, {static_cast<uint8_t>(v ? 1 : 0)});
}

int64_t CheckedMulInt64(int64_t a, int64_t b, const std::string &where) {
  if (a < 0 || b < 0) {
    throw std::invalid_argument("RunNode: " + where + " encountered a negative dimension (" +
                                std::to_string(a) + ", " + std::to_string(b) + ").");
  }
  if (a != 0 && b > std::numeric_limits<int64_t>::max() / a) {
    throw std::invalid_argument("RunNode: " + where + " overflows INT64 shape arithmetic.");
  }
  return a * b;
}

int64_t Product(const std::vector<int64_t> &shape, size_t begin, size_t end,
                const std::string &where) {
  int64_t p = 1;
  for (size_t i = begin; i < end; ++i) {
    p = CheckedMulInt64(p, shape[i], where);
  }
  return p;
}

int64_t ResolveAxis(int64_t axis, size_t rank, const std::string &op_name) {
  int64_t a = axis;
  const int64_t r = static_cast<int64_t>(rank);
  if (a < 0) {
    a += r;
  }
  if (a < 0 || a >= r) {
    throw std::invalid_argument("RunNode: op '" + op_name + "' axis is out of range.");
  }
  return a;
}

Tensor SliceTensorAlongAxis(const Tensor &t, int64_t axis, int64_t index,
                            const std::string &op_name) {
  if (t.shape.empty()) {
    throw std::invalid_argument("RunNode: op '" + op_name + "' cannot slice a rank-0 scan input.");
  }
  const int64_t dim = t.shape[static_cast<size_t>(axis)];
  if (index < 0 || index >= dim) {
    throw std::invalid_argument("RunNode: op '" + op_name + "' scan index is out of range.");
  }
  std::vector<int64_t> out_shape;
  out_shape.reserve(t.shape.size() - 1);
  for (size_t i = 0; i < t.shape.size(); ++i) {
    if (static_cast<int64_t>(i) != axis) {
      out_shape.push_back(t.shape[i]);
    }
  }

  const int64_t outer = Product(t.shape, 0, static_cast<size_t>(axis), op_name);
  const int64_t inner = Product(t.shape, static_cast<size_t>(axis) + 1, t.shape.size(), op_name);
  const size_t elem_bytes = t.element_size();
  const int64_t elements_per_slice = CheckedMulInt64(outer, inner, op_name);
  if (inner > 0 && static_cast<uint64_t>(inner) > std::numeric_limits<size_t>::max() / elem_bytes) {
    throw std::invalid_argument("RunNode: op '" + op_name + "' exceeds addressable buffer size.");
  }
  const size_t inner_bytes = static_cast<size_t>(inner) * elem_bytes;
  if (elements_per_slice > 0 &&
      static_cast<uint64_t>(elements_per_slice) > std::numeric_limits<size_t>::max() / elem_bytes) {
    throw std::invalid_argument("RunNode: op '" + op_name + "' exceeds addressable buffer size.");
  }
  std::vector<uint8_t> out_data(static_cast<size_t>(elements_per_slice) * elem_bytes);

  for (int64_t o = 0; o < outer; ++o) {
    const size_t src_offset = static_cast<size_t>((o * dim + index) * inner) * elem_bytes;
    const size_t dst_offset = static_cast<size_t>(o * inner) * elem_bytes;
    if (inner_bytes > 0) {
      std::memcpy(out_data.data() + dst_offset, t.bytes() + src_offset, inner_bytes);
    }
  }

  return Tensor("", t.data_type, std::move(out_shape), std::move(out_data));
}

std::vector<Tensor> RunSubgraph(const GraphProto &graph,
                                const std::vector<std::pair<std::string, Tensor>> &bindings,
                                RuntimeContext &rt) {
  RuntimeContext child(rt.kernel_ctx());
  child.functions() = rt.functions();
  child.tensors() = rt.tensors();
  for (const auto &kv : bindings) {
    child.tensors()[kv.first] = kv.second;
  }
  RunGraph(graph, child);

  std::vector<Tensor> outputs;
  outputs.reserve(graph.output().size());
  for (size_t i = 0; i < graph.output().size(); ++i) {
    const std::string out_name = graph.output()[i].name().as_string();
    if (out_name.empty()) {
      throw std::invalid_argument("RunNode: a subgraph output has an empty name.");
    }
    auto it = child.tensors().find(out_name);
    if (it == child.tensors().end()) {
      throw std::invalid_argument("RunNode: subgraph output '" + out_name + "' was not produced.");
    }
    outputs.push_back(it->second);
  }
  return outputs;
}

void PropagateOutputsToCaller(const NodeProto &node, const std::vector<Tensor> &outputs,
                              RuntimeContext &rt) {
  if (outputs.size() != node.output_size()) {
    throw std::invalid_argument("RunNode: op '" + node.op_type().as_string() + "' produced " +
                                std::to_string(outputs.size()) + " output(s), node declares " +
                                std::to_string(node.output_size()) + ".");
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    const std::string caller_name = node.output(i).as_string();
    if (caller_name.empty()) {
      continue;
    }
    Tensor t = outputs[i];
    t.name = caller_name;
    rt.tensors()[caller_name] = std::move(t);
  }
}

void RunIfNode(const NodeProto &node, RuntimeContext &rt) {
  RequireInputCount(node, 1);

  const Tensor &cond = GetInput(node, 0, rt.tensors());
  const GraphProto &then_branch = GetRequiredGraphAttribute(node, "then_branch");
  const GraphProto &else_branch = GetRequiredGraphAttribute(node, "else_branch");
  kernel::If if_kernel(rt.kernel_ctx());
  std::vector<Tensor> outputs = if_kernel(cond, then_branch, else_branch, rt);
  PropagateOutputsToCaller(node, outputs, rt);
}

void RunLoopNode(const NodeProto &node, RuntimeContext &rt) {
  if (node.input_size() < 2) {
    throw std::invalid_argument("RunNode: op 'Loop' expects at least 2 inputs (M, cond).");
  }
  const GraphProto &body = GetRequiredGraphAttribute(node, "body");

  Tensor m_tensor;
  bool has_m = false;
  if (!node.input(0).as_string().empty()) {
    has_m = true;
    m_tensor = GetInput(node, 0, rt.tensors());
  }
  Tensor cond_tensor;
  bool has_cond_input = false;
  if (!node.input(1).as_string().empty()) {
    has_cond_input = true;
    cond_tensor = GetInput(node, 1, rt.tensors());
  }

  std::vector<Tensor> v_initial;
  v_initial.reserve(static_cast<size_t>(node.input_size() - 2));
  for (int i = 2; i < node.input_size(); ++i) {
    if (node.input(i).as_string().empty()) {
      throw std::invalid_argument(
          "RunNode: Loop does not support empty placeholders in loop-carried inputs.");
    }
    v_initial.push_back(GetInput(node, i, rt.tensors()));
  }

  if (body.input_size() < static_cast<int>(2 + v_initial.size())) {
    throw std::invalid_argument("RunNode: Loop body graph does not declare enough inputs.");
  }
  const size_t n = v_initial.size();
  if (body.output_size() < static_cast<int>(1 + n)) {
    throw std::invalid_argument("RunNode: Loop body graph does not declare enough outputs.");
  }
  const size_t k = body.output_size() - 1 - n;
  if (node.output_size() != static_cast<int>(n + k)) {
    throw std::invalid_argument("RunNode: Loop node output count does not match body outputs.");
  }

  int64_t max_trip = std::numeric_limits<int64_t>::max();
  if (has_m) {
    max_trip = ParseInt64Scalar(m_tensor, "Loop input 'M'");
    if (max_trip < 0) {
      throw std::invalid_argument("RunNode: Loop input 'M' must be non-negative.");
    }
  }
  bool cond = has_cond_input ? ParseBoolScalar(cond_tensor, "Loop input 'cond'") : true;

  std::vector<Tensor> state = v_initial;
  std::vector<std::vector<Tensor>> scan_values(k);

  for (int64_t iter = 0; iter < max_trip && cond; ++iter) {
    std::vector<std::pair<std::string, Tensor>> bindings;
    bindings.reserve(2 + n);
    bindings.emplace_back(body.input(0).name().as_string(),
                          MakeInt64Scalar(body.input(0).name().as_string(), iter));
    bindings.emplace_back(body.input(1).name().as_string(),
                          MakeBoolScalar(body.input(1).name().as_string(), cond));
    for (size_t i = 0; i < n; ++i) {
      Tensor t = state[i];
      t.name = body.input(2 + i).name().as_string();
      bindings.emplace_back(t.name, std::move(t));
    }

    const std::vector<Tensor> body_outputs = RunSubgraph(body, bindings, rt);
    cond = ParseBoolScalar(body_outputs[0], "Loop body output #0");

    std::vector<Tensor> next_state;
    next_state.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      next_state.push_back(body_outputs[1 + i]);
    }
    state = std::move(next_state);
    for (size_t i = 0; i < k; ++i) {
      scan_values[i].push_back(body_outputs[1 + n + i]);
    }
  }

  kernel::Loop loop_kernel(rt.kernel_ctx());
  std::vector<Tensor> outputs =
      loop_kernel(has_m ? m_tensor : Tensor(), has_cond_input ? cond_tensor : Tensor(), v_initial,
                  state, scan_values);
  PropagateOutputsToCaller(node, outputs, rt);
}

void RunScanNode(const NodeProto &node, RuntimeContext &rt) {
  const GraphProto &body = GetRequiredGraphAttribute(node, "body");
  const int64_t num_scan_inputs = GetAttributeIntOrDefault(node, "num_scan_inputs", 1);
  if (num_scan_inputs <= 0) {
    throw std::invalid_argument("RunNode: Scan attribute 'num_scan_inputs' must be positive.");
  }
  if (node.input_size() < num_scan_inputs) {
    throw std::invalid_argument(
        "RunNode: Scan does not have enough inputs for the declared num_scan_inputs.");
  }

  const size_t n = static_cast<size_t>(node.input_size() - num_scan_inputs);
  const size_t m = static_cast<size_t>(num_scan_inputs);
  if (body.input_size() < static_cast<int>(n + m)) {
    throw std::invalid_argument("RunNode: Scan body graph does not declare enough inputs.");
  }
  if (body.output_size() < static_cast<int>(n)) {
    throw std::invalid_argument("RunNode: Scan body graph does not declare enough outputs.");
  }
  const size_t k = body.output_size() - n;
  if (node.output_size() != static_cast<int>(n + k)) {
    throw std::invalid_argument("RunNode: Scan node output count does not match body outputs.");
  }

  std::vector<Tensor> initial_state;
  initial_state.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (node.input(static_cast<int>(i)).as_string().empty()) {
      throw std::invalid_argument(
          "RunNode: Scan does not support empty placeholders in state inputs.");
    }
    initial_state.push_back(GetInput(node, static_cast<int>(i), rt.tensors()));
  }

  std::vector<Tensor> scan_inputs;
  scan_inputs.reserve(m);
  for (size_t i = 0; i < m; ++i) {
    const int idx = static_cast<int>(n + i);
    if (node.input(idx).as_string().empty()) {
      throw std::invalid_argument(
          "RunNode: Scan does not support empty placeholders in scan inputs.");
    }
    scan_inputs.push_back(GetInput(node, idx, rt.tensors()));
  }

  std::vector<int64_t> scan_input_axes = GetAttributeIntsOrDefault(node, "scan_input_axes", {});
  std::vector<int64_t> scan_input_directions =
      GetAttributeIntsOrDefault(node, "scan_input_directions", {});
  std::vector<int64_t> scan_output_axes = GetAttributeIntsOrDefault(node, "scan_output_axes", {});
  std::vector<int64_t> scan_output_directions =
      GetAttributeIntsOrDefault(node, "scan_output_directions", {});

  if (scan_input_axes.empty()) {
    scan_input_axes.assign(m, 0);
  } else if (scan_input_axes.size() != m) {
    throw std::invalid_argument(
        "RunNode: Scan attribute 'scan_input_axes' must have num_scan_inputs entries.");
  }
  if (scan_input_directions.empty()) {
    scan_input_directions.assign(m, 0);
  } else if (scan_input_directions.size() != m) {
    throw std::invalid_argument(
        "RunNode: Scan attribute 'scan_input_directions' must have num_scan_inputs entries.");
  }
  for (size_t i = 0; i < m; ++i) {
    if (scan_input_directions[i] != 0 && scan_input_directions[i] != 1) {
      throw std::invalid_argument(
          "RunNode: Scan attribute 'scan_input_directions' entries must be 0 or 1.");
    }
  }

  int64_t trip_count = -1;
  std::vector<int64_t> resolved_scan_input_axes;
  resolved_scan_input_axes.reserve(m);
  for (size_t i = 0; i < m; ++i) {
    const Tensor &scan = scan_inputs[i];
    if (scan.shape.empty()) {
      throw std::invalid_argument("RunNode: Scan input rank must be >= 1.");
    }
    const int64_t axis = ResolveAxis(scan_input_axes[i], scan.shape.size(), "Scan");
    resolved_scan_input_axes.push_back(axis);
    const int64_t dim = scan.shape[static_cast<size_t>(axis)];
    if (trip_count < 0) {
      trip_count = dim;
    } else if (trip_count != dim) {
      throw std::invalid_argument("RunNode: all Scan inputs must have the same trip count.");
    }
  }
  if (trip_count < 0) {
    trip_count = 0;
  }

  std::vector<Tensor> state = initial_state;
  std::vector<std::vector<Tensor>> scan_values(k);
  for (int64_t iter = 0; iter < trip_count; ++iter) {
    std::vector<std::pair<std::string, Tensor>> bindings;
    bindings.reserve(n + m);
    for (size_t i = 0; i < n; ++i) {
      Tensor t = state[i];
      t.name = body.input(i).name().as_string();
      bindings.emplace_back(t.name, std::move(t));
    }
    for (size_t i = 0; i < m; ++i) {
      const int64_t index = (scan_input_directions[i] == 0) ? iter : (trip_count - 1 - iter);
      Tensor slice =
          SliceTensorAlongAxis(scan_inputs[i], resolved_scan_input_axes[i], index, "Scan");
      slice.name = body.input(n + i).name().as_string();
      bindings.emplace_back(slice.name, std::move(slice));
    }

    const std::vector<Tensor> body_outputs = RunSubgraph(body, bindings, rt);
    std::vector<Tensor> next_state;
    next_state.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      next_state.push_back(body_outputs[i]);
    }
    state = std::move(next_state);
    for (size_t i = 0; i < k; ++i) {
      scan_values[i].push_back(body_outputs[n + i]);
    }
  }

  kernel::Scan scan_kernel(rt.kernel_ctx());
  std::vector<Tensor> outputs = scan_kernel(trip_count, initial_state, state, scan_values,
                                            scan_output_axes, scan_output_directions);
  PropagateOutputsToCaller(node, outputs, rt);
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

  if (domain == kDefaultOnnxDomain && op_type == "If") {
    RunIfNode(node, rt);
    return;
  }
  if (domain == kDefaultOnnxDomain && op_type == "Loop") {
    RunLoopNode(node, rt);
    return;
  }
  if (domain == kDefaultOnnxDomain && op_type == "Scan") {
    RunScanNode(node, rt);
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
