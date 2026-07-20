// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/run_nodes.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnx_core/runtime/controlflow/include_controlflow_kernels.h"
#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/node_helpers.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

namespace {

int64_t ParseInt64Scalar(const Tensor &t, const std::string &where) {
  EXT_ENFORCE_INVALID(!(t.data_type != DataType::INT64 || t.element_count() != 1),
                      "RunNode: ", where, " must be an INT64 scalar.");
  return t.AsInt64()[0];
}

bool ParseBoolScalar(const Tensor &t, const std::string &where) {
  EXT_ENFORCE_INVALID(!(t.data_type != DataType::BOOL || t.element_count() != 1),
                      "RunNode: ", where, " must be a BOOL scalar.");
  return t.AsBool()[0] != 0;
}

Tensor MakeInt64Scalar(const std::string &name, int64_t v, RawBufferAllocator *allocator) {
  return Tensor::FromInt64(name, {}, {v}, allocator);
}

Tensor MakeBoolScalar(const std::string &name, bool v, RawBufferAllocator *allocator) {
  return Tensor::FromBool(name, {}, {static_cast<uint8_t>(v ? 1 : 0)}, allocator);
}

Tensor CloneTensor(const Tensor &tensor, RawBufferAllocator *allocator = nullptr);

/**
 * Creates a deep copy of a tensor before it leaves a child RuntimeContext.
 *
 * @param tensor Tensor to clone.
 * @param allocator Optional allocator for the cloned raw buffer.
 *     Passing `nullptr` keeps the legacy inline `std::vector<uint8_t>`
 *     storage for numeric tensors.
 * @return A deep copy of `tensor` with owned storage.
 *
 * The copy avoids dangling pointers when the child held allocator-backed or
 * borrowed storage, and it preserves duplicate subgraph outputs that name the
 * same tensor more than once. When `allocator` is non-null, numeric
 * tensors are materialized in allocator-backed storage immediately instead of
 * waiting for a later `RuntimeContext::Put` migration.
 */
Tensor CloneTensor(const Tensor &tensor, RawBufferAllocator *allocator) {
  if (static_cast<DataType>(tensor.data_type) == DataType::STRING) {
    return Tensor::MakeString(tensor.name, tensor.shape, tensor.string_data);
  }
  if (allocator == nullptr) {
    std::vector<uint8_t> data(tensor.size_bytes());
    if (tensor.size_bytes() > 0) {
      std::memcpy(data.data(), tensor.bytes(), tensor.size_bytes());
    }
    return Tensor(tensor.name, tensor.data_type, tensor.shape, std::move(data));
  }
  Tensor clone = MakeOutputTensor(tensor.data_type, tensor.shape, tensor.size_bytes(), allocator);
  clone.name = tensor.name;
  if (tensor.size_bytes() > 0) {
    std::memcpy(clone.mutable_bytes(), tensor.bytes(), tensor.size_bytes());
  }
  return clone;
}

int64_t CheckedMulInt64(int64_t a, int64_t b, const std::string &where) {
  EXT_ENFORCE_INVALID(!(a < 0 || b < 0), "RunNode: ", where, " encountered a negative dimension (",
                      a, ", ", b, ").");
  EXT_ENFORCE_INVALID(!(a != 0 && b > std::numeric_limits<int64_t>::max() / a), "RunNode: ", where,
                      " overflows INT64 shape arithmetic.");
  return a * b;
}

} // namespace

namespace {

// Builds the canonical "<domain>:<op_type>:<overload>" key used to
// look up model-local FunctionProto definitions in
// ``RuntimeContext::functions``. The default ONNX domain (empty
// ``NodeProto::domain``) is normalised to ``ai.onnx``.
std::string FunctionLookupKey(const std::string &domain, const std::string &op_type,
                              const std::string &overload) {
  if (domain.empty())
    return std::string(":") + op_type + ":" + overload;
  return domain + ":" + op_type + ":" + overload;
}

template <class NameCollection> std::string FormatNameList(const NameCollection &names) {
  std::ostringstream oss;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << names[i];
  }
  return oss.str();
}

// Emits the ReferenceEvaluator verbose progress line for one node dispatch.
// The format is
// ``[ReferenceEvaluator] #<node_index> Domain::OpType(inputs) -> (outputs)``.
// Nothing is printed when ``rt.verbose() <= 0``.
void PrintNodeProgress(const RuntimeContext &rt, const NodeProto &node, const std::string &domain,
                       const std::string &op_type) {
  if (rt.verbose() <= 0) {
    return;
  }
  std::cout << "[ReferenceEvaluator] ";
  if (rt.current_subgraph_node_index() >= 0) {
    std::cout << rt.current_subgraph_attr_name() << "@" << rt.current_subgraph_node_index() << "/";
  }
  if (rt.current_node_index() >= 0) {
    std::cout << "#" << rt.current_node_index() << " ";
  }
  if (domain != kDefaultOnnxDomain) {
    std::cout << domain << "::";
  }
  std::cout << op_type << "(" << FormatNameList(node.input()) << ") -> ("
            << FormatNameList(node.output()) << ")" << std::endl;
}

} // namespace

int64_t ResolveAxis(int64_t axis, size_t rank, const std::string &op_name) {
  int64_t a = axis;
  const int64_t r = static_cast<int64_t>(rank);
  if (a < 0) {
    a += r;
  }
  EXT_ENFORCE_INVALID(!(a < 0 || a >= r), "RunNode: op '", op_name, "' axis is out of range.");
  return a;
}

Tensor SliceTensorAlongAxis(const Tensor &t, int64_t axis, int64_t index,
                            const std::string &op_name) {
  EXT_ENFORCE_INVALID(!(t.shape.empty()), "RunNode: op '", op_name,
                      "' cannot slice a rank-0 scan input.");
  const int64_t dim = t.shape[static_cast<size_t>(axis)];
  EXT_ENFORCE_INVALID(!(index < 0 || index >= dim), "RunNode: op '", op_name,
                      "' scan index is out of range.");
  std::vector<int64_t> out_shape;
  out_shape.reserve(t.shape.size() - 1);
  for (size_t i = 0; i < t.shape.size(); ++i) {
    if (static_cast<int64_t>(i) != axis) {
      out_shape.push_back(t.shape[i]);
    }
  }

  const int64_t outer = t.shape.product(0, static_cast<size_t>(axis), op_name);
  const int64_t inner = t.shape.product(static_cast<size_t>(axis) + 1, t.shape.size(), op_name);
  const size_t elem_bytes = t.element_size();
  const int64_t elements_per_slice = CheckedMulInt64(outer, inner, op_name);
  EXT_ENFORCE_INVALID(!(inner > 0 && static_cast<uint64_t>(inner) >
                                         std::numeric_limits<size_t>::max() / elem_bytes),
                      "RunNode: op '", op_name, "' exceeds addressable buffer size.");
  const size_t inner_bytes = static_cast<size_t>(inner) * elem_bytes;
  EXT_ENFORCE_INVALID(
      !(elements_per_slice > 0 && static_cast<uint64_t>(elements_per_slice) >
                                      std::numeric_limits<size_t>::max() / elem_bytes),
      "RunNode: op '", op_name, "' exceeds addressable buffer size.");
  const size_t out_n_bytes = static_cast<size_t>(elements_per_slice) * elem_bytes;
  Tensor out;
  if (t.has_allocation()) {
    out = MakeOutputTensor(t.data_type, out_shape, out_n_bytes, t.allocation_owner());
  } else {
    out = Tensor("", t.data_type, out_shape, std::vector<uint8_t>(out_n_bytes));
  }
  if (!t.name.empty()) {
    out.name = t.name + "_slice";
  }

  for (int64_t o = 0; o < outer; ++o) {
    const size_t src_offset = static_cast<size_t>((o * dim + index) * inner) * elem_bytes;
    const size_t dst_offset = static_cast<size_t>(o * inner) * elem_bytes;
    if (inner_bytes > 0) {
      std::memcpy(out.mutable_bytes() + dst_offset, t.bytes() + src_offset, inner_bytes);
    }
  }

  return out;
}

std::vector<Tensor> RunSubgraph(const GraphProto &graph,
                                std::vector<std::pair<std::string, Tensor>> bindings,
                                RuntimeContext &rt, const std::string &attr_name) {
  RuntimeContext child = rt.MakeSubgraphContext(attr_name);
  for (auto &kv : bindings) {
    child.Put(kv.first, std::move(kv.second), RuntimeEventKind::kInput);
  }
  RunGraph(graph, child);

  if (rt.events_enabled()) {
    for (auto &ev : child.events()) {
      rt.events().push_back(std::move(ev));
    }
  }

  std::vector<Tensor> outputs;
  outputs.reserve(graph.output().size());
  for (size_t i = 0; i < graph.output().size(); ++i) {
    const std::string out_name = graph.output()[i].name();
    EXT_ENFORCE_INVALID(!(out_name.empty()), "RunNode: a subgraph output has an empty name.");
    auto it = child.tensors().find(out_name);
    EXT_ENFORCE_INVALID(it != child.tensors().end(), "RunNode: subgraph output '", out_name,
                        "' was not produced.");
    outputs.push_back(CloneTensor(it->second));
  }
  return outputs;
}

namespace {

void PropagateOutputsToCaller(const NodeProto &node, std::vector<Tensor> &&outputs,
                              RuntimeContext &rt) {
  EXT_ENFORCE_INVALID(outputs.size() == node.output_size(), "RunNode: op '", node.op_type(),
                      "' produced ", outputs.size(), " output(s), node declares ",
                      node.output_size(), ".");
  for (size_t i = 0; i < outputs.size(); ++i) {
    const std::string caller_name = node.output(i);
    if (caller_name.empty()) {
      continue;
    }
    Tensor t = std::move(outputs[i]);
    t.name = caller_name;
    rt.Put(caller_name, std::move(t), RuntimeEventKind::kOutput);
  }
}

void RunIfNode(const NodeProto &node, RuntimeContext &rt) {
  RequireInputCount(node, 1);

  const Tensor &cond = GetInput(node, 0, rt.tensors());
  EXT_ENFORCE_INVALID(cond.data_type == DataType::BOOL, "RunNode: If input 'cond' must be BOOL.");
  EXT_ENFORCE_INVALID(cond.element_count() == 1,
                      "RunNode: If input 'cond' must contain a single element.");

  const GraphProto &then_branch = GetRequiredGraphAttribute(node, "then_branch");
  const GraphProto &else_branch = GetRequiredGraphAttribute(node, "else_branch");
  EXT_ENFORCE_INVALID(
      then_branch.output_size() == else_branch.output_size(),
      "RunNode: If 'then_branch' and 'else_branch' must declare the same number of outputs.");
  EXT_ENFORCE_INVALID(node.output_size() == then_branch.output_size(),
                      "RunNode: If node output count does not match branch output count.");

  const bool taken = cond.bytes()[0] != 0;
  const GraphProto &branch = taken ? then_branch : else_branch;
  const std::string branch_attr = taken ? "then_branch" : "else_branch";

  RuntimeContext child = rt.MakeSubgraphContext(branch_attr);
  RunGraph(branch, child);

  if (rt.events_enabled()) {
    for (auto &ev : child.events()) {
      rt.events().push_back(std::move(ev));
    }
  }

  for (int i = 0; i < branch.output_size(); ++i) {
    const std::string out_name = branch.output()[i].name();
    EXT_ENFORCE_INVALID(!(out_name.empty()), "RunNode: If: a subgraph output has an empty name.");
    const std::string caller_name = node.output(i);
    if (caller_name.empty()) {
      continue;
    }
    if (child.HasSequence(out_name)) {
      rt.PutSequence(caller_name, child.GetSequence(out_name));
    } else {
      auto it = child.tensors().find(out_name);
      EXT_ENFORCE_INVALID(it != child.tensors().end(), "RunNode: If: subgraph output '", out_name,
                          "' was not produced by the selected branch.");
      Tensor t = CloneTensor(it->second, rt.allocator());
      t.name = caller_name;
      rt.Put(caller_name, std::move(t), RuntimeEventKind::kOutput);
    }
  }
}

// Runs ``body`` as a Loop iteration body where loop-carried state may
// contain sequence-typed values in addition to tensors. Used by
// :func:`RunLoopNode` to support the ``test_cc_loop13_seq``-style case
// (``SequenceEmpty`` -> ``Loop`` with a sequence-typed loop-carried
// state). The orchestration is performed directly here because
// :class:`Loop` only operates on tensor state.
void RunLoopWithSequenceState(const NodeProto &node, const GraphProto &body, const Tensor &m_tensor,
                              const Tensor &cond_tensor, const std::vector<bool> &is_seq_state,
                              std::vector<Tensor> tensor_state,
                              std::vector<Sequence> sequence_state, std::size_t k,
                              RuntimeContext &rt) {
  const std::size_t n = is_seq_state.size();

  int64_t max_trip = std::numeric_limits<int64_t>::max();
  if (m_tensor.data_type != DataType::UNDEFINED) {
    max_trip = ParseInt64Scalar(m_tensor, "Loop input 'M'");
    EXT_ENFORCE_INVALID(!(max_trip < 0), "RunNode: Loop input 'M' must be non-negative.");
  }
  bool cond_value = true;
  if (cond_tensor.data_type != DataType::UNDEFINED) {
    cond_value = ParseBoolScalar(cond_tensor, "Loop input 'cond'");
  }

  std::vector<std::vector<Tensor>> scan_values(k);
  int64_t trip_count = 0;
  for (int64_t iter = 0; iter < max_trip && cond_value; ++iter) {
    RuntimeContext child = rt.MakeSubgraphContext("body");

    const std::string &iter_name = body.input(0).name();
    const std::string &cond_name = body.input(1).name();
    child.Put(iter_name, MakeInt64Scalar(iter_name, iter, rt.allocator()),
              RuntimeEventKind::kInput);
    child.Put(cond_name, MakeBoolScalar(cond_name, cond_value, rt.allocator()),
              RuntimeEventKind::kInput);
    for (std::size_t i = 0; i < n; ++i) {
      const std::string &bname = body.input(static_cast<int>(2 + i)).name();
      if (is_seq_state[i]) {
        child.PutSequence(bname, sequence_state[i]);
      } else {
        Tensor t = tensor_state[i];
        t.name = bname;
        child.Put(bname, std::move(t), RuntimeEventKind::kInput);
      }
    }
    RunGraph(body, child);

    if (rt.events_enabled()) {
      for (auto &ev : child.events()) {
        rt.events().push_back(std::move(ev));
      }
    }

    const std::string &cond_out_name = body.output(0).name();
    auto cond_it = child.tensors().find(cond_out_name);
    EXT_ENFORCE_INVALID(cond_it != child.tensors().end(),
                        "RunNode: Loop body did not produce 'cond_out' output '", cond_out_name,
                        "'.");
    cond_value = ParseBoolScalar(cond_it->second, "Loop body output 'cond_out'");

    for (std::size_t i = 0; i < n; ++i) {
      const std::string &oname = body.output(static_cast<int>(1 + i)).name();
      if (is_seq_state[i]) {
        EXT_ENFORCE_INVALID(
            child.HasSequence(oname),
            "RunNode: Loop body did not produce sequence-typed loop-carried output '", oname, "'.");
        sequence_state[i] = child.GetSequence(oname);
      } else {
        auto it = child.tensors().find(oname);
        EXT_ENFORCE_INVALID(it != child.tensors().end(),
                            "RunNode: Loop body did not produce tensor-typed loop-carried output '",
                            oname, "'.");
        tensor_state[i] = CloneTensor(it->second);
      }
    }
    for (std::size_t j = 0; j < k; ++j) {
      const std::string &oname = body.output(static_cast<int>(1 + n + j)).name();
      auto it = child.tensors().find(oname);
      EXT_ENFORCE_INVALID(it != child.tensors().end(),
                          "RunNode: Loop body did not produce scan output '", oname, "'.");
      scan_values[j].push_back(CloneTensor(it->second));
    }
    ++trip_count;
  }

  // Propagate loop-carried outputs (sequence- or tensor-typed) to caller.
  for (std::size_t i = 0; i < n; ++i) {
    const std::string &caller_name = node.output(static_cast<int>(i));
    if (caller_name.empty()) {
      continue;
    }
    if (is_seq_state[i]) {
      rt.PutSequence(caller_name, sequence_state[i]);
    } else {
      Tensor t = tensor_state[i];
      t.name = caller_name;
      rt.Put(caller_name, std::move(t), RuntimeEventKind::kOutput);
    }
  }

  // Stack scan outputs along a new leading axis and propagate to caller.
  for (std::size_t j = 0; j < k; ++j) {
    const std::string &caller_name = node.output(static_cast<int>(n + j));
    if (caller_name.empty()) {
      continue;
    }
    const auto &row = scan_values[j];
    int32_t dtype = static_cast<int32_t>(DataType::UNDEFINED);
    std::vector<int64_t> base_shape;
    std::size_t elem_bytes = 0;
    if (!row.empty()) {
      const Tensor &first = row[0];
      dtype = first.data_type;
      base_shape = first.shape;
      elem_bytes = first.element_count() == 0 ? 0 : first.size_bytes() / first.element_count();
    }
    std::vector<int64_t> stacked_shape;
    stacked_shape.reserve(base_shape.size() + 1);
    stacked_shape.push_back(trip_count);
    stacked_shape.insert(stacked_shape.end(), base_shape.begin(), base_shape.end());

    std::vector<uint8_t> stacked_data;
    if (trip_count > 0 && elem_bytes > 0 && row[0].size_bytes() > 0) {
      stacked_data.reserve(static_cast<std::size_t>(trip_count) * row[0].size_bytes());
      for (int64_t t = 0; t < trip_count; ++t) {
        const Tensor &it = row[static_cast<std::size_t>(t)];
        stacked_data.insert(stacked_data.end(), it.bytes(), it.bytes() + it.size_bytes());
      }
    }
    Tensor stacked(caller_name, dtype, std::move(stacked_shape), std::move(stacked_data));

    // When the loop runs zero iterations the kernel has no template to seed
    // dtype/shape from. Patch using the body's declared output value-info so
    // the downstream pipeline (ReferenceEvaluator, numpy conversion, ...)
    // sees a well-typed empty tensor of the expected element type and
    // per-iteration trailing shape.
    if (stacked.data_type == static_cast<int32_t>(DataType::UNDEFINED)) {
      const auto &vi = body.output(static_cast<int>(1 + n + j));
      if (vi.has_type() && vi.type().has_tensor_type()) {
        const auto &tt = vi.type().tensor_type();
        stacked.data_type = static_cast<int32_t>(tt.elem_type());
        std::vector<int64_t> per_iter_shape;
        if (tt.has_shape()) {
          per_iter_shape.reserve(tt.shape().dim().size());
          for (int d = 0; d < tt.shape().dim().size(); ++d) {
            const auto &dim = tt.shape().dim()[d];
            per_iter_shape.push_back(dim.has_dim_value() ? static_cast<int64_t>(dim.dim_value())
                                                         : 0);
          }
        }
        stacked.shape.assign(1, 0);
        stacked.shape.insert(stacked.shape.end(), per_iter_shape.begin(), per_iter_shape.end());
      }
    }
    rt.Put(caller_name, std::move(stacked), RuntimeEventKind::kOutput);
  }
}

void RunLoopNode(const NodeProto &node, RuntimeContext &rt) {
  EXT_ENFORCE_INVALID(!(node.input_size() < 2),
                      "RunNode: op 'Loop' expects at least 2 inputs (M, cond).");
  const GraphProto &body = GetRequiredGraphAttribute(node, "body");

  Tensor m_tensor;
  if (!node.input(0).empty()) {
    m_tensor = GetInput(node, 0, rt.tensors());
  }
  Tensor cond_tensor;
  if (!node.input(1).empty()) {
    cond_tensor = GetInput(node, 1, rt.tensors());
  }

  // Classify each loop-carried input as either sequence-typed (looked up
  // in ``rt.sequences()``) or tensor-typed (in ``rt.tensors()``). When
  // any state slot is sequence-typed we route through a dispatcher-local
  // orchestrator since :class:`Loop` only operates on tensors.
  const std::size_t n_inputs = static_cast<std::size_t>(node.input_size() - 2);
  std::vector<bool> is_seq_state(n_inputs, false);
  std::vector<Tensor> tensor_state(n_inputs);
  std::vector<Sequence> sequence_state(n_inputs);
  bool any_sequence_state = false;
  for (std::size_t i = 0; i < n_inputs; ++i) {
    const int idx = static_cast<int>(2 + i);
    EXT_ENFORCE_INVALID(
        !(node.input(idx).empty()),
        "RunNode: Loop does not support empty placeholders in loop-carried inputs.");
    const std::string name = node.input(idx);
    if (rt.HasSequence(name)) {
      is_seq_state[i] = true;
      sequence_state[i] = rt.GetSequence(name);
      any_sequence_state = true;
    } else {
      tensor_state[i] = GetInput(node, idx, rt.tensors());
    }
  }

  EXT_ENFORCE_INVALID(!(body.input_size() < static_cast<int>(2 + n_inputs)),
                      "RunNode: Loop body graph does not declare enough inputs.");
  EXT_ENFORCE_INVALID(!(body.output_size() < static_cast<int>(1 + n_inputs)),
                      "RunNode: Loop body graph does not declare enough outputs.");
  const std::size_t k = body.output_size() - 1 - n_inputs;
  EXT_ENFORCE_INVALID(!(node.output_size() != static_cast<int>(n_inputs + k)),
                      "RunNode: Loop node output count does not match body outputs.");

  if (any_sequence_state) {
    RunLoopWithSequenceState(node, body, m_tensor, cond_tensor, is_seq_state,
                             std::move(tensor_state), std::move(sequence_state), k, rt);
    return;
  }

  const std::size_t n = n_inputs;
  std::vector<Tensor> v_initial = std::move(tensor_state);

  auto run_body = [&](int64_t iter, bool cond_in,
                      const std::vector<Tensor> &state) -> std::vector<Tensor> {
    std::vector<std::pair<std::string, Tensor>> bindings;
    bindings.reserve(2 + n);
    bindings.emplace_back(body.input(0).name(),
                          MakeInt64Scalar(body.input(0).name(), iter, rt.allocator()));
    bindings.emplace_back(body.input(1).name(),
                          MakeBoolScalar(body.input(1).name(), cond_in, rt.allocator()));
    for (size_t i = 0; i < n; ++i) {
      Tensor t = state[i];
      t.name = body.input(2 + i).name();
      bindings.emplace_back(t.name, std::move(t));
    }
    return RunSubgraph(body, std::move(bindings), rt, "body");
  };

  Loop loop_kernel(rt.kernel_ctx());
  std::vector<Tensor> outputs = loop_kernel(rt, m_tensor, cond_tensor, v_initial, k, run_body);

  // When the loop runs zero iterations the kernel produces UNDEFINED-typed
  // empty scan outputs (it has no template to seed dtype/shape from). Patch
  // each scan output using the body's declared output value-info so the
  // downstream pipeline (ReferenceEvaluator, numpy conversion, ...) sees a
  // well-typed empty tensor of the expected element type and per-iteration
  // trailing shape.
  for (size_t i = 0; i < k; ++i) {
    Tensor &t = outputs[n + i];
    if (t.data_type != static_cast<int32_t>(DataType::UNDEFINED)) {
      continue;
    }
    const auto &vi = body.output(static_cast<int>(1 + n + i));
    if (!vi.has_type() || !vi.type().has_tensor_type()) {
      continue;
    }
    const auto &tt = vi.type().tensor_type();
    t.data_type = static_cast<int32_t>(tt.elem_type());
    std::vector<int64_t> per_iter_shape;
    if (tt.has_shape()) {
      per_iter_shape.reserve(tt.shape().dim().size());
      for (int d = 0; d < tt.shape().dim().size(); ++d) {
        const auto &dim = tt.shape().dim()[d];
        per_iter_shape.push_back(dim.has_dim_value() ? static_cast<int64_t>(dim.dim_value()) : 0);
      }
    }
    t.shape.assign(1, 0);
    t.shape.insert(t.shape.end(), per_iter_shape.begin(), per_iter_shape.end());
  }
  PropagateOutputsToCaller(node, std::move(outputs), rt);
}

void RunScanNode(const NodeProto &node, RuntimeContext &rt) {
  const GraphProto &body = GetRequiredGraphAttribute(node, "body");
  const int64_t num_scan_inputs = GetAttributeIntOrDefault(node, "num_scan_inputs", 1);
  EXT_ENFORCE_INVALID(!(num_scan_inputs <= 0),
                      "RunNode: Scan attribute 'num_scan_inputs' must be positive.");

  // Scan-8 (opset versions [1, 8]) prepends an optional ``sequence_lens``
  // input and wraps every state / scan input/output in an outer batch
  // dimension. We detect it from the model's default-domain opset version
  // and, when active, skip the leading ``sequence_lens`` slot, run the
  // Scan-9+ kernel once per batch element, and stack the per-batch
  // outputs along a new leading axis.
  const int64_t opset_version = rt.kernel_ctx().opset.version;
  const bool is_scan8 = opset_version >= 1 && opset_version <= 8;
  const int scan8_offset = is_scan8 ? 1 : 0;

  EXT_ENFORCE_INVALID(
      !(node.input_size() < num_scan_inputs + scan8_offset),
      "RunNode: Scan does not have enough inputs for the declared num_scan_inputs.");

  const size_t n = static_cast<size_t>(node.input_size() - num_scan_inputs - scan8_offset);
  const size_t m = static_cast<size_t>(num_scan_inputs);
  EXT_ENFORCE_INVALID(!(body.input_size() < static_cast<int>(n + m)),
                      "RunNode: Scan body graph does not declare enough inputs.");
  EXT_ENFORCE_INVALID(!(body.output_size() < static_cast<int>(n)),
                      "RunNode: Scan body graph does not declare enough outputs.");
  const size_t k = body.output_size() - n;
  EXT_ENFORCE_INVALID(!(node.output_size() != static_cast<int>(n + k)),
                      "RunNode: Scan node output count does not match body outputs.");

  if (is_scan8 && !node.input(0).empty()) {
    // Per-batch sequence lengths are not yet supported; the only
    // exercised form (e.g. backend test ``test_scan_sum``) passes an
    // empty placeholder, meaning "use the full sequence length for
    // every batch element".
    EXT_THROW_INVALID(
        "RunNode: Scan opset 8 with a non-empty 'sequence_lens' input is not supported.");
  }

  std::vector<Tensor> initial_state;
  initial_state.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const int idx = static_cast<int>(scan8_offset + i);
    EXT_ENFORCE_INVALID(!(node.input(idx).empty()),
                        "RunNode: Scan does not support empty placeholders in state inputs.");
    initial_state.push_back(GetInput(node, idx, rt.tensors()));
  }

  std::vector<Tensor> scan_inputs;
  scan_inputs.reserve(m);
  for (size_t i = 0; i < m; ++i) {
    const int idx = static_cast<int>(scan8_offset + n + i);
    EXT_ENFORCE_INVALID(!(node.input(idx).empty()),
                        "RunNode: Scan does not support empty placeholders in scan inputs.");
    scan_inputs.push_back(GetInput(node, idx, rt.tensors()));
  }

  // Scan-8 attribute ``directions`` was split into
  // ``scan_input_directions`` / ``scan_output_directions`` (the latter
  // defaulting to all-forward) in Scan-9; ``scan_*_axes`` did not exist
  // (the scan axis was always 1 of the batched tensor, i.e. axis 0 of
  // each per-batch slice).
  std::vector<int64_t> scan_input_axes;
  std::vector<int64_t> scan_input_directions;
  std::vector<int64_t> scan_output_axes;
  std::vector<int64_t> scan_output_directions;
  if (is_scan8) {
    scan_input_directions = GetAttributeIntsOrDefault(node, "directions", {});
  } else {
    scan_input_axes = GetAttributeIntsOrDefault(node, "scan_input_axes", {});
    scan_input_directions = GetAttributeIntsOrDefault(node, "scan_input_directions", {});
    scan_output_axes = GetAttributeIntsOrDefault(node, "scan_output_axes", {});
    scan_output_directions = GetAttributeIntsOrDefault(node, "scan_output_directions", {});
  }

  // The Scan kernel now owns the iteration loop (including the body
  // subgraph evaluation): RunScanNode is reduced to validating the node
  // shape and forwarding inputs / attributes to the kernel.
  Scan scan_kernel(rt.kernel_ctx());
  if (!is_scan8) {
    std::vector<Tensor> outputs =
        scan_kernel(rt, body, initial_state, scan_inputs, scan_input_axes, scan_input_directions,
                    scan_output_axes, scan_output_directions);
    PropagateOutputsToCaller(node, std::move(outputs), rt);
    return;
  }

  // Scan-8: every state / scan input has a leading batch dim ``B`` and
  // all inputs must agree on it. Run the Scan-9 kernel once per batch
  // index on the batch-stripped slices and stack the resulting per-
  // batch outputs along a new leading axis.
  int64_t batch = -1;
  auto check_batch = [&](const Tensor &t) {
    EXT_ENFORCE_INVALID(!(t.shape.empty()),
                        "RunNode: Scan opset 8 requires inputs to have a leading batch dimension.");
    const int64_t b = t.shape[0];
    if (batch < 0) {
      batch = b;
    } else if (batch != b) {
      EXT_THROW_INVALID("RunNode: Scan opset 8 inputs must agree on the leading batch dimension.");
    }
  };
  for (const auto &t : initial_state) {
    check_batch(t);
  }
  for (const auto &t : scan_inputs) {
    check_batch(t);
  }
  if (batch < 0) {
    batch = 0;
  }

  // Slot ``o`` of ``per_output`` collects the per-batch tensors for the
  // ``o``-th Scan-9 output, in batch order; reusing the stacking-only
  // overload of ``Scan`` then concatenates them along a fresh
  // leading axis to recover the batch dimension.
  std::vector<std::vector<Tensor>> per_output(n + k);
  for (int64_t b = 0; b < batch; ++b) {
    std::vector<Tensor> batch_state;
    batch_state.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      batch_state.push_back(SliceTensorAlongAxis(initial_state[i], 0, b, "Scan"));
    }
    std::vector<Tensor> batch_scan;
    batch_scan.reserve(m);
    for (size_t i = 0; i < m; ++i) {
      batch_scan.push_back(SliceTensorAlongAxis(scan_inputs[i], 0, b, "Scan"));
    }
    std::vector<Tensor> batch_outputs =
        scan_kernel(rt, body, batch_state, batch_scan, /*scan_input_axes=*/{},
                    scan_input_directions, /*scan_output_axes=*/{},
                    /*scan_output_directions=*/{});
    EXT_ENFORCE_INVALID(
        batch_outputs.size() == n + k,
        "RunNode: Scan opset 8 inner Scan-9 call produced an unexpected output count.");
    for (size_t o = 0; o < n + k; ++o) {
      per_output[o].push_back(std::move(batch_outputs[o]));
    }
  }

  // Stack each output column along a new leading axis (default axis 0).
  std::vector<Tensor> outputs = scan_kernel(rt, batch, /*initial_state=*/{}, /*final_state=*/{},
                                            per_output, /*scan_output_axes=*/{},
                                            /*scan_output_directions=*/{});
  PropagateOutputsToCaller(node, std::move(outputs), rt);
}

void RunSequenceMapNode(const NodeProto &node, RuntimeContext &rt) {
  // ai.onnx::SequenceMap (since opset 17): applies a sub-graph (``body``)
  // to each element of the first input sequence. Additional inputs are
  // either further sequences (which must have the same length as the
  // first one and are iterated element-wise) or tensors (which are
  // broadcast unchanged to every iteration). The body produces ``M``
  // tensors per iteration which are assembled into ``M`` output
  // sequences via :cpp:class:`kernel::SequenceMap`.
  EXT_ENFORCE_INVALID(!(node.input_size() < 1),
                      "RunNode: SequenceMap expects at least 1 input (the input sequence).");
  const GraphProto &body = GetRequiredGraphAttribute(node, "body");

  // The first input must always be a sequence; its length sets ``N``.
  const Sequence &input_sequence = GetInputSequence(node, 0, rt);
  const std::size_t n = input_sequence.size();

  // Classify every additional input as either a sequence input (must
  // have length ``N``) or a broadcast tensor input.
  const std::size_t num_additional = static_cast<std::size_t>(node.input_size() - 1);
  std::vector<const Sequence *> additional_sequences(num_additional, nullptr);
  std::vector<const Tensor *> additional_tensors(num_additional, nullptr);
  for (std::size_t k = 0; k < num_additional; ++k) {
    const int idx = static_cast<int>(1 + k);
    const std::string name = node.input(idx);
    EXT_ENFORCE_INVALID(
        !(name.empty()),
        "RunNode: SequenceMap does not support empty placeholders in additional inputs.");
    if (rt.HasSequence(name)) {
      const Sequence &seq = rt.GetSequence(name);
      EXT_ENFORCE_INVALID(seq.size() == n, "RunNode: SequenceMap additional sequence input '", name,
                          "' has length ", seq.size(), ", expected ", n,
                          " (matching the first input sequence length).");
      additional_sequences[k] = &seq;
    } else {
      additional_tensors[k] = &GetInput(node, idx, rt.tensors());
    }
  }

  // ``body`` must declare one input per SequenceMap input.
  EXT_ENFORCE_INVALID(!(static_cast<std::size_t>(body.input_size()) != 1u + num_additional),
                      "RunNode: SequenceMap body graph declares ", body.input_size(),
                      " input(s), expected ", 1u + num_additional, ".");

  const std::size_t m = static_cast<std::size_t>(body.output_size());
  EXT_ENFORCE_INVALID(m != 0u, "RunNode: SequenceMap body graph must declare at least 1 output.");
  EXT_ENFORCE_INVALID(!(static_cast<std::size_t>(node.output_size()) != m),
                      "RunNode: SequenceMap node declares ", node.output_size(),
                      " output(s), but the body graph produces ", m, ".");

  // ``body_outputs_per_iter[k][i]`` = body output ``k`` for iteration ``i``.
  std::vector<std::vector<Tensor>> body_outputs_per_iter(m);
  for (std::size_t k = 0; k < m; ++k) {
    body_outputs_per_iter[k].reserve(n);
  }

  for (std::size_t i = 0; i < n; ++i) {
    std::vector<std::pair<std::string, Tensor>> bindings;
    bindings.reserve(1u + num_additional);

    Tensor elem0 = input_sequence.values[i];
    elem0.name = body.input(0).name();
    bindings.emplace_back(elem0.name, std::move(elem0));

    for (std::size_t k = 0; k < num_additional; ++k) {
      const std::string param_name = body.input(static_cast<int>(1 + k)).name();
      Tensor t;
      if (additional_sequences[k] != nullptr) {
        t = additional_sequences[k]->values[i];
      } else {
        t = *additional_tensors[k];
      }
      t.name = param_name;
      bindings.emplace_back(param_name, std::move(t));
    }

    std::vector<Tensor> iter_outputs = RunSubgraph(body, std::move(bindings), rt, "body");
    EXT_ENFORCE_INVALID(iter_outputs.size() == m, "RunNode: SequenceMap body produced ",
                        iter_outputs.size(), " output(s) at iteration ", i, ", expected ", m, ".");
    for (std::size_t k = 0; k < m; ++k) {
      body_outputs_per_iter[k].push_back(std::move(iter_outputs[k]));
    }
  }

  // The final packaging step ("stack the per-iteration body outputs
  // into M output sequences") is implemented by
  // ``onnx_kernels::kernel::SequenceMap``, which must stay in
  // ``onnx_kernels`` (only control-flow kernels move alongside the
  // runtime engine). It is invoked here through the registered
  // :cpp:func:`GetSequenceMapPackFn` callback instead of being called
  // directly so this translation unit (part of ``onnx_core``) never
  // includes an ``onnx_kernels`` kernel header.
  const SequenceMapPackFn &pack_fn = GetSequenceMapPackFn();
  EXT_ENFORCE_INVALID(static_cast<bool>(pack_fn),
                      "RunNode: SequenceMap: no SequenceMap packing function has been "
                      "registered; call onnx_kernels::RegisterKernelFunctions() before "
                      "running a model that uses SequenceMap.");
  std::vector<Sequence> outputs = pack_fn(rt, input_sequence, body_outputs_per_iter);
  EXT_ENFORCE_INVALID(outputs.size() == m,
                      "RunNode: kernel::SequenceMap returned an unexpected number of "
                      "output sequences.");
  for (std::size_t k = 0; k < m; ++k) {
    SetOutputSequence(node, static_cast<int>(k), std::move(outputs[k]), rt);
  }
}

// Replaces every attribute of ``node`` (and recursively in any
// sub-graph attribute) carrying a non-empty ``ref_attr_name`` with the
// corresponding entry from ``attr_map``. When the call-site does not
// supply a value for a referenced attribute, the referenced attribute
// is removed -- matching the ONNX function-inliner semantics.
//
// The local attribute name declared inside the function body is
// preserved; only the attribute's value is taken from the call-site.
void BindRefAttributes(NodeProto &node,
                       const std::unordered_map<std::string, const AttributeProto *> &attr_map) {
  auto &attributes = node.attribute();
  for (auto it = attributes.begin(); it != attributes.end();) {
    AttributeProto &attr = *it;
    if (!attr.ref_attr_name().empty()) {
      auto found = attr_map.find(attr.ref_attr_name());
      if (found != attr_map.end()) {
        const std::string local_name = attr.name();
        attr.CopyFrom(*found->second);
        attr.set_name(local_name);
        ++it;
      } else {
        it = attributes.erase(it);
      }
    } else {
      // Recurse into any sub-graph attributes so that nested nodes
      // also receive the bound attribute values.
      if (attr.has_g()) {
        GraphProto &gp = attr.ref_g();
        for (size_t i = 0; i < gp.node().size(); ++i) {
          BindRefAttributes(gp.ref_node()[i], attr_map);
        }
      }
      auto &gs = attr.graphs();
      for (size_t gi = 0; gi < gs.size(); ++gi) {
        GraphProto &gp = gs[gi];
        for (size_t i = 0; i < gp.node().size(); ++i) {
          BindRefAttributes(gp.ref_node()[i], attr_map);
        }
      }
      ++it;
    }
  }
}

// Invokes a model-local FunctionProto in response to a call site
// ``node``. The function is executed in a child RuntimeContext so its
// local names cannot collide with the caller's tensor map; only the
// formal outputs are propagated back to the caller under the names
// declared by ``node.output(i)``.
//
// Before execution, every attribute reference (``ref_attr_name``) in
// the function body is resolved against the call-site attributes,
// falling back to the typed defaults declared in
// ``FunctionProto::attribute_proto`` when the call-site omits a value.
void CallModelLocalFunction(const NodeProto &node, const FunctionProto &func, RuntimeContext &rt) {
  const std::string op_type = node.op_type();
  EXT_ENFORCE_INVALID(!(static_cast<int>(node.input_size()) != static_cast<int>(func.input_size())),
                      "RunNode: call to model-local function '", op_type, "' expects ",
                      func.input_size(), " input(s), got ", node.input_size(), ".");
  EXT_ENFORCE_INVALID(
      !(static_cast<int>(node.output_size()) != static_cast<int>(func.output_size())),
      "RunNode: call to model-local function '", op_type, "' expects ", func.output_size(),
      " output(s), got ", node.output_size(), ".");

  // Build a child runtime context that shares the kernel construction
  // context and the function registry (so nested function calls work)
  // but starts with a fresh, isolated tensor map.
  RuntimeContext child = rt.MakeFunctionContext();

  // Bind formal function inputs to the caller's actuals.
  for (size_t i = 0; i < func.input_size(); ++i) {
    const std::string caller_name = node.input(i);
    const std::string param_name = func.input(i);
    // Optional/unused function inputs (empty actual or formal name) are skipped.
    if (caller_name.empty() || param_name.empty()) {
      continue;
    }
    auto it = rt.tensors().find(caller_name);
    EXT_ENFORCE_INVALID(it != rt.tensors().end(), "RunNode: input '", caller_name,
                        "' of call to model-local "
                        "function '",
                        op_type, "' is missing from the tensor map.");
    Tensor bound = it->second;
    bound.name = param_name;
    child.Put(param_name, std::move(bound), RuntimeEventKind::kInput);
  }

  // Resolve attribute references (``ref_attr_name``) inside the
  // function body. Resolution proceeds in two steps: the call-site
  // attributes take precedence, and any unresolved reference falls
  // back to the typed default declared in
  // ``FunctionProto::attribute_proto``. Resolution is performed on a
  // local copy of the function so the caller's ModelProto is not
  // mutated and the runtime stays thread-safe with respect to the
  // shared function registry.
  std::unordered_map<std::string, const AttributeProto *> attr_map;
  for (size_t i = 0; i < func.attribute_proto().size(); ++i) {
    const AttributeProto &a = func.attribute_proto()[i];
    attr_map[a.name()] = &a;
  }
  for (size_t i = 0; i < node.attribute().size(); ++i) {
    const AttributeProto &a = node.attribute()[i];
    attr_map[a.name()] = &a;
  }
  FunctionProto bound_func;
  bound_func.CopyFrom(func);
  for (size_t i = 0; i < bound_func.node().size(); ++i) {
    BindRefAttributes(bound_func.ref_node()[i], attr_map);
  }
  RunFunction(bound_func, child);

  // Copy the function's formal outputs back into the caller's tensor
  // map under the names declared by the node's output list.
  for (size_t i = 0; i < func.output_size(); ++i) {
    const std::string caller_name = node.output(i);
    const std::string param_name = func.output(i);
    if (caller_name.empty()) {
      // The caller does not want this output; skip it.
      continue;
    }
    auto it = child.tensors().find(param_name);
    EXT_ENFORCE_INVALID(it != child.tensors().end(), "RunNode: output '", param_name,
                        "' of model-local function '", op_type,
                        "' was not produced by the function body.");
    Tensor result = CloneTensor(it->second, rt.allocator());
    result.name = caller_name;
    rt.Put(caller_name, std::move(result), RuntimeEventKind::kOutput);
  }
}

} // namespace

void RunNode(const NodeProto &node, RuntimeContext &rt) {
  auto domain = ONNX_LIGHT_NAMESPACE::NormaliseDispatchDomain(node);
  const std::string &op_type = node.op_type().value();
  PrintNodeProgress(rt, node, domain, op_type);

  // Only capture timing and input names when event logging is active.
  const bool logging = rt.events_enabled();
  int64_t start_time_ns = 0;
  std::chrono::steady_clock::time_point t0;
  if (logging) {
    start_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    t0 = std::chrono::steady_clock::now();
  }

  // A node referring to a model-local FunctionProto (registered by
  // ``RunModel`` from ``ModelProto::functions()``) takes priority over
  // the built-in kernel dispatch table so that user-defined functions
  // override same-named built-ins, matching the ONNX runtime semantics
  // for model-local functions.
  if (!rt.functions().empty()) {
    const std::string fkey = FunctionLookupKey(domain, op_type, node.overload());
    auto fit = rt.functions().find(fkey);
    if (fit != rt.functions().end()) {
      CallModelLocalFunction(node, *fit->second, rt);
      if (logging) {
        const int64_t duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - t0)
                                        .count();
        std::vector<std::string> inputs;
        inputs.reserve(static_cast<size_t>(node.input_size()));
        for (size_t i = 0; i < static_cast<size_t>(node.input_size()); ++i) {
          inputs.push_back(node.input(i));
        }
        rt.AppendRunNodeEvent(domain, op_type, std::move(inputs), start_time_ns, duration_ns);
      }
      return;
    }
  }

  if (domain == kDefaultOnnxDomain && op_type == "If") {
    RunIfNode(node, rt);
  } else if (domain == kDefaultOnnxDomain && op_type == "Loop") {
    RunLoopNode(node, rt);
  } else if (domain == kDefaultOnnxDomain && op_type == "Scan") {
    RunScanNode(node, rt);
  } else if (domain == kDefaultOnnxDomain && op_type == "SequenceMap") {
    RunSequenceMapNode(node, rt);
  } else {
    const std::string key = domain + ":" + op_type;
    // User-registered custom kernels take precedence over built-in
    // kernel dispatch table entries so callers can override (or
    // extend) the runtime with their own implementations.
    auto ckit = rt.custom_kernels().find(key);
    if (ckit != rt.custom_kernels().end()) {
      ckit->second(node, rt);
    } else {
      const auto &table = KernelDispatchTable();
      auto it = table.find(key);
      EXT_ENFORCE_INVALID(it != table.end(), "RunNode: unsupported op_type '", op_type,
                          "' in domain '", domain, "'.");
      it->second(node, rt);
    }
  }

  if (logging) {
    const int64_t duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
            .count();
    std::vector<std::string> inputs;
    inputs.reserve(static_cast<size_t>(node.input_size()));
    for (size_t i = 0; i < static_cast<size_t>(node.input_size()); ++i) {
      inputs.push_back(node.input(i));
    }
    rt.AppendRunNodeEvent(domain, op_type, std::move(inputs), start_time_ns, duration_ns);
  }
}

namespace {

// Runs every node of ``nodes`` in order and, after each node, delegates
// to :cpp:func:`ExecutionPlan::ReleaseAfter` to free any intermediates
// the plan has scheduled for release at that node.
template <class NodeRange>
void RunNodesAndRelease(const NodeRange &nodes, RuntimeContext &rt, const ExecutionPlan &plan) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    rt.set_current_node_index(static_cast<int64_t>(i));
    RunNode(nodes[i], rt);
    plan.ReleaseAfter(nodes[i], rt);
  }
  rt.set_current_node_index(-1);
}

} // namespace

void RunNodes(const utils::RepeatedProtoField<NodeProto> &nodes, RuntimeContext &rt) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    rt.set_current_node_index(static_cast<int64_t>(i));
    RunNode(nodes[i], rt);
  }
  rt.set_current_node_index(-1);
}

void RunNodes(const utils::RepeatedProtoField<NodeProto> &nodes, RuntimeContext &rt,
              const ExecutionPlan &plan) {
  RunNodesAndRelease(nodes, rt, plan);
}

void RunGraph(const GraphProto &graph, RuntimeContext &rt) {
  // Seed the tensor map with all graph initializers.
  const auto &inits = graph.initializer();
  for (size_t i = 0; i < inits.size(); ++i) {
    const TensorProto &tp = inits[i];
    const std::string init_name = tp.name();
    // Only insert if the caller has not already provided a value for this
    // name (i.e. runtime overrides of initializers are respected).
    if (!rt.Has(init_name)) {
      rt.Set(init_name, TensorFromProto(tp), RuntimeEventKind::kInitializer);
    }
  }
  if (!rt.release_intermediates()) {
    RunNodes(graph.node(), rt);
    return;
  }
  // Reuse the cached :cpp:class:`ExecutionPlan` for ``graph`` (built
  // on first use) so the release analysis is paid only once across
  // every invocation of the same model.
  RunNodes(graph.node(), rt, rt.GetExecutionPlan(graph));
}

void RunFunction(const FunctionProto &func, RuntimeContext &rt) {
  if (!rt.release_intermediates()) {
    RunNodes(func.node(), rt);
    return;
  }
  RunNodes(func.node(), rt, rt.GetExecutionPlan(func));
}

void RunModel(const ModelProto &model, RuntimeContext &rt) {
  EXT_ENFORCE_INVALID(model.has_graph(), "RunModel: the ModelProto does not contain a graph.");
  // Register every model-local function so that nodes referring to
  // them by (domain, op_type, overload) are dispatched to
  // :cpp:func:`RunFunction` rather than rejected as unsupported ops.
  const auto &fns = model.functions();
  for (size_t i = 0; i < fns.size(); ++i) {
    const FunctionProto &f = fns[i];
    const std::string key = FunctionLookupKey(f.domain(), f.name(), f.overload());
    rt.functions()[key] = &f;
  }
  RunGraph(model.ref_graph(), rt);
}

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
