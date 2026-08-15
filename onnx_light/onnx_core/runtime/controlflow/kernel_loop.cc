// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/controlflow/include_controlflow_kernels.h"
#include "onnx_core/runtime/run_nodes.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

namespace {

// Returns the effective trip count, applying ONNX's Loop termination rules:
//   * cond, when present, must be a scalar BOOL; if false, the loop runs 0
//     iterations regardless of M;
//   * M, when present, must be a scalar INT64 trip-count upper bound;
//   * the number of caller-supplied per-iteration scan values caps the
//     trip count when M is absent or larger.
int64_t EffectiveTripCount(const Tensor &M, const Tensor &cond, int64_t per_iter_count) {
  // ``cond`` empty means "omitted" -> treat as true.
  if (cond.data_type != DataType::UNDEFINED) {
    EXT_ENFORCE_INVALID(cond.data_type == DataType::BOOL,
                        "kernel::Loop: 'cond' must be a BOOL tensor when provided.");
    EXT_ENFORCE_INVALID(cond.element_count() == 1,
                        "kernel::Loop: 'cond' must contain a single element when provided.");
    if (cond.data[0] == 0) {
      return 0;
    }
  }
  int64_t limit = per_iter_count;
  if (M.data_type != DataType::UNDEFINED) {
    EXT_ENFORCE_INVALID(M.data_type == DataType::INT64,
                        "kernel::Loop: 'M' must be an INT64 tensor when provided.");
    EXT_ENFORCE_INVALID(M.element_count() == 1,
                        "kernel::Loop: 'M' must contain a single element when provided.");
    EXT_ENFORCE_INVALID(M.size_bytes() >= sizeof(int64_t),
                        "kernel::Loop: 'M' buffer is too small to hold an INT64.");
    const int64_t m_value = M.AsInt64()[0];
    EXT_ENFORCE_INVALID(m_value >= 0, "kernel::Loop: 'M' must be non-negative.");
    if (m_value < limit) {
      limit = m_value;
    }
  }
  return limit;
}

// Parses the INT64 scalar ``M`` (when provided) into a non-negative
// trip-count upper bound. Returns ``INT64_MAX`` when ``M`` is omitted
// (``data_type == UNDEFINED``).
int64_t ParseMaxTripCount(const Tensor &M) {
  if (M.data_type == DataType::UNDEFINED) {
    return std::numeric_limits<int64_t>::max();
  }
  EXT_ENFORCE_INVALID(M.data_type == DataType::INT64,
                      "kernel::Loop: 'M' must be an INT64 tensor when provided.");
  EXT_ENFORCE_INVALID(M.element_count() == 1,
                      "kernel::Loop: 'M' must contain a single element when provided.");
  EXT_ENFORCE_INVALID(M.size_bytes() >= sizeof(int64_t),
                      "kernel::Loop: 'M' buffer is too small to hold an INT64.");
  const int64_t m_value = M.AsInt64()[0];
  EXT_ENFORCE_INVALID(m_value >= 0, "kernel::Loop: 'M' must be non-negative.");
  return m_value;
}

// Parses the optional BOOL scalar ``cond`` into the initial termination
// condition. Returns ``true`` when ``cond`` is omitted.
bool ParseInitialCond(const Tensor &cond) {
  if (cond.data_type == DataType::UNDEFINED) {
    return true;
  }
  EXT_ENFORCE_INVALID(cond.data_type == DataType::BOOL,
                      "kernel::Loop: 'cond' must be a BOOL tensor when provided.");
  EXT_ENFORCE_INVALID(cond.element_count() == 1,
                      "kernel::Loop: 'cond' must contain a single element when provided.");
  return cond.bytes()[0] != 0;
}

// Stacks ``per_iter`` along a new leading axis of length ``trip_count``.
// All entries must share the same data type and shape; only the first
// ``trip_count`` entries are consumed (later entries, if any, are ignored
// because the loop terminated early). When ``allocator`` is non-null, the
// returned tensor stores its bytes in an allocator-owned ``RawBuffer``.
Tensor StackScanOutput(const Tensors &per_iter, int64_t trip_count,
                       RawBufferAllocator *allocator = nullptr) {
  EXT_ENFORCE_INVALID(static_cast<int64_t>(per_iter.size()) >= trip_count,
                      "kernel::Loop: scan-output row is shorter than the effective trip count.");
  DataType dtype = DataType::UNDEFINED;
  Shape base_shape;
  std::size_t elem_bytes = 0;
  // Seed dtype/shape from the first available per-iteration tensor so that
  // even when the loop runs zero iterations the stacked output keeps its
  // trailing shape and dtype (only the leading axis becomes zero).
  if (!per_iter.empty()) {
    const Tensor &first = per_iter[0];
    dtype = static_cast<DataType>(first.data_type);
    base_shape = first.shape;
    EXT_ENFORCE_INVALID(first.element_count() == 0 ||
                            first.size_bytes() % first.element_count() == 0,
                        "kernel::Loop: scan-output tensor data is not a multiple of its element "
                        "count.");
    elem_bytes = first.element_count() == 0 ? 0 : first.size_bytes() / first.element_count();
  }
  for (int64_t t = 0; t < trip_count; ++t) {
    const Tensor &it = per_iter[static_cast<std::size_t>(t)];
    if (t == 0) {
      // Already seeded above.
    } else {
      EXT_ENFORCE_INVALID(it.data_type == dtype,
                          "kernel::Loop: scan-output tensors must share the same data type "
                          "across iterations.");
      EXT_ENFORCE_INVALID(it.shape == base_shape,
                          "kernel::Loop: scan-output tensors must share the same shape "
                          "across iterations.");
    }
  }
  Shape stacked_shape;
  stacked_shape.reserve(base_shape.size() + 1);
  stacked_shape.push_back(trip_count);
  stacked_shape.insert(stacked_shape.end(), base_shape.begin(), base_shape.end());

  const std::size_t stacked_n_bytes =
      trip_count > 0 ? static_cast<std::size_t>(trip_count) * per_iter[0].size_bytes() : 0;
  Tensor stacked =
      MakeOutputTensor(static_cast<int32_t>(dtype), stacked_shape, stacked_n_bytes, allocator);
  if (trip_count > 0 && elem_bytes > 0 && per_iter[0].size_bytes() > 0) {
    uint8_t *dst = stacked.mutable_bytes();
    for (int64_t t = 0; t < trip_count; ++t) {
      const Tensor &it = per_iter[static_cast<std::size_t>(t)];
      std::memcpy(dst, it.bytes(), it.size_bytes());
      dst += it.size_bytes();
    }
  }
  return stacked;
}

// Assembles the final Loop outputs from the final loop-carried state and the collected
// per-iteration scan values.
//   * ``trip_count`` specifies the number of executed iterations.
//   * ``v_initial`` supplies the initial loop-carried tensors returned when
//     ``trip_count == 0``.
//   * ``final_state`` supplies the final loop-carried tensors produced by the loop body.
//   * ``scan_values_per_iter`` supplies the collected per-iteration scan-output tensors.
//   * ``allocator`` specifies the optional allocator used for stacked scan outputs.
Tensors AssembleLoopOutputs(int64_t trip_count, const Tensors &v_initial,
                            const Tensors &final_state,
                            const std::vector<Tensors> &scan_values_per_iter,
                            RawBufferAllocator *allocator) {
  Tensors out;
  out.reserve(final_state.size() + scan_values_per_iter.size());
  for (std::size_t i = 0; i < final_state.size(); ++i) {
    out.push_back(trip_count == 0 ? v_initial[i] : final_state[i]);
  }
  for (const auto &row : scan_values_per_iter) {
    out.push_back(StackScanOutput(row, trip_count, allocator));
  }
  return out;
}

// Runs the Loop body callback until termination, then assembles the final outputs.
//   * ``M`` supplies the optional Loop trip-count tensor.
//   * ``cond`` supplies the optional Loop initial-condition tensor.
//   * ``v_initial`` supplies the initial loop-carried tensors.
//   * ``num_scan_outputs`` specifies the number of per-iteration scan outputs produced by
//     ``run_body``.
//   * ``run_body`` implements one Loop body iteration.
//   * ``allocator`` specifies the optional allocator used for stacked scan outputs.
Tensors RunLoopBody(const Tensor &M, const Tensor &cond, const Tensors &v_initial,
                    std::size_t num_scan_outputs, const Loop::BodyRunner &run_body,
                    RawBufferAllocator *allocator) {
  EXT_ENFORCE_INVALID(static_cast<bool>(run_body),
                      "kernel::Loop: 'run_body' callback must be callable.");

  const int64_t max_trip = ParseMaxTripCount(M);
  bool cond_value = ParseInitialCond(cond);

  const std::size_t n = v_initial.size();
  Tensors state = v_initial;
  std::vector<Tensors> scan_values(num_scan_outputs);

  int64_t trip_count = 0;
  for (int64_t iter = 0; iter < max_trip && cond_value; ++iter) {
    Tensors body_outputs = run_body(iter, cond_value, state);
    EXT_ENFORCE_INVALID(body_outputs.size() == 1 + n + num_scan_outputs,
                        "kernel::Loop: body returned the wrong number of outputs (expected "
                        "1 + N + num_scan_outputs).");
    EXT_ENFORCE_INVALID(body_outputs[0].data_type == DataType::BOOL &&
                            body_outputs[0].element_count() == 1,
                        "kernel::Loop: body output #0 ('cond_out') must be a BOOL scalar.");
    cond_value = body_outputs[0].bytes()[0] != 0;
    body_outputs[0].ResetAllocation();

    Tensors next_state;
    next_state.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      EXT_ENFORCE_INVALID(body_outputs[1 + i].data_type == v_initial[i].data_type,
                          "kernel::Loop: body's updated loop-carried output must keep the "
                          "data type of the matching 'v_initial' tensor.");
      next_state.push_back(std::move(body_outputs[1 + i]));
    }
    state = std::move(next_state);
    for (std::size_t i = 0; i < num_scan_outputs; ++i) {
      scan_values[i].push_back(std::move(body_outputs[1 + n + i]));
    }
    ++trip_count;
  }

  return AssembleLoopOutputs(trip_count, v_initial, state, scan_values, allocator);
}

} // namespace

Tensors Loop::operator()(const Tensor &M, const Tensor &cond, const Tensors &v_initial,
                         const Tensors &final_state,
                         const std::vector<Tensors> &scan_values_per_iter) const {
  EXT_ENFORCE_INVALID(v_initial.size() == final_state.size(),
                      "kernel::Loop: 'final_state' must have the same number of tensors "
                      "as 'v_initial'.");
  for (std::size_t i = 0; i < v_initial.size(); ++i) {
    EXT_ENFORCE_INVALID(v_initial[i].data_type == final_state[i].data_type,
                        "kernel::Loop: 'final_state' tensor data type must match the "
                        "corresponding 'v_initial' tensor.");
  }

  // Determine the per-iteration row length used to cap the trip count.
  int64_t per_iter_count = -1;
  for (const auto &row : scan_values_per_iter) {
    const int64_t row_len = static_cast<int64_t>(row.size());
    if (per_iter_count < 0) {
      per_iter_count = row_len;
    } else {
      EXT_ENFORCE_INVALID(row_len == per_iter_count,
                          "kernel::Loop: all scan-output rows must have the same length "
                          "(one per actually executed iteration).");
    }
  }
  // When there are no scan outputs we fall back to M (or 'unbounded' when M
  // is also missing). Using INT64_MAX as the sentinel keeps the
  // ``min(M, per_iter_count)`` rule expressible without a special case.
  if (per_iter_count < 0) {
    per_iter_count = INT64_MAX;
  }

  const int64_t trip_count = EffectiveTripCount(M, cond, per_iter_count);

  return AssembleLoopOutputs(trip_count, v_initial, final_state, scan_values_per_iter, nullptr);
}

Tensors Loop::operator()(RuntimeContext &rt, const Tensor &M, const Tensor &cond,
                         const Tensors &v_initial, const Tensors &final_state,
                         const std::vector<Tensors> &scan_values_per_iter) const {
  EXT_ENFORCE_INVALID(v_initial.size() == final_state.size(),
                      "kernel::Loop: 'final_state' must have the same number of tensors "
                      "as 'v_initial'.");
  for (std::size_t i = 0; i < v_initial.size(); ++i) {
    EXT_ENFORCE_INVALID(v_initial[i].data_type == final_state[i].data_type,
                        "kernel::Loop: 'final_state' tensor data type must match the "
                        "corresponding 'v_initial' tensor.");
  }

  int64_t per_iter_count = -1;
  for (const auto &row : scan_values_per_iter) {
    const int64_t row_len = static_cast<int64_t>(row.size());
    if (per_iter_count < 0) {
      per_iter_count = row_len;
    } else {
      EXT_ENFORCE_INVALID(row_len == per_iter_count,
                          "kernel::Loop: all scan-output rows must have the same length "
                          "(one per actually executed iteration).");
    }
  }
  if (per_iter_count < 0) {
    per_iter_count = INT64_MAX;
  }

  const int64_t trip_count = EffectiveTripCount(M, cond, per_iter_count);
  return AssembleLoopOutputs(trip_count, v_initial, final_state, scan_values_per_iter,
                             rt.allocator());
}

Tensors Loop::operator()(const Tensor &M, const Tensor &cond, const Tensors &v_initial,
                         std::size_t num_scan_outputs, const BodyRunner &run_body) const {
  return RunLoopBody(M, cond, v_initial, num_scan_outputs, run_body, nullptr);
}

Tensors Loop::operator()(RuntimeContext &rt, const Tensor &M, const Tensor &cond,
                         const Tensors &v_initial, std::size_t num_scan_outputs,
                         const BodyRunner &run_body) const {
  return RunLoopBody(M, cond, v_initial, num_scan_outputs, run_body, rt.allocator());
}

Tensors Loop::operator()(RuntimeContext &rt, const GraphProto &body, const Tensor &M,
                         const Tensor &cond, const Tensors &v_initial) const {
  // Build a single session over the body once and reuse it for every
  // iteration instead of re-resolving the body's kernels on every call; the
  // body itself is only needed here, not kept around afterwards.
  SubgraphSession session(rt, body);
  return (*this)(rt, body, session, M, cond, v_initial);
}

Tensors Loop::operator()(RuntimeContext &rt, const GraphProto &body, SubgraphSession &session,
                         const Tensor &M, const Tensor &cond, const Tensors &v_initial) const {
  const std::size_t n = v_initial.size();
  EXT_ENFORCE_INVALID(!(body.input_size() < static_cast<int>(2 + n)),
                      "kernel::Loop: body graph does not declare enough inputs.");
  EXT_ENFORCE_INVALID(!(body.output_size() < static_cast<int>(1 + n)),
                      "kernel::Loop: body graph does not declare enough outputs.");
  const std::size_t k = static_cast<std::size_t>(body.output_size()) - 1 - n;

  auto run_body = [&](int64_t iter, bool cond_in, const Tensors &state) -> Tensors {
    std::vector<std::pair<std::string, Tensor>> bindings;
    bindings.reserve(2 + n);
    bindings.emplace_back(body.input(0).name(),
                          MakeInt64Scalar(body.input(0).name(), iter, rt.allocator()));
    bindings.emplace_back(body.input(1).name(),
                          MakeBoolScalar(body.input(1).name(), cond_in, rt.allocator()));
    for (std::size_t i = 0; i < n; ++i) {
      Tensor t = state[i];
      t.name = body.input(static_cast<int>(2 + i)).name();
      bindings.emplace_back(t.name, std::move(t));
    }
    return session.Run(std::move(bindings), rt, "body");
  };

  Tensors outputs = RunLoopBody(M, cond, v_initial, k, run_body, rt.allocator());

  // When the loop runs zero iterations the kernel produces UNDEFINED-typed
  // empty scan outputs (it has no template to seed dtype/shape from). Patch
  // each scan output using the body's declared output value-info so the
  // downstream pipeline (ReferenceEvaluator, numpy conversion, ...) sees a
  // well-typed empty tensor of the expected element type and per-iteration
  // trailing shape.
  for (std::size_t i = 0; i < k; ++i) {
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
  return outputs;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
