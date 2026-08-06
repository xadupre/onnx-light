// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_tensor.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

// Forward declaration to avoid pulling the full proto definitions into
// every translation unit that needs only the kernel signatures.
class GraphProto;

namespace core::runtime {

// Forward declaration; full definition lives in run_nodes.h. Kept as a
// forward declaration here (rather than including run_nodes.h) since only
// a reference to an already-built instance is needed by the ``SubgraphSession``-
// aware overloads below.
class SubgraphSession;

// ---------------------------------------------------------------------------
// Reference implementations of the ``controlflow`` backend test kernels.
//
// Each kernel is exposed as a small class whose constructor takes a
// :ref:`KernelContext` (carrying the opset against which the kernel must
// behave) and whose ``operator()`` performs the computation.
//
// Two flavors of ``operator()`` are provided:
//
//   * The returning overload (``Tensor operator()(...) const``) allocates a
//     fresh ``Tensor`` whose data buffer is owned by the returned value.
//   * The in-place overload (``void operator()(..., Tensor &output) const``)
//     writes results into a caller-supplied output tensor whose buffer has
//     already been allocated. The caller is responsible for setting
//     ``output.data_type``, ``output.shape`` and sizing ``output.data`` to
//     match the operator's expected output; the kernel validates these
//     attributes and throws ``std::invalid_argument`` on mismatch.
//
// ``If`` mirrors the ONNX ``If`` operator. Two flavors of selection are
// provided:
//
//   * The "precomputed branches" overloads
//     (``operator()(cond, then_value, else_value [, output])``) consume
//     already-evaluated branch outputs and simply select one based on the
//     scalar BOOL ``cond``. They are convenient for unit tests that do not
//     need a graph executor. The returning overload returns a copy of the
//     selected branch tensor directly, avoiding any extra allocation or
//     explicit buffer copy.
//   * The "branch graphs" overload
//     (``operator()(rt, cond, then_branch, else_branch)``) executes the
//     selected ``GraphProto`` subgraph through a :cpp:class:`onnx_kernels::SubgraphSession`
//     using the caller-provided :cpp:class:`RuntimeContext` and returns the
//     subgraph's outputs in declaration order. The caller's tensors,
//     sequences, and verbosity level are all propagated to the child
//     context so subgraphs can reference outer-scope values.
//     This overload is retained for tests that want to exercise branch
//     selection through the kernel directly; :cpp:func:`RunIfNode` (the
//     runtime dispatcher entry point) additionally supports sequence-typed
//     branch outputs, so it drives the selected branch itself rather than
//     going through this overload.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias one of
// the input tensors' buffers. ``If`` simply copies the selected branch into
// the output, so aliasing with that branch's buffer is permitted.
// ---------------------------------------------------------------------------

/// Selects ``then_value`` when the scalar BOOL ``cond`` is true,
/// otherwise returns ``else_value``. Both branch values must share the
/// same data type and shape.
class If : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &cond, const Tensor &then_value, const Tensor &else_value) const;
  void operator()(const Tensor &cond, const Tensor &then_value, const Tensor &else_value,
                  Tensor &output) const;

  /// Branch-graph overload.
  ///
  /// Selects the ``then_branch`` :cpp:class:`GraphProto` when the scalar
  /// BOOL ``cond`` is true and ``else_branch`` otherwise, then executes
  /// the selected subgraph through a :cpp:class:`onnx_kernels::SubgraphSession`
  /// using ``rt`` (a child :cpp:class:`RuntimeContext` is created
  /// internally so the caller's tensor map is left untouched apart from
  /// being inherited as the outer scope of the subgraph). The child
  /// context also inherits ``rt.sequences()`` and ``rt.verbose()`` so
  /// that outer-scope sequence values and diagnostic verbosity are
  /// visible inside the branch. The returned vector contains the
  /// subgraph outputs in the order declared by ``branch.output()``.
  /// Unlike ``Loop`` / ``Scan``, ``If`` selects and runs a branch exactly
  /// once per call, so the session is not reused across repeated
  /// invocations; only the selected branch's ``GraphProto`` is read (not
  /// retained) while building it.
  ///
  /// @throws std::invalid_argument if ``cond`` is not a BOOL scalar, if a
  ///         declared subgraph output is missing, if ``then_branch`` and
  ///         ``else_branch`` declare a different number of outputs, or if
  ///         any node of the executed subgraph fails to dispatch.
  Tensors operator()(RuntimeContext &rt, const Tensor &cond, const GraphProto &then_branch,
                     const GraphProto &else_branch) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of the ONNX ``Loop`` operator.
///
/// Like :class:`If`, this kernel does not execute the loop ``body`` subgraph
/// itself; it consumes already-evaluated per-iteration values and merely
/// validates and assembles the operator's outputs:
///
///   * each of the ``N`` final loop-carried dependency values is forwarded
///     verbatim from the caller-provided ``final_state`` tensors;
///   * each of the ``K`` scan outputs is built by stacking the
///     caller-provided per-iteration values along a new leading axis whose
///     length equals the actually executed trip count.
///
/// The kernel honors ONNX's termination rules: the effective trip count is
/// ``min(M, len(scan_values_per_iter))`` when ``M`` is provided, otherwise
/// ``len(scan_values_per_iter)``; when ``cond`` is false on entry, the trip
/// count is zero and scan outputs are zero-length along the new axis. The
/// kernel is therefore a faithful reference for the operator's
/// composition/stacking semantics that is useful for shape and
/// type-propagation tests while keeping the implementation independent
/// from any graph executor.
class Loop : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returning overload.
  ///
  /// @param M           Optional INT64 scalar maximum trip-count
  ///                    (``empty Tensor`` means ``omitted``).
  /// @param cond        Optional BOOL scalar initial termination
  ///                    condition (``empty Tensor`` means ``omitted``,
  ///                    treated as ``true``).
  /// @param v_initial   Initial loop-carried dependency values (size N,
  ///                    may be empty). Returned as-is when the loop
  ///                    executes zero iterations.
  /// @param final_state Final loop-carried dependency values (size N).
  ///                    Their data types must match the corresponding
  ///                    ``v_initial`` tensor's data type.
  /// @param scan_values_per_iter Per-iteration scan-output values, given
  ///                    as ``scan_values_per_iter[k][t]`` &mdash; ``K``
  ///                    scan outputs each with one tensor per iteration
  ///                    (rectangular). All entries within a scan-output
  ///                    row must share the same data type and shape.
  /// @return ``N + K`` tensors: the final loop-carried dependency values
  ///         followed by the stacked scan outputs.
  Tensors operator()(const Tensor &M, const Tensor &cond, const Tensors &v_initial,
                     const Tensors &final_state,
                     const std::vector<Tensors> &scan_values_per_iter) const;

  /// Allocator-aware stacking-only overload used by the runtime dispatcher.
  Tensors operator()(RuntimeContext &rt, const Tensor &M, const Tensor &cond,
                     const Tensors &v_initial, const Tensors &final_state,
                     const std::vector<Tensors> &scan_values_per_iter) const;

  /// Body-runner callback executed by the new ``operator()`` overload once
  /// per iteration. ``iter`` is the 0-based iteration index, ``cond_in`` is
  /// the BOOL termination condition entering this iteration, and ``state``
  /// is the current loop-carried state (``v_initial`` for ``iter == 0``,
  /// otherwise the previous iteration's body outputs ``[1..1+N)``). The
  /// callback must return ``1 + N + K`` tensors matching the ONNX ``Loop``
  /// body output convention: the new ``cond_out`` (BOOL scalar) at index 0,
  /// the ``N`` updated loop-carried values at indices ``[1, 1+N)``, and the
  /// ``K`` per-iteration scan values at indices ``[1+N, 1+N+K)``.
  using BodyRunner = std::function<Tensors(int64_t iter, bool cond_in, const Tensors &state)>;

  /// Body-aware overload.
  ///
  /// Performs the full ONNX ``Loop`` semantics: honors ``M`` /
  /// ``cond`` termination rules, invokes ``run_body`` once per
  /// iteration, threads the loop-carried state across iterations,
  /// collects each iteration's scan values, and assembles the
  /// ``N + num_scan_outputs`` outputs (final loop-carried state
  /// followed by stacked scan outputs).
  ///
  /// @param M                 Optional INT64 scalar maximum trip-count
  ///                          (``empty Tensor`` means ``omitted``).
  /// @param cond              Optional BOOL scalar initial termination
  ///                          condition (``empty Tensor`` means
  ///                          ``omitted``, treated as ``true``).
  /// @param v_initial         Initial loop-carried dependency values
  ///                          (size ``N``).
  /// @param num_scan_outputs  Number of per-iteration scan outputs the
  ///                          body produces (``K``).
  /// @param run_body          Callback executed once per iteration; see
  ///                          :type:`BodyRunner`.
  /// @return ``N + num_scan_outputs`` tensors.
  Tensors operator()(const Tensor &M, const Tensor &cond, const Tensors &v_initial,
                     std::size_t num_scan_outputs, const BodyRunner &run_body) const;

  /// Allocator-aware body-aware overload used by the runtime dispatcher.
  Tensors operator()(RuntimeContext &rt, const Tensor &M, const Tensor &cond,
                     const Tensors &v_initial, std::size_t num_scan_outputs,
                     const BodyRunner &run_body) const;

  /// Body-aware overload driven directly by the Loop ``body`` subgraph.
  ///
  /// Performs the full ONNX ``Loop`` semantics like the ``BodyRunner``
  /// overload above, but builds and owns the :cpp:class:`SubgraphSession`
  /// that drives ``body`` itself (mirroring :cpp:class:`Scan`'s body-aware
  /// overload): the session — and the initializers / output names cached
  /// from ``body`` at construction — is built once and reused for every
  /// iteration, so ``body`` itself is only needed for this call, not kept
  /// around afterwards.
  ///
  /// @param rt         Runtime context used to build the body's
  ///                   :cpp:class:`SubgraphSession` and to evaluate it. The
  ///                   body is executed in a fresh child context per
  ///                   iteration so it cannot mutate ``rt.tensors()``.
  /// @param body       The Loop body subgraph. Its first two formal inputs
  ///                   are the iteration number (INT64 scalar) and the
  ///                   incoming condition (BOOL scalar); its next ``N``
  ///                   formal inputs are bound to the current loop-carried
  ///                   state. Its first output is the outgoing condition,
  ///                   its next ``N`` outputs become the next state, and its
  ///                   remaining ``K`` outputs are the per-iteration scan
  ///                   outputs.
  /// @param M          Optional INT64 scalar maximum trip-count (``empty
  ///                   Tensor`` means ``omitted``).
  /// @param cond       Optional BOOL scalar initial termination condition
  ///                   (``empty Tensor`` means ``omitted``, treated as
  ///                   ``true``).
  /// @param v_initial  Initial loop-carried dependency values (size ``N``).
  /// @return ``N + K`` tensors: the final loop-carried dependency values
  ///         followed by the stacked scan outputs.
  Tensors operator()(RuntimeContext &rt, const GraphProto &body, const Tensor &M,
                     const Tensor &cond, const Tensors &v_initial) const;

  /// Body-aware overload driven by an externally-owned :cpp:class:`SubgraphSession`.
  ///
  /// Identical to the ``body``-only overload above except that ``session``
  /// (already built over ``body``, typically once per node by the runtime
  /// dispatcher and cached across every invocation of that node) is reused
  /// as-is instead of being constructed locally, so the body's kernels are
  /// resolved once per node rather than once per call.
  ///
  /// @param session    Already-built :cpp:class:`SubgraphSession` over
  ///                   ``body``. Its lifetime is owned by the caller.
  Tensors operator()(RuntimeContext &rt, const GraphProto &body, SubgraphSession &session,
                     const Tensor &M, const Tensor &cond, const Tensors &v_initial) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``Scan`` operator.
///
/// Two ``operator()`` overloads are provided:
///
///   * a *body-aware* overload that takes the Scan ``body`` subgraph, the
///     initial state and the per-axis scan inputs, executes the body once
///     per iteration through :cpp:class:`onnx_kernels::SubgraphSession`, and
///     returns the operator's full output list. This is the overload used
///     when the kernel is invoked from the runtime dispatcher
///     (:cpp:func:`onnx_kernels::RunNode`);
///   * a *stacking-only* overload that consumes already-evaluated state
///     and per-iteration values and merely assembles the operator's
///     outputs. It is retained for legacy callers and for tests that want
///     to exercise the stacking semantics in isolation, without standing
///     up a runtime context.
///
/// For both overloads:
///
///   * each of the ``N`` final state values is returned as either the
///     final body-produced state (when the trip count is non-zero) or
///     the corresponding ``initial_state`` entry (when the trip count
///     is zero);
///   * each of the ``K`` scan outputs is built by stacking the
///     per-iteration values along a new axis whose position is
///     ``scan_output_axes[k]`` (default 0 &mdash; new leading axis) and
///     whose length equals the trip count. When
///     ``scan_output_directions[k]`` equals ``1`` the per-iteration
///     values are reversed before stacking (prepend semantics).
class Scan : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Body-aware overload.
  ///
  /// Iterates the Scan ``body`` subgraph for the trip count implied by
  /// the scan inputs, threading the state forward across iterations and
  /// collecting the per-iteration scan outputs, then returns the
  /// stacked operator outputs.
  ///
  /// @param body                    The Scan body subgraph. Its first
  ///                                ``N`` formal inputs are bound to the
  ///                                current state, its next ``M`` formal
  ///                                inputs are bound to the per-iteration
  ///                                scan-input slices, its first ``N``
  ///                                outputs become the next state, and its
  ///                                remaining ``K`` outputs are the
  ///                                per-iteration scan outputs.
  /// @param initial_state           Initial state values (size ``N``).
  /// @param scan_inputs             Per-axis scan inputs (size ``M``).
  ///                                Each must have rank at least 1; the
  ///                                axis given by ``scan_input_axes`` is
  ///                                consumed one element at a time and
  ///                                determines the trip count.
  /// @param rt                      Runtime context used to evaluate
  ///                                ``body``. The body is executed in a
  ///                                fresh child context per iteration so
  ///                                it cannot mutate ``rt.tensors()``.
  /// @param scan_input_axes         Per-scan-input axis along which the
  ///                                input is sliced. When empty, axis 0
  ///                                is used for every scan input.
  ///                                Negative values count from the back
  ///                                of the scan input's rank. When
  ///                                non-empty, must have ``M`` entries.
  /// @param scan_input_directions   Per-scan-input direction (0 =
  ///                                forward, 1 = reverse). When empty,
  ///                                all scan inputs use the forward
  ///                                direction. When non-empty, must
  ///                                have ``M`` entries.
  /// @param scan_output_axes        Per-scan-output axis at which the
  ///                                new stacking axis is inserted (see
  ///                                the stacking-only overload).
  /// @param scan_output_directions  Per-scan-output direction (see the
  ///                                stacking-only overload).
  /// @return ``N + K`` tensors: the final state values followed by the
  ///         stacked scan outputs.
  Tensors operator()(RuntimeContext &rt, const GraphProto &body, const Tensors &initial_state,
                     const Tensors &scan_inputs, const ParamInts &scan_input_axes = {},
                     const ParamInts &scan_input_directions = {},
                     const ParamInts &scan_output_axes = {},
                     const ParamInts &scan_output_directions = {}) const;

  /// Body-aware overload driven by an externally-owned :cpp:class:`SubgraphSession`.
  ///
  /// Identical to the ``body``-only overload above except that ``session``
  /// (already built over ``body``, typically once per node by the runtime
  /// dispatcher and cached across every invocation of that node, including
  /// every per-batch call in the Scan-8 legacy path) is reused as-is instead
  /// of being constructed locally, so the body's kernels are resolved once
  /// per node rather than once per call.
  ///
  /// @param session Already-built :cpp:class:`SubgraphSession` over ``body``.
  ///                Its lifetime is owned by the caller.
  Tensors operator()(RuntimeContext &rt, const GraphProto &body, SubgraphSession &session,
                     const Tensors &initial_state, const Tensors &scan_inputs,
                     const ParamInts &scan_input_axes = {},
                     const ParamInts &scan_input_directions = {},
                     const ParamInts &scan_output_axes = {},
                     const ParamInts &scan_output_directions = {}) const;

  /// Allocator-aware stacking-only overload used by the runtime dispatcher.
  Tensors operator()(RuntimeContext &rt, int64_t trip_count, const Tensors &initial_state,
                     const Tensors &final_state, const std::vector<Tensors> &scan_values_per_iter,
                     const ParamInts &scan_output_axes = {},
                     const ParamInts &scan_output_directions = {}) const;

  /// Stacking-only overload.
  ///
  /// @param trip_count               Number of iterations actually executed
  ///                                 (non-negative).
  /// @param initial_state            Initial state values (size N).
  /// @param final_state              Final state values (size N). Their
  ///                                 data types must match the
  ///                                 corresponding ``initial_state``
  ///                                 entry's data type.
  /// @param scan_values_per_iter     Per-iteration scan-output values, given
  ///                                 as ``scan_values_per_iter[k][t]`` — K
  ///                                 scan outputs each with one tensor per
  ///                                 iteration (rectangular). All entries
  ///                                 within a scan-output row must share
  ///                                 the same data type and shape.
  /// @param scan_output_axes         Per-scan-output axis at which the new
  ///                                 stacking axis is inserted. When
  ///                                 empty, axis 0 is used for every scan
  ///                                 output. Negative values count from
  ///                                 the back of the stacked output (with
  ///                                 rank ``elt.rank + 1``). When
  ///                                 non-empty, must have ``K`` entries.
  /// @param scan_output_directions   Per-scan-output direction (0 =
  ///                                 append, 1 = prepend / reverse before
  ///                                 stacking). When empty, all scan
  ///                                 outputs use the append direction.
  ///                                 When non-empty, must have ``K``
  ///                                 entries.
  /// @return ``N + K`` tensors: the final state values followed by the
  ///         stacked scan outputs.
  Tensors operator()(int64_t trip_count, const Tensors &initial_state, const Tensors &final_state,
                     const std::vector<Tensors> &scan_values_per_iter,
                     const ParamInts &scan_output_axes = {},
                     const ParamInts &scan_output_directions = {}) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace core::runtime
} // namespace ONNX_LIGHT_NAMESPACE
