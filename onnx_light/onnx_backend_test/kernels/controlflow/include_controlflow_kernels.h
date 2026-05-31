// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

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
// ``If`` mirrors the ONNX ``If`` operator: it selects one of two precomputed
// branch values based on a scalar BOOL condition. The kernel does not
// execute the branch subgraphs itself — it consumes their already-evaluated
// outputs, which keeps this reference implementation independent from any
// graph-executor machinery while still exercising the operator's selection
// semantics.
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
  std::vector<Tensor>
  operator()(const Tensor &M, const Tensor &cond, const std::vector<Tensor> &v_initial,
             const std::vector<Tensor> &final_state,
             const std::vector<std::vector<Tensor>> &scan_values_per_iter) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
