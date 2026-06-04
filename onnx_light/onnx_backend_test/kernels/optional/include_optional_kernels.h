// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_sequence.h"
#include "onnx_backend_test/simple_tensor.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

// ---------------------------------------------------------------------------
// Reference implementations of the ``optional`` backend test kernels.
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
// ``Optional`` mirrors the ONNX ``Optional`` operator (since opset 15 in the
// ai.onnx domain) restricted to the "tensor element, present" case: it
// wraps a single tensor ``input`` into an optional-of-tensor value. Because
// the project's runtime :ref:`Tensor` type does not model optional/sequence
// values, the kernel implements the present case as a passthrough: the
// output's element type, shape and bytes are an exact copy of ``input``.
// The ``type`` attribute that ONNX requires on the ``Optional`` node is
// structural and is set by the test-case builder; the kernel itself does not
// consume attributes.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias the
// input tensor's buffer. ``Optional`` simply copies the input bytes into the
// output, so aliasing the input buffer is permitted.
// ---------------------------------------------------------------------------

/// Wraps a single tensor ``input`` into an optional-of-tensor value. The
/// output's element type, shape and bytes are an exact copy of ``input``.
class Optional : public KernelBase {
public:
  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &input) const;
  void operator()(const Tensor &input, Tensor &output) const;

  /// Output is a byte-for-byte copy of ``input``, so storage may safely be
  /// shared with the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of the ONNX ``OptionalGetElement`` operator
/// (since opset 15 in the ai.onnx domain). It extracts the element from
/// an optional-type input and returns it. Because the project's runtime
/// :ref:`Tensor` type does not model optional values, the kernel treats
/// the input as the "present" element and behaves as a passthrough:
///
///   * the tensor overload returns a byte-for-byte copy of the input
///     tensor (same ``data_type`` and ``shape``);
///   * the sequence overload returns a copy of the input sequence.
///
/// Since opset 18, ``OptionalGetElement`` also accepts non-optional
/// tensor or sequence inputs as a no-op; the same passthrough behavior
/// is correct in both cases. The kernel does not consume attributes.
class OptionalGetElement : public KernelBase {
public:
  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &input) const;
  void operator()(const Tensor &input, Tensor &output) const;
  Sequence operator()(const Sequence &input) const;

  /// Output is a byte-for-byte copy of ``input``, so storage may safely be
  /// shared with the input buffer when both are tensors.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of the ONNX ``OptionalHasElement`` operator
/// (since opset 15 in the ai.onnx domain). Returns a scalar boolean
/// tensor: ``true`` when the input contains an element, ``false`` when
/// the input is an empty optional or (since opset 18) when no input is
/// provided.
///
/// Because the project's runtime :ref:`Tensor` and :ref:`Sequence`
/// types do not model "empty optional" values, this kernel treats any
/// concrete tensor or sequence input as containing an element (returns
/// ``true``). The zero-input overload returns ``false`` and mirrors the
/// opset-18 behavior where the input is omitted.
class OptionalHasElement : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Tensor input: always returns ``Tensor<bool, {}>{true}`` (the input
  /// is assumed to be the present element).
  Tensor operator()(const Tensor &input) const;
  /// Sequence input: always returns ``Tensor<bool, {}>{true}``.
  Tensor operator()(const Sequence &input) const;
  /// No input (opset 18 only): returns ``Tensor<bool, {}>{false}``.
  Tensor operator()() const;

  /// Output is a fresh scalar bool tensor unrelated to any input
  /// storage; aliasing is not permitted.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
