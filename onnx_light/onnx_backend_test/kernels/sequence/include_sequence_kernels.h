// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_sequence.h"
#include "onnx_backend_test/simple_tensor.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

// ---------------------------------------------------------------------------
// Reference implementations of the ``sequence`` backend test kernels.
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
// ``SequenceConstruct`` mirrors the ONNX ``SequenceConstruct`` operator
// (since opset 11 in the ai.onnx domain): it builds a tensor sequence from
// ``N`` input tensors that share the same element type. Because the
// project's runtime :ref:`Tensor` type does not natively model sequence
// values, the kernel materializes the constructed sequence as a single
// stacked ``Tensor`` whose outer dimension is the sequence length: given
// ``N`` inputs of identical shape ``[d0, ..., dk]`` and element type ``T``,
// the output has shape ``[N, d0, ..., dk]`` and element type ``T``, and its
// byte buffer is the row-major concatenation of the per-element byte
// buffers. The N == 0 case yields a tensor with shape ``[0]`` and an empty
// byte buffer; the element type defaults to ``UNDEFINED`` since it cannot
// be inferred from inputs alone.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias any
// input tensor's buffer. ``SequenceConstruct`` produces an output whose
// layout fundamentally differs from any single input, so aliasing is not
// permitted.
// ---------------------------------------------------------------------------

/// Stacks ``inputs`` along a new outer axis 0 to materialize the constructed
/// tensor sequence. All inputs must share the same ``data_type`` and
/// ``shape``; the output has shape ``[N, *inputs[0].shape]`` and contains
/// the concatenation of the per-input byte buffers.
class SequenceConstruct : public KernelBase {
public:
  using KernelBase::KernelBase;

  Tensor operator()(const std::vector<Tensor> &inputs) const;
  void operator()(const std::vector<Tensor> &inputs, Tensor &output) const;

  /// Sequence-returning overload. Builds an :cpp:struct:`Sequence`
  /// whose ``elem_type`` is the common element type of ``inputs`` (or
  /// ``UNDEFINED`` when ``inputs`` is empty) and whose ``values``
  /// preserves the input order. Unlike the ``Tensor``-returning
  /// overloads, this overload does not stack the inputs into a single
  /// buffer and does not require the inputs to share a common shape.
  Sequence AsSequence(const std::vector<Tensor> &inputs) const;

  /// Output layout is a stacked concatenation of input byte buffers, which
  /// cannot share storage with any single input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Concatenates the tensor elements of an input sequence along ``axis``
/// (when ``new_axis == 0``, the default) or stacks them along a new
/// axis at position ``axis`` (when ``new_axis == 1``), mirroring the
/// ONNX ``ConcatFromSequence`` operator (since opset 11 in the
/// ai.onnx domain). The project's runtime ``Tensor`` type does not
/// natively model sequence values, so this kernel takes the sequence
/// as a ``std::vector<Tensor>`` of identically-typed tensors. All
/// inputs must share the same ``data_type`` and rank, and must agree
/// on every dimension except (for ``new_axis == 0``) the concat axis,
/// which must be equal across inputs (for ``new_axis == 1``) every
/// dimension must match exactly. The output element type is the
/// common input element type; the output shape follows the ONNX
/// schema (rank ``r`` for ``new_axis == 0``, rank ``r + 1`` for
/// ``new_axis == 1``).
class ConcatFromSequence : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Returning overload. ``axis`` follows the ONNX convention and may
  /// be negative; ``new_axis`` must be 0 (concat) or 1 (stack).
  Tensor operator()(const std::vector<Tensor> &inputs, int64_t axis, int64_t new_axis = 0) const;

  /// In-place overload. ``output`` must already carry the expected
  /// ``data_type``, ``shape`` and sufficiently-sized ``data`` buffer.
  void operator()(const std::vector<Tensor> &inputs, int64_t axis, int64_t new_axis,
                  Tensor &output) const;

  /// Output layout fundamentally differs from any single input
  /// (concatenation/stacking along an axis), so aliasing is not
  /// permitted.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Produces a scalar INT64 tensor containing the number of tensor
/// elements in ``input_sequence``. This mirrors ONNX
/// ``SequenceLength`` (since opset 11 in the ai.onnx domain).
class SequenceLength : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// Computes ``Tensor<int64, {}>{ len(input_sequence) }``.
  Tensor operator()(const Sequence &input_sequence) const;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
