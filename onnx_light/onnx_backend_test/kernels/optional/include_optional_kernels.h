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
class Optional {
public:
  explicit Optional(const KernelContext &ctx) : ctx_(ctx) {}

  Tensor operator()(const Tensor &input) const;
  void operator()(const Tensor &input, Tensor &output) const;

  /// Output is a byte-for-byte copy of ``input``, so storage may safely be
  /// shared with the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
