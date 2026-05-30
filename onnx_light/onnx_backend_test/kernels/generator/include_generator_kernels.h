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
// Reference implementations of the ``generator`` backend test kernels.
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
// ``Constant`` mirrors the ONNX ``Constant`` operator: it has no inputs and
// produces an output tensor whose data type, shape and bytes are taken from
// the operator's ``value`` attribute. The kernel does not parse attributes
// itself — it consumes the already-decoded ``value`` tensor, which keeps
// this reference implementation independent from any attribute-decoding
// machinery while still exercising the operator's value-producing
// semantics.
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias the
// supplied ``value`` tensor's buffer. ``Constant`` simply copies the
// ``value`` bytes into the output, so aliasing with the value buffer is
// permitted.
// ---------------------------------------------------------------------------

/// Returns a copy of the ``value`` attribute of the ``Constant`` op.
class Constant : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &value) const;
  void operator()(const Tensor &value, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of the ONNX ``ConstantOfShape`` operator
/// (since opset 9 in the ``ai.onnx`` domain). Produces an output tensor
/// of the shape given by the 1-D ``int64`` ``shape`` input, filled with
/// the (single) scalar value of the ``value`` tensor.
///
/// When ``value`` is empty (``data_type == 0`` and ``data.empty()``),
/// the output defaults to a ``FLOAT`` tensor filled with zeros, per the
/// schema.
///
/// Supported numeric ``value`` dtypes match the upstream
/// ``test_constantofshape_int_zeros`` / ``test_constantofshape_float_ones``
/// node tests: every fixed-width whole-byte numeric type as well as
/// ``BOOL``. Other dtypes throw ``std::invalid_argument``.
class ConstantOfShape : public KernelBase {
public:
  using KernelBase::KernelBase;
  /// ``shape`` must be a 1-D INT64 tensor whose entries describe the
  /// shape of the output. ``value`` is the (single-element) fill value
  /// taken from the operator's ``value`` attribute; pass a
  /// default-constructed ``Tensor`` to use the schema default
  /// (FLOAT 0.0).
  Tensor operator()(const Tensor &shape, const Tensor &value) const;
  void operator()(const Tensor &shape, const Tensor &value, Tensor &output) const;

  /// The output buffer has a different size than the inputs, so storage
  /// can not generally be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``Bernoulli`` operator (since
/// opset 15 in the ``ai.onnx`` domain). Draws binary samples ``y[i]`` from
/// a Bernoulli distribution with probability ``input[i]`` (a value in
/// ``[0, 1]``); returns ``1`` with probability ``input[i]`` and ``0``
/// otherwise.
///
/// ``Bernoulli`` is non-deterministic; this reference implementation uses
/// a ``std::mt19937`` engine seeded either with the value of the optional
/// ``seed`` attribute (interpreted by truncating to ``uint32_t``) or, when
/// the attribute is absent, with a fixed default seed so the kernel is
/// reproducible for testing. Output dtype is controlled by the optional
/// ``dtype`` attribute: when absent the output element type matches the
/// input; when present it overrides the output element type.
///
/// Supported input dtypes are ``FLOAT``, ``DOUBLE`` and ``FLOAT16``;
/// supported output dtypes are ``FLOAT``, ``DOUBLE``, ``FLOAT16``,
/// ``INT8``, ``INT16``, ``INT32``, ``INT64``, ``UINT8``, ``UINT16``,
/// ``UINT32``, ``UINT64`` and ``BOOL`` (every type for which the produced
/// 0/1 value has a natural representation).
class Bernoulli : public KernelBase {
public:
  using KernelBase::KernelBase;
  /// Draws Bernoulli samples for every element of ``input``. ``seed`` is
  /// the value of the ``seed`` attribute when present (truncated to
  /// ``uint32_t``); pass ``kNoSeed`` to use the kernel's default seed.
  /// ``dtype`` is the value of the ``dtype`` attribute when present (a
  /// :cpp:class:`TensorProto::DataType` value); pass ``0`` to keep the
  /// output dtype identical to ``input.data_type``.
  Tensor operator()(const Tensor &input, int64_t seed = kNoSeed, int32_t dtype = 0) const;
  void operator()(const Tensor &input, int64_t seed, int32_t dtype, Tensor &output) const;

  /// Sentinel value indicating the ``seed`` attribute is absent. Picked
  /// outside the ``uint32_t`` range so any 32-bit seed (including 0) is
  /// representable as a regular value.
  static constexpr int64_t kNoSeed = -1;

  /// The output buffer has the same byte size as the input only when
  /// ``dtype`` is identical to the input dtype, which we do not require;
  /// disable in-place support to keep the contract simple.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
