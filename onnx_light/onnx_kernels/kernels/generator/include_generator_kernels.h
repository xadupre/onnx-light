// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/simple_tensor.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
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

/// Reference implementation of ONNX ``EyeLike`` (since opset 9):
/// returns a matrix with ones on diagonal ``k`` and zeros elsewhere, with
/// output shape copied from the 2-D ``input`` tensor.
class EyeLike : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &input, int64_t k = 0, int32_t dtype = 0) const;
  void operator()(const Tensor &input, int64_t k, int32_t dtype, Tensor &output) const;

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
  /// :cpp:class:`DataType` value); pass ``0`` to keep the
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

/// Reference implementation of the ONNX ``RandomNormal`` operator (since
/// opset 1 in the ``ai.onnx`` domain). Produces an output tensor of the
/// given ``shape`` filled with samples drawn from a normal distribution
/// with mean ``mean`` and standard deviation ``scale``.
///
/// ``RandomNormal`` is non-deterministic; this reference kernel uses the
/// deterministic :cpp:func:`Randn` helper (SplitMix64 + Irwin-Hall) so
/// expected outputs are bit-stable for testing. When ``seed`` is
/// :cpp:var:`kNoSeed` the kernel's default seed is used; otherwise the
/// optional ``seed`` attribute (declared as FLOAT in the schema) is
/// truncated to ``uint64_t`` and forwarded to :cpp:func:`Randn`.
///
/// Supported output dtypes are ``FLOAT`` (default) and ``DOUBLE``; any
/// other value of ``dtype`` triggers ``std::invalid_argument``.
class RandomNormal : public KernelBase {
public:
  using KernelBase::KernelBase;
  /// Produces a ``shape``-shaped tensor of normal samples. ``mean`` and
  /// ``scale`` come from the operator attributes (defaults ``0.0`` /
  /// ``1.0``). ``dtype`` defaults to ``DataType::FLOAT`` to mirror the
  /// schema default. ``seed`` selects the RNG seed; pass ``kNoSeed`` to
  /// use the kernel's default.
  Tensor operator()(const std::vector<int64_t> &shape, double mean = 0.0, double scale = 1.0,
                    int64_t seed = kNoSeed, int32_t dtype = 0) const;
  void operator()(const std::vector<int64_t> &shape, double mean, double scale, int64_t seed,
                  int32_t dtype, Tensor &output) const;

  /// Sentinel meaning ``seed`` attribute is absent.
  static constexpr int64_t kNoSeed = -1;
  /// No input tensor to alias with: in-place semantics are not relevant.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``RandomUniform`` operator (since
/// opset 1 in the ``ai.onnx`` domain). Produces an output tensor of the
/// given ``shape`` filled with samples drawn uniformly from the half-open
/// interval ``[low, high)``.
///
/// Uses the deterministic :cpp:func:`Rand` helper (SplitMix64). Supported
/// output dtypes are ``FLOAT`` (default) and ``DOUBLE``; any other value
/// of ``dtype`` triggers ``std::invalid_argument``.
class RandomUniform : public KernelBase {
public:
  using KernelBase::KernelBase;
  /// Produces a ``shape``-shaped tensor of uniform samples in
  /// ``[low, high)``. ``low`` defaults to ``0.0`` and ``high`` to
  /// ``1.0`` to mirror the schema. ``dtype`` defaults to
  /// ``DataType::FLOAT``. ``seed`` selects the RNG seed; pass ``kNoSeed``
  /// to use the kernel's default.
  Tensor operator()(const std::vector<int64_t> &shape, double low = 0.0, double high = 1.0,
                    int64_t seed = kNoSeed, int32_t dtype = 0) const;
  void operator()(const std::vector<int64_t> &shape, double low, double high, int64_t seed,
                  int32_t dtype, Tensor &output) const;

  /// Sentinel meaning ``seed`` attribute is absent.
  static constexpr int64_t kNoSeed = -1;
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``RandomNormalLike`` operator
/// (since opset 1 in the ``ai.onnx`` domain). Produces an output tensor
/// whose shape matches ``input`` filled with samples drawn from a normal
/// distribution. When ``dtype`` is ``0`` the output dtype equals the
/// input dtype (which must be one of the supported output dtypes);
/// otherwise the value overrides the output dtype.
class RandomNormalLike : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &input, double mean = 0.0, double scale = 1.0,
                    int64_t seed = kNoSeed, int32_t dtype = 0) const;
  void operator()(const Tensor &input, double mean, double scale, int64_t seed, int32_t dtype,
                  Tensor &output) const;

  static constexpr int64_t kNoSeed = -1;
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``RandomUniformLike`` operator
/// (since opset 1 in the ``ai.onnx`` domain). Same semantics as
/// :cpp:class:`RandomUniform` but the output shape is copied from
/// ``input``.
class RandomUniformLike : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &input, double low = 0.0, double high = 1.0,
                    int64_t seed = kNoSeed, int32_t dtype = 0) const;
  void operator()(const Tensor &input, double low, double high, int64_t seed, int32_t dtype,
                  Tensor &output) const;

  static constexpr int64_t kNoSeed = -1;
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``Multinomial`` operator (since
/// opset 7 in the ``ai.onnx`` domain). Draws ``sample_size`` samples from
/// a multinomial distribution per batch row, where the per-row
/// unnormalized log-probabilities are given by the 2-D input tensor of
/// shape ``[batch_size, class_size]``. The output is a 2-D tensor of
/// shape ``[batch_size, sample_size]`` of integer class indices.
///
/// ``Multinomial`` is non-deterministic; this reference implementation
/// uses a ``std::mt19937`` engine seeded either with the value of the
/// optional ``seed`` attribute (interpreted by truncating to
/// ``uint32_t``) or, when the attribute is absent, with a fixed default
/// seed so the kernel is reproducible for testing.
///
/// Supported input dtypes are ``FLOAT``, ``DOUBLE`` and ``FLOAT16``.
/// Supported output dtypes are ``INT32`` (default) and ``INT64``.
class Multinomial : public KernelBase {
public:
  using KernelBase::KernelBase;
  /// Draws ``sample_size`` samples per batch row of ``input``. ``seed`` is
  /// the value of the ``seed`` attribute when present (truncated to
  /// ``uint32_t``); pass ``kNoSeed`` to use the kernel's default seed.
  /// ``dtype`` is the value of the ``dtype`` attribute; pass ``0`` (or
  /// :cpp:enumerator:`DataType::INT32`) to produce ``INT32`` output, or
  /// :cpp:enumerator:`DataType::INT64` for ``INT64`` output.
  Tensor operator()(const Tensor &input, int64_t sample_size = 1, int64_t seed = kNoSeed,
                    int32_t dtype = 0) const;
  void operator()(const Tensor &input, int64_t sample_size, int64_t seed, int32_t dtype,
                  Tensor &output) const;

  /// Sentinel meaning ``seed`` attribute is absent.
  static constexpr int64_t kNoSeed = -1;
  /// The output buffer has a different shape than the input, so storage
  /// cannot generally be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``Range`` operator (since opset 11
/// in the ``ai.onnx`` domain, with opset 27 widening the supported types
/// to include ``FLOAT16``/``BFLOAT16``). Generates a 1-D tensor with
/// values ``[start, start + delta, start + 2*delta, ...]`` up to (but
/// excluding) ``limit``.
///
/// The three inputs ``start``, ``limit`` and ``delta`` must all be
/// scalar tensors of the same element type. Supported element types are
/// ``FLOAT``, ``DOUBLE``, ``INT16``, ``INT32``, ``INT64``, ``FLOAT16``
/// and ``BFLOAT16`` (matching the upstream ``Range`` schema's ``T``
/// type-constraint at opset 27). For ``FLOAT16`` and ``BFLOAT16`` the
/// loop accumulator runs in ``float`` (the ``stash_type = 1`` default
/// introduced at opset 27).
///
/// The number of output elements is
/// ``max(ceil((limit - start) / delta), 0)``. When the result would be
/// empty (e.g. ``start >= limit`` with a positive ``delta``), the kernel
/// returns an empty 1-D tensor of length 0.
class Range : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &start, const Tensor &limit, const Tensor &delta) const;
  void operator()(const Tensor &start, const Tensor &limit, const Tensor &delta,
                  Tensor &output) const;

  /// The output buffer has a different size than the inputs, so storage
  /// cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
