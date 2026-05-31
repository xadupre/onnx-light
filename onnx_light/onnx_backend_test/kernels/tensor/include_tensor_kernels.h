// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace kernel {

// ---------------------------------------------------------------------------
// Reference implementations of the ``tensor`` backend test kernels.
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
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias one of
// the input tensors' buffers. ``Concat`` returns ``false`` because its
// output is strictly larger (along ``axis``) than any single input and
// therefore cannot share storage with one.
// ---------------------------------------------------------------------------

/// Concatenates a list of tensors along ``axis`` (since opset 13). All input
/// tensors must share the same data type and the same shape except along the
/// concatenation axis. ``axis`` may be negative, in which case it counts from
/// the back of the input rank.
class Concat : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const std::vector<Tensor> &inputs, int64_t axis) const;
  void operator()(const std::vector<Tensor> &inputs, int64_t axis, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Performs element-wise type conversion of an input tensor ``x`` to the
/// data type specified by ``to`` (a ``TensorProto::DataType`` value,
/// mirroring the ``Cast`` operator's required ``to`` attribute). The output
/// shape always matches the input shape.
///
/// The reference implementation supports the numeric element types in the
/// backend test library — ``FLOAT``, ``DOUBLE``, ``INT32``, ``INT64``,
/// ``INT8``, ``UINT8``, ``INT16``, ``UINT16`` and ``BOOL`` — as well as
/// ``STRING`` in either direction (numeric ↔ STRING uses the canonical
/// decimal representation). Other dtypes will cause the kernel to throw
/// ``std::invalid_argument``: this is sufficient for the backend test
/// cases registered today and keeps the implementation small.
/// Out-of-range floating-point values when casting to an integer dtype
/// follow C++ ``static_cast`` semantics, which matches the behaviour
/// exercised by the upstream ``test_cast_FLOAT_to_*`` node tests for the
/// supported conversions.
class Cast : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, int32_t to) const;
  void operator()(const Tensor &x, int32_t to, Tensor &output) const;

  /// Output element type may differ from the input element type, so storage
  /// can not be shared in general.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``AffineGrid`` operator (since
/// opset 20 in the ``ai.onnx`` domain). Generates a flow field of
/// sampling coordinates by applying a batch of affine matrices ``theta``
/// to a regular grid of size ``size``.
///
/// Inputs:
///   * ``theta``: FLOAT tensor of shape ``(N, 2, 3)`` for 2D or
///     ``(N, 3, 4)`` for 3D.
///   * ``size``: INT64 1-D tensor of length 4 (``(N, C, H, W)``) for 2D
///     or 5 (``(N, C, D, H, W)``) for 3D. Only the spatial dimensions
///     ``(H, W)`` or ``(D, H, W)`` are used; ``N`` is taken from
///     ``theta`` (and must match ``size[0]``) and ``C`` is ignored.
///
/// Attribute ``align_corners`` (int, default 0): when 1, the normalised
/// coordinates ``-1`` and ``+1`` refer to the centres of the corner
/// pixels; when 0 they refer to the outer edges (the convention matching
/// ``torch.nn.functional.affine_grid``).
///
/// Output shape: ``(N, H, W, 2)`` for 2D or ``(N, D, H, W, 3)`` for 3D.
/// The element type follows the ``theta`` input (FLOAT in this
/// implementation).
class AffineGrid : public KernelBase {
public:
  /// Attributes carried by the ONNX ``AffineGrid`` operator.
  struct Attributes {
    int64_t align_corners = 0;
  };

  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &theta, const Tensor &size, const Attributes &attrs) const;
  void operator()(const Tensor &theta, const Tensor &size, const Attributes &attrs,
                  Tensor &output) const;

  /// Output shape and element layout differ from both inputs, so the
  /// output cannot share storage with any input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Performs element-wise type conversion of an input tensor ``x`` to the
/// data type carried by ``target_type`` (a second tensor whose values are
/// ignored). This mirrors the ONNX ``CastLike`` operator (since opset 15
/// in the ai.onnx domain), which is equivalent to ``Cast`` with
/// ``to = target_type.data_type``.
///
/// The reference implementation forwards to :ref:`kernel::Cast` and so
/// supports the same element-type matrix.
class CastLike : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &target_type) const;
  void operator()(const Tensor &x, const Tensor &target_type, Tensor &output) const;

  /// Output element type may differ from the input element type, so storage
  /// can not be shared in general.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Broadcasts the ``input`` tensor to the shape given by the 1-D INT64
/// ``shape`` tensor, following the ONNX numpy-style broadcasting rules
/// (ONNX ``Expand`` operator, since opset 8 in the ``ai.onnx`` domain).
///
/// The output shape is computed as ``broadcast(input.shape, shape_values)``.
/// A dimension in ``input`` of size 1 is expanded (repeated) to match the
/// corresponding target dimension; a dimension equal to the target is
/// left unchanged. The output dtype always matches the input dtype.
///
/// The reference implementation supports all whole-byte element types
/// supported by :cpp:func:`ElementSize`. String and sub-byte dtypes
/// (INT4/UINT4/INT2/UINT2) are not supported and will cause the kernel
/// to throw ``std::invalid_argument``.
class Expand : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &input, const Tensor &shape) const;
  void operator()(const Tensor &input, const Tensor &shape, Tensor &output) const;

  /// The output may be larger than either input, so storage cannot be
  /// shared in general.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Constructs a tensor by tiling the ``input`` tensor a number of times along
/// each axis given by the 1-D INT64 ``repeats`` tensor (ONNX ``Tile``
/// operator, since opset 6 in the ``ai.onnx`` domain).
///
/// ``repeats`` must have the same length as ``input``'s rank, and every entry
/// must be non-negative. The output has the same rank and dtype as ``input``;
/// its dimension ``i`` is ``input.shape[i] * repeats[i]``.
///
/// The reference implementation supports all whole-byte element types
/// supported by :cpp:func:`ElementSize`. String and sub-byte dtypes
/// (INT4/UINT4/INT2/UINT2) are not supported and will cause the kernel to
/// throw ``std::invalid_argument``.
class Tile : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &input, const Tensor &repeats) const;
  void operator()(const Tensor &input, const Tensor &repeats, Tensor &output) const;

  /// The output is generally larger than the input, so storage cannot be
  /// shared in general.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Permutes the axes of the input tensor according to ``perm`` (ONNX
/// ``Transpose`` operator). When ``perm`` is empty, the axis order is
/// reversed.
///
/// The reference implementation supports whole-byte tensor element types
/// supported by :cpp:func:`ElementSize`.
class Transpose : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &data, const std::vector<int64_t> &perm) const;
  void operator()(const Tensor &data, const std::vector<int64_t> &perm, Tensor &output) const;

  /// Output shape differs from input shape in general.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
