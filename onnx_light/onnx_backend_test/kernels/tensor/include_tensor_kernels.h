// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

#include <string>
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
/// data type specified by ``to`` (a ``DataType`` value,
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

/// Reference implementation of the ONNX ``BitCast`` operator (since
/// opset 26). Reinterprets the bit pattern of the input tensor as the
/// data type ``to`` without value conversion. ``to`` must be a non-string
/// type with the same element bit-width as ``x.data_type``; otherwise the
/// kernel throws ``std::invalid_argument``. Implementations treat the
/// underlying bytes as little endian, which matches the host ABIs
/// targeted by the backend test library.
class BitCast : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, int32_t to) const;
  void operator()(const Tensor &x, int32_t to, Tensor &output) const;

  /// In-place execution is permitted only when ``to == x.data_type``; the
  /// kernel itself does not enforce aliasing constraints so this flag is
  /// conservatively ``false``.
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

/// Reference implementation of the ONNX ``GridSample`` operator (since
/// opset 16 in the ``ai.onnx`` domain; extended to N-D in opset 20).
/// Performs sampling of an input tensor ``X`` at the positions given by
/// the flow field ``grid``.
///
/// Inputs:
///   * ``X``: tensor of shape ``(N, C, D1, D2, ..., Dr)`` with ``r`` spatial
///     dimensions (``r >= 1``).
///   * ``grid``: floating-point tensor of shape
///     ``(N, D1_out, D2_out, ..., Dr_out, r)`` carrying normalised
///     sampling coordinates.
///
/// Attributes:
///   * ``mode`` (string, default ``"linear"`` / ``"bilinear"``):
///     interpolation mode, one of ``"linear"``/``"bilinear"``,
///     ``"nearest"`` or ``"cubic"``/``"bicubic"``.
///   * ``padding_mode`` (string, default ``"zeros"``): one of ``"zeros"``,
///     ``"border"`` or ``"reflection"``.
///   * ``align_corners`` (int, default 0): when 1, the normalised
///     coordinates ``-1`` and ``+1`` refer to the centres of the corner
///     pixels; when 0 they refer to the outer edges.
///
/// Output shape: ``(N, C, D1_out, D2_out, ..., Dr_out)``. The element
/// type follows ``X``; this implementation supports the FLOAT/DOUBLE
/// element types of ``X`` and ``grid``.
class GridSample : public KernelBase {
public:
  /// Attributes carried by the ONNX ``GridSample`` operator.
  struct Attributes {
    std::string mode = "linear";
    std::string padding_mode = "zeros";
    int64_t align_corners = 0;
  };

  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &X, const Tensor &grid, const Attributes &attrs) const;
  void operator()(const Tensor &X, const Tensor &grid, const Attributes &attrs,
                  Tensor &output) const;

  /// Output shape differs from both inputs, so the output cannot share
  /// storage with any input buffer.
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

/// Reshapes ``data`` to the target shape described by the 1-D INT64 ``shape``
/// tensor (ONNX ``Reshape`` operator, since opset 5; with ``allowzero`` input
/// semantics unchanged in newer opsets).
///
/// Output dtype always matches ``data``. The output shape follows ONNX rules:
/// positive values are copied, one ``-1`` is inferred from element count, and
/// ``0`` copies the corresponding input dim unless ``allowzero`` is set.
class Reshape : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &data, const Tensor &shape, int64_t allowzero = 0) const;
  void operator()(const Tensor &data, const Tensor &shape, int64_t allowzero, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Slices ``data`` according to ONNX ``Slice`` semantics (since opset 10+):
/// ``starts`` and ``ends`` are required; ``axes`` and ``steps`` are optional.
///
/// Supports positive and negative steps, negative indices, omitted axes/steps,
/// and clamping behavior aligned with ONNX shape-inference rules.
class Slice : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &data, const Tensor &starts, const Tensor &ends,
                    const Tensor *axes = nullptr, const Tensor *steps = nullptr) const;
  void operator()(const Tensor &data, const Tensor &starts, const Tensor &ends, const Tensor *axes,
                  const Tensor *steps, Tensor &output) const;

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

/// Removes dimensions of size 1 from ``data`` according to ``axes`` (ONNX
/// ``Squeeze`` operator). When ``axes`` is empty, all dimensions with size 1
/// are removed.
class Squeeze : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &data, const std::vector<int64_t> &axes) const;
  void operator()(const Tensor &data, const std::vector<int64_t> &axes, Tensor &output) const;

  /// Rank may change after squeezing.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Inserts dimensions of size 1 into ``data`` at positions given by ``axes``
/// (ONNX ``Unsqueeze`` operator).
class Unsqueeze : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &data, const std::vector<int64_t> &axes) const;
  void operator()(const Tensor &data, const std::vector<int64_t> &axes, Tensor &output) const;

  /// Rank changes after unsqueezing.
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

/// Returns the indices of the elements that are non-zero of the input tensor
/// ``X`` in row-major order, as a 2-D ``INT64`` tensor of shape ``(rank, nnz)``
/// (ONNX ``NonZero`` operator, since opset 9 in the ``ai.onnx`` domain). For
/// scalar input the output shape is ``(0, nnz)`` (mirroring the upstream
/// specification, which differs from NumPy).
///
/// The reference implementation supports the numeric and ``BOOL`` element
/// types in the backend test library — ``FLOAT``, ``DOUBLE``, ``INT8``,
/// ``UINT8``, ``INT16``, ``UINT16``, ``INT32``, ``INT64``, ``UINT32``,
/// ``UINT64`` and ``BOOL``. Other dtypes will cause the kernel to throw
/// ``std::invalid_argument``.
class NonZero : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;

  /// The output has a different dtype (INT64) and a different shape
  /// from the input, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``Gather`` operator (since opset 1 in
/// the ``ai.onnx`` domain). Gathers entries of the ``axis`` dimension of
/// ``data`` indexed by ``indices``, producing an output tensor of rank
/// ``q + (r - 1)`` where ``r = rank(data)`` and ``q = rank(indices)``.
///
/// ``indices`` may be INT32 or INT64; negative values count from the back of
/// the gathered axis. The output dtype always matches ``data``.
class Gather : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &data, const Tensor &indices, int64_t axis = 0) const;
  void operator()(const Tensor &data, const Tensor &indices, int64_t axis, Tensor &output) const;

  /// Output shape differs from input shape in general.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``GatherElements`` operator (since
/// opset 11 in the ``ai.onnx`` domain). ``data`` and ``indices`` must have the
/// same rank ``r`` and the output has the same shape as ``indices``.
///
/// In the 3-D case: ``out[i][j][k] = data[indices[i][j][k]][j][k]`` when
/// ``axis == 0`` (and analogously for other axes). ``indices`` may be INT32 or
/// INT64; negative values count from the back of the gathered axis.
class GatherElements : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &data, const Tensor &indices, int64_t axis = 0) const;
  void operator()(const Tensor &data, const Tensor &indices, int64_t axis, Tensor &output) const;

  /// Output shape matches ``indices`` and differs from ``data`` in general.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``GatherND`` operator (since opset 11
/// in the ``ai.onnx`` domain; ``batch_dims`` attribute added in opset 12).
/// Gathers slices from ``data`` at the index tuples encoded by ``indices`` and
/// produces an output of rank ``q + r - indices_shape[-1] - 1 - b`` where
/// ``q = rank(indices)``, ``r = rank(data)`` and ``b = batch_dims``.
///
/// ``indices`` must be INT64. Negative index values count from the back of the
/// corresponding ``data`` axis.
class GatherND : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &data, const Tensor &indices, int64_t batch_dims = 0) const;
  void operator()(const Tensor &data, const Tensor &indices, int64_t batch_dims,
                  Tensor &output) const;

  /// Output shape differs from both inputs in general.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``DepthToSpace`` operator (since opset
/// 1; ``mode`` attribute added in opset 11; type set extended in opset 13).
/// Rearranges (permutes) data from depth into blocks of spatial data — the
/// inverse of ``SpaceToDepth``. The input must be a 4-D tensor of shape
/// ``(N, C, H, W)`` with ``C`` divisible by ``blocksize * blocksize``. The
/// output has shape ``(N, C/(blocksize*blocksize), H*blocksize, W*blocksize)``.
///
/// ``mode`` is either ``"DCR"`` (default; depth-column-row order) or ``"CRD"``
/// (column-row-depth order); the reference implementation matches the
/// upstream NumPy equivalents in the operator spec.
///
/// The reference implementation supports whole-byte tensor element types
/// supported by :cpp:func:`ElementSize`.
class DepthToSpace : public KernelBase {
public:
  /// Attributes carried by the ONNX ``DepthToSpace`` operator.
  struct Attributes {
    int64_t blocksize = 0;
    std::string mode = "DCR";
  };

  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &input, const Attributes &attrs) const;
  void operator()(const Tensor &input, const Attributes &attrs, Tensor &output) const;

  /// Output shape differs from input shape in general.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
