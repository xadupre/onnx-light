// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

#include <optional>
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

/// Splits ``input`` along ``axis`` into a list of tensors (ONNX ``Split``
/// operator). When ``split`` is empty the input dimension on ``axis`` is
/// divided into ``num_outputs`` chunks of equal size (the last chunk being
/// smaller when the dimension is not evenly divisible). When ``split`` is
/// non-empty its entries give the size of each output along ``axis`` and
/// must sum to ``input.shape[axis]``. ``axis`` may be negative, in which
/// case it counts from the back of ``input``'s rank.
///
/// The reference implementation supports all whole-byte element types
/// supported by :cpp:func:`ElementSize`. STRING and sub-byte element types
/// are not supported and will cause the kernel to throw
/// ``std::invalid_argument``.
class Split : public KernelBase {
public:
  using KernelBase::KernelBase;
  /// Computes the split. Exactly one of ``split`` (non-empty) or
  /// ``num_outputs`` (> 0) must be provided.
  std::vector<Tensor> operator()(const Tensor &input, int64_t axis,
                                 const std::vector<int64_t> &split, int64_t num_outputs) const;

  /// Output buffers are strict subsets of the input and have a different
  /// shape, so storage can not generally be shared.
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

/// Reference implementation of the ONNX ``Shape`` operator (since opset 1 in
/// the ``ai.onnx`` domain; extended with ``start``/``end`` attributes in
/// opset 15). Returns the shape of the input tensor as an ``INT64`` 1-D
/// tensor.
///
/// Attributes ``start`` (int, default 0) and ``end`` (int, optional) bound
/// the slice ``input.shape[start:end]`` (using numpy-style indexing).
/// Negative values count from the back; out-of-range values are clamped to
/// ``[0, r]`` where ``r`` is the rank of the input. When ``start > end``
/// (after normalisation) the output is empty.
///
/// The kernel reads only the input shape, never its data buffer, so it
/// accepts an input of any element type.
class Shape : public KernelBase {
public:
  /// Attributes carried by the ONNX ``Shape`` operator.
  struct Attributes {
    int64_t start = 0;
    /// ``std::nullopt`` means "use the input rank" (no slicing on the right).
    std::optional<int64_t> end = std::nullopt;
  };

  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &data) const;
  Tensor operator()(const Tensor &data, const Attributes &attrs) const;
  void operator()(const Tensor &data, const Attributes &attrs, Tensor &output) const;

  /// Output has a different dtype (INT64) and shape from the input, so
  /// storage cannot be shared.
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

/// Reference implementation of the ONNX ``TensorScatter`` operator (since
/// opset 24 in the ``ai.onnx`` domain). Performs a functional update of a
/// KV-cache style tensor by writing slices of ``update`` into a copy of
/// ``past_cache`` along the ``axis`` (sequence) dimension, starting at the
/// per-batch positions given by ``write_indices``.
///
/// Inputs:
///   * ``past_cache``: tensor of rank ``r >= 2`` with shape
///     ``(batch_size, D1, ..., max_sequence_length, ..., Dn)``.
///   * ``update``: tensor of the same dtype and rank as ``past_cache``;
///     the only dimension that may differ is the ``axis`` (sequence)
///     dimension which holds ``sequence_length``.
///   * ``write_indices``: optional 1-D ``INT64`` tensor of length
///     ``batch_size``. When omitted (``nullptr``) it defaults to all zeros.
///
/// Attributes ``axis`` (int, default ``-2``) and ``mode`` (string,
/// default ``"linear"``): ``axis`` selects the sequence dimension and
/// cannot be the batch dimension (``0``). When ``mode`` is ``"linear"``
/// every ``write_indices[batch] + sequence_idx`` must remain in
/// ``[0, max_sequence_length)``; when ``mode`` is ``"circular"`` the
/// write index wraps around modulo ``max_sequence_length``.
///
/// Output dtype and shape always match ``past_cache``.
class TensorScatter : public KernelBase {
public:
  /// Attributes carried by the ONNX ``TensorScatter`` operator.
  struct Attributes {
    int64_t axis = -2;
    std::string mode = "linear";
  };

  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &past_cache, const Tensor &update, const Tensor *write_indices,
                    const Attributes &attrs) const;
  void operator()(const Tensor &past_cache, const Tensor &update, const Tensor *write_indices,
                  const Attributes &attrs, Tensor &output) const;

  /// Output shape matches ``past_cache`` so the output buffer could share
  /// storage with the first input; the reference implementation always
  /// writes into a freshly allocated buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of the ONNX ``Compress`` operator (since opset 9
/// in the ``ai.onnx`` domain; axis may be negative since opset 11). Selects
/// slices from ``input`` along a given ``axis`` where the corresponding entry
/// of ``condition`` (a rank-1 BOOL tensor) is ``true``. When ``axis`` is not
/// supplied the input is first flattened and individual elements are selected;
/// the output is then a 1-D tensor.
///
/// The ``condition`` length may be shorter than the input size along the axis
/// (or the flattened size when no axis is given); excess slices are discarded.
///
/// The output dtype always matches ``input``. The selected count is a runtime
/// value and is therefore unknown at shape-inference time.
class Compress : public KernelBase {
public:
  using KernelBase::KernelBase;
  /// ``axis`` is an ``std::optional<int64_t>``: pass ``std::nullopt`` to
  /// compress the flattened input, or the axis index to compress along.
  Tensor operator()(const Tensor &input, const Tensor &condition,
                    std::optional<int64_t> axis) const;
  void operator()(const Tensor &input, const Tensor &condition, std::optional<int64_t> axis,
                  Tensor &output) const;

  /// Output size is data-dependent and cannot be inferred without evaluating
  /// ``condition`` at runtime.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``Trilu`` operator (since opset 14
/// in the ``ai.onnx`` domain). Returns the upper (``upper == 1``, default) or
/// lower (``upper == 0``) triangular part of the input tensor, keeping the
/// elements on/above (resp. on/below) the ``k``-th diagonal and zeroing the
/// others.
///
/// Inputs:
///   * ``input``: tensor of rank ``>= 2``. The last two dimensions
///     ``(N, M)`` are interpreted as a (batch of) matrix; any leading
///     dimensions are treated as batch dimensions and processed
///     independently.
///   * ``k``: optional ``INT64`` scalar (0-D tensor) selecting the
///     diagonal to keep/exclude. Defaults to ``0`` when omitted (signalled
///     by passing ``nullptr``).
///
/// Attribute ``upper`` (int, default 1): when 1 the upper triangular part
/// is retained; when 0 the lower triangular part is retained.
///
/// Output shape and dtype always match ``input``. Elements outside the
/// selected triangular region are set to zero (for ``BOOL`` to ``false``;
/// for ``STRING`` to the empty string).
class Trilu : public KernelBase {
public:
  /// Attributes carried by the ONNX ``Trilu`` operator.
  struct Attributes {
    int64_t upper = 1;
  };

  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &input, const Tensor *k, const Attributes &attrs) const;
  void operator()(const Tensor &input, const Tensor *k, const Attributes &attrs,
                  Tensor &output) const;

  /// Output shape matches input shape so the output buffer could in theory
  /// share storage with the input; the reference implementation does not
  /// rely on that and always writes into a freshly allocated buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of the ONNX ``ReverseSequence`` operator (since
/// opset 10 in the ``ai.onnx`` domain). Reverses, for each batch slice ``i``,
/// the first ``sequence_lens[i]`` elements along the time axis. Elements past
/// that prefix are copied unchanged.
///
/// Inputs:
///   * ``input``: tensor of rank ``>= 2``. The ``time_axis`` and
///     ``batch_axis`` attributes (each one of ``0`` or ``1``) select which
///     of the first two dimensions plays the time / batch role. Any
///     remaining dimensions are treated as inner (feature) dimensions and
///     copied unchanged.
///   * ``sequence_lens``: rank-1 ``INT64`` tensor of length
///     ``input.shape[batch_axis]``. Each entry must satisfy
///     ``0 <= sequence_lens[i] <= input.shape[time_axis]``.
///
/// Attributes ``time_axis`` (int, default ``0``) and ``batch_axis``
/// (int, default ``1``): must be ``0`` or ``1`` and must differ.
///
/// Output shape and dtype always match ``input``.
class ReverseSequence : public KernelBase {
public:
  /// Attributes carried by the ONNX ``ReverseSequence`` operator.
  struct Attributes {
    int64_t time_axis = 0;
    int64_t batch_axis = 1;
  };

  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &input, const Tensor &sequence_lens,
                    const Attributes &attrs) const;
  void operator()(const Tensor &input, const Tensor &sequence_lens, const Attributes &attrs,
                  Tensor &output) const;

  /// Output shape matches input shape so the kernel can in theory share
  /// storage with the input; the reference implementation always writes into
  /// a freshly allocated buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
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

/// Reference implementation of the (deprecated) ONNX ``Upsample`` operator
/// (since opset 1 in the ``ai.onnx`` domain, last refreshed at opset 10 and
/// replaced by ``Resize`` from opset 10 onwards).
///
/// The kernel takes the input tensor ``X`` and a 1-D ``FLOAT`` ``scales``
/// tensor of length ``rank(X)`` and produces a tensor whose dim ``i`` is
/// ``floor(X.shape[i] * scales[i])``. The ``mode`` attribute selects the
/// interpolation rule:
///
///   * ``"nearest"`` (default) — every output element is copied from the
///     nearest input element using ``floor(out / scale)`` to map output
///     coordinates back to input coordinates. Supports any input rank.
///   * ``"linear"`` / ``"bilinear"`` — supported only for 4-D NCHW input,
///     with scales equal to ``1`` on the ``N`` and ``C`` axes. The two
///     spatial axes are upsampled with the "asymmetric" coordinate
///     transformation used by Upsample v7/9/10 in upstream ONNX.
///
/// The reference implementation supports the same whole-byte element
/// types as :cpp:func:`ElementSize` for ``"nearest"`` mode; ``"linear"``
/// mode requires a floating-point input (``FLOAT16``/``FLOAT``/``DOUBLE``).
class Upsample : public KernelBase {
public:
  /// Attributes carried by the ONNX ``Upsample`` operator.
  struct Attributes {
    std::string mode = "nearest";
  };

  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &X, const Tensor &scales, const Attributes &attrs) const;
  void operator()(const Tensor &X, const Tensor &scales, const Attributes &attrs,
                  Tensor &output) const;

  /// Output shape generally differs from input shape, so storage cannot be
  /// shared with the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``Unique`` operator (since opset 11
/// in the ``ai.onnx`` domain). Returns the unique values or subtensors of
/// the input tensor and three optional companion outputs:
///
///   * ``Y``               — unique values (1-D when ``axis`` is not provided)
///                           or unique subtensors sliced along ``axis``;
///   * ``indices``         — 1-D INT64 indices of the first occurrence of
///                           every ``Y`` element in ``X``;
///   * ``inverse_indices`` — 1-D INT64 indices that map every ``X`` element
///                           back to its position in ``Y``;
///   * ``counts``          — 1-D INT64 count of each ``Y`` element in ``X``.
///
/// When ``sorted`` is ``true`` (the default) outputs are ordered by ascending
/// value (or lexicographic order when ``axis`` is provided). When ``sorted``
/// is ``false`` the order of first occurrence in ``X`` is preserved.
///
/// The reference implementation supports the same whole-byte element types
/// as :cpp:func:`ElementSize` (FLOAT, DOUBLE, INT8/16/32/64, UINT8/16/32/64,
/// BOOL) and STRING.
class Unique : public KernelBase {
public:
  /// Attributes carried by the ONNX ``Unique`` operator.
  struct Attributes {
    /// Whether to sort unique elements in ascending order. Defaults to
    /// ``true`` (matching the schema default of 1).
    bool sorted = true;
    /// Axis along which to take unique subtensors. ``std::nullopt`` means
    /// "flatten the input" (matching the schema default behaviour).
    std::optional<int64_t> axis = std::nullopt;
  };

  /// Aggregated output of :cpp:class:`Unique`.
  struct Outputs {
    Tensor y;
    Tensor indices;
    Tensor inverse_indices;
    Tensor counts;
  };

  using KernelBase::KernelBase;

  Outputs operator()(const Tensor &x) const;
  Outputs operator()(const Tensor &x, const Attributes &attrs) const;

  /// The outputs have different dtypes (Y matches X; the others are INT64)
  /// and different shapes, so storage cannot be shared with the input.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
