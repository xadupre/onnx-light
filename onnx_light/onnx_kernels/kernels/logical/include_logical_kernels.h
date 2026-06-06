// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/kernels/kernel_context.h"
#include "onnx_kernels/simple_tensor.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

// ---------------------------------------------------------------------------
// Reference implementations of the ``logical`` backend test kernels.
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
// And/Or/Xor operate on ``BOOL`` tensors (one byte per element, 0 or 1)
// and support multidirectional broadcasting per the standard NumPy/ONNX
// rules — mirroring the broadcasting behavior exercised elsewhere in the
// backend test library (see ``kernel::Add``).
//
// Each kernel class also exposes a ``static constexpr bool CanRunInPlace()``
// query indicating whether the output tensor's data buffer may alias one of
// the input tensors' buffers. All three element-wise logical kernels here
// support in-place execution (the output element depends only on the
// corresponding input elements at the same index).
// ---------------------------------------------------------------------------

/// Element-wise logical AND on BOOL tensors with multidirectional broadcasting.
class And : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise logical OR on BOOL tensors with multidirectional broadcasting.
class Or : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise logical XOR on BOOL tensors with multidirectional broadcasting.
class Xor : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise logical NOT on a BOOL tensor (opset 1). The output has the
/// same BOOL dtype and shape as the input. Mirrors the upstream ONNX
/// ``Not`` reference implementation (``np.logical_not``).
class Not : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise ``IsNaN``: returns a BOOL tensor with the same shape as the
/// input, where each element is ``true`` iff the corresponding input value
/// is NaN. Mirrors the upstream ONNX ``IsNaN`` reference implementation
/// (``np.isnan``). Only the FLOAT input dtype is supported.
class IsNaN : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Input is FLOAT (4 bytes/elt) and output is BOOL (1 byte/elt), so the
  /// output buffer cannot alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Element-wise ``IsInf``: returns a BOOL tensor with the same shape as the
/// input, where each element is ``true`` iff the corresponding input value
/// is +/- infinity. The two boolean attributes ``detect_positive`` and
/// ``detect_negative`` (both default to 1) toggle whether +inf and -inf
/// are reported respectively. Mirrors the upstream ONNX ``IsInf`` reference
/// implementation (``np.isinf`` / ``np.isposinf`` / ``np.isneginf``). Only
/// the FLOAT input dtype is supported.
class IsInf : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, int64_t detect_positive = 1,
                    int64_t detect_negative = 1) const;
  void operator()(const Tensor &x, int64_t detect_positive, int64_t detect_negative,
                  Tensor &output) const;

  /// Input is FLOAT (4 bytes/elt) and output is BOOL (1 byte/elt), so the
  /// output buffer cannot alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Element-wise ``Greater`` comparison with multidirectional broadcasting.
/// Inputs may be FLOAT, INT8, INT16, UINT8, UINT16, UINT32 or UINT64 (both
/// inputs must share the same dtype); the output is BOOL (one byte per
/// element, ``0`` or ``1``). Mirrors the upstream ONNX ``Greater`` reference
/// implementation (``np.greater``).
class Greater : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Element-wise ``Less`` comparison with multidirectional broadcasting.
/// Inputs may be FLOAT, INT8, INT16, UINT8, UINT16, UINT32 or UINT64 (both
/// inputs must share the same dtype); the output is BOOL (one byte per
/// element, ``0`` or ``1``). Mirrors the upstream ONNX ``Less`` reference
/// implementation (``np.less``).
class Less : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Element-wise ``GreaterOrEqual`` comparison with multidirectional
/// broadcasting. Inputs may be FLOAT, INT8, INT16, UINT8, UINT16, UINT32 or
/// UINT64 (both inputs must share the same dtype); the output is BOOL (one
/// byte per element, ``0`` or ``1``). Mirrors the upstream ONNX
/// ``GreaterOrEqual`` reference implementation (``np.greater_equal``).
class GreaterOrEqual : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Element-wise ``Equal`` comparison with multidirectional broadcasting.
/// Inputs may be BOOL, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8,
/// UINT16, UINT32, UINT64 or STRING (both inputs must share the same dtype);
/// the output is BOOL (one byte per element, ``0`` or ``1``). Mirrors the
/// upstream ONNX ``Equal`` reference implementation (``np.equal``). STRING
/// support matches the ``Equal`` opset 19 type expansion and is restricted
/// to equal-shape inputs or scalar broadcasting.
class Equal : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Element-wise ``Where`` selection with multidirectional broadcasting.
/// ``condition`` must be BOOL; ``x`` and ``y`` must share the same dtype and
/// may be BOOL, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16,
/// UINT32, UINT64 or STRING. Output dtype equals ``x``/``y`` dtype.
class Where : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &condition, const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &condition, const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Element-wise bitwise AND on integer tensors with multidirectional
/// broadcasting (opset 18). Inputs may be INT8, INT16, INT32, INT64, UINT8,
/// UINT16, UINT32 or UINT64 (both inputs must share the same dtype); the
/// output has the same dtype. Mirrors the upstream ONNX ``BitwiseAnd``
/// reference implementation (``np.bitwise_and``).
class BitwiseAnd : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise bitwise OR on integer tensors with multidirectional
/// broadcasting (opset 18). Inputs may be INT8, INT16, INT32, INT64, UINT8,
/// UINT16, UINT32 or UINT64 (both inputs must share the same dtype); the
/// output has the same dtype. Mirrors the upstream ONNX ``BitwiseOr``
/// reference implementation (``np.bitwise_or``).
class BitwiseOr : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise bitwise XOR on integer tensors with multidirectional
/// broadcasting (opset 18). Inputs may be INT8, INT16, INT32, INT64, UINT8,
/// UINT16, UINT32 or UINT64 (both inputs must share the same dtype); the
/// output has the same dtype. Mirrors the upstream ONNX ``BitwiseXor``
/// reference implementation (``np.bitwise_xor``).
class BitwiseXor : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise bitwise NOT on integer tensors (opset 18). Input may be
/// INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32 or UINT64; the output
/// has the same dtype and shape. Mirrors the upstream ONNX ``BitwiseNot``
/// reference implementation (``np.bitwise_not``).
class BitwiseNot : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise ``BitShift`` on unsigned integer tensors (opset 11). Both
/// inputs must share the same dtype (UINT8, UINT16, UINT32 or UINT64); the
/// output has the same dtype. Multidirectional (Numpy-style) broadcasting
/// is supported. The required ``direction`` attribute selects ``"LEFT"`` or
/// ``"RIGHT"`` and is passed to ``operator()``. Mirrors the upstream
/// ``np.left_shift`` / ``np.right_shift`` reference implementations.
class BitShift : public KernelBase {
public:
  /// Direction of the bitwise shift.
  enum class Direction { kLeft, kRight };

  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y, Direction direction) const;
  void operator()(const Tensor &x, const Tensor &y, Direction direction, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
