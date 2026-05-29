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
class And {
public:
  explicit And(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise logical OR on BOOL tensors with multidirectional broadcasting.
class Or {
public:
  explicit Or(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise logical XOR on BOOL tensors with multidirectional broadcasting.
class Xor {
public:
  explicit Xor(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise ``Greater`` comparison with multidirectional broadcasting.
/// Inputs may be FLOAT, INT8, INT16, UINT8, UINT16, UINT32 or UINT64 (both
/// inputs must share the same dtype); the output is BOOL (one byte per
/// element, ``0`` or ``1``). Mirrors the upstream ONNX ``Greater`` reference
/// implementation (``np.greater``).
class Greater {
public:
  explicit Greater(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  const KernelContext &ctx_;
};

/// Element-wise ``Less`` comparison with multidirectional broadcasting.
/// Inputs may be FLOAT, INT8, INT16, UINT8, UINT16, UINT32 or UINT64 (both
/// inputs must share the same dtype); the output is BOOL (one byte per
/// element, ``0`` or ``1``). Mirrors the upstream ONNX ``Less`` reference
/// implementation (``np.less``).
class Less {
public:
  explicit Less(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  const KernelContext &ctx_;
};

/// Element-wise ``Equal`` comparison with multidirectional broadcasting.
/// Inputs may be BOOL, FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8,
/// UINT16, UINT32, UINT64 or STRING (both inputs must share the same dtype);
/// the output is BOOL (one byte per element, ``0`` or ``1``). Mirrors the
/// upstream ONNX ``Equal`` reference implementation (``np.equal``). STRING
/// support matches the ``Equal`` opset 19 type expansion and is restricted
/// to equal-shape inputs or scalar broadcasting.
class Equal {
public:
  explicit Equal(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  const KernelContext &ctx_;
};

/// Element-wise bitwise AND on integer tensors with multidirectional
/// broadcasting (opset 18). Inputs may be INT8, INT16, INT32, INT64, UINT8,
/// UINT16, UINT32 or UINT64 (both inputs must share the same dtype); the
/// output has the same dtype. Mirrors the upstream ONNX ``BitwiseAnd``
/// reference implementation (``np.bitwise_and``).
class BitwiseAnd {
public:
  explicit BitwiseAnd(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise bitwise OR on integer tensors with multidirectional
/// broadcasting (opset 18). Inputs may be INT8, INT16, INT32, INT64, UINT8,
/// UINT16, UINT32 or UINT64 (both inputs must share the same dtype); the
/// output has the same dtype. Mirrors the upstream ONNX ``BitwiseOr``
/// reference implementation (``np.bitwise_or``).
class BitwiseOr {
public:
  explicit BitwiseOr(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise bitwise XOR on integer tensors with multidirectional
/// broadcasting (opset 18). Inputs may be INT8, INT16, INT32, INT64, UINT8,
/// UINT16, UINT32 or UINT64 (both inputs must share the same dtype); the
/// output has the same dtype. Mirrors the upstream ONNX ``BitwiseXor``
/// reference implementation (``np.bitwise_xor``).
class BitwiseXor {
public:
  explicit BitwiseXor(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise bitwise NOT on integer tensors (opset 18). Input may be
/// INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32 or UINT64; the output
/// has the same dtype and shape. Mirrors the upstream ONNX ``BitwiseNot``
/// reference implementation (``np.bitwise_not``).
class BitwiseNot {
public:
  explicit BitwiseNot(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
