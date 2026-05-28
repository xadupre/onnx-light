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
// Reference implementations of the ``math`` backend test kernels.
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
// query indicating whether the kernel may be invoked with an output tensor
// whose data buffer aliases one of the input tensors' buffers (i.e. the
// caller is allowed to reuse an input buffer as the output buffer). The
// query returns the operator-level capability; concrete invocations must
// still verify that input and output shapes/types match. Operators whose
// output cannot share storage with any input (different element type, or
// larger output than every input — e.g. ``BlackmanWindow``, ``Concat``)
// return ``false``.
// ---------------------------------------------------------------------------

/// Element-wise absolute value.
class Abs {
public:
  explicit Abs(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise arc cosine: y = acos(x), with x in [-1, 1] and y in [0, pi].
class Acos {
public:
  explicit Acos(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise inverse hyperbolic cosine: y = acosh(x), with x >= 1.
class Acosh {
public:
  explicit Acosh(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise arc sine: y = asin(x), with x in [-1, 1] and y in [-pi/2, pi/2].
class Asin {
public:
  explicit Asin(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise inverse hyperbolic sine: y = asinh(x), defined for all real x.
class Asinh {
public:
  explicit Asinh(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise arc tangent: y = atan(x), defined for all real x.
class Atan {
public:
  explicit Atan(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise inverse hyperbolic tangent: y = atanh(x), with x in (-1, 1).
class Atanh {
public:
  explicit Atanh(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise cosine: y = cos(x), defined for all real x with y in [-1, 1].
class Cos {
public:
  explicit Cos(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise hyperbolic cosine: y = cosh(x), defined for all real x with y >= 1.
class Cosh {
public:
  explicit Cosh(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise addition with NumPy-style broadcasting.
class Add {
public:
  explicit Add(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  /// Element-wise binary kernel: the output buffer may alias an input buffer
  /// when that input is not broadcast-expanded (i.e. its shape equals the
  /// output shape).
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise subtraction with NumPy-style broadcasting.
class Sub {
public:
  explicit Sub(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  /// Element-wise binary kernel: the output buffer may alias an input buffer
  /// when that input is not broadcast-expanded (i.e. its shape equals the
  /// output shape).
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise multiplication with NumPy-style broadcasting.
class Mul {
public:
  explicit Mul(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  /// Element-wise binary kernel: the output buffer may alias an input buffer
  /// when that input is not broadcast-expanded (i.e. its shape equals the
  /// output shape).
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// Element-wise division with NumPy-style broadcasting.
class Div {
public:
  explicit Div(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  /// Element-wise binary kernel: the output buffer may alias an input buffer
  /// when that input is not broadcast-expanded (i.e. its shape equals the
  /// output shape).
  static constexpr bool CanRunInPlace() noexcept { return true; }

private:
  const KernelContext &ctx_;
};

/// BlackmanWindow function evaluated at ``size`` integer samples. When
/// ``periodic`` is true the window is computed as if of length ``size+1`` and
/// the last sample is discarded (matches NumPy/ONNX conventions).
class BlackmanWindow {
public:
  explicit BlackmanWindow(const KernelContext &ctx) : ctx_(ctx) {}
  Tensor operator()(const Tensor &size, bool periodic = true) const;
  void operator()(const Tensor &size, bool periodic, Tensor &output) const;

  /// Output is a float vector while the input is an int scalar: storage
  /// can never be shared with an input.
  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  const KernelContext &ctx_;
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
