// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/simple_tensor.h"

#include <string>
#include <utility>
#include <vector>

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
class Abs : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise arc cosine: y = acos(x), with x in [-1, 1] and y in [0, pi].
class Acos : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise inverse hyperbolic cosine: y = acosh(x), with x >= 1.
class Acosh : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise arc sine: y = asin(x), with x in [-1, 1] and y in [-pi/2, pi/2].
class Asin : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise inverse hyperbolic sine: y = asinh(x), defined for all real x.
class Asinh : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise arc tangent: y = atan(x), defined for all real x.
class Atan : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise inverse hyperbolic tangent: y = atanh(x), with x in (-1, 1).
class Atanh : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise cosine: y = cos(x), defined for all real x with y in [-1, 1].
class Cos : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise hyperbolic cosine: y = cosh(x), defined for all real x with y >= 1.
class Cosh : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Determinant of a square matrix or batches of square matrices.
///
/// The input must have shape ``[*, M, M]`` (rank >= 2) with the inner-most
/// two dimensions forming square matrices. The output has shape ``[*]``
/// (the leading batch dimensions); when the input is 2-D the output is a
/// scalar (rank-0). Both input and output are FLOAT tensors.
class Det : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// The output buffer is smaller than the input (drops the trailing two
  /// dimensions); storage can never be shared with the input.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Element-wise ceiling: y = ceil(x), the smallest integer >= x.
class Ceil : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise floor: y = floor(x), the largest integer <= x.
class Floor : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise clip: y = min(max(x, min_val), max_val) (since opset 6).
///
/// Both ``min`` and ``max`` are optional scalar tensors (since opset 11);
/// when omitted, the corresponding bound defaults to
/// ``std::numeric_limits<T>::lowest()`` and ``std::numeric_limits<T>::max()``.
/// When ``min > max`` the operator returns ``max`` for every element,
/// matching the ONNX specification (``Min(max, Max(input, min))``).
class Clip : public KernelBase {
public:
  using KernelBase::KernelBase;
  /// Computes ``y = clip(x, min, max)``. ``min``/``max`` may be ``nullptr``
  /// to use the dtype-specific default bound. When provided, each must be a
  /// 0-D (scalar) tensor whose dtype matches ``x``.
  Tensor operator()(const Tensor &x, const Tensor *min = nullptr,
                    const Tensor *max = nullptr) const;
  void operator()(const Tensor &x, const Tensor *min, const Tensor *max, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise round to nearest integer, ties to even (banker's rounding).
class Round : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise exponential: y = exp(x), defined for all real x.
class Exp : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise error function: y = erf(x), defined for all real x.
class Erf : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise natural logarithm: y = log(x), with x > 0.
class Log : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise square root: y = sqrt(x), with x >= 0 (NaN otherwise).
class Sqrt : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise logistic sigmoid: y = 1 / (1 + exp(-x)).
class Sigmoid : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise sine: y = sin(x), defined for all real x with y in [-1, 1].
class Sin : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Softmax normalized exponential along a selected axis.
class Softmax : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, int64_t axis) const;
  void operator()(const Tensor &x, int64_t axis, Tensor &output) const;

  /// Softmax needs the full input slice to compute each output value; aliasing
  /// input/output would overwrite values needed for later positions.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Element-wise softplus activation: y = ln(1 + exp(x)).
class Softplus : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise softsign activation: y = x / (1 + |x|).
class Softsign : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// SoftmaxCrossEntropyLoss computes the cross-entropy loss between the
/// (un-normalized) softmax distribution of ``scores`` and integer class
/// indices given by ``labels``. Optionally supports per-class weights and
/// an ``ignore_index`` value used to mask labels that should not contribute
/// to the loss. The kernel always returns the loss tensor and the
/// log-probability tensor (same shape and dtype as ``scores``); the caller
/// is free to ignore ``log_prob`` when the node does not request it.
class SoftmaxCrossEntropyLoss : public KernelBase {
public:
  using KernelBase::KernelBase;
  /// @param scores Input scores of shape ``(N, C)`` or ``(N, C, D1, ..., Dk)``.
  /// @param labels Integer class indices of shape ``(N)`` or ``(N, D1, ..., Dk)``.
  /// @param weights Optional rank-1 tensor of length ``C``; ``nullptr`` means unweighted.
  /// @param reduction One of ``"none"``, ``"sum"``, or ``"mean"``.
  /// @param has_ignore_index Whether ``ignore_index`` is set.
  /// @param ignore_index Class index to ignore (only used when ``has_ignore_index`` is true).
  /// @return Pair ``(loss, log_prob)`` with the loss tensor (shape per ``reduction``) and the
  ///         log-probability tensor (same shape and dtype as ``scores``).
  std::pair<Tensor, Tensor> operator()(const Tensor &scores, const Tensor &labels,
                                       const Tensor *weights, const std::string &reduction,
                                       bool has_ignore_index, int64_t ignore_index) const;

  /// The kernel allocates new outputs; it does not support input/output aliasing.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Element-wise hyperbolic sine: y = sinh(x), defined for all real x.
class Sinh : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise tangent: y = tan(x); undefined at x = (2k+1) * pi/2.
class Tan : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise hyperbolic tangent: y = tanh(x), with y in (-1, 1).
class Tanh : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x) const;
  void operator()(const Tensor &x, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise Swish activation: y = x * sigmoid(alpha * x). ``alpha``
/// defaults to 1.0 to match the ONNX schema (opset 24).
class Swish : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, float alpha = 1.0f) const;
  void operator()(const Tensor &x, float alpha, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise thresholded rectified linear unit: y = x for x > alpha, y = 0
/// otherwise. ``alpha`` defaults to 1.0 to match the ONNX schema.
class ThresholdedRelu : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, float alpha = 1.0f) const;
  void operator()(const Tensor &x, float alpha, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise scaled exponential linear unit:
/// ``y = gamma * x`` for ``x > 0`` and
/// ``y = gamma * (alpha * exp(x) - alpha)`` for ``x <= 0``.
/// ``alpha`` and ``gamma`` default to the ONNX schema defaults
/// (~1.6732632 and ~1.0507009 respectively).
class Selu : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, float alpha = 1.67326319217681884765625f,
                    float gamma = 1.05070102214813232421875f) const;
  void operator()(const Tensor &x, float alpha, float gamma, Tensor &output) const;

  /// Element-wise unary kernel: the output buffer may alias the input buffer.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise addition with NumPy-style broadcasting.
class Add : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  /// Element-wise binary kernel: the output buffer may alias an input buffer
  /// when that input is not broadcast-expanded (i.e. its shape equals the
  /// output shape).
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise subtraction with NumPy-style broadcasting.
class Sub : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  /// Element-wise binary kernel: the output buffer may alias an input buffer
  /// when that input is not broadcast-expanded (i.e. its shape equals the
  /// output shape).
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise multiplication with NumPy-style broadcasting.
class Mul : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  /// Element-wise binary kernel: the output buffer may alias an input buffer
  /// when that input is not broadcast-expanded (i.e. its shape equals the
  /// output shape).
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise parametric ReLU: ``y = x`` when ``x >= 0`` and
/// ``y = slope * x`` otherwise. The ``slope`` tensor is unidirectionally
/// broadcastable to ``x`` (the standard NumPy-style multidirectional
/// broadcasting rules apply here as well — the unidirectional case is a
/// strict subset). Float-input semantics preserve ``+inf``/``-inf`` because
/// the kernel branches on the sign of ``x`` rather than evaluating
/// ``max(0, x) + slope * min(0, x)`` (which would yield ``NaN`` for
/// infinite inputs; see microsoft/onnxruntime#28732).
class PRelu : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &slope) const;
  void operator()(const Tensor &x, const Tensor &slope, Tensor &output) const;

  /// Element-wise binary kernel: the output buffer may alias an input buffer
  /// when that input is not broadcast-expanded (i.e. its shape equals the
  /// output shape).
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise division with NumPy-style broadcasting.
class Div : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  /// Element-wise binary kernel: the output buffer may alias an input buffer
  /// when that input is not broadcast-expanded (i.e. its shape equals the
  /// output shape).
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise modulo with NumPy-style broadcasting. The ``fmod`` flag
/// controls the semantics:
///   * ``fmod == 0`` (default): integer modulo whose sign follows the divisor
///     (Python ``%`` / ``numpy.mod``). Only valid for integer dtypes.
///   * ``fmod == 1``: C ``fmod`` semantics whose sign follows the dividend.
///     Required when either input is floating point and also accepted for
///     integer inputs (where it coincides with C ``%`` truncated modulo).
class Mod : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y, int64_t fmod = 0) const;
  void operator()(const Tensor &x, const Tensor &y, int64_t fmod, Tensor &output) const;

  /// Element-wise binary kernel: the output buffer may alias an input buffer
  /// when that input is not broadcast-expanded (i.e. its shape equals the
  /// output shape).
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Element-wise exponentiation ``z = x ^ y`` with NumPy-style multidirectional
/// broadcasting (since opset 7). Unlike most binary element-wise kernels,
/// ``Pow`` allows ``x`` and ``y`` to have different dtypes: the output dtype
/// always matches the dtype of the base ``x`` (per the ONNX schema, ``Z`` is
/// constrained to ``T`` while ``Y`` is constrained to the broader ``T1``).
///
/// Supported base dtypes (``T``): FLOAT, INT32, INT64.
/// Supported exponent dtypes (``T1``): FLOAT, INT32, INT64, UINT32, UINT64.
/// Integer base / integer exponent pairs evaluate the power in ``double``
/// precision and cast the result back to the base dtype, matching the
/// reference behaviour of NumPy's ``numpy.power`` and the upstream ONNX
/// backend test cases.
class Pow : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &y) const;
  void operator()(const Tensor &x, const Tensor &y, Tensor &output) const;

  /// Element-wise binary kernel: the output buffer may alias the base input
  /// buffer when that input is not broadcast-expanded (i.e. its shape equals
  /// the output shape). Aliasing the exponent is generally not safe because
  /// it may have a different dtype than the output.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// BlackmanWindow function evaluated at ``size`` integer samples. When
/// ``periodic`` is true the window is computed as if of length ``size+1`` and
/// the last sample is discarded (matches NumPy/ONNX conventions).
class BlackmanWindow : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &size, bool periodic = true) const;
  void operator()(const Tensor &size, bool periodic, Tensor &output) const;

  /// Output is a float vector while the input is an int scalar: storage
  /// can never be shared with an input.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// HannWindow function evaluated at ``size`` integer samples. When
/// ``periodic`` is true the window is computed as if of length ``size+1`` and
/// the last sample is discarded (matches NumPy/ONNX conventions).
class HannWindow : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &size, bool periodic = true) const;
  void operator()(const Tensor &size, bool periodic, Tensor &output) const;

  /// Output is a float vector while the input is an int scalar: storage
  /// can never be shared with an input.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// HammingWindow function evaluated at ``size`` integer samples. When
/// ``periodic`` is true the window is computed as if of length ``size+1`` and
/// the last sample is discarded (matches NumPy/ONNX conventions).
class HammingWindow : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &size, bool periodic = true) const;
  void operator()(const Tensor &size, bool periodic, Tensor &output) const;

  /// Output is a float vector while the input is an int scalar: storage
  /// can never be shared with an input.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// General matrix multiplication: Y = alpha * op(A) * op(B) + beta * C.
/// ``transA``/``transB`` control whether A and B are transposed (0 = no,
/// non-zero = yes); ``alpha`` and ``beta`` are scalar multipliers.
/// When ``c`` is ``nullptr`` the bias term is omitted (treated as zero).
class Gemm : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &a, const Tensor &b, const Tensor *c, float alpha, float beta,
                    int64_t transA, int64_t transB) const;
  void operator()(const Tensor &a, const Tensor &b, const Tensor *c, float alpha, float beta,
                  int64_t transA, int64_t transB, Tensor &output) const;

  /// Gemm produces a new matrix that cannot alias any input.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Matrix product that behaves like NumPy/ONNX ``matmul``.
///
/// Supports rank-1 and rank-N inputs:
/// - rank-1 x rank-1 -> scalar
/// - rank-2 x rank-2 -> matrix
/// - higher-rank prefixes are broadcast, then batched matrix multiply
class MatMul : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &a, const Tensor &b) const;
  void operator()(const Tensor &a, const Tensor &b, Tensor &output) const;

  /// MatMul generally changes shape and cannot alias inputs safely.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Element-wise sum of a list of tensors with NumPy-style (multidirectional)
/// broadcasting. At least one input is required. All inputs must share the
/// same float dtype (FLOAT or DOUBLE); the output has the broadcast shape of
/// all inputs and the same dtype.
class Sum : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const std::vector<Tensor> &inputs) const;
  void operator()(const std::vector<Tensor> &inputs, Tensor &output) const;

  /// Variadic element-wise kernel: the output buffer may alias an input
  /// buffer when that input is not broadcast-expanded (i.e. its shape equals
  /// the output shape).
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Cumulative sum of the input tensor along ``axis``. ``axis`` is a 0-D
/// INT32 or INT64 tensor whose value selects the dimension along which the
/// cumulative sum is computed (negative values count from the back). The
/// ``exclusive`` flag, when true, excludes the current element from each
/// position's running sum (the j-th output element is the sum of the first
/// j-1 elements; the 0-th element becomes 0). The ``reverse`` flag, when
/// true, performs the summation in the opposite direction along ``axis``.
class CumSum : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &axis, bool exclusive = false,
                    bool reverse = false) const;
  void operator()(const Tensor &x, const Tensor &axis, bool exclusive, bool reverse,
                  Tensor &output) const;

  /// Each output element depends on a previous output element along
  /// ``axis``; aliasing input and output buffers is safe because we
  /// always read the current element before writing the output at
  /// the same position. The ``exclusive`` mode reads the *previous*
  /// position's input before writing, which is also safe in row-major
  /// order when iterating outward from the starting end.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Cumulative product of the input tensor along ``axis``. Semantics mirror
/// :class:`CumSum` with addition replaced by multiplication; in
/// ``exclusive`` mode the starting value is 1 (multiplicative identity).
class CumProd : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &x, const Tensor &axis, bool exclusive = false,
                    bool reverse = false) const;
  void operator()(const Tensor &x, const Tensor &axis, bool exclusive, bool reverse,
                  Tensor &output) const;

  /// See :class:`CumSum`.
  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of the ``Einsum`` operator (opset 12).
///
/// Evaluates the Einstein summation expressed by ``equation`` over the list
/// of input tensors. The equation may contain ellipsis (``...``) to broadcast
/// leading dimensions, and may be given either in explicit form (``->``
/// followed by the output term) or implicit form. All inputs must share the
/// same dtype (FLOAT or DOUBLE); the output has the same dtype.
class Einsum : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const std::vector<Tensor> &inputs, const std::string &equation) const;
  void operator()(const std::vector<Tensor> &inputs, const std::string &equation,
                  Tensor &output) const;

  /// Einsum generally changes shape and cannot alias inputs safely.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``DFT`` operator (since opset 17;
/// in opset 20 the ``axis`` attribute moved to an optional third input).
///
/// Computes the (one-dimensional) discrete Fourier Transform of ``input``
/// along ``axis``. ``input`` must be a floating-point tensor of rank >= 2,
/// where the trailing dimension is ``1`` (real-valued samples) or ``2``
/// (interleaved real/imaginary parts). The returned tensor has the same
/// rank as ``input``; its last dimension is ``2`` (complex output) except
/// for IRFFT (``onesided=1`` and ``inverse=1``), in which case it is ``1``.
///
/// When ``dft_length`` is specified the signal is zero-padded or truncated
/// along ``axis``; otherwise the axis dimension is used (or
/// ``2 * (signal_dim_axis - 1)`` for the IRFFT default).
class DFT : public KernelBase {
public:
  using KernelBase::KernelBase;
  /// ``axis`` is the signal axis (must satisfy ``-rank <= axis``, ``axis !=
  /// -1`` and ``axis < rank - 1``). ``dft_length`` is a pointer to a 0-D
  /// INT32/INT64 tensor; pass ``nullptr`` to use the default.
  Tensor operator()(const Tensor &input, const Tensor *dft_length, int64_t axis,
                    bool onesided = false, bool inverse = false) const;
  void operator()(const Tensor &input, const Tensor *dft_length, int64_t axis, bool onesided,
                  bool inverse, Tensor &output) const;

  /// The output shape (last dim and signal axis) generally differs from the
  /// input shape, so storage cannot be shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ONNX ``STFT`` operator (opset 17).
///
/// Computes the Short-time Fourier Transform of ``signal`` by sliding a
/// (optionally windowed) DFT of length ``frame_length`` over the signal axis
/// with step ``frame_step``. ``signal`` must be a floating-point tensor of
/// shape ``[batch_size, signal_length, 1]`` (real input) or
/// ``[batch_size, signal_length, 2]`` (interleaved real/imaginary parts).
/// The output has shape ``[batch_size, n_frames, dft_unique_bins, 2]`` where
/// ``n_frames = (signal_length - frame_length) / frame_step + 1`` and
/// ``dft_unique_bins`` is ``floor(frame_length / 2) + 1`` when ``onesided``
/// is true (the default for real input) or ``frame_length`` otherwise.
///
/// ``window`` is an optional 1-D tensor with shape ``[frame_length]``; pass
/// ``nullptr`` to skip the windowing step. ``frame_length`` is an optional
/// pointer to a 0-D INT32/INT64 tensor; pass ``nullptr`` to derive
/// ``frame_length`` from the ``window`` shape. At least one of ``window``
/// or ``frame_length`` must be provided.
class STFT : public KernelBase {
public:
  using KernelBase::KernelBase;
  Tensor operator()(const Tensor &signal, const Tensor &frame_step, const Tensor *window,
                    const Tensor *frame_length, bool onesided = true) const;
  void operator()(const Tensor &signal, const Tensor &frame_step, const Tensor *window,
                  const Tensor *frame_length, bool onesided, Tensor &output) const;

  /// The output shape generally differs from the input, so storage cannot be
  /// shared.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ``TopK`` operator (opsets 1, 10, 11).
///
/// Selects the ``k`` largest (or smallest, when ``largest`` is false) values
/// along ``axis`` from the input tensor ``x`` and returns them together with
/// the corresponding indices. ``axis`` is interpreted modulo the rank of
/// ``x`` (negative values count from the back). When ``sorted`` is true (the
/// default) the returned values are sorted descending (or ascending when
/// ``largest`` is false); ties along ``axis`` are broken by the lower index.
/// When ``sorted`` is false the order of the returned values is unspecified
/// by the ONNX schema; this reference implementation still returns them
/// sorted to keep the output deterministic.
///
/// Returns a ``std::pair<Tensor, Tensor>`` where the first tensor (``Values``)
/// has the same dtype as ``x`` and the second tensor (``Indices``) is an
/// ``INT64`` tensor. Both share the shape of ``x`` with the ``axis``
/// dimension replaced by ``k``.
class TopK : public KernelBase {
public:
  using KernelBase::KernelBase;
  std::pair<Tensor, Tensor> operator()(const Tensor &x, int64_t k, int64_t axis = -1,
                                       bool largest = true, bool sorted = true) const;
  void operator()(const Tensor &x, int64_t k, int64_t axis, bool largest, bool sorted,
                  Tensor &values, Tensor &indices) const;

  /// TopK output shape differs from the input along ``axis`` (k vs. axis_dim)
  /// and the Indices output has a different dtype (int64), so the output
  /// buffers cannot safely alias either input buffer.
  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
