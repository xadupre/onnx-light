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
// Reference implementations of the ``traditionalml`` backend test kernels
// (``ai.onnx.ml`` domain).
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
// the input tensors' buffers. ``LabelEncoder`` reports ``false`` because in
// the general case the input and output element types differ. ``Binarizer``
// reports ``true`` because its output has the same dtype and shape as its
// input.
// ---------------------------------------------------------------------------

/// Reference implementation of the ``ai.onnx.ml`` ``Binarizer`` operator
/// (since opset 1 in the ``ai.onnx.ml`` domain).
///
/// For every input element ``x[i]``, the output element ``y[i]`` is ``1`` if
/// ``x[i] > threshold`` and ``0`` otherwise. The output tensor has the same
/// shape and element type as the input.
///
/// The kernel supports the four numeric element types listed in the ONNX
/// schema via explicit template instantiations:
///
///   * ``float``
///   * ``double``
///   * ``int64_t``
///   * ``int32_t``
///
/// The in-place overload throws ``std::invalid_argument`` if the
/// preallocated output's dtype/shape/byte size do not match the input's.
class Binarizer : public KernelBase {
public:
  using KernelBase::KernelBase;

  template <typename T> Tensor operator()(const Tensor &x, T threshold) const;

  template <typename T> void operator()(const Tensor &x, T threshold, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of the ``ai.onnx.ml`` ``ArrayFeatureExtractor``
/// operator (since opset 1 in the ``ai.onnx.ml`` domain).
///
/// For each logical "row" in the input tensor (all dimensions except the last
/// one), this kernel gathers values from the last axis according to ``indices``
/// (input ``Y``). The output type matches the input type. The output shape is
/// the input shape with the last dimension replaced by ``numel(indices)``.
class ArrayFeatureExtractor : public KernelBase {
public:
  using KernelBase::KernelBase;

  template <typename T> Tensor operator()(const Tensor &x, const Tensor &indices) const;

  template <typename T>
  void operator()(const Tensor &x, const Tensor &indices, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Maps each element of the input tensor through a pair of parallel
/// ``keys``/``values`` arrays — the reference behaviour of the ``ai.onnx.ml``
/// ``LabelEncoder`` operator (since opset 4 in the ``ai.onnx.ml`` domain).
///
/// For every input element ``x[i]``, the output element ``y[i]`` is
/// ``values[k]`` where ``k`` is the index of the first ``keys[k]`` that
/// matches ``x[i]``; if no key matches, ``y[i]`` is ``default_value``.
///
/// The output tensor has the same shape as the input tensor. The kernel
/// supports the following ``(KeyT, ValueT)`` element-type combinations via
/// explicit template instantiations:
///
///   * ``(int64_t, int64_t)``
///   * ``(int64_t, float)``
///   * ``(float,   int64_t)``
///   * ``(float,   float)``
///   * ``(std::string, int64_t)``
///   * ``(std::string, int16_t)``
///
/// ``keys.size()`` must match ``values.size()``. The kernel throws
/// ``std::invalid_argument`` if the input element type does not match
/// ``KeyT`` or, for the in-place overload, if the preallocated output's
/// type/shape do not match the resolved value type and the input shape.
class LabelEncoder : public KernelBase {
public:
  using KernelBase::KernelBase;

  template <typename KeyT, typename ValueT>
  Tensor operator()(const Tensor &x, const std::vector<KeyT> &keys,
                    const std::vector<ValueT> &values, ValueT default_value) const;

  template <typename KeyT, typename ValueT>
  void operator()(const Tensor &x, const std::vector<KeyT> &keys, const std::vector<ValueT> &values,
                  ValueT default_value, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ``ai.onnx.ml`` ``OneHotEncoder`` operator
/// (since opset 1 in the ``ai.onnx.ml`` domain).
///
/// For every input element ``x[i]``, the operator emits a one-hot row of length
/// ``cats.size()``: position ``k`` is ``1.0`` when ``cats[k] == x[i]`` and
/// ``0.0`` everywhere else. If ``x[i]`` matches no category and ``zeros`` is
/// ``true``, the entire row is zero; if ``zeros`` is ``false`` the kernel
/// throws ``std::invalid_argument``.
///
/// The output is always ``float`` with shape equal to the input shape with an
/// additional trailing dimension of size ``cats.size()``. The kernel supports
/// the following category/element types via explicit template instantiations:
///
///   * ``int64_t`` categories with ``int64_t``, ``int32_t``, ``float`` or
///     ``double`` input element types (numeric inputs are cast to ``int64_t``
///     per the ONNX schema).
///   * ``std::string`` categories with a ``std::string`` input element type.
///
/// The in-place overload throws ``std::invalid_argument`` if the preallocated
/// output's dtype/shape/byte size do not match the expected one-hot output.
class OneHotEncoder : public KernelBase {
public:
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, const std::vector<int64_t> &cats, bool zeros) const;

  Tensor operator()(const Tensor &x, const std::vector<std::string> &cats, bool zeros) const;

  template <typename T>
  void operator()(const Tensor &x, const std::vector<int64_t> &cats, bool zeros,
                  Tensor &output) const;

  void operator()(const Tensor &x, const std::vector<std::string> &cats, bool zeros,
                  Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation helper for the ``ai.onnx.ml`` ``SVMClassifier``
/// operator (since opset 1).
///
/// This implementation supports the common binary-classification path and
/// outputs:
///   * ``Y``: class labels (int64 or string)
///   * ``Z``: one raw decision score per sample
class SVMClassifier : public KernelBase {
public:
  using KernelBase::KernelBase;

  template <typename T>
  std::pair<Tensor, Tensor>
  operator()(const Tensor &x, const std::vector<float> &support_vectors,
             const std::vector<float> &coefficients, const std::vector<float> &rho,
             const std::vector<int64_t> &vectors_per_class,
             const std::vector<int64_t> &class_labels, const char *kernel_type, float gamma,
             float coef0, float degree) const;

  template <typename T>
  std::pair<Tensor, Tensor>
  operator()(const Tensor &x, const std::vector<float> &support_vectors,
             const std::vector<float> &coefficients, const std::vector<float> &rho,
             const std::vector<int64_t> &vectors_per_class,
             const std::vector<std::string> &class_labels, const char *kernel_type, float gamma,
             float coef0, float degree) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation helper for the ``ai.onnx.ml`` ``SVMRegressor``
/// operator (since opset 1).
class SVMRegressor : public KernelBase {
public:
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, const std::vector<float> &support_vectors,
                    const std::vector<float> &coefficients, const std::vector<float> &rho,
                    const char *kernel_type, float gamma, float coef0, float degree) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation helper for the ``ai.onnx.ml`` ``ZipMap`` operator
/// (since opset 1 in the ``ai.onnx.ml`` domain).
///
/// ``ZipMap`` returns a ``sequence<map<...>>`` value, which the backend test
/// runtime materializes as a float tensor containing the map values:
///
///   * 1-D input ``[C]`` -> tensor ``[1, C]``
///   * 2-D input ``[N, C]`` -> tensor ``[N, C]``
///
/// The map keys come from either ``classlabels_int64s`` or
/// ``classlabels_strings`` and are validated by this helper through the
/// ``class_labels`` argument size.
class ZipMap : public KernelBase {
public:
  using KernelBase::KernelBase;

  Tensor operator()(const Tensor &x, const std::vector<int64_t> &class_labels) const;
  Tensor operator()(const Tensor &x, const std::vector<std::string> &class_labels) const;

  void operator()(const Tensor &x, const std::vector<int64_t> &class_labels, Tensor &output) const;
  void operator()(const Tensor &x, const std::vector<std::string> &class_labels,
                  Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
