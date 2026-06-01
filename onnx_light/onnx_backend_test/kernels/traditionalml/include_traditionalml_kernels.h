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

/// Reference implementation of the ``ai.onnx.ml`` ``Imputer`` operator
/// (since opset 1 in the ``ai.onnx.ml`` domain).
///
/// Replaces each element ``x[i]`` that matches the ``replaced_value`` with the
/// corresponding ``imputed_value[i % stride]``, where ``stride`` is the length
/// of the ``imputed_value`` vector (either 1 for broadcast or equal to the size
/// of the last dimension of ``x``). Elements that do not match are left
/// unchanged. For float/double inputs, NaN equality is used for
/// ``replaced_value_float`` when the replaced value is NaN.
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
class Imputer : public KernelBase {
public:
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, const std::vector<T> &imputed_values, T replaced_value) const;

  template <typename T>
  void operator()(const Tensor &x, const std::vector<T> &imputed_values, T replaced_value,
                  Tensor &output) const;

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

/// Reference implementation helper for the ``ai.onnx.ml`` ``LinearClassifier``
/// operator (since opset 1).
///
/// Computes raw scores ``Z = X @ W^T + b`` where ``W`` has shape
/// ``[E, C]`` (``coefficients`` is a flat ``E*C`` array, classes contiguous)
/// and ``b`` has shape ``[E]`` (``intercepts``). The predicted label ``Y``
/// is selected from ``class_labels`` based on the argmax of ``Z`` (or the
/// sign of ``Z`` for binary classifiers with a single intercept and two
/// labels, in which case ``Z`` is expanded to ``[N, 2]`` with the canonical
/// ``[-z, z]`` convention).
class LinearClassifier : public KernelBase {
public:
  using KernelBase::KernelBase;

  template <typename T>
  std::pair<Tensor, Tensor> operator()(const Tensor &x, const std::vector<float> &coefficients,
                                       const std::vector<float> &intercepts,
                                       const std::vector<int64_t> &class_labels,
                                       const std::string &post_transform) const;

  template <typename T>
  std::pair<Tensor, Tensor> operator()(const Tensor &x, const std::vector<float> &coefficients,
                                       const std::vector<float> &intercepts,
                                       const std::vector<std::string> &class_labels,
                                       const std::string &post_transform) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation helper for the ``ai.onnx.ml`` ``LinearRegressor``
/// operator (since opset 1).
///
/// Computes ``Y = X @ W^T + b`` where ``W`` has shape ``[targets, C]`` and
/// ``b`` has shape ``[targets]`` (or is empty, in which case zeros are
/// assumed). Only ``post_transform == "NONE"`` is supported.
class LinearRegressor : public KernelBase {
public:
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, const std::vector<float> &coefficients,
                    const std::vector<float> &intercepts, int64_t targets,
                    const std::string &post_transform) const;

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

/// Reference implementation of the ``ai.onnx.ml`` ``Scaler`` operator
/// (since opset 1 in the ``ai.onnx.ml`` domain).
///
/// For every input element ``x[i]``, the output element ``y[i]`` is
/// ``(static_cast<float>(x[i]) - offset_k) * scale_k`` where ``k`` is
/// ``i % stride`` and ``stride`` is the length of the ``offset``/``scale``
/// vectors. ``offset`` and ``scale`` must have the same length, which must
/// either be ``1`` (broadcast to every element) or equal to the size of the
/// last dimension of ``x``.
///
/// The output is always ``float`` with the same shape as the input. The
/// kernel supports the four numeric input element types listed in the ONNX
/// schema via explicit template instantiations:
///
///   * ``float``
///   * ``double``
///   * ``int64_t``
///   * ``int32_t``
///
/// The in-place overload throws ``std::invalid_argument`` if the
/// preallocated output's dtype/shape/byte size do not match the expected
/// float output.
class Scaler : public KernelBase {
public:
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, const std::vector<float> &offset,
                    const std::vector<float> &scale) const;

  template <typename T>
  void operator()(const Tensor &x, const std::vector<float> &offset,
                  const std::vector<float> &scale, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ``ai.onnx.ml`` ``Normalizer`` operator
/// (since opset 1 in the ``ai.onnx.ml`` domain).
///
/// Normalizes the input ``X`` along its last (feature) axis using one of
/// three modes selected by the ``norm`` attribute:
///
///   * ``"MAX"`` — ``y = x / max(abs(x))`` per row.
///   * ``"L1"``  — ``y = x / sum(abs(x))`` per row.
///   * ``"L2"``  — ``y = x / sqrt(sum(x^2))`` per row.
///
/// If the per-row divisor is zero, the row is copied through unchanged
/// (``y == x``). The input may be ``[C]`` (a single row) or ``[N, C]``
/// (a batch of rows normalized independently).
///
/// The output is always ``float`` with the same shape as the input. The
/// kernel supports the four numeric input element types listed in the ONNX
/// schema via explicit template instantiations:
///
///   * ``float``
///   * ``double``
///   * ``int64_t``
///   * ``int32_t``
///
/// The in-place overload throws ``std::invalid_argument`` if the
/// preallocated output's dtype/shape/byte size do not match the expected
/// float output.
class Normalizer : public KernelBase {
public:
  using KernelBase::KernelBase;

  template <typename T> Tensor operator()(const Tensor &x, const std::string &norm) const;

  template <typename T>
  void operator()(const Tensor &x, const std::string &norm, Tensor &output) const;

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

/// Reference implementation helper for the ``ai.onnx.ml``
/// ``TreeEnsembleRegressor`` operator (opsets 1, 3, 5 in the ``ai.onnx.ml``
/// domain).
///
/// Traverses a classic-encoding decision tree ensemble and returns a float
/// tensor of regression scores with shape ``[N, n_targets]``.
///
/// This implementation supports ``aggregate_function`` values "SUM" (default),
/// "AVERAGE", "MIN", and "MAX", and ``post_transform`` value "NONE".
class TreeEnsembleRegressor : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// @param x               Input feature matrix, shape ``[N, F]`` or ``[F]``.
  /// @param nodes_treeids   Tree id per node.
  /// @param nodes_nodeids   Node id per node (root is 0 per tree).
  /// @param nodes_featureids Feature index per node.
  /// @param nodes_values    Split threshold per node (float variant).
  /// @param nodes_modes     Node mode strings per node (e.g. "BRANCH_LEQ").
  /// @param nodes_truenodeids  True-branch child node id per node.
  /// @param nodes_falsenodeids False-branch child node id per node.
  /// @param nodes_missing   1 if a NaN input follows the true branch; 0 for
  ///                        false. May be empty (treated as all 0).
  /// @param target_treeids  Tree id per leaf entry.
  /// @param target_nodeids  Node id per leaf entry.
  /// @param target_ids      Target index per leaf entry.
  /// @param target_weights  Weight contribution per leaf entry.
  /// @param n_targets       Total number of regression targets.
  /// @param aggregate_function  "SUM" | "AVERAGE" | "MIN" | "MAX".
  /// @param post_transform  "NONE".
  /// @param base_values     Added to the aggregated output; empty means 0.
  template <typename T>
  Tensor operator()(
      const Tensor &x, const std::vector<int64_t> &nodes_treeids,
      const std::vector<int64_t> &nodes_nodeids, const std::vector<int64_t> &nodes_featureids,
      const std::vector<float> &nodes_values, const std::vector<std::string> &nodes_modes,
      const std::vector<int64_t> &nodes_truenodeids, const std::vector<int64_t> &nodes_falsenodeids,
      const std::vector<int64_t> &nodes_missing, const std::vector<int64_t> &target_treeids,
      const std::vector<int64_t> &target_nodeids, const std::vector<int64_t> &target_ids,
      const std::vector<float> &target_weights, int64_t n_targets,
      const std::string &aggregate_function, const std::string &post_transform,
      const std::vector<float> &base_values) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation helper for the ``ai.onnx.ml``
/// ``TreeEnsembleClassifier`` operator (opsets 1, 3, 5 in the ``ai.onnx.ml``
/// domain).
///
/// Traverses a classic-encoding decision tree ensemble and returns:
///   - ``Y``: top class label tensor of shape ``[N]``.
///   - ``Z``: class score tensor of shape ``[N, E]``.
///
/// Supports integer and string class labels.
class TreeEnsembleClassifier : public KernelBase {
public:
  using KernelBase::KernelBase;

  template <typename T>
  std::pair<Tensor, Tensor> operator()(
      const Tensor &x, const std::vector<int64_t> &nodes_treeids,
      const std::vector<int64_t> &nodes_nodeids, const std::vector<int64_t> &nodes_featureids,
      const std::vector<float> &nodes_values, const std::vector<std::string> &nodes_modes,
      const std::vector<int64_t> &nodes_truenodeids, const std::vector<int64_t> &nodes_falsenodeids,
      const std::vector<int64_t> &nodes_missing, const std::vector<int64_t> &class_treeids,
      const std::vector<int64_t> &class_nodeids, const std::vector<int64_t> &class_ids,
      const std::vector<float> &class_weights, const std::vector<int64_t> &classlabels_int64s,
      const std::vector<float> &base_values, const std::string &post_transform) const;

  template <typename T>
  std::pair<Tensor, Tensor> operator()(
      const Tensor &x, const std::vector<int64_t> &nodes_treeids,
      const std::vector<int64_t> &nodes_nodeids, const std::vector<int64_t> &nodes_featureids,
      const std::vector<float> &nodes_values, const std::vector<std::string> &nodes_modes,
      const std::vector<int64_t> &nodes_truenodeids, const std::vector<int64_t> &nodes_falsenodeids,
      const std::vector<int64_t> &nodes_missing, const std::vector<int64_t> &class_treeids,
      const std::vector<int64_t> &class_nodeids, const std::vector<int64_t> &class_ids,
      const std::vector<float> &class_weights, const std::vector<std::string> &classlabels_strings,
      const std::vector<float> &base_values, const std::string &post_transform) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation helper for the ``ai.onnx.ml`` ``TreeEnsemble``
/// operator (opset 5 in the ``ai.onnx.ml`` domain).
///
/// Uses the new ``TreeEnsemble`` encoding: ``tree_roots``, ``nodes_splits``
/// (a tensor), ``leaf_targetids``, and ``leaf_weights``.
///
/// Only ``post_transform`` 0 (NONE) and 1 (SOFTMAX) are supported.
class TreeEnsemble : public KernelBase {
public:
  using KernelBase::KernelBase;

  /// @param x                    Input feature matrix ``[N, F]``.
  /// @param tree_roots           Index into nodes_* arrays for each tree root.
  /// @param nodes_featureids     Feature index per interior node.
  /// @param nodes_splits         Threshold per interior node (same type as x).
  /// @param nodes_modes          Comparison mode per node (uint8 encoding).
  /// @param nodes_truenodeids    True-branch index per node.
  /// @param nodes_falsenodeids   False-branch index per node.
  /// @param nodes_trueleafs      1 if true branch is a leaf.
  /// @param nodes_falseleafs     1 if false branch is a leaf.
  /// @param nodes_missing        1 if NaN follows true branch.
  /// @param leaf_targetids       Target index per leaf.
  /// @param leaf_weights         Weight per leaf (same type as x).
  /// @param n_targets            Number of regression targets.
  /// @param aggregate_function   0=AVERAGE, 1=SUM (default), 2=MIN, 3=MAX.
  /// @param post_transform       0=NONE (default), 1=SOFTMAX.
  template <typename T>
  Tensor operator()(const Tensor &x, const std::vector<int64_t> &tree_roots,
                    const std::vector<int64_t> &nodes_featureids,
                    const std::vector<T> &nodes_splits, const std::vector<uint8_t> &nodes_modes,
                    const std::vector<int64_t> &nodes_truenodeids,
                    const std::vector<int64_t> &nodes_falsenodeids,
                    const std::vector<int64_t> &nodes_trueleafs,
                    const std::vector<int64_t> &nodes_falseleafs,
                    const std::vector<int64_t> &nodes_missing,
                    const std::vector<int64_t> &leaf_targetids, const std::vector<T> &leaf_weights,
                    int64_t n_targets, int64_t aggregate_function, int64_t post_transform) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
