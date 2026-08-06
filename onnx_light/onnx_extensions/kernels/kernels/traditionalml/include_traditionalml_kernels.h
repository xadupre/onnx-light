// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_extensions/kernels/kernels/traditionalml/kernel_tree_ensemble_common.h"

#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels {
// Re-exports the runtime types moved to ``onnx_core::runtime`` so
// kernel implementations below can keep referring to them
// unqualified, matching pre-move code.
using namespace ::onnx_light::core::runtime;

namespace kernel {
using ::onnx_light::core::runtime::DefaultOpset;
using ::onnx_light::core::runtime::KernelBase;
using ::onnx_light::core::runtime::KernelContext;
using ::onnx_light::core::runtime::OpsetId;

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
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, T threshold, RuntimeContext *rt = nullptr) const;

  template <typename T> void operator()(const Tensor &x, T threshold, Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return true; }
};

/// Reference implementation of the ``ai.onnx.ml`` ``CategoryMapper`` operator
/// (since opset 1 in the ``ai.onnx.ml`` domain).
///
/// Converts strings to integers (or vice versa) using two parallel lookup
/// arrays ``cats_strings`` and ``cats_int64s`` of identical length. When the
/// input element type is ``std::string``, the kernel emits an ``int64`` tensor
/// whose element at position ``i`` is ``cats_int64s[k]`` where ``k`` is the
/// index of the first matching ``cats_strings[k] == x[i]`` (and
/// ``default_int64`` when no key matches). When the input element type is
/// ``int64_t``, the kernel emits a ``std::string`` tensor whose element at
/// position ``i`` is ``cats_strings[k]`` where ``k`` is the index of the first
/// matching ``cats_int64s[k] == x[i]`` (and ``default_string`` when no key
/// matches).
///
/// The output tensor has the same shape as the input tensor. The kernel
/// supports the two element-type combinations listed in the ONNX schema via
/// explicit template instantiations:
///
///   * ``std::string`` input → ``int64_t`` output
///   * ``int64_t``     input → ``std::string`` output
///
/// The in-place overload throws ``std::invalid_argument`` if the
/// preallocated output's dtype/shape/byte size do not match the expected
/// output.
class CategoryMapper : public KernelBase {
public:
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  template <typename InT, typename OutT>
  Tensor operator()(const Tensor &x, const ParamStrings &cats_strings,
                    const std::vector<int64_t> &cats_int64s, OutT default_value,
                    RuntimeContext *rt = nullptr) const;

  template <typename InT, typename OutT>
  void operator()(const Tensor &x, const ParamStrings &cats_strings,
                  const std::vector<int64_t> &cats_int64s, OutT default_value,
                  Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
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
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, const std::vector<T> &imputed_values, T replaced_value,
                    RuntimeContext *rt = nullptr) const;

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
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, const Tensor &indices, RuntimeContext *rt = nullptr) const;

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
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  /// Primary overload accepting zero-copy span views of the keys and values
  /// (preferred when the data comes directly from tensor buffers).
  template <typename KeyT, typename ValueT>
  Tensor operator()(const Tensor &x, std::span<const KeyT> keys, std::span<const ValueT> values,
                    ValueT default_value, RuntimeContext *rt = nullptr) const;

  /// Convenience overload for callers that already own ``std::vector`` objects;
  /// forwards to the span overload without an extra allocation.
  template <typename KeyT, typename ValueT>
  Tensor operator()(const Tensor &x, const std::vector<KeyT> &keys,
                    const std::vector<ValueT> &values, ValueT default_value,
                    RuntimeContext *rt = nullptr) const {
    return (*this).template operator()<KeyT, ValueT>(
        x, std::span<const KeyT>{keys}, std::span<const ValueT>{values}, default_value, rt);
  }

  /// Primary in-place overload accepting zero-copy span views.
  template <typename KeyT, typename ValueT>
  void operator()(const Tensor &x, std::span<const KeyT> keys, std::span<const ValueT> values,
                  ValueT default_value, Tensor &output) const;

  /// Convenience in-place overload forwarding to the span overload.
  template <typename KeyT, typename ValueT>
  void operator()(const Tensor &x, const std::vector<KeyT> &keys, const std::vector<ValueT> &values,
                  ValueT default_value, Tensor &output) const {
    (*this).template operator()<KeyT, ValueT>(
        x, std::span<const KeyT>{keys}, std::span<const ValueT>{values}, default_value, output);
  }

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
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, const std::vector<int64_t> &cats, bool zeros,
                    RuntimeContext *rt = nullptr) const;

  Tensor operator()(const Tensor &x, const ParamStrings &cats, bool zeros,
                    RuntimeContext *rt = nullptr) const;

  template <typename T>
  void operator()(const Tensor &x, const std::vector<int64_t> &cats, bool zeros,
                  Tensor &output) const;

  void operator()(const Tensor &x, const ParamStrings &cats, bool zeros, Tensor &output) const;

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
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  template <typename T>
  std::pair<Tensor, Tensor>
  operator()(const Tensor &x, const ParamFloats &coefficients, const ParamFloats &intercepts,
             const std::vector<int64_t> &class_labels, const std::string &post_transform) const;

  template <typename T>
  std::pair<Tensor, Tensor>
  operator()(const Tensor &x, const ParamFloats &coefficients, const ParamFloats &intercepts,
             const ParamStrings &class_labels, const std::string &post_transform) const;

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
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, const ParamFloats &coefficients, const ParamFloats &intercepts,
                    int64_t targets, const std::string &post_transform,
                    RuntimeContext *rt = nullptr) const;

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
  void Run(RuntimeContext &rt) override;
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
             const std::vector<int64_t> &vectors_per_class, const ParamStrings &class_labels,
             const char *kernel_type, float gamma, float coef0, float degree) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation helper for the ``ai.onnx.ml`` ``SVMRegressor``
/// operator (since opset 1).
class SVMRegressor : public KernelBase {
public:
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, const std::vector<float> &support_vectors,
                    const std::vector<float> &coefficients, const std::vector<float> &rho,
                    const char *kernel_type, float gamma, float coef0, float degree,
                    RuntimeContext *rt = nullptr) const;

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
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, const std::vector<float> &offset,
                    const std::vector<float> &scale, RuntimeContext *rt = nullptr) const;

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
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  template <typename T>
  Tensor operator()(const Tensor &x, const std::string &norm, RuntimeContext *rt = nullptr) const;

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

  Tensor operator()(const Tensor &x, const std::vector<int64_t> &class_labels,
                    RuntimeContext *rt = nullptr) const;
  Tensor operator()(const Tensor &x, const ParamStrings &class_labels,
                    RuntimeContext *rt = nullptr) const;

  void operator()(const Tensor &x, const std::vector<int64_t> &class_labels, Tensor &output) const;
  void operator()(const Tensor &x, const ParamStrings &class_labels, Tensor &output) const;

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
  void Run(RuntimeContext &rt) override;
  /// Inherits the context-only constructor so the dispatch factory can
  /// build the kernel; ``Run`` parses the ONNX attributes and constructs the
  /// fully-initialized instance it delegates to.
  using KernelBase::KernelBase;
  /// Builds the classic node and leaf maps from the parallel ``nodes_*`` and
  /// ``target_*`` attribute arrays so the tree structure is constructed once,
  /// ahead of any call to ``operator()``.
  ///
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
  TreeEnsembleRegressor(
      const KernelContext &ctx, const std::vector<int64_t> &nodes_treeids,
      const std::vector<int64_t> &nodes_nodeids, const std::vector<int64_t> &nodes_featureids,
      const std::vector<float> &nodes_values, const ParamStrings &nodes_modes,
      const std::vector<int64_t> &nodes_truenodeids, const std::vector<int64_t> &nodes_falsenodeids,
      const std::vector<int64_t> &nodes_missing, const std::vector<int64_t> &target_treeids,
      const std::vector<int64_t> &target_nodeids, const std::vector<int64_t> &target_ids,
      const std::vector<float> &target_weights);

  /// Traverses the pre-built decision tree ensemble and returns a float tensor
  /// of regression scores with shape ``[N, n_targets]``.
  ///
  /// @param x               Input feature matrix, shape ``[N, F]`` or ``[F]``.
  /// @param n_targets       Total number of regression targets.
  /// @param aggregate_function  "SUM" | "AVERAGE" | "MIN" | "MAX".
  /// @param post_transform  "NONE".
  /// @param base_values     Added to the aggregated output; empty means 0.
  template <typename T>
  Tensor operator()(const Tensor &x, int64_t n_targets, const std::string &aggregate_function,
                    const std::string &post_transform, const std::vector<float> &base_values,
                    RuntimeContext *rt = nullptr) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  ClassicNodeMap node_map_;
  ClassicLeafMap leaf_map_;
  std::vector<int64_t> tree_ids_;
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
  void Run(RuntimeContext &rt) override;
  /// Inherits the context-only constructor so the dispatch factory can
  /// build the kernel; ``Run`` parses the ONNX attributes and constructs the
  /// fully-initialized instance it delegates to.
  using KernelBase::KernelBase;
  /// Builds the classic node and leaf maps from the parallel ``nodes_*`` and
  /// ``class_*`` attribute arrays so the tree structure is constructed once,
  /// ahead of any call to ``operator()``.
  TreeEnsembleClassifier(
      const KernelContext &ctx, const std::vector<int64_t> &nodes_treeids,
      const std::vector<int64_t> &nodes_nodeids, const std::vector<int64_t> &nodes_featureids,
      const std::vector<float> &nodes_values, const ParamStrings &nodes_modes,
      const std::vector<int64_t> &nodes_truenodeids, const std::vector<int64_t> &nodes_falsenodeids,
      const std::vector<int64_t> &nodes_missing, const std::vector<int64_t> &class_treeids,
      const std::vector<int64_t> &class_nodeids, const std::vector<int64_t> &class_ids,
      const std::vector<float> &class_weights);

  template <typename T>
  std::pair<Tensor, Tensor>
  operator()(const Tensor &x, const std::vector<int64_t> &classlabels_int64s,
             const std::vector<float> &base_values, const std::string &post_transform) const;

  template <typename T>
  std::pair<Tensor, Tensor> operator()(const Tensor &x, const ParamStrings &classlabels_strings,
                                       const std::vector<float> &base_values,
                                       const std::string &post_transform) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  ClassicNodeMap node_map_;
  ClassicLeafMap leaf_map_;
  std::vector<int64_t> tree_ids_;
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
  void Run(RuntimeContext &rt) override;
  /// Inherits the context-only constructor so the dispatch factory can
  /// build the kernel; ``Run`` parses the ONNX attributes and constructs the
  /// fully-initialized instance it delegates to.
  using KernelBase::KernelBase;
  /// Stores the tree structure and precomputes the per-node membership sets so
  /// the tree is constructed once, ahead of any call to ``operator()``.
  ///
  /// @param tree_roots           Index into nodes_* arrays for each tree root.
  /// @param nodes_featureids     Feature index per interior node.
  /// @param nodes_splits         Threshold per interior node.
  /// @param nodes_modes          Comparison mode per node (uint8 encoding).
  /// @param nodes_truenodeids    True-branch index per node.
  /// @param nodes_falsenodeids   False-branch index per node.
  /// @param nodes_trueleafs      1 if true branch is a leaf.
  /// @param nodes_falseleafs     1 if false branch is a leaf.
  /// @param nodes_missing        1 if NaN follows true branch.
  /// @param leaf_targetids       Target index per leaf.
  /// @param leaf_weights         Weight per leaf.
  /// @param membership_values    Flat list of set-member values consumed in
  ///                             ``nodes_modes`` order for every node with
  ///                             ``BRANCH_MEMBER`` (6) mode. Each node's set
  ///                             is delimited by a ``NaN`` sentinel. May be
  ///                             empty if no node uses ``BRANCH_MEMBER``.
  ///
  /// ``nodes_splits``, ``leaf_weights``, and ``membership_values`` are taken as
  /// ``double`` so the kernel owns its tree data (independent of the input
  /// element type) and remains safely copy-constructible.
  TreeEnsemble(const KernelContext &ctx, const std::vector<int64_t> &tree_roots,
               const std::vector<int64_t> &nodes_featureids,
               const std::vector<double> &nodes_splits, const std::vector<uint8_t> &nodes_modes,
               const std::vector<int64_t> &nodes_truenodeids,
               const std::vector<int64_t> &nodes_falsenodeids,
               const std::vector<int64_t> &nodes_trueleafs,
               const std::vector<int64_t> &nodes_falseleafs,
               const std::vector<int64_t> &nodes_missing,
               const std::vector<int64_t> &leaf_targetids, const std::vector<double> &leaf_weights,
               const std::vector<double> &membership_values);

  /// Traverses the pre-built tree ensemble and returns a tensor of shape
  /// ``[N, n_targets]`` with the same element type as ``x``.
  ///
  /// @param x                    Input feature matrix ``[N, F]``.
  /// @param n_targets            Number of regression targets.
  /// @param aggregate_function   0=AVERAGE, 1=SUM (default), 2=MIN, 3=MAX.
  /// @param post_transform       0=NONE (default), 1=SOFTMAX.
  template <typename T>
  Tensor operator()(const Tensor &x, int64_t n_targets, int64_t aggregate_function,
                    int64_t post_transform, RuntimeContext *rt = nullptr) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }

private:
  ParamInts tree_roots_;
  ParamInts nodes_featureids_;
  std::vector<double> nodes_splits_;
  std::vector<uint8_t> nodes_modes_;
  ParamInts nodes_truenodeids_;
  ParamInts nodes_falsenodeids_;
  ParamInts nodes_trueleafs_;
  ParamInts nodes_falseleafs_;
  ParamInts nodes_missing_;
  ParamInts leaf_targetids_;
  std::vector<double> leaf_weights_;
  /// Per-node set-membership values for ``BRANCH_MEMBER`` nodes; other nodes
  /// hold an empty vector.
  std::vector<std::vector<double>> node_member_sets_;
};

/// Reference implementation of the ``ai.onnx.ml`` ``DictVectorizer`` operator
/// (since opset 1 in the ``ai.onnx.ml`` domain).
///
/// Converts a dictionary, represented here as parallel ``keys``/``values``
/// arrays, into a 1-D output tensor of shape ``[len(vocabulary)]``. For every
/// key in the input dictionary, the kernel looks up the key's index in the
/// supplied vocabulary and writes the corresponding value into that slot. Keys
/// missing from the input dictionary leave a ``0`` (or empty string) at their
/// vocabulary index in the output.
///
/// Two helper template parameters control the variant:
///
///   * ``K`` — element type of the dictionary keys (``int64_t`` or
///     ``std::string``);
///   * ``V`` — element type of the dictionary values (``int64_t``, ``float``,
///     ``double`` or ``std::string``).
///
/// All keys in ``input_keys`` must be present in ``vocabulary``; the kernel
/// throws ``std::invalid_argument`` otherwise. ``input_keys`` and
/// ``input_values`` must have the same length.
class DictVectorizer : public KernelBase {
public:
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  /// Primary overload accepting zero-copy span views of the input keys and
  /// values (preferred when the data comes directly from tensor buffers).
  /// The vocabulary is always sourced from operator attributes and is passed
  /// by reference.
  template <typename K, typename V>
  Tensor operator()(std::span<const K> input_keys, std::span<const V> input_values,
                    const std::vector<K> &vocabulary, RuntimeContext *rt = nullptr) const;

  /// Convenience overload for callers that already own ``std::vector`` objects;
  /// forwards to the span overload without an extra allocation.
  template <typename K, typename V>
  Tensor operator()(const std::vector<K> &input_keys, const std::vector<V> &input_values,
                    const std::vector<K> &vocabulary, RuntimeContext *rt = nullptr) const {
    return (*this).template operator()<K, V>(std::span<const K>{input_keys},
                                             std::span<const V>{input_values}, vocabulary, rt);
  }

  /// Primary in-place overload accepting zero-copy span views.
  template <typename K, typename V>
  void operator()(std::span<const K> input_keys, std::span<const V> input_values,
                  const std::vector<K> &vocabulary, Tensor &output) const;

  /// Convenience in-place overload forwarding to the span overload.
  template <typename K, typename V>
  void operator()(const std::vector<K> &input_keys, const std::vector<V> &input_values,
                  const std::vector<K> &vocabulary, Tensor &output) const {
    (*this).template operator()<K, V>(std::span<const K>{input_keys},
                                      std::span<const V>{input_values}, vocabulary, output);
  }

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ``ai.onnx.ml`` ``CastMap`` operator
/// (since opset 1 in the ``ai.onnx.ml`` domain).
///
/// Converts a map ``input_keys``/``input_values`` (key type ``int64_t``,
/// value type ``std::string`` or ``float``) into a 1-D output tensor. The
/// packing of the output is controlled by ``map_form``:
///
///   * ``"DENSE"`` — the output has length ``input_keys.size()``; entries
///     are written in ascending order of their input keys.
///   * ``"SPARSE"`` — the output has length ``max_map``; entries are
///     written at position ``key`` for every key in the input map and the
///     remaining positions are zero (or empty string for string outputs).
///
/// The output element type is determined by ``cast_to``:
///
///   * ``"TO_FLOAT"``  → ``float``  output (string values are parsed,
///                                          float values are forwarded);
///   * ``"TO_INT64"``  → ``int64_t`` output (string values are parsed,
///                                          float values are truncated);
///   * ``"TO_STRING"`` → ``std::string`` output (float values are formatted
///                                               using ``std::to_string``,
///                                               string values are forwarded).
///
/// ``input_keys.size()`` must match ``input_values.size()``. The kernel
/// throws ``std::invalid_argument`` for unsupported attribute values, when
/// ``map_form == "SPARSE"`` and a key is ``< 0`` or ``>= max_map``, or
/// when the in-place output's dtype/shape do not match the expected output.
class CastMap : public KernelBase {
public:
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  /// Primary overload accepting zero-copy span views of the input keys and
  /// values (preferred when the data comes directly from tensor buffers).
  template <typename V, typename OutT>
  Tensor operator()(std::span<const int64_t> input_keys, std::span<const V> input_values,
                    const std::string &cast_to, const std::string &map_form, int64_t max_map,
                    RuntimeContext *rt = nullptr) const;

  /// Convenience overload for callers that already own ``std::vector`` objects;
  /// forwards to the span overload without an extra allocation.
  template <typename V, typename OutT>
  Tensor operator()(const std::vector<int64_t> &input_keys, const std::vector<V> &input_values,
                    const std::string &cast_to, const std::string &map_form, int64_t max_map,
                    RuntimeContext *rt = nullptr) const {
    return (*this).template operator()<V, OutT>(std::span<const int64_t>{input_keys},
                                                std::span<const V>{input_values}, cast_to, map_form,
                                                max_map, rt);
  }

  /// Primary in-place overload accepting zero-copy span views.
  template <typename V, typename OutT>
  void operator()(std::span<const int64_t> input_keys, std::span<const V> input_values,
                  const std::string &cast_to, const std::string &map_form, int64_t max_map,
                  Tensor &output) const;

  /// Convenience in-place overload forwarding to the span overload.
  template <typename V, typename OutT>
  void operator()(const std::vector<int64_t> &input_keys, const std::vector<V> &input_values,
                  const std::string &cast_to, const std::string &map_form, int64_t max_map,
                  Tensor &output) const {
    (*this).template operator()<V, OutT>(std::span<const int64_t>{input_keys},
                                         std::span<const V>{input_values}, cast_to, map_form,
                                         max_map, output);
  }

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

/// Reference implementation of the ``ai.onnx.ml`` ``FeatureVectorizer``
/// operator (since opset 1 in the ``ai.onnx.ml`` domain).
///
/// Concatenates a variadic list of 1-D or 2-D input tensors along the second
/// (feature) dimension, casting every value to ``float``. 1-D inputs are
/// treated as ``[1, C]``. When the ``inputdimensions`` attribute is provided
/// each input is truncated or zero-padded to its declared feature width before
/// concatenation. The output element type is always ``float`` and the output
/// shape is ``[N, sum(input_dims)]`` where ``N`` is the common batch size of
/// the inputs (taken as the maximum reported batch size; smaller inputs are
/// broadcast to row 0).
///
/// The kernel supports the four numeric input element types listed in the
/// ONNX schema via explicit template instantiations:
///
///   * ``float``
///   * ``double``
///   * ``int64_t``
///   * ``int32_t``
///
/// Each input may use any of the four supported element types — they need not
/// agree across inputs. The kernel validates each input's dtype against the
/// caller-supplied per-input template parameters; the convenience overload
/// that takes a homogeneous ``Tensors`` plus the per-input element
/// type ``DataType`` dispatches to the right template specialization.
class FeatureVectorizer : public KernelBase {
public:
  void Run(RuntimeContext &rt) override;
  using KernelBase::KernelBase;

  /// Concatenates the input tensors (each cast to ``float``) along the
  /// trailing feature dimension. ``inputdimensions``, when non-empty, must
  /// have the same length as ``inputs`` and gives the declared feature width
  /// per input. When empty the feature width is taken from each input's last
  /// dimension.
  Tensor operator()(const Tensors &inputs, const std::vector<int64_t> &inputdimensions,
                    RuntimeContext *rt = nullptr) const;

  void operator()(const Tensors &inputs, const std::vector<int64_t> &inputdimensions,
                  Tensor &output) const;

  static constexpr bool CanRunInPlace() noexcept { return false; }
};

} // namespace kernel
} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels
