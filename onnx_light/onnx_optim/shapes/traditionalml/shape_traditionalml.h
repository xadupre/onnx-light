// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"

#include <string>
#include <vector>

/**
 * @file shape_traditionalml.h
 * @brief Shape-inference functions for ONNX operators in the
 *        ``ai.onnx.ml`` (traditional machine-learning) family.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_optim {
namespace shapes {
namespace traditionalml {

/// Canonical domain string for the ``ai.onnx.ml`` operator set.
inline constexpr const char *kOnnxMlDomain = "ai.onnx.ml";

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Binarizer`` node
 * and stores it in ``ctx``.
 *
 * ``Binarizer`` (``ai.onnx.ml``) is an element-wise operator: the output
 * tensor has the exact same shape and element type as the input — only
 * the values change (each element becomes ``1`` if it is strictly
 * greater than the ``threshold`` attribute, ``0`` otherwise).
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Binarizer`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Binarizer"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must be
 *              present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Binarizer"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeBinarizer(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``CategoryMapper`` node
 * and stores it in ``ctx``.
 *
 * ``CategoryMapper`` (``ai.onnx.ml``) is a one-to-one mapping between
 * strings and integers: the output tensor has the exact same shape as the
 * input, while its element type is determined by the input element type
 * (``STRING`` input → ``INT64`` output; ``INT64`` input → ``STRING``
 * output).
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``CategoryMapper`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"CategoryMapper"``
 *              and ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must be
 *              present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"CategoryMapper"``, if ``node`` has no output, or if the
 *         input element type is neither ``STRING`` nor ``INT64``.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeCategoryMapper(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``Imputer`` node
 * and stores it in ``ctx``.
 *
 * ``Imputer`` (``ai.onnx.ml``) is an element-wise operator: the output
 * tensor has the exact same shape and element type as the input — only
 * the values change (elements matching ``replaced_value_float`` or
 * ``replaced_value_int64`` are replaced by the corresponding imputed
 * values).
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Imputer`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Imputer"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must be
 *              present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Imputer"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeImputer(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of an
 * ``ArrayFeatureExtractor`` node and stores it in ``ctx``.
 *
 * ``ArrayFeatureExtractor`` selects values from the last axis of ``x``
 * using indices from ``y``:
 *
 *   - output dtype always matches ``x``;
 *   - all leading dimensions of ``x`` are preserved;
 *   - the last output dimension equals the flattened element count of
 *     ``y`` (product of ``y``'s dimensions) when that value can be
 *     inferred from ``y``'s shape.
 *
 * @param ctx   In/out context. Must already contain entries for ``x``
 *              and ``y``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``ArrayFeatureExtractor`` ``NodeProto`` whose output
 *              should be described. ``node.op_type()`` must be
 *              ``"ArrayFeatureExtractor"`` and ``node`` must declare at
 *              least one output.
 * @param x     Name of the data input tensor.
 * @param y     Name of the indices input tensor.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"ArrayFeatureExtractor"``, if ``node`` has no output, or if
 *         ``y`` is not an ``int64`` tensor.
 * @throws std::out_of_range     if ``x`` or ``y`` is not present in
 *         ``ctx``.
 */
void ComputeShapeArrayFeatureExtractor(ShapesContext &ctx, const NodeProto &node, const char *x,
                                       const char *y);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``LabelEncoder``
 * node and stores it in ``ctx``.
 *
 * ``LabelEncoder`` (``ai.onnx.ml``) is a one-to-one mapping from input
 * keys to output values, so the output shape always matches the input
 * shape. The output dtype is determined by which of the ``values_*``
 * attributes is set:
 *
 *   - ``values_tensor`` — dtype is the tensor's ``data_type``;
 *   - ``values_strings`` — dtype is ``string``;
 *   - ``values_int64s``  — dtype is ``int64``;
 *   - ``values_floats``  — dtype is ``float``.
 *
 * Exactly one of these attributes must be set; an error is raised
 * otherwise.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``LabelEncoder`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"LabelEncoder"`` and ``node`` must declare at least
 *              one output.
 * @param x     Name of the input value to read from ``ctx``. Must be
 *              present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"LabelEncoder"``, if ``node`` has no output, or if the
 *         attributes do not specify exactly one of the allowed
 *         ``values_*`` forms.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeLabelEncoder(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``OneHotEncoder``
 * node and stores it in ``ctx``.
 *
 * ``OneHotEncoder`` (``ai.onnx.ml``) emits a one-hot encoding of the
 * input: the output is a ``float`` tensor whose shape equals the input
 * shape extended by a trailing dimension of size ``len(cats_*)``. Exactly
 * one of ``cats_int64s`` or ``cats_strings`` must be set on the node.
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``OneHotEncoder`` ``NodeProto`` whose output should
 *              be described. ``node.op_type()`` must be
 *              ``"OneHotEncoder"`` and ``node`` must declare at least
 *              one output.
 * @param x     Name of the input value to read from ``ctx``. Must be
 *              present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"OneHotEncoder"``, if ``node`` has no output, or if the
 *         attributes do not specify exactly one of the allowed
 *         ``cats_*`` forms.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeOneHotEncoder(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` objects of a ``LinearClassifier``
 * node and stores them in ``ctx``.
 *
 * ``LinearClassifier`` (``ai.onnx.ml``) consumes either a single feature vector
 * ``[C]`` or a batch ``[N,C]`` and emits:
 *
 *   - ``Y``: predicted labels (``string`` when ``classlabels_strings`` is
 *     provided, ``int64`` otherwise), shape ``[N]`` (or ``[1]`` for rank-1
 *     input);
 *   - ``Z``: classification scores, shape ``[N,E]`` where ``E`` is the number
 *     of classes inferred from the ``intercepts``/``classlabels_*`` attributes
 *     (binary classifiers with a single intercept and two labels expose
 *     ``E == 2``).
 *
 * @param ctx   In/out context with input ``x`` already present.
 * @param node  ``LinearClassifier`` node.
 * @param x     Name of the input value to read from ``ctx``.
 */
void ComputeShapeLinearClassifier(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``LinearRegressor`` node and
 * stores it in ``ctx``.
 *
 * ``LinearRegressor`` (``ai.onnx.ml``) consumes either ``[C]`` or ``[N,C]`` and
 * emits a float tensor of regression scores with shape ``[N, targets]`` where
 * ``targets`` is taken from the operator's ``targets`` attribute (default 1).
 *
 * @param ctx   In/out context with input ``x`` already present.
 * @param node  ``LinearRegressor`` node.
 * @param x     Name of the input value to read from ``ctx``.
 */
void ComputeShapeLinearRegressor(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Scaler`` node and
 * stores it in ``ctx``.
 *
 * ``Scaler`` (``ai.onnx.ml``) is an element-wise operator: the output
 * tensor has the same shape as the input but its element type is always
 * ``float`` (per the ONNX schema, ``Y`` is ``tensor(float)`` regardless
 * of the input element type).
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Scaler`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Scaler"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must be
 *              present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Scaler"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeScaler(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``Normalizer`` node and
 * stores it in ``ctx``.
 *
 * ``Normalizer`` (``ai.onnx.ml``) normalizes its input along the last
 * (feature) axis. The output tensor has the same shape as the input but
 * its element type is always ``float`` (per the ONNX schema, ``Y`` is
 * ``tensor(float)`` regardless of the input element type).
 *
 * @param ctx   In/out context. Must already contain an entry for
 *              ``x``; on return it also contains an entry for
 *              ``node.output(0)``.
 * @param node  The ``Normalizer`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"Normalizer"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must be
 *              present in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not
 *         ``"Normalizer"`` or if ``node`` has no output.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeNormalizer(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` objects of an ``SVMClassifier``
 * node and stores them in ``ctx``.
 *
 * ``SVMClassifier`` (``ai.onnx.ml``) consumes either a single feature vector
 * ``[C]`` or a batch ``[N,C]`` and emits:
 *
 *   - ``Y``: predicted labels (``string`` when ``classlabels_strings`` is
 *     provided, ``int64`` otherwise), shape ``[N]`` (or ``[1]`` for rank-1
 *     input);
 *   - ``Z``: class scores/probabilities, shape ``[N,S]`` where ``S`` is
 *     inferred from class labels when available.
 *
 * @param ctx   In/out context with input ``x`` already present.
 * @param node  ``SVMClassifier`` node.
 * @param x     Name of the input value to read from ``ctx``.
 */
void ComputeShapeSVMClassifier(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of an ``SVMRegressor`` node and
 * stores it in ``ctx``.
 *
 * ``SVMRegressor`` (``ai.onnx.ml``) consumes either ``[C]`` or ``[N,C]`` and
 * emits a float tensor of regression scores with shape ``[N,1]`` (or ``[1,1]``
 * for rank-1 input).
 *
 * @param ctx   In/out context with input ``x`` already present.
 * @param node  ``SVMRegressor`` node.
 * @param x     Name of the input value to read from ``ctx``.
 */
void ComputeShapeSVMRegressor(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``TreeEnsembleRegressor``
 * node and stores it in ``ctx``.
 *
 * ``TreeEnsembleRegressor`` (``ai.onnx.ml``) consumes either ``[C]`` or
 * ``[N,C]`` and emits a float tensor of regression scores with shape
 * ``[N, n_targets]``, where ``n_targets`` is taken from the operator's
 * ``n_targets`` attribute (default 1).
 *
 * @param ctx   In/out context with input ``x`` already present.
 * @param node  ``TreeEnsembleRegressor`` node.
 * @param x     Name of the input value to read from ``ctx``.
 */
void ComputeShapeTreeEnsembleRegressor(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` objects of a
 * ``TreeEnsembleClassifier`` node and stores them in ``ctx``.
 *
 * ``TreeEnsembleClassifier`` (``ai.onnx.ml``) consumes either ``[C]`` or
 * ``[N,C]`` and emits:
 *
 *   - ``Y``: predicted labels (``string`` when ``classlabels_strings`` is
 *     provided, ``int64`` otherwise), shape ``[N]`` (or ``[1]`` for rank-1
 *     input);
 *   - ``Z``: classification scores, shape ``[N,E]`` where ``E`` is the
 *     number of classes from the ``classlabels_*`` attribute.
 *
 * @param ctx   In/out context with input ``x`` already present.
 * @param node  ``TreeEnsembleClassifier`` node.
 * @param x     Name of the input value to read from ``ctx``.
 */
void ComputeShapeTreeEnsembleClassifier(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``TreeEnsemble`` node
 * and stores it in ``ctx``.
 *
 * ``TreeEnsemble`` (``ai.onnx.ml``, opset 5) consumes ``[N, F]`` and emits
 * a tensor of shape ``[N, n_targets]`` with the same element type as the
 * input, where ``n_targets`` is taken from the ``n_targets`` attribute
 * (default 1).
 *
 * @param ctx   In/out context with input ``x`` already present.
 * @param node  ``TreeEnsemble`` node.
 * @param x     Name of the input value to read from ``ctx``.
 */
void ComputeShapeTreeEnsemble(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``ZipMap`` node and
 * stores it in ``ctx``.
 *
 * ``ZipMap`` (``ai.onnx.ml``) converts each row of the float input tensor
 * into a map keyed by either ``classlabels_strings`` or
 * ``classlabels_int64s``. The output dtype is therefore inferred as either
 * ``seq(map(string, float))`` or ``seq(map(int64, float))``.
 *
 * @param ctx   In/out context. Must already contain an entry for ``x``; on
 *              return it also contains an entry for ``node.output(0)``.
 * @param node  The ``ZipMap`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"ZipMap"`` and
 *              ``node`` must declare at least one output.
 * @param x     Name of the input value to read from ``ctx``. Must be present
 *              in ``ctx``.
 *
 * @throws std::invalid_argument if ``node.op_type()`` is not ``"ZipMap"``,
 *         if ``node`` has no output, if ``x`` is not a float tensor, or if
 *         the class-label attributes are invalid.
 * @throws std::out_of_range     if ``x`` is not present in ``ctx``.
 */
void ComputeShapeZipMap(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``DictVectorizer`` node
 * and stores it in ``ctx``.
 *
 * ``DictVectorizer`` (``ai.onnx.ml``) converts a dictionary input into a 1-D
 * tensor whose length matches the vocabulary attribute length. Exactly one of
 * ``string_vocabulary`` or ``int64_vocabulary`` must be set on the node. The
 * output element type is inferred from the value type of the input map (or
 * from a non-empty ``T2`` ``ValueInfoProto`` annotation when ``x`` is not a
 * tensor); when neither is available, the output dtype is left undefined and
 * only the shape ``[C]`` is propagated.
 *
 * @param ctx   In/out context. May contain an entry for ``x`` (a map); on
 *              return it contains an entry for ``node.output(0)``.
 * @param node  The ``DictVectorizer`` ``NodeProto`` whose output should be
 *              described. ``node.op_type()`` must be ``"DictVectorizer"``.
 * @param x     Name of the input value (a map).
 */
void ComputeShapeDictVectorizer(ShapesContext &ctx, const NodeProto &node, const char *x);

/**
 * Computes the output :cpp:class:`OptimTensor` of a ``FeatureVectorizer`` node
 * and stores it in ``ctx``.
 *
 * ``FeatureVectorizer`` (``ai.onnx.ml``) concatenates a variadic list of
 * tensors along the trailing feature dimension; the output is always a
 * ``float`` tensor of shape ``[N, sum(inputdimensions)]`` where ``N`` is the
 * common batch size of the inputs. When the ``inputdimensions`` attribute is
 * absent the per-input feature widths are taken from each input's last
 * dimension; if any feature width or the batch size is unknown, the
 * corresponding output dimension is left symbolic.
 *
 * @param ctx     In/out context.
 * @param node    The ``FeatureVectorizer`` ``NodeProto``.
 * @param inputs  Names of the variadic input values, in declaration order.
 */
void ComputeShapeFeatureVectorizer(ShapesContext &ctx, const NodeProto &node,
                                   const std::vector<std::string> &inputs);

} // namespace traditionalml
} // namespace shapes
} // namespace onnx_optim
} // namespace ONNX_LIGHT_NAMESPACE
