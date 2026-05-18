// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file tensor_util.h
 * @brief Declares tensor conversion helpers used by ONNX schema definitions.
 *
 * This header provides utility templates to decode typed values from a Tensor
 * wrapper and to construct TensorProto constants from scalar and vector C++
 * values.
 */

#pragma once

#include <vector>

#include "onnx/common/tensor.h"

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Extracts typed element data from a Tensor wrapper.
 *
 * @tparam T Element type to decode.
 * @param tensor Source tensor wrapper.
 * @return A vector containing tensor data converted to T.
 */
template <typename T> std::vector<T> ParseData(const TensorProto *tensor);

/**
 * Extracts typed element data from a Tensor wrapper.
 *
 * @tparam T Element type to decode.
 * @param tensor Source tensor wrapper.
 * @return A vector containing tensor data converted to T.
 */
template <typename T> std::vector<T> ParseData(const Tensor *tensor);

/**
 * Creates a scalar TensorProto from a C++ value.
 *
 * @tparam T Scalar value type.
 * @param value Scalar value to encode.
 * @return A TensorProto containing one element.
 */
template <typename T> TensorProto ToTensor(const T &value);

/**
 * Creates a 1D TensorProto from a vector of values.
 *
 * @tparam T Element value type.
 * @param values Values to encode.
 * @return A TensorProto containing values in order.
 */
template <typename T> TensorProto ToTensor(const std::vector<T> &values);

/**
 * Returns a TensorProto input unchanged.
 *
 * @param tensor TensorProto to pass through.
 * @return tensor.
 */
inline TensorProto ToTensor(const TensorProto &tensor) { return tensor; }

} // namespace ONNX_LIGHT_NAMESPACE
