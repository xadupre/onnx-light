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

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

#include "onnx_lib/common/platform_helpers.h"
#include "onnx_lib/common/safe_math.h"
#include "onnx_lib/common/tensor.h"

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
 * Returns the element count implied by tensor dimensions after validating raw data size.
 *
 * @param tensor Source tensor.
 * @param element_size Size of one decoded element.
 * @param exact_fit Whether raw data must contain exactly the implied number of elements.
 * @return The number of elements implied by the tensor dimensions.
 */
int64_t RawDataElementCount(const TensorProto &tensor, size_t element_size, bool exact_fit);

/**
 * Decodes little-endian raw tensor data as values of type T.
 *
 * @tparam T Arithmetic element type to decode.
 * @param tensor Source tensor.
 * @param exact_fit Whether raw data must contain exactly the implied number of elements.
 * @return A vector containing the decoded elements.
 */
template <typename T>
std::vector<T> ParseRawData(const TensorProto &tensor, bool exact_fit = false) {
  static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type");
  const auto num_elements = static_cast<size_t>(RawDataElementCount(tensor, sizeof(T), exact_fit));
  std::vector<T> values(num_elements);
  if (num_elements != 0) {
    std::memcpy(values.data(), tensor.ref_raw_data().data(), num_elements * sizeof(T));
    if (!is_processor_little_endian()) {
      for (T &value : values) {
        auto *start = reinterpret_cast<std::byte *>(&value);
        std::reverse(start, start + sizeof(T));
      }
    }
  }
  return values;
}

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
