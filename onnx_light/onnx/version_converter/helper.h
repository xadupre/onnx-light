// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/// @file helper.h
/// @brief Helper utilities for version-conversion adapters.
///
/// This header provides broadcasting compatibility checks and input-validity
/// assertions that are shared across the per-opset adapter implementations in
/// @c onnx::version_conversion.  All functions follow NumPy broadcasting rules.

#pragma once

#include <cstdint>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/common/ir.h"
#include "onnx/defs/tensor_util.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace version_conversion {

/**
 * @brief Checks whether @p input2_sizes is unidirectionally broadcastable into
 *        @p input1_sizes under NumPy rules, and whether broadcasting is
 *        actually required.
 *
 * "Unidirectional" means that @p input2 is broadcast *into* @p input1
 * (i.e. @p input1 must be at least as large as @p input2 in every dimension).
 * This is the semantics used by pre-opset-7 binary operators that accepted an
 * explicit @c axis attribute.
 *
 * @param input1_sizes  Shape dimensions of the larger (target) input tensor.
 * @param input2_sizes  Shape dimensions of the smaller (source) input tensor.
 *                      Its rank must not exceed that of @p input1_sizes.
 *
 * @returns
 *   -  @c -1 if the inputs are not unidirectionally broadcastable (the shapes
 *      are incompatible or @p input2 has higher rank than @p input1).
 *   -  @c  0 if the shapes are identical (no broadcasting needed).
 *   -  @c  1 if the shapes are compatible and broadcasting is required.
 */
int check_numpy_unibroadcastable_and_require_broadcast(const std::vector<Dimension> &input1_sizes,
                                                       const std::vector<Dimension> &input2_sizes);

/**
 * @brief Asserts that @p input1_sizes and @p input2_sizes are
 *        multidirectionally broadcastable under NumPy rules.
 *
 * "Multidirectional" broadcasting (opset ≥ 7) allows either input to be
 * expanded; each dimension pair must satisfy: the values are equal, or at
 * least one of them is 1.  The shorter shape is right-aligned before
 * comparison.
 *
 * @param input1_sizes  Shape dimensions of the first input tensor.
 * @param input2_sizes  Shape dimensions of the second input tensor.
 *
 * @throws OnnxReleaseError Throws via @c ONNX_ASSERTM when any dimension pair
 *         is incompatible (neither equal nor 1).
 */
void assert_numpy_multibroadcastable(const std::vector<Dimension> &input1_sizes,
                                     const std::vector<Dimension> &input2_sizes);

/**
 * @brief Asserts that every dimension in @p sizes is a concrete integer value.
 *
 * Symbolic (parametric) dimensions, such as those created from named
 * dimension parameters, are rejected.  This is used before opset adapters
 * that need to reason about exact dimension values.
 *
 * @param sizes  Shape dimensions to validate.
 *
 * @throws OnnxReleaseError Throws via @c ONNX_ASSERTM when any dimension is
 *         a symbolic parameter rather than a concrete integer.
 */
void assertNotParams(const std::vector<Dimension> &sizes);

/**
 * @brief Asserts that the given @p inputs collection has exactly
 *        @p num_inputs elements and that each input has a known shape.
 *
 * All inputs must also pass @c assertNotParams(), i.e. their shapes must
 * consist entirely of concrete integer dimensions.
 *
 * @param inputs      Collection of input @c Value pointers to validate.
 * @param name        Human-readable operator name used in error messages.
 * @param num_inputs  Expected number of inputs.
 *
 * @throws OnnxReleaseError Throws via @c ONNX_ASSERTM when the input count
 *         does not match @p num_inputs, when any input lacks shape
 *         information, or when any input shape contains symbolic dimensions.
 */
void assertInputsAvailable(const ArrayRef<Value *> &inputs, const char *name, uint64_t num_inputs);

/**
 * @brief Decodes an @c INT64 tensor into a @c std::vector<int64_t>.
 *
 * Validates that @p tensor has element type @c INT64 and, when stored as raw
 * bytes, that the byte count is consistent with the tensor's declared
 * dimensions.  The actual value extraction is delegated to @c ParseData.
 *
 * @param tensor  Source tensor to decode.  Must have element type
 *                @c TensorProto_DataType_INT64.
 *
 * @returns A @c std::vector<int64_t> containing all elements of @p tensor in
 *          row-major order.  Returns a single-element vector for scalar
 *          tensors (empty dims).
 *
 * @throws OnnxReleaseError Throws via @c ONNX_ASSERTM when the element type
 *         is not @c INT64 or when the raw byte count does not match the
 *         number of elements implied by the tensor's dimensions.
 */
inline std::vector<int64_t> ReadInt64Tensor(const Tensor &tensor) {
  ONNX_ASSERTM(tensor.elem_type() == ONNX_LIGHT_NAMESPACE::TensorProto_DataType_INT64,
               "expected INT64 tensor, got elem_type=%d", tensor.elem_type())
  if (tensor.is_raw_data()) {
    const size_t raw_bytes = tensor.raw().size();
    // elem_num() returns 1 for scalars, so covers dims=[].
    ONNX_ASSERTM(raw_bytes == static_cast<size_t>(tensor.elem_num()) * sizeof(int64_t),
                 "INT64 tensor: %zu raw bytes does not match dims (%lld elements)", raw_bytes,
                 static_cast<long long>(tensor.elem_num()))
  }
  return ParseData<int64_t>(&tensor);
}

} // namespace version_conversion
} // namespace ONNX_LIGHT_NAMESPACE
