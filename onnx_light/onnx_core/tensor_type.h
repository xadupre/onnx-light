// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file tensor_type.h
 * @brief Declares the ``TensorType`` enumeration and the ``ToTypeString``
 *        converter used across the onnx-light library stack.
 *
 * ``TensorType`` lives in ``onnx_core`` so that both ``onnx_op`` (which
 * builds operator schemas) and ``onnx_optim`` (which runs shape inference)
 * can use it without either library depending on the other.
 *
 * ``onnx_op`` re-exports ``TensorType`` via a ``using`` declaration so that
 * existing code that references ``onnx_op::TensorType`` or
 * ``onnx_op::ToTypeString`` continues to compile unchanged.
 */

#pragma once

#include <cstdint>

#include "onnx_light_helpers.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_core {

/**
 * Identifies an element or sequence tensor type supported by onnx-light.
 *
 * Each enumerator corresponds to a concrete ONNX element type or to a
 * sequence-of-tensor or optional-tensor type used in type-constraint
 * definitions. The mapping from an enumerator to its canonical ONNX type
 * string is implemented exhaustively by ToTypeString(); the naming
 * convention is:
 *
 * - `kXxx` → `"tensor(xxx)"`, e.g. `kFloat` → `"tensor(float)"`,
 *   `kInt64` → `"tensor(int64)"`, `kBfloat16` → `"tensor(bfloat16)"`.
 * - `kSeqXxx` → `"seq(tensor(xxx))"`, e.g. `kSeqFloat` →
 *   `"seq(tensor(float))"`. The two map-valued sequences are
 *   `kSeqMapStringFloat` → `"seq(map(string, float))"` and
 *   `kSeqMapInt64Float` → `"seq(map(int64, float))"`.
 * - `kMapXxxYyy` → `"map(xxx, yyy)"`, e.g. `kMapStringInt64` →
 *   `"map(string, int64)"`. Used for operators like `DictVectorizer`
 *   whose inputs are dictionaries.
 * - `kOptXxx` → `"optional(tensor(xxx))"` and `kOptSeqXxx` →
 *   `"optional(seq(tensor(xxx)))"`.
 * - `kUndefined` → `"tensor(undefined)"`.
 */
enum class TensorType : uint8_t {
  kBool,
  kString,
  kUint8,
  kUint16,
  kUint32,
  kUint64,
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kFloat16,
  kFloat,
  kDouble,
  kBfloat16,
  kFloat8e4m3fn,
  kFloat8e4m3fnuz,
  kFloat8e5m2,
  kFloat8e5m2fnuz,
  kFloat8e8m0,
  kFloat4e2m1,
  kUint4,
  kInt4,
  kUint2,
  kInt2,
  kComplex64,
  kComplex128,
  kSeqBool,
  kSeqString,
  kSeqUint8,
  kSeqUint16,
  kSeqUint32,
  kSeqUint64,
  kSeqInt8,
  kSeqInt16,
  kSeqInt32,
  kSeqInt64,
  kSeqFloat16,
  kSeqFloat,
  kSeqDouble,
  kSeqComplex64,
  kSeqComplex128,
  kSeqMapStringFloat,
  kSeqMapInt64Float,
  kMapStringInt64,
  kMapInt64String,
  kMapInt64Float,
  kMapInt64Double,
  kMapStringFloat,
  kMapStringDouble,
  kOptSeqBool,
  kOptSeqString,
  kOptSeqUint8,
  kOptSeqUint16,
  kOptSeqUint32,
  kOptSeqUint64,
  kOptSeqInt8,
  kOptSeqInt16,
  kOptSeqInt32,
  kOptSeqInt64,
  kOptSeqFloat16,
  kOptSeqFloat,
  kOptSeqDouble,
  kOptSeqComplex64,
  kOptSeqComplex128,
  kOptBool,
  kOptString,
  kOptUint8,
  kOptUint16,
  kOptUint32,
  kOptUint64,
  kOptInt8,
  kOptInt16,
  kOptInt32,
  kOptInt64,
  kOptFloat16,
  kOptFloat,
  kOptDouble,
  kOptComplex64,
  kOptComplex128,
  kUndefined,
};

/**
 * Returns the ONNX type-string representation of a TensorType value.
 *
 * @param type Tensor type enumerator to convert.
 * @return Null-terminated string such as `"tensor(float)"` or
 *         `"seq(tensor(int64))"`.
 */
const char *ToTypeString(TensorType type);

} // namespace onnx_core
} // namespace ONNX_LIGHT_NAMESPACE
