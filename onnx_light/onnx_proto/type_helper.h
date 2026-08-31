// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file type_helper.h
 * @brief Provides the ``TensorType`` enumeration, the ``ToTypeString``
 *        converter, and the ``SeqTypeOf``/``OptTypeOf``/``OptSeqTypeOf``
 *        constexpr helpers used across the onnx-light library stack.
 */

#pragma once

#include <cstdint>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::onnx_proto {

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
  kFloat6e2m3,
  kFloat6e3m2,
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
inline constexpr const char *ToTypeString(TensorType type) {
  switch (type) {
  case TensorType::kBool:
    return "tensor(bool)";
  case TensorType::kString:
    return "tensor(string)";
  case TensorType::kUint8:
    return "tensor(uint8)";
  case TensorType::kUint16:
    return "tensor(uint16)";
  case TensorType::kUint32:
    return "tensor(uint32)";
  case TensorType::kUint64:
    return "tensor(uint64)";
  case TensorType::kInt8:
    return "tensor(int8)";
  case TensorType::kInt16:
    return "tensor(int16)";
  case TensorType::kInt32:
    return "tensor(int32)";
  case TensorType::kInt64:
    return "tensor(int64)";
  case TensorType::kFloat16:
    return "tensor(float16)";
  case TensorType::kFloat:
    return "tensor(float)";
  case TensorType::kDouble:
    return "tensor(double)";
  case TensorType::kBfloat16:
    return "tensor(bfloat16)";
  case TensorType::kFloat8e4m3fn:
    return "tensor(float8e4m3fn)";
  case TensorType::kFloat8e4m3fnuz:
    return "tensor(float8e4m3fnuz)";
  case TensorType::kFloat8e5m2:
    return "tensor(float8e5m2)";
  case TensorType::kFloat8e5m2fnuz:
    return "tensor(float8e5m2fnuz)";
  case TensorType::kFloat8e8m0:
    return "tensor(float8e8m0)";
  case TensorType::kFloat4e2m1:
    return "tensor(float4e2m1)";
  case TensorType::kUint4:
    return "tensor(uint4)";
  case TensorType::kInt4:
    return "tensor(int4)";
  case TensorType::kUint2:
    return "tensor(uint2)";
  case TensorType::kInt2:
    return "tensor(int2)";
  case TensorType::kFloat6e2m3:
    return "tensor(float6e2m3)";
  case TensorType::kFloat6e3m2:
    return "tensor(float6e3m2)";
  case TensorType::kComplex64:
    return "tensor(complex64)";
  case TensorType::kComplex128:
    return "tensor(complex128)";
  case TensorType::kSeqBool:
    return "seq(tensor(bool))";
  case TensorType::kSeqString:
    return "seq(tensor(string))";
  case TensorType::kSeqUint8:
    return "seq(tensor(uint8))";
  case TensorType::kSeqUint16:
    return "seq(tensor(uint16))";
  case TensorType::kSeqUint32:
    return "seq(tensor(uint32))";
  case TensorType::kSeqUint64:
    return "seq(tensor(uint64))";
  case TensorType::kSeqInt8:
    return "seq(tensor(int8))";
  case TensorType::kSeqInt16:
    return "seq(tensor(int16))";
  case TensorType::kSeqInt32:
    return "seq(tensor(int32))";
  case TensorType::kSeqInt64:
    return "seq(tensor(int64))";
  case TensorType::kSeqFloat16:
    return "seq(tensor(float16))";
  case TensorType::kSeqFloat:
    return "seq(tensor(float))";
  case TensorType::kSeqDouble:
    return "seq(tensor(double))";
  case TensorType::kSeqComplex64:
    return "seq(tensor(complex64))";
  case TensorType::kSeqComplex128:
    return "seq(tensor(complex128))";
  case TensorType::kSeqMapStringFloat:
    return "seq(map(string, float))";
  case TensorType::kSeqMapInt64Float:
    return "seq(map(int64, float))";
  case TensorType::kMapStringInt64:
    return "map(string, int64)";
  case TensorType::kMapInt64String:
    return "map(int64, string)";
  case TensorType::kMapInt64Float:
    return "map(int64, float)";
  case TensorType::kMapInt64Double:
    return "map(int64, double)";
  case TensorType::kMapStringFloat:
    return "map(string, float)";
  case TensorType::kMapStringDouble:
    return "map(string, double)";
  case TensorType::kOptSeqBool:
    return "optional(seq(tensor(bool)))";
  case TensorType::kOptSeqString:
    return "optional(seq(tensor(string)))";
  case TensorType::kOptSeqUint8:
    return "optional(seq(tensor(uint8)))";
  case TensorType::kOptSeqUint16:
    return "optional(seq(tensor(uint16)))";
  case TensorType::kOptSeqUint32:
    return "optional(seq(tensor(uint32)))";
  case TensorType::kOptSeqUint64:
    return "optional(seq(tensor(uint64)))";
  case TensorType::kOptSeqInt8:
    return "optional(seq(tensor(int8)))";
  case TensorType::kOptSeqInt16:
    return "optional(seq(tensor(int16)))";
  case TensorType::kOptSeqInt32:
    return "optional(seq(tensor(int32)))";
  case TensorType::kOptSeqInt64:
    return "optional(seq(tensor(int64)))";
  case TensorType::kOptSeqFloat16:
    return "optional(seq(tensor(float16)))";
  case TensorType::kOptSeqFloat:
    return "optional(seq(tensor(float)))";
  case TensorType::kOptSeqDouble:
    return "optional(seq(tensor(double)))";
  case TensorType::kOptSeqComplex64:
    return "optional(seq(tensor(complex64)))";
  case TensorType::kOptSeqComplex128:
    return "optional(seq(tensor(complex128)))";
  case TensorType::kOptBool:
    return "optional(tensor(bool))";
  case TensorType::kOptString:
    return "optional(tensor(string))";
  case TensorType::kOptUint8:
    return "optional(tensor(uint8))";
  case TensorType::kOptUint16:
    return "optional(tensor(uint16))";
  case TensorType::kOptUint32:
    return "optional(tensor(uint32))";
  case TensorType::kOptUint64:
    return "optional(tensor(uint64))";
  case TensorType::kOptInt8:
    return "optional(tensor(int8))";
  case TensorType::kOptInt16:
    return "optional(tensor(int16))";
  case TensorType::kOptInt32:
    return "optional(tensor(int32))";
  case TensorType::kOptInt64:
    return "optional(tensor(int64))";
  case TensorType::kOptFloat16:
    return "optional(tensor(float16))";
  case TensorType::kOptFloat:
    return "optional(tensor(float))";
  case TensorType::kOptDouble:
    return "optional(tensor(double))";
  case TensorType::kOptComplex64:
    return "optional(tensor(complex64))";
  case TensorType::kOptComplex128:
    return "optional(tensor(complex128))";
  case TensorType::kUndefined:
    return "tensor(undefined)";
  }
  throw std::logic_error("Unknown TensorType.");
}

/// Maps a scalar tensor TensorType to the corresponding sequence TensorType
/// (e.g. kFloat → kSeqFloat). Returns kUndefined when no sequence variant
/// exists for the given element type (low-precision / extended float types
/// that ONNX does not define sequence forms for).
inline constexpr TensorType SeqTypeOf(TensorType elem) {
  switch (elem) {
  case TensorType::kBool:
    return TensorType::kSeqBool;
  case TensorType::kString:
    return TensorType::kSeqString;
  case TensorType::kUint8:
    return TensorType::kSeqUint8;
  case TensorType::kUint16:
    return TensorType::kSeqUint16;
  case TensorType::kUint32:
    return TensorType::kSeqUint32;
  case TensorType::kUint64:
    return TensorType::kSeqUint64;
  case TensorType::kInt8:
    return TensorType::kSeqInt8;
  case TensorType::kInt16:
    return TensorType::kSeqInt16;
  case TensorType::kInt32:
    return TensorType::kSeqInt32;
  case TensorType::kInt64:
    return TensorType::kSeqInt64;
  case TensorType::kFloat16:
    return TensorType::kSeqFloat16;
  case TensorType::kFloat:
    return TensorType::kSeqFloat;
  case TensorType::kDouble:
    return TensorType::kSeqDouble;
  case TensorType::kComplex64:
    return TensorType::kSeqComplex64;
  case TensorType::kComplex128:
    return TensorType::kSeqComplex128;
  default:
    return TensorType::kUndefined;
  }
}

/// Maps a scalar tensor TensorType to the corresponding optional TensorType
/// (e.g. kFloat → kOptFloat). Returns kUndefined when no optional variant
/// exists for the given element type.
inline constexpr TensorType OptTypeOf(TensorType elem) {
  switch (elem) {
  case TensorType::kBool:
    return TensorType::kOptBool;
  case TensorType::kString:
    return TensorType::kOptString;
  case TensorType::kUint8:
    return TensorType::kOptUint8;
  case TensorType::kUint16:
    return TensorType::kOptUint16;
  case TensorType::kUint32:
    return TensorType::kOptUint32;
  case TensorType::kUint64:
    return TensorType::kOptUint64;
  case TensorType::kInt8:
    return TensorType::kOptInt8;
  case TensorType::kInt16:
    return TensorType::kOptInt16;
  case TensorType::kInt32:
    return TensorType::kOptInt32;
  case TensorType::kInt64:
    return TensorType::kOptInt64;
  case TensorType::kFloat16:
    return TensorType::kOptFloat16;
  case TensorType::kFloat:
    return TensorType::kOptFloat;
  case TensorType::kDouble:
    return TensorType::kOptDouble;
  case TensorType::kComplex64:
    return TensorType::kOptComplex64;
  case TensorType::kComplex128:
    return TensorType::kOptComplex128;
  default:
    return TensorType::kUndefined;
  }
}

/// Maps a scalar tensor TensorType to the corresponding optional-sequence
/// TensorType (e.g. kFloat → kOptSeqFloat). Returns kUndefined when no
/// optional-sequence variant exists for the given element type.
inline constexpr TensorType OptSeqTypeOf(TensorType elem) {
  switch (elem) {
  case TensorType::kBool:
    return TensorType::kOptSeqBool;
  case TensorType::kString:
    return TensorType::kOptSeqString;
  case TensorType::kUint8:
    return TensorType::kOptSeqUint8;
  case TensorType::kUint16:
    return TensorType::kOptSeqUint16;
  case TensorType::kUint32:
    return TensorType::kOptSeqUint32;
  case TensorType::kUint64:
    return TensorType::kOptSeqUint64;
  case TensorType::kInt8:
    return TensorType::kOptSeqInt8;
  case TensorType::kInt16:
    return TensorType::kOptSeqInt16;
  case TensorType::kInt32:
    return TensorType::kOptSeqInt32;
  case TensorType::kInt64:
    return TensorType::kOptSeqInt64;
  case TensorType::kFloat16:
    return TensorType::kOptSeqFloat16;
  case TensorType::kFloat:
    return TensorType::kOptSeqFloat;
  case TensorType::kDouble:
    return TensorType::kOptSeqDouble;
  case TensorType::kComplex64:
    return TensorType::kOptSeqComplex64;
  case TensorType::kComplex128:
    return TensorType::kOptSeqComplex128;
  default:
    return TensorType::kUndefined;
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_proto
