// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/tensor_type.h"

#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_proto {

const char *ToTypeString(TensorType type) {
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

} // namespace onnx_core
} // namespace ONNX_LIGHT_NAMESPACE
