// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {

const char *ToTypeString(TensorType type) {
  switch (type) {
  case TensorType::kBool:
    return "tensor(bool)";
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
  case TensorType::kString:
    return "tensor(string)";
  case TensorType::kComplex64:
    return "tensor(complex64)";
  case TensorType::kComplex128:
    return "tensor(complex128)";
  }
  throw std::logic_error("Unknown TensorType.");
}

std::vector<TensorType> FloatTypes() {
  return {
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
  };
}

std::vector<TensorType> NumericTypesForMathReduction() {
  return {
      TensorType::kUint32,  TensorType::kUint64, TensorType::kInt32,  TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
  };
}

std::vector<TensorType> NumericTypesForMathReductionIr4() {
  return {
      TensorType::kUint32,  TensorType::kUint64, TensorType::kInt32,  TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble, TensorType::kBfloat16,
  };
}

std::vector<TensorType> AllNumericTypes() {
  return {
      TensorType::kUint8,   TensorType::kUint16, TensorType::kUint32, TensorType::kUint64,
      TensorType::kInt8,    TensorType::kInt16,  TensorType::kInt32,  TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
  };
}

std::vector<TensorType> AllNumericTypesIr4() {
  return {
      TensorType::kUint8,   TensorType::kUint16, TensorType::kUint32, TensorType::kUint64,
      TensorType::kInt8,    TensorType::kInt16,  TensorType::kInt32,  TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble, TensorType::kBfloat16,
  };
}

std::vector<TensorType> EqualTypesV1V7() {
  return {
      TensorType::kBool,
      TensorType::kInt32,
      TensorType::kInt64,
  };
}

std::vector<TensorType> EqualTypesV11() {
  return {
      TensorType::kBool,   TensorType::kUint8,   TensorType::kUint16, TensorType::kUint32,
      TensorType::kUint64, TensorType::kInt8,    TensorType::kInt16,  TensorType::kInt32,
      TensorType::kInt64,  TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
  };
}

std::vector<TensorType> EqualTypesV13() {
  return {
      TensorType::kBool,     TensorType::kUint8,   TensorType::kUint16, TensorType::kUint32,
      TensorType::kUint64,   TensorType::kInt8,    TensorType::kInt16,  TensorType::kInt32,
      TensorType::kInt64,    TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
      TensorType::kBfloat16,
  };
}

std::vector<TensorType> EqualTypesV19() {
  return {
      TensorType::kBool,     TensorType::kUint8,   TensorType::kUint16, TensorType::kUint32,
      TensorType::kUint64,   TensorType::kInt8,    TensorType::kInt16,  TensorType::kInt32,
      TensorType::kInt64,    TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
      TensorType::kBfloat16, TensorType::kString,
  };
}

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
