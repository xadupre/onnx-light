// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {

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

std::vector<TensorType> CastTypesV1V6() {
  return {TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble, TensorType::kInt8,
          TensorType::kInt16,   TensorType::kInt32,  TensorType::kInt64,  TensorType::kUint8,
          TensorType::kUint16,  TensorType::kUint32, TensorType::kUint64, TensorType::kBool};
}

std::vector<TensorType> CastTypesV9() {
  return {TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble, TensorType::kInt8,
          TensorType::kInt16,   TensorType::kInt32,  TensorType::kInt64,  TensorType::kUint8,
          TensorType::kUint16,  TensorType::kUint32, TensorType::kUint64, TensorType::kBool,
          TensorType::kString};
}

std::vector<TensorType> CastTypesV13() {
  return {TensorType::kFloat16, TensorType::kFloat,   TensorType::kDouble, TensorType::kInt8,
          TensorType::kInt16,   TensorType::kInt32,   TensorType::kInt64,  TensorType::kUint8,
          TensorType::kUint16,  TensorType::kUint32,  TensorType::kUint64, TensorType::kBool,
          TensorType::kString,  TensorType::kBfloat16};
}

std::vector<TensorType> CastTypesV19() {
  return {TensorType::kFloat16,        TensorType::kFloat,      TensorType::kDouble,
          TensorType::kInt8,           TensorType::kInt16,      TensorType::kInt32,
          TensorType::kInt64,          TensorType::kUint8,      TensorType::kUint16,
          TensorType::kUint32,         TensorType::kUint64,     TensorType::kBool,
          TensorType::kString,         TensorType::kBfloat16,   TensorType::kFloat8e4m3fn,
          TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz};
}

std::vector<TensorType> CastTypesV21() {
  return {TensorType::kFloat16,        TensorType::kFloat,      TensorType::kDouble,
          TensorType::kInt8,           TensorType::kInt16,      TensorType::kInt32,
          TensorType::kInt64,          TensorType::kUint8,      TensorType::kUint16,
          TensorType::kUint32,         TensorType::kUint64,     TensorType::kBool,
          TensorType::kString,         TensorType::kBfloat16,   TensorType::kFloat8e4m3fn,
          TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
          TensorType::kUint4,          TensorType::kInt4};
}

std::vector<TensorType> CastTypesV23() {
  return {TensorType::kUint8,          TensorType::kUint16,     TensorType::kUint32,
          TensorType::kUint64,         TensorType::kInt8,       TensorType::kInt16,
          TensorType::kInt32,          TensorType::kInt64,      TensorType::kBfloat16,
          TensorType::kFloat16,        TensorType::kFloat,      TensorType::kDouble,
          TensorType::kString,         TensorType::kBool,       TensorType::kFloat8e4m3fn,
          TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
          TensorType::kUint4,          TensorType::kInt4,       TensorType::kFloat4e2m1};
}

std::vector<TensorType> CastTypesV24() {
  return {TensorType::kUint8,          TensorType::kUint16,     TensorType::kUint32,
          TensorType::kUint64,         TensorType::kInt8,       TensorType::kInt16,
          TensorType::kInt32,          TensorType::kInt64,      TensorType::kBfloat16,
          TensorType::kFloat16,        TensorType::kFloat,      TensorType::kDouble,
          TensorType::kString,         TensorType::kBool,       TensorType::kFloat8e4m3fn,
          TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
          TensorType::kUint4,          TensorType::kInt4,       TensorType::kFloat4e2m1,
          TensorType::kFloat8e8m0};
}

std::vector<TensorType> CastTypesV25() {
  return {TensorType::kUint8,          TensorType::kUint16,     TensorType::kUint32,
          TensorType::kUint64,         TensorType::kInt8,       TensorType::kInt16,
          TensorType::kInt32,          TensorType::kInt64,      TensorType::kBfloat16,
          TensorType::kFloat16,        TensorType::kFloat,      TensorType::kDouble,
          TensorType::kString,         TensorType::kBool,       TensorType::kFloat8e4m3fn,
          TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
          TensorType::kUint4,          TensorType::kInt4,       TensorType::kFloat4e2m1,
          TensorType::kFloat8e8m0,     TensorType::kUint2,      TensorType::kInt2};
}

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
