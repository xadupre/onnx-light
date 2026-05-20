// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "light_op_schema.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {

const std::string &ToTypeString(const TypeConstraintParam::AllowedType &type) {
  return type.type_str;
}

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
  }
  throw std::logic_error("Unknown TensorType.");
}

std::vector<TypeConstraintParam::AllowedType> FloatTypes() {
  return {
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
  };
}

std::vector<TypeConstraintParam::AllowedType> NumericTypesForMathReduction() {
  return {
      TensorType::kUint32,  TensorType::kUint64, TensorType::kInt32,  TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
  };
}

std::vector<TypeConstraintParam::AllowedType> NumericTypesForMathReductionIr4() {
  return {
      TensorType::kUint32,  TensorType::kUint64, TensorType::kInt32,  TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble, TensorType::kBfloat16,
  };
}

std::vector<TypeConstraintParam::AllowedType> AllNumericTypes() {
  return {
      TensorType::kUint8,   TensorType::kUint16, TensorType::kUint32, TensorType::kUint64,
      TensorType::kInt8,    TensorType::kInt16,  TensorType::kInt32,  TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
  };
}

std::vector<TypeConstraintParam::AllowedType> AllNumericTypesIr4() {
  return {
      TensorType::kUint8,   TensorType::kUint16, TensorType::kUint32, TensorType::kUint64,
      TensorType::kInt8,    TensorType::kInt16,  TensorType::kInt32,  TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble, TensorType::kBfloat16,
  };
}

std::vector<TypeConstraintParam::AllowedType> CastTypesVer1And6() {
  return {
      TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble, TensorType::kInt8,
      TensorType::kInt16,   TensorType::kInt32,  TensorType::kInt64,  TensorType::kUint8,
      TensorType::kUint16,  TensorType::kUint32, TensorType::kUint64, TensorType::kBool,
  };
}

std::vector<TypeConstraintParam::AllowedType> CastTypesVer9() {
  std::vector<TypeConstraintParam::AllowedType> types = CastTypesVer1And6();
  types.emplace_back(TensorType::kString);
  return types;
}

std::vector<TypeConstraintParam::AllowedType> CastTypesVer13() {
  std::vector<TypeConstraintParam::AllowedType> types = CastTypesVer9();
  types.emplace_back(TensorType::kBfloat16);
  return types;
}

std::vector<TypeConstraintParam::AllowedType> CastTypesVer19() {
  std::vector<TypeConstraintParam::AllowedType> types = CastTypesVer13();
  types.emplace_back(TensorType::kFloat8e4m3fn);
  types.emplace_back(TensorType::kFloat8e4m3fnuz);
  types.emplace_back(TensorType::kFloat8e5m2);
  types.emplace_back(TensorType::kFloat8e5m2fnuz);
  return types;
}

std::vector<TypeConstraintParam::AllowedType> CastTypesVer21() {
  std::vector<TypeConstraintParam::AllowedType> types = CastTypesVer19();
  types.emplace_back(TensorType::kUint4);
  types.emplace_back(TensorType::kInt4);
  return types;
}

std::vector<TypeConstraintParam::AllowedType> CastTypesVer23() {
  return {
      TensorType::kUint8,          TensorType::kUint16,     TensorType::kUint32,
      TensorType::kUint64,         TensorType::kInt8,       TensorType::kInt16,
      TensorType::kInt32,          TensorType::kInt64,      TensorType::kBfloat16,
      TensorType::kFloat16,        TensorType::kFloat,      TensorType::kDouble,
      TensorType::kString,         TensorType::kBool,       TensorType::kFloat8e4m3fn,
      TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
      TensorType::kUint4,          TensorType::kInt4,       TensorType::kFloat4e2m1,
  };
}

std::vector<TypeConstraintParam::AllowedType> CastTypesVer24() {
  return {
      TensorType::kUint8,          TensorType::kUint16,     TensorType::kUint32,
      TensorType::kUint64,         TensorType::kInt8,       TensorType::kInt16,
      TensorType::kInt32,          TensorType::kInt64,      TensorType::kBfloat16,
      TensorType::kFloat16,        TensorType::kFloat,      TensorType::kDouble,
      TensorType::kString,         TensorType::kBool,       TensorType::kFloat8e4m3fn,
      TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
      TensorType::kUint4,          TensorType::kInt4,       TensorType::kFloat4e2m1,
      TensorType::kFloat8e8m0,
  };
}

std::vector<TypeConstraintParam::AllowedType> CastTypesVer25() {
  return {
      TensorType::kUint8,          TensorType::kUint16,     TensorType::kUint32,
      TensorType::kUint64,         TensorType::kInt8,       TensorType::kInt16,
      TensorType::kInt32,          TensorType::kInt64,      TensorType::kBfloat16,
      TensorType::kFloat16,        TensorType::kFloat,      TensorType::kDouble,
      TensorType::kString,         TensorType::kBool,       TensorType::kFloat8e4m3fn,
      TensorType::kFloat8e4m3fnuz, TensorType::kFloat8e5m2, TensorType::kFloat8e5m2fnuz,
      TensorType::kUint4,          TensorType::kInt4,       TensorType::kFloat4e2m1,
      TensorType::kFloat8e8m0,     TensorType::kUint2,      TensorType::kInt2,
  };
}

std::vector<TypeConstraintParam::AllowedType> EqualTypesV1V7() {
  return {
      TensorType::kBool,
      TensorType::kInt32,
      TensorType::kInt64,
  };
}

std::vector<TypeConstraintParam::AllowedType> EqualTypesV11() {
  return {
      TensorType::kBool,   TensorType::kUint8,   TensorType::kUint16, TensorType::kUint32,
      TensorType::kUint64, TensorType::kInt8,    TensorType::kInt16,  TensorType::kInt32,
      TensorType::kInt64,  TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
  };
}

std::vector<TypeConstraintParam::AllowedType> EqualTypesV13() {
  return {
      TensorType::kBool,     TensorType::kUint8,   TensorType::kUint16, TensorType::kUint32,
      TensorType::kUint64,   TensorType::kInt8,    TensorType::kInt16,  TensorType::kInt32,
      TensorType::kInt64,    TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
      TensorType::kBfloat16,
  };
}

std::vector<TypeConstraintParam::AllowedType> EqualTypesV19() {
  return {
      TensorType::kBool,     TensorType::kUint8,   TensorType::kUint16, TensorType::kUint32,
      TensorType::kUint64,   TensorType::kInt8,    TensorType::kInt16,  TensorType::kInt32,
      TensorType::kInt64,    TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble,
      TensorType::kBfloat16, TensorType::kString,
  };
}

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
