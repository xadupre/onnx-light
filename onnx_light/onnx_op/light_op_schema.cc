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

std::vector<TensorType> CastTypesVer1And6() {
  return {
      TensorType::kFloat16, TensorType::kFloat,  TensorType::kDouble, TensorType::kInt8,
      TensorType::kInt16,   TensorType::kInt32,  TensorType::kInt64,  TensorType::kUint8,
      TensorType::kUint16,  TensorType::kUint32, TensorType::kUint64, TensorType::kBool,
  };
}

std::vector<TensorType> CastTypesVer9() {
  std::vector<TensorType> types = CastTypesVer1And6();
  types.push_back(TensorType::kString);
  return types;
}

std::vector<TensorType> CastTypesVer13() {
  std::vector<TensorType> types = CastTypesVer9();
  types.push_back(TensorType::kBfloat16);
  return types;
}

std::vector<TensorType> CastTypesVer19() {
  std::vector<TensorType> types = CastTypesVer13();
  types.push_back(TensorType::kFloat8e4m3fn);
  types.push_back(TensorType::kFloat8e4m3fnuz);
  types.push_back(TensorType::kFloat8e5m2);
  types.push_back(TensorType::kFloat8e5m2fnuz);
  return types;
}

std::vector<TensorType> CastTypesVer21() {
  std::vector<TensorType> types = CastTypesVer19();
  types.push_back(TensorType::kUint4);
  types.push_back(TensorType::kInt4);
  return types;
}

std::vector<TensorType> CastTypesVer23() {
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

std::vector<TensorType> CastTypesVer24() {
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

std::vector<TensorType> CastTypesVer25() {
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
