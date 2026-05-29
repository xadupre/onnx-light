// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "light_op_schema.h"

#include <sstream>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {

namespace {

template <typename T> std::string JoinList(const std::vector<T> &v) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      os << ", ";
    os << v[i];
  }
  os << "]";
  return os.str();
}

std::string JoinStringList(const std::vector<std::string> &v) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      os << ", ";
    os << "'" << v[i] << "'";
  }
  os << "]";
  return os.str();
}

} // namespace

std::string AttributeDefaultRepr(const AttributeDefault &d) {
  return std::visit(
      [](const auto &v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return std::string();
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return std::to_string(v);
        } else if constexpr (std::is_same_v<T, double>) {
          std::ostringstream os;
          os << v;
          return os.str();
        } else if constexpr (std::is_same_v<T, std::string>) {
          return v;
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                             std::is_same_v<T, std::vector<double>>) {
          return JoinList(v);
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          return JoinStringList(v);
        } else {
          return std::string();
        }
      },
      d);
}

const char *AttributeType_Name(AttributeType t) {
  switch (t) {
  case AttributeType::FLOAT:
    return "FLOAT";
  case AttributeType::INT:
    return "INT";
  case AttributeType::STRING:
    return "STRING";
  case AttributeType::TENSOR:
    return "TENSOR";
  case AttributeType::GRAPH:
    return "GRAPH";
  case AttributeType::FLOATS:
    return "FLOATS";
  case AttributeType::INTS:
    return "INTS";
  case AttributeType::STRINGS:
    return "STRINGS";
  case AttributeType::TENSORS:
    return "TENSORS";
  case AttributeType::GRAPHS:
    return "GRAPHS";
  case AttributeType::SPARSE_TENSOR:
    return "SPARSE_TENSOR";
  case AttributeType::SPARSE_TENSORS:
    return "SPARSE_TENSORS";
  case AttributeType::TYPE_PROTO:
    return "TYPE_PROTO";
  case AttributeType::TYPE_PROTOS:
    return "TYPE_PROTOS";
  case AttributeType::UNDEFINED:
  default:
    return "UNDEFINED";
  }
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

std::vector<TensorType> AllTensorTypes() {
  return {
      TensorType::kUint8,   TensorType::kUint16,    TensorType::kUint32,     TensorType::kUint64,
      TensorType::kInt8,    TensorType::kInt16,     TensorType::kInt32,      TensorType::kInt64,
      TensorType::kFloat16, TensorType::kFloat,     TensorType::kDouble,     TensorType::kString,
      TensorType::kBool,    TensorType::kComplex64, TensorType::kComplex128,
  };
}

std::vector<TensorType> AllTensorSequenceTypes() {
  return {
      TensorType::kSeqUint8,  TensorType::kSeqUint16,    TensorType::kSeqUint32,
      TensorType::kSeqUint64, TensorType::kSeqInt8,      TensorType::kSeqInt16,
      TensorType::kSeqInt32,  TensorType::kSeqInt64,     TensorType::kSeqFloat16,
      TensorType::kSeqFloat,  TensorType::kSeqDouble,    TensorType::kSeqString,
      TensorType::kSeqBool,   TensorType::kSeqComplex64, TensorType::kSeqComplex128,
  };
}

std::vector<TensorType> AllOptionalTypes() {
  return {
      TensorType::kOptSeqUint8,  TensorType::kOptSeqUint16,    TensorType::kOptSeqUint32,
      TensorType::kOptSeqUint64, TensorType::kOptSeqInt8,      TensorType::kOptSeqInt16,
      TensorType::kOptSeqInt32,  TensorType::kOptSeqInt64,     TensorType::kOptSeqFloat16,
      TensorType::kOptSeqFloat,  TensorType::kOptSeqDouble,    TensorType::kOptSeqString,
      TensorType::kOptSeqBool,   TensorType::kOptSeqComplex64, TensorType::kOptSeqComplex128,
      TensorType::kOptUint8,     TensorType::kOptUint16,       TensorType::kOptUint32,
      TensorType::kOptUint64,    TensorType::kOptInt8,         TensorType::kOptInt16,
      TensorType::kOptInt32,     TensorType::kOptInt64,        TensorType::kOptFloat16,
      TensorType::kOptFloat,     TensorType::kOptDouble,       TensorType::kOptString,
      TensorType::kOptBool,      TensorType::kOptComplex64,    TensorType::kOptComplex128,
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

std::vector<TensorType> ConcatTypesVer1() {
  return {
      TensorType::kFloat16,
      TensorType::kFloat,
      TensorType::kDouble,
  };
}

std::vector<TensorType> ConcatTypesVer4And11() { return AllTensorTypes(); }

std::vector<TensorType> ConcatTypesVer13() {
  return {
      TensorType::kUint8,    TensorType::kUint16,  TensorType::kUint32,    TensorType::kUint64,
      TensorType::kInt8,     TensorType::kInt16,   TensorType::kInt32,     TensorType::kInt64,
      TensorType::kBfloat16, TensorType::kFloat16, TensorType::kFloat,     TensorType::kDouble,
      TensorType::kString,   TensorType::kBool,    TensorType::kComplex64, TensorType::kComplex128,
  };
}

std::vector<LightOpSchema> StripDocs(const std::vector<LightOpSchema> &schemas) {
  std::vector<LightOpSchema> result;
  result.reserve(schemas.size());
  for (const LightOpSchema &s : schemas) {
    LightOpSchema stripped(s.name(), s.domain(), s.since_version(), std::string(), s.inputs(),
                           s.outputs(), s.type_constraints(), s.attributes(),
                           s.has_function_implementation(),
                           /*init_doc=*/false);
    stripped.set_min_output(s.min_output())
        .set_max_output(s.max_output())
        .set_deprecated(s.deprecated());
    result.emplace_back(std::move(stripped));
  }
  return result;
}

std::vector<LightOpSchema> FilterSchemasByOpType(std::vector<LightOpSchema> schemas,
                                                 const std::string &op_type) {
  if (op_type.empty()) {
    return schemas;
  }
  std::vector<LightOpSchema> filtered;
  for (LightOpSchema &schema : schemas) {
    if (schema.name() == op_type) {
      filtered.emplace_back(std::move(schema));
    }
  }
  return filtered;
}

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
