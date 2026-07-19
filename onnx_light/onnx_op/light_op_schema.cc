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
        .set_deprecated(s.deprecated())
        .set_node_determinism(s.node_determinism());
    result.emplace_back(std::move(stripped));
  }
  return result;
}

std::vector<LightOpSchema>
CollectSchemasFromBuilders(const std::map<std::string, SchemaBuilder> &builders,
                           const std::string &op_type, bool init_doc) {
  std::vector<LightOpSchema> result;
  if (op_type.empty()) {
    for (const auto &entry : builders) {
      std::vector<LightOpSchema> built = entry.second();
      result.insert(result.end(), std::make_move_iterator(built.begin()),
                    std::make_move_iterator(built.end()));
    }
  } else {
    const auto it = builders.find(op_type);
    if (it != builders.end()) {
      result = it->second();
    }
  }
  return init_doc ? result : StripDocs(result);
}

} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE
