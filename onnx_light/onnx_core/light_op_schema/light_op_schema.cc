// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "light_op_schema.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE::core::schema {

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

using onnx_proto::OptSeqTypeOf;
using onnx_proto::OptTypeOf;
using onnx_proto::SeqTypeOf;

/// Resolves the concrete TensorType described by @p type for LightOpSchema::Verify's
/// type-constraint checking. Returns std::nullopt when @p type uses a category that
/// TensorType cannot represent in this context (map_type, opaque_type, sparse_tensor_type, or a
/// nested sequence/map inside a Sequence/Optional); callers should skip the check in that case.
std::optional<TensorType> TensorTypeFromTypeProto(const TypeProto &type) {
  switch (type.value_case()) {
  case TypeProto::kTensorType:
    return symbolic::DataTypeToTensorType(type.tensor_type().elem_type());
  case TypeProto::kSequenceType: {
    const TypeProto &elem = type.sequence_type().elem_type();
    if (!elem.has_tensor_type()) {
      return std::nullopt;
    }
    const TensorType seq =
        SeqTypeOf(symbolic::DataTypeToTensorType(elem.tensor_type().elem_type()));
    return seq == TensorType::kUndefined ? std::nullopt : std::optional<TensorType>(seq);
  }
  case TypeProto::kOptionalType: {
    const TypeProto &elem = type.optional_type().elem_type();
    if (elem.has_tensor_type()) {
      const TensorType opt =
          OptTypeOf(symbolic::DataTypeToTensorType(elem.tensor_type().elem_type()));
      return opt == TensorType::kUndefined ? std::nullopt : std::optional<TensorType>(opt);
    }
    if (elem.has_sequence_type()) {
      const TypeProto &seq_elem = elem.sequence_type().elem_type();
      if (!seq_elem.has_tensor_type()) {
        return std::nullopt;
      }
      const TensorType opt_seq =
          OptSeqTypeOf(symbolic::DataTypeToTensorType(seq_elem.tensor_type().elem_type()));
      return opt_seq == TensorType::kUndefined ? std::nullopt : std::optional<TensorType>(opt_seq);
    }
    return std::nullopt;
  }
  default:
    return std::nullopt;
  }
}

/// Resolves the concrete TensorType carried by @p value (see SchemaInputValue), or
/// std::nullopt when it cannot be determined (see TensorTypeFromTypeProto).
std::optional<TensorType> TensorTypeFromInputValue(const SchemaInputValue &value) {
  return std::visit(
      [](const auto &v) -> std::optional<TensorType> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, ValueInfoProto>) {
          return v.has_type() ? TensorTypeFromTypeProto(v.type()) : std::nullopt;
        } else if constexpr (std::is_same_v<T, symbolic::SymTensor>) {
          return v.Dtype() == TensorType::kUndefined ? std::nullopt
                                                     : std::optional<TensorType>(v.Dtype());
        } else {
          static_assert(std::is_same_v<T, symbolic::SymSequence>);
          if (!v.HasElemDtype()) {
            return std::nullopt;
          }
          const TensorType seq = SeqTypeOf(v.ElemDtype());
          return seq == TensorType::kUndefined ? std::nullopt : std::optional<TensorType>(seq);
        }
      },
      value);
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

void LightOpSchema::Verify(const NodeProto &node,
                           const std::vector<std::optional<SchemaInputValue>> *inputs) const {
  if (deprecated()) {
    throw SchemaError(onnx_light_helpers::MakeString(
        "Operator '", name(), "' has been deprecated since version ", since_version(), "."));
  }

  if (node.op_type() != name()) {
    throw SchemaError(onnx_light_helpers::MakeString(
        "Node '", node.name(), "' has op_type '", node.op_type(), "', expected '", name(), "'."));
  }

  const std::string node_domain =
      node.domain().empty() ? std::string(kOnnxDomain) : std::string(node.domain());
  const std::string schema_domain = domain().empty() ? std::string(kOnnxDomain) : domain();
  if (node_domain != schema_domain) {
    throw SchemaError(onnx_light_helpers::MakeString("Node '", node.name(), "' (op_type '",
                                                     node.op_type(), "') has domain '", node_domain,
                                                     "', expected '", schema_domain, "'."));
  }

  const int num_outputs = static_cast<int>(node.output().size());
  if (num_outputs < min_output() || num_outputs > max_output()) {
    throw SchemaError(onnx_light_helpers::MakeString(
        "Node '", node.name(), "' (op_type '", node.op_type(), "') has ", num_outputs,
        " output(s), expected between ", min_output(), " and ", max_output(), "."));
  }

  // Attributes: unrecognized names are rejected (except the "__"-prefixed internal-symbol
  // convention), types must match the declared AttributeParam, and every required attribute
  // must be present.
  std::unordered_set<std::string> seen_attr_names;
  for (const auto &attr : node.attribute()) {
    const std::string attr_name(attr.name().sv());
    if (!seen_attr_names.insert(attr_name).second) {
      throw SchemaError(onnx_light_helpers::MakeString("Node '", node.name(), "' (op_type '",
                                                       node.op_type(), "') has attribute '",
                                                       attr_name, "' more than once."));
    }

    const bool is_internal_symbol =
        attr_name.size() >= 2 && attr_name[0] == '_' && attr_name[1] == '_';
    const auto found = std::find_if(attributes().begin(), attributes().end(),
                                    [&](const AttributeParam &p) { return p.name == attr_name; });
    if (found == attributes().end()) {
      if (is_internal_symbol) {
        continue;
      }
      throw SchemaError(
          onnx_light_helpers::MakeString("Node '", node.name(), "' (op_type '", node.op_type(),
                                         "') has unrecognized attribute '", attr_name, "'."));
    }

    const AttributeType expected_type = found->type;
    const auto actual_type = static_cast<AttributeType>(attr.type());
    if (actual_type != expected_type) {
      throw SchemaError(onnx_light_helpers::MakeString(
          "Node '", node.name(), "' (op_type '", node.op_type(), "') attribute '", attr_name,
          "' has type ", AttributeType_Name(actual_type), ", expected ",
          AttributeType_Name(expected_type), "."));
    }
  }

  for (const auto &attr_param : attributes()) {
    if (attr_param.required && !seen_attr_names.count(attr_param.name)) {
      throw SchemaError(onnx_light_helpers::MakeString(
          "Node '", node.name(), "' (op_type '", node.op_type(),
          "') is missing required attribute '", attr_param.name, "'."));
    }
  }

  if (inputs == nullptr || this->inputs().empty()) {
    return;
  }

  for (std::size_t i = 0; i < inputs->size(); ++i) {
    const std::optional<SchemaInputValue> &entry = (*inputs)[i];
    if (!entry.has_value()) {
      continue;
    }

    // Beyond the declared formal inputs, reuse the last one (the conventional way ONNX
    // describes variadic inputs), since LightOpSchema does not track a separate arity option.
    const FormalParameter &formal =
        i < this->inputs().size() ? this->inputs()[i] : this->inputs().back();
    const auto ctype =
        std::find_if(type_constraints().begin(), type_constraints().end(),
                     [&](const TypeConstraintParam &c) { return c.type_param_str == formal.type; });
    if (ctype == type_constraints().end()) {
      continue; // formal.type does not reference a known type-constraint parameter.
    }

    const std::optional<TensorType> actual = TensorTypeFromInputValue(*entry);
    if (!actual.has_value()) {
      continue; // The concrete type could not be determined; nothing to check.
    }

    const auto &allowed = ctype->allowed_type_strs;
    if (std::find(allowed.begin(), allowed.end(), *actual) == allowed.end()) {
      throw SchemaError(onnx_light_helpers::MakeString(
          "Node '", node.name(), "' (op_type '", node.op_type(), "') input ", i, " ('", formal.name,
          "') has type '", schema::ToTypeString(*actual),
          "', which does not satisfy type constraint '", formal.type, "'."));
    }
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::core::schema
