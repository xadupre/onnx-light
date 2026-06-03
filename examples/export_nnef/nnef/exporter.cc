// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "nnef/exporter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace nnef {

// ---------------------------------------------------------------------------
// Identifier / literal formatting (matches the Python implementation).
// ---------------------------------------------------------------------------

namespace {

const std::set<std::string> &ReservedWords() {
  static const std::set<std::string> kReserved{
      "graph",   "fragment", "tensor",   "integer",  "scalar", "logical", "string", "true",
      "false",   "if",       "else",     "for",      "do",     "while",   "yield",  "extension",
      "version", "external", "variable", "constant", "shape",  "label",   "dtype",  "value"};
  return kReserved;
}

inline bool IsIdentChar(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

inline bool IsAlphaOrUnderscore(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

std::string SanitizeIdent(const std::string &name) {
  if (name.empty())
    return "t";
  std::string out;
  out.reserve(name.size());
  for (char c : name) {
    out.push_back(IsIdentChar(c) ? c : '_');
  }
  return out;
}

std::string ToIdentifier(const std::string &name, std::map<std::string, std::string> &used) {
  auto it = used.find(name);
  if (it != used.end())
    return it->second;
  std::string cleaned = name.empty() ? std::string("t") : SanitizeIdent(name);
  if (cleaned.empty() || !IsAlphaOrUnderscore(cleaned[0])) {
    cleaned = "t_" + cleaned;
  }
  if (ReservedWords().count(cleaned)) {
    cleaned.push_back('_');
  }
  std::set<std::string> existing;
  for (const auto &kv : used)
    existing.insert(kv.second);
  std::string base = cleaned;
  int suffix = 0;
  while (existing.count(cleaned)) {
    ++suffix;
    cleaned = base + "_" + std::to_string(suffix);
  }
  used[name] = cleaned;
  return cleaned;
}

std::string FormatScalarDouble(double v) {
  if (!std::isfinite(v)) {
    throw NNEFExportError("Non-finite literal cannot be encoded in NNEF");
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%g", v);
  return std::string(buf);
}

std::string FormatScalarString(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('\'');
  for (char c : s) {
    if (c == '\\') {
      out.append("\\\\");
    } else if (c == '\'') {
      out.append("\\'");
    } else {
      out.push_back(c);
    }
  }
  out.push_back('\'');
  return out;
}

} // namespace

std::string FormatValue(const AttributeValue &value) {
  return std::visit(
      [](auto &&v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          throw NNEFExportError("Cannot format an unset attribute value");
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return std::to_string(v);
        } else if constexpr (std::is_same_v<T, double>) {
          return FormatScalarDouble(v);
        } else if constexpr (std::is_same_v<T, std::string>) {
          return FormatScalarString(v);
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
          std::string out = "[";
          for (size_t i = 0; i < v.size(); ++i) {
            if (i)
              out += ", ";
            out += std::to_string(v[i]);
          }
          out.push_back(']');
          return out;
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          std::string out = "[";
          for (size_t i = 0; i < v.size(); ++i) {
            if (i)
              out += ", ";
            out += FormatScalarDouble(v[i]);
          }
          out.push_back(']');
          return out;
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          std::string out = "[";
          for (size_t i = 0; i < v.size(); ++i) {
            if (i)
              out += ", ";
            out += FormatScalarString(v[i]);
          }
          out.push_back(']');
          return out;
        } else {
          throw NNEFExportError("Cannot format value as NNEF literal");
        }
      },
      value);
}

namespace {

// Formats a list of int (for nested list-of-list e.g. padding=[[1,1],[1,1]]).
std::string FormatIntList(const std::vector<int64_t> &v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      out += ", ";
    out += std::to_string(v[i]);
  }
  out.push_back(']');
  return out;
}

std::string FormatIntListOfList(const std::vector<std::vector<int64_t>> &v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      out += ", ";
    out += FormatIntList(v[i]);
  }
  out.push_back(']');
  return out;
}

std::string EmitAssign(const std::vector<std::string> &outputs, const std::string &rhs) {
  if (outputs.size() == 1)
    return outputs[0] + " = " + rhs + ";";
  std::string lhs;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (i)
      lhs += ", ";
    lhs += outputs[i];
  }
  return "[" + lhs + "] = " + rhs + ";";
}

// Simple kv pair used by emit_call.
struct KV {
  std::string key;
  std::string formatted_value;
};

std::string EmitCallRaw(const std::string &op, const std::vector<std::string> &args,
                        const std::vector<KV> &kwargs) {
  std::string out = op + "(";
  bool first = true;
  for (const auto &a : args) {
    if (!first)
      out += ", ";
    out += a;
    first = false;
  }
  for (const auto &kv : kwargs) {
    if (!first)
      out += ", ";
    out += kv.key + " = " + kv.formatted_value;
    first = false;
  }
  out.push_back(')');
  return out;
}

std::string EmitCall(const std::string &op, const std::vector<std::string> &args,
                     const std::vector<std::pair<std::string, AttributeValue>> &kwargs) {
  std::vector<KV> kv;
  kv.reserve(kwargs.size());
  for (const auto &p : kwargs) {
    kv.push_back({p.first, FormatValue(p.second)});
  }
  return EmitCallRaw(op, args, kv);
}

} // namespace

// ---------------------------------------------------------------------------
// ExportContext
// ---------------------------------------------------------------------------

ExportContext::ExportContext() = default;

std::string ExportContext::MapName(const std::string &onnx_name) {
  return ToIdentifier(onnx_name, name_map_);
}

std::string ExportContext::MakeTemp(const std::string &base) {
  ++tmp_counter_;
  std::string cleaned = SanitizeIdent(base);
  if (cleaned.empty())
    cleaned = "t";
  std::string name = cleaned + "_tmp" + std::to_string(tmp_counter_);
  return ToIdentifier(name, name_map_);
}

const NNEFTensor *ExportContext::GetInitializer(const std::string &onnx_name) const {
  auto it = initializers_by_onnx_.find(onnx_name);
  if (it == initializers_by_onnx_.end())
    return nullptr;
  return &it->second;
}

// ---------------------------------------------------------------------------
// NNEFGraph::ToText
// ---------------------------------------------------------------------------

std::string NNEFGraph::ToText() const {
  std::ostringstream oss;
  oss << "version " << version_major << "." << version_minor << ";\n";
  for (const auto &ext : extensions) {
    oss << "extension " << ext << ";\n";
  }
  oss << "\n";
  std::string params;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i)
      params += ", ";
    params += inputs[i];
  }
  std::string results;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (i)
      results += ", ";
    results += outputs[i];
  }
  oss << "graph " << name << "(" << params << ") -> (" << results << ")\n{\n";
  for (const auto &stmt : statements) {
    oss << "    " << stmt << "\n";
  }
  oss << "}\n";
  return oss.str();
}

// ---------------------------------------------------------------------------
// TensorProto → NNEFTensor (and dtype helpers).
// ---------------------------------------------------------------------------

namespace {

// Map ONNX TensorProto::DataType → (item_type, bits, bytes_per_elem).
struct DTypeInfo {
  int item_type;
  int bits;
  int bytes; // bytes per element when stored in raw_data
};

bool ResolveDType(int onnx_dtype, DTypeInfo &out) {
  switch (onnx_dtype) {
  case TensorProto::FLOAT:
    out = {kItemTypeFloat, 32, 4};
    return true;
  case TensorProto::FLOAT16:
    out = {kItemTypeFloat, 16, 2};
    return true;
  case TensorProto::DOUBLE:
    out = {kItemTypeFloat, 64, 8};
    return true;
  case TensorProto::INT8:
    out = {kItemTypeSigned, 8, 1};
    return true;
  case TensorProto::INT16:
    out = {kItemTypeSigned, 16, 2};
    return true;
  case TensorProto::INT32:
    out = {kItemTypeSigned, 32, 4};
    return true;
  case TensorProto::INT64:
    out = {kItemTypeSigned, 64, 8};
    return true;
  case TensorProto::UINT8:
    out = {kItemTypeUnsigned, 8, 1};
    return true;
  case TensorProto::UINT16:
    out = {kItemTypeUnsigned, 16, 2};
    return true;
  case TensorProto::UINT32:
    out = {kItemTypeUnsigned, 32, 4};
    return true;
  case TensorProto::UINT64:
    out = {kItemTypeUnsigned, 64, 8};
    return true;
  case TensorProto::BOOL:
    out = {kItemTypeBool, 8, 1};
    return true;
  default:
    return false;
  }
}

template <typename SrcT, typename DstT>
void CopyCast(const utils::RepeatedField<SrcT> &src, std::vector<uint8_t> &dst) {
  dst.resize(src.size() * sizeof(DstT));
  DstT *dp = reinterpret_cast<DstT *>(dst.data());
  for (size_t i = 0; i < src.size(); ++i) {
    dp[i] = static_cast<DstT>(src[i]);
  }
}

} // namespace

NNEFTensor TensorProtoToNNEFTensor(const TensorProto &tensor) {
  DTypeInfo info;
  if (!ResolveDType(static_cast<int>(tensor.data_type()), info)) {
    throw NNEFExportError("Unsupported ONNX tensor data_type=" +
                          std::to_string(static_cast<int>(tensor.data_type())));
  }
  NNEFTensor t;
  t.item_type = info.item_type;
  t.bits = info.bits;
  for (auto d : tensor.dims()) {
    t.shape.push_back(static_cast<int64_t>(d));
  }
  int64_t nelem = 1;
  for (auto d : t.shape)
    nelem *= d;
  if (t.shape.empty())
    nelem = 1;

  const auto &raw = tensor.raw_data();
  if (!raw.empty()) {
    t.data.assign(raw.data(), raw.data() + raw.size());
    return t;
  }
  // Decode from typed fields.
  switch (tensor.data_type()) {
  case TensorProto::FLOAT:
    if (tensor.float_data().size() > 0) {
      const auto &f = tensor.float_data();
      t.data.resize(f.size() * sizeof(float));
      std::memcpy(t.data.data(), &f[0], t.data.size());
      return t;
    }
    break;
  case TensorProto::DOUBLE:
    if (tensor.double_data().size() > 0) {
      const auto &f = tensor.double_data();
      t.data.resize(f.size() * sizeof(double));
      std::memcpy(t.data.data(), &f[0], t.data.size());
      return t;
    }
    break;
  case TensorProto::INT64:
    if (tensor.int64_data().size() > 0) {
      const auto &f = tensor.int64_data();
      t.data.resize(f.size() * sizeof(int64_t));
      std::memcpy(t.data.data(), &f[0], t.data.size());
      return t;
    }
    break;
  case TensorProto::UINT32:
  case TensorProto::UINT64:
    if (tensor.uint64_data().size() > 0) {
      const auto &f = tensor.uint64_data();
      if (tensor.data_type() == TensorProto::UINT64) {
        t.data.resize(f.size() * sizeof(uint64_t));
        std::memcpy(t.data.data(), &f[0], t.data.size());
      } else {
        CopyCast<uint64_t, uint32_t>(f, t.data);
      }
      return t;
    }
    break;
  case TensorProto::INT32:
  case TensorProto::INT16:
  case TensorProto::INT8:
  case TensorProto::UINT16:
  case TensorProto::UINT8:
  case TensorProto::BOOL:
  case TensorProto::FLOAT16:
    if (tensor.int32_data().size() > 0) {
      const auto &f = tensor.int32_data();
      switch (tensor.data_type()) {
      case TensorProto::INT32:
        CopyCast<int32_t, int32_t>(f, t.data);
        break;
      case TensorProto::INT16:
        CopyCast<int32_t, int16_t>(f, t.data);
        break;
      case TensorProto::INT8:
        CopyCast<int32_t, int8_t>(f, t.data);
        break;
      case TensorProto::UINT16:
        CopyCast<int32_t, uint16_t>(f, t.data);
        break;
      case TensorProto::UINT8:
        CopyCast<int32_t, uint8_t>(f, t.data);
        break;
      case TensorProto::BOOL:
        CopyCast<int32_t, uint8_t>(f, t.data);
        break;
      case TensorProto::FLOAT16:
        CopyCast<int32_t, uint16_t>(f, t.data);
        break;
      default:
        break;
      }
      return t;
    }
    break;
  default:
    break;
  }
  // No data anywhere: zero buffer of the right size.
  t.data.assign(static_cast<size_t>(nelem) * static_cast<size_t>(info.bytes), 0);
  return t;
}

// ---------------------------------------------------------------------------
// Attribute extraction.
// ---------------------------------------------------------------------------

namespace {

AttributeValue ExtractAttribute(const AttributeProto &attr) {
  switch (attr.type()) {
  case AttributeProto::INT:
    return static_cast<int64_t>(attr.i());
  case AttributeProto::FLOAT:
    return static_cast<double>(attr.f());
  case AttributeProto::STRING:
    return attr.s().as_string();
  case AttributeProto::INTS: {
    std::vector<int64_t> v;
    for (auto x : attr.ints())
      v.push_back(static_cast<int64_t>(x));
    return v;
  }
  case AttributeProto::FLOATS: {
    std::vector<double> v;
    for (auto x : attr.floats())
      v.push_back(static_cast<double>(x));
    return v;
  }
  case AttributeProto::STRINGS: {
    std::vector<std::string> v;
    for (const auto &x : attr.strings())
      v.push_back(x.as_string());
    return v;
  }
  case AttributeProto::TENSOR:
    return TensorProtoToNNEFTensor(attr.t());
  default:
    throw NNEFExportError("Unsupported ONNX attribute type: " +
                          std::to_string(static_cast<int>(attr.type())));
  }
}

std::map<std::string, AttributeValue> AttrsToDict(const NodeProto &node) {
  std::map<std::string, AttributeValue> out;
  for (const auto &a : node.attribute()) {
    out[a.name().as_string()] = ExtractAttribute(a);
  }
  return out;
}

// Reads a 1-D int64 tensor (after Decode) into a vector<int64_t>.
std::vector<int64_t> NNEFTensorToInt64Vec(const NNEFTensor &t) {
  std::vector<int64_t> out;
  if (t.item_type == kItemTypeSigned && t.bits == 64) {
    size_t n = t.data.size() / 8;
    out.resize(n);
    std::memcpy(out.data(), t.data.data(), n * 8);
  } else if (t.item_type == kItemTypeSigned && t.bits == 32) {
    size_t n = t.data.size() / 4;
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
      int32_t v;
      std::memcpy(&v, t.data.data() + i * 4, 4);
      out[i] = v;
    }
  } else {
    throw NNEFExportError("Expected integer tensor for shape-like initializer");
  }
  return out;
}

double NNEFTensorFirstScalarAsDouble(const NNEFTensor &t) {
  if (t.data.empty())
    throw NNEFExportError("Empty initializer tensor");
  if (t.item_type == kItemTypeFloat && t.bits == 32) {
    float v;
    std::memcpy(&v, t.data.data(), 4);
    return static_cast<double>(v);
  }
  if (t.item_type == kItemTypeFloat && t.bits == 64) {
    double v;
    std::memcpy(&v, t.data.data(), 8);
    return v;
  }
  if (t.item_type == kItemTypeSigned && t.bits == 64) {
    int64_t v;
    std::memcpy(&v, t.data.data(), 8);
    return static_cast<double>(v);
  }
  if (t.item_type == kItemTypeSigned && t.bits == 32) {
    int32_t v;
    std::memcpy(&v, t.data.data(), 4);
    return static_cast<double>(v);
  }
  throw NNEFExportError("Cannot interpret initializer as a scalar");
}

// ---------------------------------------------------------------------------
// Helpers used by builtin converters.
// ---------------------------------------------------------------------------

const std::vector<int64_t> *GetInts(const std::map<std::string, AttributeValue> &attrs,
                                    const std::string &k) {
  auto it = attrs.find(k);
  if (it == attrs.end())
    return nullptr;
  return std::get_if<std::vector<int64_t>>(&it->second);
}

const int64_t *GetInt(const std::map<std::string, AttributeValue> &attrs, const std::string &k) {
  auto it = attrs.find(k);
  if (it == attrs.end())
    return nullptr;
  return std::get_if<int64_t>(&it->second);
}

const double *GetFloat(const std::map<std::string, AttributeValue> &attrs, const std::string &k) {
  auto it = attrs.find(k);
  if (it == attrs.end())
    return nullptr;
  return std::get_if<double>(&it->second);
}

const std::string *GetString(const std::map<std::string, AttributeValue> &attrs,
                             const std::string &k) {
  auto it = attrs.find(k);
  if (it == attrs.end())
    return nullptr;
  return std::get_if<std::string>(&it->second);
}

std::vector<std::vector<int64_t>> ConvPadding(const std::map<std::string, AttributeValue> &attrs,
                                              int rank) {
  const auto *pads = GetInts(attrs, "pads");
  if (pads == nullptr) {
    const std::string *ap = GetString(attrs, "auto_pad");
    if (ap && (*ap == "SAME_UPPER" || *ap == "SAME_LOWER")) {
      return {};
    }
    return std::vector<std::vector<int64_t>>(rank, std::vector<int64_t>{0, 0});
  }
  if (static_cast<int>(pads->size()) != 2 * rank) {
    throw NNEFExportError("Expected " + std::to_string(2 * rank) + " pad values, got " +
                          std::to_string(pads->size()));
  }
  std::vector<std::vector<int64_t>> out;
  out.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    out.push_back({(*pads)[i], (*pads)[i + rank]});
  }
  return out;
}

// Builtin converters --------------------------------------------------------

ConverterFn MakeUnary(const std::string &nnef_op) {
  return [nnef_op](
             ExportContext &ctx, const NodeProto &, const std::map<std::string, AttributeValue> &,
             const std::vector<std::string> &inputs, const std::vector<std::string> &outputs) {
    ctx.AddStatement(EmitAssign(outputs, EmitCall(nnef_op, {inputs[0]}, {})));
  };
}

ConverterFn MakeBinary(const std::string &nnef_op) {
  return [nnef_op](
             ExportContext &ctx, const NodeProto &, const std::map<std::string, AttributeValue> &,
             const std::vector<std::string> &inputs, const std::vector<std::string> &outputs) {
    ctx.AddStatement(EmitAssign(outputs, EmitCall(nnef_op, {inputs[0], inputs[1]}, {})));
  };
}

void ConvConverter(ExportContext &ctx, const NodeProto &node,
                   const std::map<std::string, AttributeValue> &attrs,
                   const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs) {
  if (inputs.size() < 2)
    throw NNEFExportError("Conv requires at least input and weight");
  const NNEFTensor *init_w = nullptr;
  if (node.input().size() >= 2) {
    init_w = ctx.GetInitializer(node.input(1).as_string());
  }
  int spatial_rank = 0;
  if (init_w != nullptr) {
    int rank = static_cast<int>(init_w->shape.size());
    spatial_rank = std::max(0, rank - 2);
  } else if (const auto *ks = GetInts(attrs, "kernel_shape")) {
    spatial_rank = static_cast<int>(ks->size());
  }
  std::string bias = inputs.size() >= 3 ? inputs[2] : std::string("0.0");

  std::vector<KV> kvs;
  if (const auto *strides = GetInts(attrs, "strides"))
    kvs.push_back({"stride", FormatIntList(*strides)});
  if (const auto *dilations = GetInts(attrs, "dilations"))
    kvs.push_back({"dilation", FormatIntList(*dilations)});
  if (spatial_rank > 0) {
    kvs.push_back({"padding", FormatIntListOfList(ConvPadding(attrs, spatial_rank))});
  }
  if (const auto *g = GetInt(attrs, "group")) {
    if (*g != 1)
      kvs.push_back({"groups", std::to_string(*g)});
  }
  std::vector<std::string> args = {inputs[0], inputs[1], bias};
  ctx.AddStatement(EmitAssign(outputs, EmitCallRaw("conv", args, kvs)));
}

ConverterFn MakePool(const std::string &nnef_op) {
  return
      [nnef_op](ExportContext &ctx, const NodeProto &node,
                const std::map<std::string, AttributeValue> &attrs,
                const std::vector<std::string> &inputs, const std::vector<std::string> &outputs) {
        const auto *kernel = GetInts(attrs, "kernel_shape");
        if (kernel == nullptr)
          throw NNEFExportError(node.op_type().as_string() + " requires kernel_shape");
        int rank = static_cast<int>(kernel->size());
        std::vector<int64_t> size = {1, 1};
        size.insert(size.end(), kernel->begin(), kernel->end());
        std::vector<KV> kvs;
        kvs.push_back({"size", FormatIntList(size)});
        if (const auto *strides = GetInts(attrs, "strides")) {
          std::vector<int64_t> s = {1, 1};
          s.insert(s.end(), strides->begin(), strides->end());
          kvs.push_back({"stride", FormatIntList(s)});
        }
        bool has_pads = attrs.find("pads") != attrs.end();
        const std::string *ap = GetString(attrs, "auto_pad");
        if (has_pads || (ap && *ap != "NOTSET")) {
          std::vector<std::vector<int64_t>> p = {{0, 0}, {0, 0}};
          auto cp = ConvPadding(attrs, rank);
          p.insert(p.end(), cp.begin(), cp.end());
          kvs.push_back({"padding", FormatIntListOfList(p)});
        }
        ctx.AddStatement(EmitAssign(outputs, EmitCallRaw(nnef_op, {inputs[0]}, kvs)));
      };
}

ConverterFn MakeGlobalPool(const std::string &nnef_op) {
  return [nnef_op](
             ExportContext &ctx, const NodeProto &, const std::map<std::string, AttributeValue> &,
             const std::vector<std::string> &inputs, const std::vector<std::string> &outputs) {
    ctx.AddStatement(EmitAssign(
        outputs, EmitCall(nnef_op, {inputs[0]}, {{"axes", std::vector<int64_t>{2, 3}}})));
  };
}

void ReshapeConverter(ExportContext &ctx, const NodeProto &node,
                      const std::map<std::string, AttributeValue> &,
                      const std::vector<std::string> &inputs,
                      const std::vector<std::string> &outputs) {
  const NNEFTensor *init = nullptr;
  if (node.input().size() >= 2)
    init = ctx.GetInitializer(node.input(1).as_string());
  if (init == nullptr)
    throw NNEFExportError("Reshape requires a constant shape initializer to export to NNEF");
  auto shape = NNEFTensorToInt64Vec(*init);
  ctx.AddStatement(EmitAssign(outputs, EmitCall("reshape", {inputs[0]}, {{"shape", shape}})));
}

void TransposeConverter(ExportContext &ctx, const NodeProto &,
                        const std::map<std::string, AttributeValue> &attrs,
                        const std::vector<std::string> &inputs,
                        const std::vector<std::string> &outputs) {
  std::vector<std::pair<std::string, AttributeValue>> kv;
  if (const auto *perm = GetInts(attrs, "perm")) {
    kv.push_back({"axes", *perm});
  }
  ctx.AddStatement(EmitAssign(outputs, EmitCall("transpose", {inputs[0]}, kv)));
}

void ConcatConverter(ExportContext &ctx, const NodeProto &,
                     const std::map<std::string, AttributeValue> &attrs,
                     const std::vector<std::string> &inputs,
                     const std::vector<std::string> &outputs) {
  int64_t axis = 0;
  if (const auto *a = GetInt(attrs, "axis"))
    axis = *a;
  std::string joined;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i)
      joined += ", ";
    joined += inputs[i];
  }
  std::string rhs = "concat([" + joined + "], axis = " + std::to_string(axis) + ")";
  ctx.AddStatement(EmitAssign(outputs, rhs));
}

void SoftmaxConverter(ExportContext &ctx, const NodeProto &,
                      const std::map<std::string, AttributeValue> &attrs,
                      const std::vector<std::string> &inputs,
                      const std::vector<std::string> &outputs) {
  int64_t axis = -1;
  if (const auto *a = GetInt(attrs, "axis"))
    axis = *a;
  ctx.AddStatement(EmitAssign(
      outputs, EmitCall("softmax", {inputs[0]}, {{"axes", std::vector<int64_t>{axis}}})));
}

void FlattenConverter(ExportContext &ctx, const NodeProto &,
                      const std::map<std::string, AttributeValue> &attrs,
                      const std::vector<std::string> &inputs,
                      const std::vector<std::string> &outputs) {
  int64_t axis = 1;
  if (const auto *a = GetInt(attrs, "axis"))
    axis = *a;
  std::vector<int64_t> shape;
  if (axis == 1) {
    shape = {0, -1};
  } else {
    shape.assign(static_cast<size_t>(axis), 0);
    shape.push_back(-1);
  }
  ctx.AddStatement(EmitAssign(outputs, EmitCall("reshape", {inputs[0]}, {{"shape", shape}})));
}

void GemmConverter(ExportContext &ctx, const NodeProto &,
                   const std::map<std::string, AttributeValue> &attrs,
                   const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs) {
  if (inputs.size() < 2)
    throw NNEFExportError("Gemm requires at least A and B");
  double alpha = 1.0, beta = 1.0;
  if (const auto *a = GetFloat(attrs, "alpha"))
    alpha = *a;
  if (const auto *a = GetFloat(attrs, "beta"))
    beta = *a;
  int64_t trans_a = 0, trans_b = 0;
  if (const auto *t = GetInt(attrs, "transA"))
    trans_a = *t;
  if (const auto *t = GetInt(attrs, "transB"))
    trans_b = *t;
  if (alpha != 1.0 || beta != 1.0)
    throw NNEFExportError("Gemm export only supports alpha=beta=1");
  std::string a = inputs[0];
  std::string b = inputs[1];
  if (trans_a) {
    std::string ta = ctx.MakeTemp("gemm_a");
    ctx.AddStatement(EmitAssign({ta}, EmitCall("transpose", {a}, {})));
    a = ta;
  }
  if (trans_b) {
    std::string tb = ctx.MakeTemp("gemm_b");
    ctx.AddStatement(EmitAssign({tb}, EmitCall("transpose", {b}, {})));
    b = tb;
  }
  std::string matmul_out = (inputs.size() < 3) ? outputs[0] : ctx.MakeTemp("gemm_mm");
  ctx.AddStatement(EmitAssign({matmul_out}, EmitCall("matmul", {a, b}, {})));
  if (inputs.size() >= 3) {
    ctx.AddStatement(EmitAssign(outputs, EmitCall("add", {matmul_out, inputs[2]}, {})));
  }
}

void MatMulConverter(ExportContext &ctx, const NodeProto &,
                     const std::map<std::string, AttributeValue> &,
                     const std::vector<std::string> &inputs,
                     const std::vector<std::string> &outputs) {
  ctx.AddStatement(EmitAssign(outputs, EmitCall("matmul", {inputs[0], inputs[1]}, {})));
}

void BatchNormConverter(ExportContext &ctx, const NodeProto &,
                        const std::map<std::string, AttributeValue> &attrs,
                        const std::vector<std::string> &inputs,
                        const std::vector<std::string> &outputs) {
  if (inputs.size() < 5)
    throw NNEFExportError("BatchNormalization requires 5 inputs");
  double eps = 1e-5;
  if (const auto *e = GetFloat(attrs, "epsilon"))
    eps = *e;
  std::vector<std::string> args = {inputs[0], inputs[3], inputs[4], inputs[2], inputs[1]};
  std::vector<std::string> out1 = {outputs[0]};
  ctx.AddStatement(EmitAssign(out1, EmitCall("batch_normalization", args, {{"epsilon", eps}})));
}

void IdentityConverter(ExportContext &ctx, const NodeProto &,
                       const std::map<std::string, AttributeValue> &,
                       const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs) {
  ctx.AddStatement(EmitAssign(outputs, EmitCall("copy", {inputs[0]}, {})));
}

void ClipConverter(ExportContext &ctx, const NodeProto &node,
                   const std::map<std::string, AttributeValue> &attrs,
                   const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs) {
  double lo = -std::numeric_limits<double>::infinity();
  double hi = std::numeric_limits<double>::infinity();
  bool has_lo = false, has_hi = false;
  if (const auto *m = GetFloat(attrs, "min")) {
    lo = *m;
    has_lo = true;
  }
  if (const auto *m = GetFloat(attrs, "max")) {
    hi = *m;
    has_hi = true;
  }
  if (!has_lo && inputs.size() >= 2) {
    const NNEFTensor *init = ctx.GetInitializer(node.input(1).as_string());
    if (init == nullptr)
      throw NNEFExportError("Clip min must be a constant initializer for NNEF export");
    lo = NNEFTensorFirstScalarAsDouble(*init);
    has_lo = true;
  }
  if (!has_hi && inputs.size() >= 3) {
    const NNEFTensor *init = ctx.GetInitializer(node.input(2).as_string());
    if (init == nullptr)
      throw NNEFExportError("Clip max must be a constant initializer for NNEF export");
    hi = NNEFTensorFirstScalarAsDouble(*init);
    has_hi = true;
  }
  if (!std::isfinite(lo) && !std::isfinite(hi)) {
    ctx.AddStatement(EmitAssign(outputs, EmitCall("copy", {inputs[0]}, {})));
    return;
  }
  std::vector<std::string> args = {inputs[0], FormatScalarDouble(lo), FormatScalarDouble(hi)};
  ctx.AddStatement(EmitAssign(outputs, EmitCallRaw("clamp", args, {})));
}

// ---------------------------------------------------------------------------
// Registry.
// ---------------------------------------------------------------------------

std::mutex &RegistryMutex() {
  static std::mutex m;
  return m;
}

std::map<std::string, ConverterFn> &Registry() {
  static std::map<std::string, ConverterFn> r;
  return r;
}

void EnsureBuiltinsRegistered() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    auto &r = Registry();
    auto add = [&](const std::string &op, ConverterFn fn) { r[op] = std::move(fn); };
    for (auto &p : std::vector<std::pair<std::string, std::string>>{
             {"Relu", "relu"},
             {"Sigmoid", "sigmoid"},
             {"Tanh", "tanh"},
             {"Softplus", "softplus"},
             {"Exp", "exp"},
             {"Log", "log"},
             {"Sqrt", "sqrt"},
             {"Neg", "neg"},
             {"Abs", "abs"},
             {"Floor", "floor"},
             {"Ceil", "ceil"},
             {"Sin", "sin"},
             {"Cos", "cos"},
             {"Not", "not"},
         }) {
      add(p.first, MakeUnary(p.second));
    }
    for (auto &p : std::vector<std::pair<std::string, std::string>>{
             {"Add", "add"},
             {"Sub", "sub"},
             {"Mul", "mul"},
             {"Div", "div"},
             {"Pow", "pow"},
             {"Min", "min"},
             {"Max", "max"},
             {"And", "and"},
             {"Or", "or"},
             {"Equal", "eq"},
             {"Less", "lt"},
             {"Greater", "gt"},
         }) {
      add(p.first, MakeBinary(p.second));
    }
    add("Conv", ConvConverter);
    add("MaxPool", MakePool("max_pool"));
    add("AveragePool", MakePool("avg_pool"));
    add("GlobalAveragePool", MakeGlobalPool("mean_reduce"));
    add("GlobalMaxPool", MakeGlobalPool("max_reduce"));
    add("Reshape", ReshapeConverter);
    add("Transpose", TransposeConverter);
    add("Concat", ConcatConverter);
    add("Softmax", SoftmaxConverter);
    add("Flatten", FlattenConverter);
    add("Gemm", GemmConverter);
    add("MatMul", MatMulConverter);
    add("BatchNormalization", BatchNormConverter);
    add("Identity", IdentityConverter);
    add("Clip", ClipConverter);
  });
}

} // namespace

void RegisterOpConverter(const std::string &op_type, ConverterFn converter) {
  EnsureBuiltinsRegistered();
  std::lock_guard<std::mutex> lk(RegistryMutex());
  Registry()[op_type] = std::move(converter);
}

bool UnregisterOpConverter(const std::string &op_type) {
  EnsureBuiltinsRegistered();
  std::lock_guard<std::mutex> lk(RegistryMutex());
  return Registry().erase(op_type) > 0;
}

std::vector<std::string> SupportedOps() {
  EnsureBuiltinsRegistered();
  std::lock_guard<std::mutex> lk(RegistryMutex());
  std::vector<std::string> out;
  out.reserve(Registry().size());
  for (const auto &kv : Registry())
    out.push_back(kv.first);
  std::sort(out.begin(), out.end());
  return out;
}

bool HasOpConverter(const std::string &op_type) {
  EnsureBuiltinsRegistered();
  std::lock_guard<std::mutex> lk(RegistryMutex());
  return Registry().count(op_type) > 0;
}

const ConverterFn &GetOpConverter(const std::string &op_type) {
  EnsureBuiltinsRegistered();
  std::lock_guard<std::mutex> lk(RegistryMutex());
  auto it = Registry().find(op_type);
  if (it == Registry().end())
    throw NNEFExportError("No NNEF converter registered for ONNX op '" + op_type + "'");
  return it->second;
}

// ---------------------------------------------------------------------------
// ExportToNNEF
// ---------------------------------------------------------------------------

namespace {

struct ValueInfoShape {
  std::vector<int64_t> shape; // -1 for unknown
  bool has_shape = false;
};

ValueInfoShape ShapeFromValueInfo(const ValueInfoProto &vi) {
  ValueInfoShape out;
  if (!vi.has_type())
    return out;
  const auto &tp = vi.type();
  if (!tp.has_tensor_type())
    return out;
  const auto &tt = tp.tensor_type();
  if (!tt.has_shape())
    return out;
  out.has_shape = true;
  for (const auto &d : tt.shape().dim()) {
    if (d.has_dim_value())
      out.shape.push_back(static_cast<int64_t>(d.dim_value()));
    else
      out.shape.push_back(-1);
  }
  return out;
}

std::string SanitizeGraphName(const std::string &name) {
  std::string cleaned;
  for (char c : name) {
    cleaned.push_back(IsIdentChar(c) ? c : '_');
  }
  if (cleaned.empty())
    cleaned = "main";
  return cleaned;
}

} // namespace

NNEFGraph ExportToNNEF(const ModelProto &model, const std::string &graph_name) {
  EnsureBuiltinsRegistered();
  if (!model.has_graph())
    throw NNEFExportError("Model has no graph");
  const auto &graph = model.graph();
  std::string name = !graph_name.empty()
                         ? graph_name
                         : (!graph.name().empty() ? graph.name().as_string() : std::string("main"));
  name = SanitizeGraphName(name);

  ExportContext ctx;

  std::set<std::string> initializer_names;
  for (const auto &init : graph.initializer()) {
    initializer_names.insert(init.name().as_string());
  }

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // Map input names (skip those that are also initializers).
  for (const auto &vi : graph.input()) {
    const std::string n = vi.name().as_string();
    if (initializer_names.count(n))
      continue;
    std::string nnef_name = ctx.MapName(n);
    input_names.push_back(nnef_name);
  }
  // Materialize initializers.
  std::vector<std::pair<std::string, NNEFTensor>> ordered_inits;
  for (const auto &init : graph.initializer()) {
    NNEFTensor t = TensorProtoToNNEFTensor(init);
    std::string n = init.name().as_string();
    std::string nnef_name = ctx.MapName(n);
    ctx.SetInitializer(n, t);
    ordered_inits.push_back({nnef_name, std::move(t)});
  }
  // Outputs.
  for (const auto &vi : graph.output()) {
    std::string nnef_name = ctx.MapName(vi.name().as_string());
    output_names.push_back(nnef_name);
  }

  // Emit external statements for graph inputs.
  for (const auto &vi : graph.input()) {
    const std::string n = vi.name().as_string();
    if (initializer_names.count(n))
      continue;
    std::string nnef_name = ctx.MapName(n);
    auto sh = ShapeFromValueInfo(vi);
    std::vector<KV> kvs;
    if (sh.has_shape) {
      std::vector<int64_t> s;
      for (auto d : sh.shape)
        s.push_back(d >= 0 ? d : 1);
      kvs.push_back({"shape", FormatIntList(s)});
    }
    ctx.AddStatement(EmitAssign({nnef_name}, EmitCallRaw("external", {}, kvs)));
  }

  // Emit variable statements for initializers (in initializer order).
  for (size_t i = 0; i < graph.initializer().size(); ++i) {
    const auto &init = graph.initializer(i);
    const auto &pair = ordered_inits[i];
    std::vector<KV> kvs;
    kvs.push_back({"shape", FormatIntList(pair.second.shape)});
    kvs.push_back({"label", FormatScalarString(init.name().as_string())});
    ctx.AddStatement(EmitAssign({pair.first}, EmitCallRaw("variable", {}, kvs)));
  }

  // Walk nodes.
  for (const auto &node : graph.node()) {
    std::string op = node.op_type().as_string();
    if (!HasOpConverter(op)) {
      throw NNEFExportError("No NNEF converter registered for ONNX op '" + op + "' (node '" +
                            node.name().as_string() +
                            "'). Use onnx_light.nnef.register_op_converter to add one.");
    }
    auto attrs = AttrsToDict(node);
    std::vector<std::string> inputs;
    for (const auto &n : node.input()) {
      const std::string s = n.as_string();
      inputs.push_back(s.empty() ? std::string("0.0") : ctx.MapName(s));
    }
    std::vector<std::string> outputs;
    for (const auto &n : node.output()) {
      const std::string s = n.as_string();
      if (s.empty()) {
        outputs.push_back(ctx.MakeTemp("unused"));
      } else {
        outputs.push_back(ctx.MapName(s));
      }
    }
    GetOpConverter(op)(ctx, node, attrs, inputs, outputs);
  }

  NNEFGraph g;
  g.name = name;
  g.inputs = std::move(input_names);
  g.outputs = std::move(output_names);
  g.statements = ctx.Statements();
  g.initializers = std::move(ordered_inits);
  return g;
}

std::string ToNNEFText(const ModelProto &model, const std::string &graph_name) {
  return ExportToNNEF(model, graph_name).ToText();
}

std::string SaveNNEF(const ModelProto &model, const std::string &out_dir,
                     const std::string &graph_name, bool overwrite) {
  auto g = ExportToNNEF(model, graph_name);
  namespace fs = std::filesystem;
  fs::path dir(out_dir);
  if (fs::is_directory(dir)) {
    if (!overwrite && !fs::is_empty(dir)) {
      throw std::runtime_error(std::string("FileExistsError: ") + dir.string());
    }
  } else {
    fs::create_directories(dir);
  }
  {
    std::ofstream f(dir / "graph.nnef");
    if (!f)
      throw std::runtime_error("Cannot write graph.nnef");
    f << g.ToText();
  }
  for (const auto &p : g.initializers) {
    WriteNNEFTensor((dir / (p.first + ".dat")).string(), p.second);
  }
  return fs::absolute(dir).string();
}

} // namespace nnef
} // namespace ONNX_LIGHT_NAMESPACE
