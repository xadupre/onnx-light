// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Experimental language syntax and parser for ONNX. Please note that the syntax as formalized
// by this parser is preliminary and may change.

#pragma once

#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx/common/onnx_pb.h"
#include "onnx/common/status.h"
#include "onnx/common/string_utils.h"

namespace ONNX_LIGHT_NAMESPACE {

/// List of identifiers used in parser productions.
using IdList = std::vector<std::string>;

/// List of parsed node definitions.
using NodeList = std::vector<NodeProto>;

/// List of parsed node attributes.
using AttrList = std::vector<AttributeProto>;

/// List of value-info records used for graph/function signatures.
using ValueInfoList = std::vector<ValueInfoProto>;

/// List of tensor literals and initializers.
using TensorList = std::vector<TensorProto>;

/// List of opset imports.
using OpsetIdList = std::vector<OperatorSetIdProto>;

/// List of key/value metadata entries.
using StringStringList = std::vector<StringStringEntryProto>;

#define CHECK_PARSER_STATUS(status)                                                                \
  {                                                                                                \
    auto local_status_ = status;                                                                   \
    if (!local_status_.IsOK())                                                                     \
      return local_status_;                                                                        \
  }

template <typename Map>
// NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
class StringIntMap {
public:
  /// Returns the singleton name-to-value map instance.
  static const std::unordered_map<std::string, int32_t> &Instance() {
    static Map instance;
    return instance.map_;
  }

  /// Looks up a value by textual name.
  static int32_t Lookup(const std::string &dtype) {
    auto it = Instance().find(dtype);
    if (it != Instance().end())
      return it->second;
    return 0;
  }

  /// Returns a textual name for an enum/integer value, or `"undefined"`.
  static const std::string &ToString(int32_t dtype) {
    static std::string undefined("undefined");
    for (const auto &[name, value] : Instance()) {
      if (value == dtype)
        return name;
    }
    return undefined;
  }

protected:
  std::unordered_map<std::string, int32_t> map_;
};

class PrimitiveTypeNameMap : public StringIntMap<PrimitiveTypeNameMap> {
public:
  PrimitiveTypeNameMap() : StringIntMap() {
    map_["float"] = static_cast<int32_t>(TensorProto::DataType::FLOAT);
    map_["uint8"] = static_cast<int32_t>(TensorProto::DataType::UINT8);
    map_["int8"] = static_cast<int32_t>(TensorProto::DataType::INT8);
    map_["uint16"] = static_cast<int32_t>(TensorProto::DataType::UINT16);
    map_["int16"] = static_cast<int32_t>(TensorProto::DataType::INT16);
    map_["int32"] = static_cast<int32_t>(TensorProto::DataType::INT32);
    map_["int64"] = static_cast<int32_t>(TensorProto::DataType::INT64);
    map_["string"] = static_cast<int32_t>(TensorProto::DataType::STRING);
    map_["bool"] = static_cast<int32_t>(TensorProto::DataType::BOOL);
    map_["float16"] = static_cast<int32_t>(TensorProto::DataType::FLOAT16);
    map_["double"] = static_cast<int32_t>(TensorProto::DataType::DOUBLE);
    map_["uint32"] = static_cast<int32_t>(TensorProto::DataType::UINT32);
    map_["uint64"] = static_cast<int32_t>(TensorProto::DataType::UINT64);
    map_["complex64"] = static_cast<int32_t>(TensorProto::DataType::COMPLEX64);
    map_["complex128"] = static_cast<int32_t>(TensorProto::DataType::COMPLEX128);
    map_["bfloat16"] = static_cast<int32_t>(TensorProto::DataType::BFLOAT16);
    map_["float8e4m3fn"] = static_cast<int32_t>(TensorProto::DataType::FLOAT8E4M3FN);
    map_["float8e4m3fnuz"] = static_cast<int32_t>(TensorProto::DataType::FLOAT8E4M3FNUZ);
    map_["float8e5m2"] = static_cast<int32_t>(TensorProto::DataType::FLOAT8E5M2);
    map_["float8e5m2fnuz"] = static_cast<int32_t>(TensorProto::DataType::FLOAT8E5M2FNUZ);
    map_["float8e8m0"] = static_cast<int32_t>(TensorProto::DataType::FLOAT8E8M0);
    map_["uint4"] = static_cast<int32_t>(TensorProto::DataType::UINT4);
    map_["int4"] = static_cast<int32_t>(TensorProto::DataType::INT4);
    map_["float4e2m1"] = static_cast<int32_t>(TensorProto::DataType::FLOAT4E2M1);
    map_["uint2"] = static_cast<int32_t>(TensorProto::DataType::UINT2);
    map_["int2"] = static_cast<int32_t>(TensorProto::DataType::INT2);
  }

  static bool IsTypeName(const std::string &dtype) { return Lookup(dtype) != 0; }
};

class AttributeTypeNameMap : public StringIntMap<AttributeTypeNameMap> {
public:
  AttributeTypeNameMap() : StringIntMap() {
    map_["float"] = static_cast<int32_t>(AttributeProto::AttributeType::FLOAT);
    map_["int"] = static_cast<int32_t>(AttributeProto::AttributeType::INT);
    map_["string"] = static_cast<int32_t>(AttributeProto::AttributeType::STRING);
    map_["tensor"] = static_cast<int32_t>(AttributeProto::AttributeType::TENSOR);
    map_["graph"] = static_cast<int32_t>(AttributeProto::AttributeType::GRAPH);
    map_["sparse_tensor"] = static_cast<int32_t>(AttributeProto::AttributeType::SPARSE_TENSOR);
    map_["type_proto"] = static_cast<int32_t>(AttributeProto::AttributeType::TYPE_PROTO);
    map_["floats"] = static_cast<int32_t>(AttributeProto::AttributeType::FLOATS);
    map_["ints"] = static_cast<int32_t>(AttributeProto::AttributeType::INTS);
    map_["strings"] = static_cast<int32_t>(AttributeProto::AttributeType::STRINGS);
    map_["tensors"] = static_cast<int32_t>(AttributeProto::AttributeType::TENSORS);
    map_["graphs"] = static_cast<int32_t>(AttributeProto::AttributeType::GRAPHS);
    map_["sparse_tensors"] = static_cast<int32_t>(AttributeProto::AttributeType::SPARSE_TENSORS);
    map_["type_protos"] = static_cast<int32_t>(AttributeProto::AttributeType::TYPE_PROTOS);
  }
};

class KeyWordMap {
public:
  enum class KeyWord : std::uint8_t {
    NONE,
    IR_VERSION,
    OPSET_IMPORT,
    PRODUCER_NAME,
    PRODUCER_VERSION,
    DOMAIN_KW,
    MODEL_VERSION,
    DOC_STRING,
    METADATA_PROPS,
    SEQ_TYPE,
    MAP_TYPE,
    OPTIONAL_TYPE,
    SPARSE_TENSOR_TYPE,
    OVERLOAD_KW
  };

  KeyWordMap() {
    map_["ir_version"] = KeyWord::IR_VERSION;
    map_["opset_import"] = KeyWord::OPSET_IMPORT;
    map_["producer_name"] = KeyWord::PRODUCER_NAME;
    map_["producer_version"] = KeyWord::PRODUCER_VERSION;
    map_["domain"] = KeyWord::DOMAIN_KW;
    map_["model_version"] = KeyWord::MODEL_VERSION;
    map_["doc_string"] = KeyWord::DOC_STRING;
    map_["metadata_props"] = KeyWord::METADATA_PROPS;
    map_["seq"] = KeyWord::SEQ_TYPE;
    map_["map"] = KeyWord::MAP_TYPE;
    map_["optional"] = KeyWord::OPTIONAL_TYPE;
    map_["sparse_tensor"] = KeyWord::SPARSE_TENSOR_TYPE;
    map_["overload"] = KeyWord::OVERLOAD_KW;
  }

  static const std::unordered_map<std::string, KeyWord> &Instance();

  static KeyWord Lookup(const std::string &id) {
    auto it = Instance().find(id);
    if (it != Instance().end())
      return it->second;
    return KeyWord::NONE;
  }

  static const std::string &ToString(KeyWord kw);

private:
  std::unordered_map<std::string, KeyWord> map_;
};

class ParserBase {
public:
  /// Initializes a parser from a string buffer.
  explicit ParserBase(const std::string &str)
      : start_(str.data()), next_(str.data()), end_(str.data() + str.length()), saved_pos_(next_) {}

  /// Initializes a parser from a null-terminated string.
  explicit ParserBase(const char *cstr)
      : start_(cstr), next_(cstr), end_(cstr + strlen(cstr)), saved_pos_(next_) {}

  void SavePos() { saved_pos_ = next_; }

  void RestorePos() { next_ = saved_pos_; }

  /// Returns the current parser position as `(line, column)`.
  std::string GetCurrentPos() {
    uint32_t line = 1, col = 1;
    for (const char *p = start_; p < next_; ++p) {
      if (*p == '\n') {
        ++line;
        col = 1;
      } else {
        ++col;
      }
    }
    return ONNX_LIGHT_NAMESPACE::MakeString("(line: ", line, " column: ", col, ")");
  }

  // Return a suitable suffix of what has been parsed to provide error message context:
  // return the line containing the last non-space character preceding the error (if it exists).
  std::string GetErrorContext() {
    // Special cases: empty input string, and parse-error at first character.
    const char *p = next_ < end_ ? next_ : next_ - 1;
    while ((p > start_) && std::isspace(static_cast<unsigned char>(*p)))
      --p;
    while ((p > start_) && (*p != '\n'))
      --p;
    // Start at character after '\n' unless we are at start of input
    const char *context_start = (p > start_) ? (p + 1) : start_;
    for (p = context_start; (p < end_) && (*p != '\n'); ++p)
      ;
    return std::string(context_start, p - context_start);
  }

  /// Builds a parse error with current position and local context.
  template <typename... Args> Common::Status ParseError(const Args &...args) {
    return Common::Status(
        Common::StatusCategory::NONE, Common::StatusCode::FAIL,
        ONNX_LIGHT_NAMESPACE::MakeString("[ParseError at position ", GetCurrentPos(), "]\n",
                                         "Error context: ", GetErrorContext(), "\n", args...));
  }

  void SkipWhiteSpace() {
    do {
      while ((next_ < end_) && std::isspace(static_cast<unsigned char>(*next_)))
        ++next_;
      if ((next_ >= end_) || ((*next_) != '#'))
        return;
      // Skip rest of the line:
      while ((next_ < end_) && ((*next_) != '\n'))
        ++next_;
    } while (true);
  }

  int NextChar(bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    return (next_ < end_) ? *next_ : 0;
  }

  bool Matches(char ch, bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    if ((next_ < end_) && (*next_ == ch)) {
      ++next_;
      return true;
    }
    return false;
  }

  Common::Status Match(char ch, bool skipspace = true) {
    if (!Matches(ch, skipspace))
      return ParseError("Expected character ", ch, " not found.");
    return Common::Status::OK();
  }

  bool EndOfInput() {
    SkipWhiteSpace();
    return (next_ >= end_);
  }

  enum class LiteralType : std::uint8_t { UNDEFINED, INT_LITERAL, FLOAT_LITERAL, STRING_LITERAL };

  struct Literal {
    LiteralType type{LiteralType::UNDEFINED};
    std::string value;
  };

  /// Parses the next scalar literal token.
  Common::Status Parse(Literal &result);

  /// Parses a required integer literal into a signed integer.
  Common::Status Parse(int64_t &val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal))
    std::string s = literal.value;
    if (literal.type != LiteralType::INT_LITERAL)
      return ParseError("Integer value expected, but not found.");
    val = std::stoll(s);
    return Common::Status::OK();
  }

  /// Parses a required integer literal into an unsigned integer.
  Common::Status Parse(uint64_t &val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal))
    std::string s = literal.value;
    if (literal.type != LiteralType::INT_LITERAL)
      return ParseError("Integer value expected, but not found.");
    val = std::stoull(s);
    return Common::Status::OK();
  }

  /// Parses an integer or floating-point literal as `float`.
  Common::Status Parse(float &val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal))
    switch (literal.type) {
    case LiteralType::INT_LITERAL:
    case LiteralType::FLOAT_LITERAL:
      val = std::stof(literal.value);
      break;
    default:
      return ParseError("Unexpected literal type.");
    }
    return Common::Status::OK();
  }

  /// Parses an integer or floating-point literal as `double`.
  Common::Status Parse(double &val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal))
    switch (literal.type) {
    case LiteralType::INT_LITERAL:
    case LiteralType::FLOAT_LITERAL:
      val = std::stod(literal.value);
      break;
    default:
      return ParseError("Unexpected literal type.");
    }
    return Common::Status::OK();
  }

  // Parse a string-literal enclosed within double-quotes.
  Common::Status Parse(std::string &val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal))
    if (literal.type != LiteralType::STRING_LITERAL)
      return ParseError("String value expected, but not found.");
    val = literal.value;
    return Common::Status::OK();
  }

  // Parse an identifier, including keywords. If none found, this will
  // return an empty-string identifier.
  std::string ParseOptionalIdentifier() {
    SkipWhiteSpace();
    const auto *from = next_;
    if ((next_ < end_) && (std::isalpha(static_cast<unsigned char>(*next_)) || (*next_ == '_'))) {
      ++next_;
      while ((next_ < end_) &&
             (std::isalnum(static_cast<unsigned char>(*next_)) || (*next_ == '_')))
        ++next_;
    }
    return std::string(from, next_ - from);
  }

  /// Parses an identifier and fails if none is found.
  Common::Status ParseIdentifier(std::string &id) {
    id = ParseOptionalIdentifier();
    if (id.empty())
      return ParseError("Identifier expected but not found.");
    return Common::Status::OK();
  }

  /// Parses a quoted identifier or an unquoted identifier.
  Common::Status ParseQuotableIdentifier(std::string &id) {
    if (NextChar() == '"') {
      return Parse(id);
    }
    return ParseIdentifier(id);
  }

  Common::Status ParseOptionalQuotableIdentifier(std::string &id) {
    if (NextChar() == '"') {
      return Parse(id);
    }
    id = ParseOptionalIdentifier();
    return Common::Status::OK();
  }

  // Parse an optional quotable identifier, and return whether an identifier was found
  // in the output parameter 'id_found'.
  // A empty string followed by a comma is considered to be a valid, but empty, identifier.
  // This helps handle the following different cases:
  // "Op()" has no operands
  // "Op(,x)" has two operands, the first being empty.
  // 'Op("")' has one operand, which is an empty string.
  // 'Op(,)' has one operand, which is an empty string.
  // Thus, this will also allow a trailing comma after a non-empty identifier with no effect.
  // 'Op(x,)' has one operand, which is 'x'.
  //
  // This is mostly for some backward compatibility. "" is a simpler way to represent an
  // empty identifier that is less confusing and is recommended.
  Common::Status ParseOptionalQuotableIdentifier(std::string &id, bool &id_found) {
    if (NextChar() == '"') {
      id_found = true;
      return Parse(id);
    }
    id = ParseOptionalIdentifier();
    id_found = !id.empty() || NextChar() == ',';
    return Common::Status::OK();
  }

  std::string PeekIdentifier() {
    SavePos();
    auto id = ParseOptionalIdentifier();
    RestorePos();
    return id;
  }

  /// Parses a keyword token.
  Common::Status Parse(KeyWordMap::KeyWord &keyword) {
    std::string id;
    CHECK_PARSER_STATUS(ParseIdentifier(id))
    keyword = KeyWordMap::Lookup(id);
    return Common::Status::OK();
  }

protected:
  const char *start_;
  const char *next_;
  const char *end_;
  const char *saved_pos_;

  bool NextIsValidFloatString();
};

class OnnxParser : public ParserBase {
public:
  /// Initializes an ONNX text parser from a null-terminated buffer.
  explicit OnnxParser(const char *cstr) : ParserBase(cstr) {}

  /// Parses a `TensorShapeProto`.
  ONNX_API Common::Status Parse(TensorShapeProto &shape);

  /// Parses a `TypeProto`.
  ONNX_API Common::Status Parse(TypeProto &typeProto);

  /// Parses a metadata key/value list.
  ONNX_API Common::Status Parse(StringStringList &stringStringList);

  /// Parses a `TensorProto`.
  ONNX_API Common::Status Parse(TensorProto &tensorProto);

  /// Parses an `AttributeProto`.
  ONNX_API Common::Status Parse(AttributeProto &attr);

  /// Parses a named `AttributeProto`.
  ONNX_API Common::Status Parse(AttributeProto &attr, std::string &name);

  /// Parses a comma-separated attribute list.
  ONNX_API Common::Status Parse(AttrList &attrlist);

  /// Parses a `NodeProto`.
  ONNX_API Common::Status Parse(NodeProto &node);

  /// Parses a list of nodes.
  ONNX_API Common::Status Parse(NodeList &nodelist);

  /// Parses a `GraphProto`.
  ONNX_API Common::Status Parse(GraphProto &graph);

  /// Parses a `FunctionProto`.
  ONNX_API Common::Status Parse(FunctionProto &fn);

  /// Parses a `ModelProto`.
  ONNX_API Common::Status Parse(ModelProto &model);

  /// Parses an ONNX text snippet into the requested protobuf type.
  template <typename T> static Common::Status Parse(T &parsedData, const char *input) {
    OnnxParser parser(input);
    return parser.Parse(parsedData);
  }

private:
  Common::Status Parse(std::string name, GraphProto &graph);

  Common::Status Parse(IdList &idlist);

  Common::Status Parse(char open, IdList &idlist, char close);

  Common::Status Parse(IdList &idlist, AttrList &attrlist);

  Common::Status Parse(char open, IdList &idlist, AttrList &attrlist, char close);

  Common::Status ParseSingleAttributeValue(AttributeProto &attr,
                                           AttributeProto::AttributeType expected);

  Common::Status Parse(ValueInfoProto &valueinfo);

  Common::Status ParseGraphInputOutput(ValueInfoList &vilist);

  Common::Status ParseFunctionInputOutput(IdList &idlist, ValueInfoList &vilist);

  Common::Status Parse(char open, ValueInfoList &vilist, char close);

  Common::Status ParseInput(ValueInfoList &inputs, TensorList &initializers);

  Common::Status ParseValueInfo(ValueInfoList &value_infos, TensorList &initializers);

  Common::Status Parse(TensorProto &tensorProto, const TypeProto &tensorTypeProto);

  Common::Status Parse(OpsetIdList &opsets);

  bool NextIsType();

  bool NextIsIdentifier();
};

} // namespace ONNX_LIGHT_NAMESPACE
