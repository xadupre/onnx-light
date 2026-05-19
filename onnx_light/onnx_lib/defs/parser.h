// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file parser.h
 * @brief Declares the ONNX text-format parser and related helpers.
 *
 * This header exposes ParserBase (cursor-based tokenizer) and OnnxParser
 * (builds protobuf structures from text) for parsing ONNX models, graphs,
 * functions, nodes, and types from their textual representation.
 * It also defines utility maps (PrimitiveTypeNameMap, AttributeTypeNameMap,
 * KeyWordMap) and convenience type aliases used throughout the parser.
 *
 * @note The ONNX text syntax is experimental and may change.
 */

#pragma once

#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx_lib/common/onnx_pb.h"
#include "onnx_lib/common/status.h"
#include "onnx_lib/common/string_utils.h"

namespace ONNX_LIGHT_NAMESPACE {

/// List of identifiers used in parser productions.
using IdList = std::vector<std::string>;

/// List of parsed node definitions.
using NodeList = utils::RepeatedProtoField<NodeProto>;

/// List of parsed node attributes.
using AttrList = utils::RepeatedProtoField<AttributeProto>;

/// List of value-info records used for graph/function signatures.
using ValueInfoList = utils::RepeatedProtoField<ValueInfoProto>;

/// List of tensor literals and initializers.
using TensorList = utils::RepeatedProtoField<TensorProto>;

/// List of opset imports.
using OpsetIdList = utils::RepeatedProtoField<OperatorSetIdProto>;

/// List of key/value metadata entries.
using StringStringList = utils::RepeatedProtoField<StringStringEntryProto>;

#define CHECK_PARSER_STATUS(status)                                                                \
  {                                                                                                \
    auto local_status_ = status;                                                                   \
    if (!local_status_.IsOK())                                                                     \
      return local_status_;                                                                        \
  }

/**
 * @brief CRTP singleton base that maps string names to integer codes.
 *
 * Subclasses populate `map_` in their constructor and gain static
 * Instance(), Lookup(), and ToString() helpers automatically.
 *
 * @tparam Map Concrete subclass (CRTP pattern).
 */
template <typename Map>
// NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
class StringIntMap {
public:
  /// Returns the singleton name-to-value map instance.
  static const std::unordered_map<std::string, int32_t> &Instance() {
    static Map instance;
    return instance.map_;
  }

  /// Finds a value by textual name.
  static int32_t Lookup(const std::string &dtype) {
    auto it = Instance().find(dtype);
    if (it != Instance().end())
      return it->second;
    return 0;
  }

  /// Returns a textual name for an enum/integer value, or "undefined".
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

/**
 * @brief Maps ONNX primitive type name strings to TensorProto::DataType values.
 *
 * Supports all scalar element types such as "float", "int64", "bfloat16",
 * and the low-precision types "float8e4m3fn", "uint4", "float4e2m1", etc.
 */
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

  /// Returns true if the given string is a recognized ONNX primitive type name.
  static bool IsTypeName(const std::string &dtype) { return Lookup(dtype) != 0; }
};

/**
 * @brief Maps ONNX attribute type name strings to AttributeProto::AttributeType values.
 *
 * Covers scalar types ("float", "int", "string", "tensor", "graph",
 * "sparse_tensor", "type_proto") and their list counterparts
 * ("floats", "ints", "strings", "tensors", "graphs", "sparse_tensors",
 * "type_protos").
 */
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

/**
 * @brief Singleton map from keyword identifier strings to KeyWord enum values.
 *
 * Recognizes ONNX model-level keywords such as "ir_version", "opset_import",
 * "domain", "seq", "map", "optional", "sparse_tensor", and "overload".
 */
class KeyWordMap {
public:
  /// Enumeration of all reserved ONNX text-format keywords.
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

  /// Constructs the keyword map and populates all reserved identifier entries.
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

  /// Returns the singleton string-to-keyword map instance.
  static const std::unordered_map<std::string, KeyWord> &Instance();

  /// Looks up a keyword by identifier string; returns KeyWord::NONE if not found.
  static KeyWord Lookup(const std::string &id) {
    auto it = Instance().find(id);
    if (it != Instance().end())
      return it->second;
    return KeyWord::NONE;
  }

  /// Converts a KeyWord enum value to its canonical string representation.
  static const std::string &ToString(KeyWord kw);

private:
  std::unordered_map<std::string, KeyWord> map_;
};

/**
 * @brief Cursor-based tokenizer that drives parsing of ONNX text format.
 *
 * Maintains a read cursor over a character buffer and provides low-level
 * helpers for skipping whitespace, matching characters, reading literals
 * (integer, float, string), and parsing identifiers.  OnnxParser inherits
 * from this class to build higher-level protobuf parsing on top.
 */
class ParserBase {
public:
  /// Creates a parser from a string buffer.
  /// The underlying buffer must remain valid and unchanged during the parser lifetime.
  explicit ParserBase(const std::string &str)
      : start_(str.data()), next_(str.data()), end_(str.data() + str.length()), saved_pos_(next_) {}

  /// Creates a parser from a null-terminated string.
  /// The referenced character buffer must outlive the parser instance.
  explicit ParserBase(const char *cstr)
      : start_(cstr), next_(cstr), end_(cstr + strlen(cstr)), saved_pos_(next_) {}

  /// Saves the current parser cursor position.
  void SavePos() { saved_pos_ = next_; }

  /// Restores the parser cursor to the most recently saved position.
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
  /// Returns a line of source context around the current cursor position for error messages.
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

  /// Generates a parse error with current position and local context.
  template <typename... Args> Common::Status ParseError(const Args &...args) {
    return Common::Status(
        Common::StatusCategory::NONE, Common::StatusCode::FAIL,
        ONNX_LIGHT_NAMESPACE::MakeString("[ParseError at position ", GetCurrentPos(), "]\n",
                                         "Error context: ", GetErrorContext(), "\n", args...));
  }

  /// Advances the cursor past whitespace characters and `#`-prefixed line comments.
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

  /// Returns the next character without consuming it; returns 0 at end of input.
  /// @param skipspace When true (default), skips whitespace before peeking.
  int NextChar(bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    return (next_ < end_) ? *next_ : 0;
  }

  /// Consumes `ch` if it is the next character and returns true; otherwise returns false.
  /// @param skipspace When true (default), skips whitespace before matching.
  bool Matches(char ch, bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    if ((next_ < end_) && (*next_ == ch)) {
      ++next_;
      return true;
    }
    return false;
  }

  /// Consumes `ch` or returns a parse error if it is not the next character.
  /// @param skipspace When true (default), skips whitespace before matching.
  Common::Status Match(char ch, bool skipspace = true) {
    if (!Matches(ch, skipspace))
      return ParseError("Expected character ", ch, " not found.");
    return Common::Status::OK();
  }

  /// Returns true when all remaining input (after skipping whitespace) has been consumed.
  bool EndOfInput() {
    SkipWhiteSpace();
    return (next_ >= end_);
  }

  /// Classifies the kind of a parsed literal token.
  enum class LiteralType : std::uint8_t { UNDEFINED, INT_LITERAL, FLOAT_LITERAL, STRING_LITERAL };

  /// Holds the raw text and classification of a parsed literal token.
  struct Literal {
    LiteralType type{LiteralType::UNDEFINED};
    std::string value;
  };

  /// Parses the next scalar literal token.
  Common::Status Parse(Literal &result);

  /// Parses a required integer literal into `int64_t`.
  /// Returns a parse error if the next token is not an integer literal.
  Common::Status Parse(int64_t &val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal))
    std::string s = literal.value;
    if (literal.type != LiteralType::INT_LITERAL)
      return ParseError("Integer value expected, but not found.");
    val = std::stoll(s);
    return Common::Status::OK();
  }

  /// Parses a required integer literal into `uint64_t`.
  /// Returns a parse error if the next token is not an integer literal.
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

  /// Parses a double-quoted string literal; returns a parse error if a string is not found.
  Common::Status Parse(std::string &val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal))
    if (literal.type != LiteralType::STRING_LITERAL)
      return ParseError("String value expected, but not found.");
    val = literal.value;
    return Common::Status::OK();
  }

  /// Parses an optional identifier (including keywords); returns an empty string if none found.
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

  /// Parses an optional identifier that may be enclosed in double quotes.
  /// Sets @p id to the identifier text (possibly empty) and returns OK.
  Common::Status ParseOptionalQuotableIdentifier(std::string &id) {
    if (NextChar() == '"') {
      return Parse(id);
    }
    id = ParseOptionalIdentifier();
    return Common::Status::OK();
  }

  /// Parses an optional quotable identifier and sets @p id_found to indicate whether one was found.
  ///
  /// An empty string followed by a comma is treated as a valid but empty identifier to support
  /// operand-list patterns such as `Op(,x)` (two operands, first empty) versus `Op()` (no
  /// operands). A trailing comma after a non-empty identifier is silently accepted. Using `""` as
  /// an explicit empty identifier is preferred over relying on this behavior.
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

  /// Returns the next identifier without advancing the cursor.
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
  /// Pointer to the beginning of the input buffer.
  const char *start_;
  /// Pointer to the current read position within the input buffer.
  const char *next_;
  /// Pointer to one past the last character of the input buffer.
  const char *end_;
  /// Cursor snapshot written by SavePos() and restored by RestorePos().
  const char *saved_pos_;

  /// Returns true if the characters at the current position form a valid floating-point literal.
  bool NextIsValidFloatString();
};

/**
 * @brief High-level ONNX text-format parser that builds protobuf structures.
 *
 * Inherits from ParserBase and provides overloaded Parse() methods for every
 * major ONNX protobuf type (ModelProto, GraphProto, FunctionProto, NodeProto,
 * TensorProto, TypeProto, AttributeProto, etc.).
 *
 * The most convenient entry point is the static Parse() helper:
 * @code
 *   ModelProto model;
 *   auto status = OnnxParser::Parse(model, text_buffer);
 * @endcode
 */
class OnnxParser : public ParserBase {
public:
  /// Creates an ONNX text parser from a null-terminated buffer.
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
