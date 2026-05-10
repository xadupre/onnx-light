// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Experimental language syntax and parser for ONNX. Please note that the syntax as formalized
// by this parser is preliminary and may change.

#include "parser.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>

#include "onnx/common/common.h"

#define PARSE_TOKEN(x) CHECK_PARSER_STATUS(ParserBase::Parse(x))
#define PARSE(...) CHECK_PARSER_STATUS(Parse(__VA_ARGS__))
#define MATCH(...) CHECK_PARSER_STATUS(Match(__VA_ARGS__))

namespace ONNX_NAMESPACE {

Common::Status ParserBase::Parse(Literal &result) {
  bool decimal_point = false;
  auto nextch = NextChar();
  const auto *from = next_;
  if (nextch == '"') {
    ++next_;
    bool has_escape = false;
    while ((next_ < end_) && (*next_ != '"')) {
      if (*next_ == '\\') {
        has_escape = true;
        ++next_;
        if (next_ >= end_)
          return ParseError("Incomplete string literal.");
      }
      ++next_;
    }
    if (next_ >= end_)
      return ParseError("Incomplete string literal.");
    ++next_;
    result.type = LiteralType::STRING_LITERAL;
    if (has_escape) {
      std::string &target = result.value;
      target.clear();
      target.reserve(static_cast<size_t>(next_ - from - 2)); // upper bound
      // *from is the starting quote. *(next_-1) is the ending quote.
      while (++from < next_ - 1) {
        target.push_back(*from != '\\' ? (*from) : *(++from));
      }
    } else {
      result.value =
          std::string(from + 1, static_cast<size_t>(next_ - from - 2)); // skip enclosing quotes
    }
    return Common::Status::OK();
  }

  // Simplify the next ifs by consuming a possible negative sign.
  if (nextch == '-') {
    ++next_;
    nextch = NextChar();
  }

  // Check for float literals that start with alphabet characters.
  if (std::isalpha(nextch)) {
    // Has to be a special float literal now: (-)*(nan|inf|infinity).
    if (NextIsValidFloatString()) {
      while (next_ < end_ && std::isalpha(static_cast<unsigned char>(*next_))) {
        ++next_;
      }
      ONNX_TRY {
        static_cast<void>(std::stof(std::string(from, static_cast<size_t>(next_ - from))));
        result.type = LiteralType::FLOAT_LITERAL;
        result.value = std::string(from, static_cast<size_t>(next_ - from));
      }
      ONNX_CATCH(...) {
        ONNX_HANDLE_EXCEPTION([&]() { return ParseError("Encountered invalid float literal!"); });
      }
    } else {
      return ParseError("Encountered invalid float literal!");
    }
    return Common::Status::OK();
  }

  // Checking for numeric ints or float literal.
  if (std::isdigit(nextch)) {
    ++next_;

    while ((next_ < end_) &&
           (std::isdigit(static_cast<unsigned char>(*next_)) || (*next_ == '.'))) {
      if (*next_ == '.') {
        if (decimal_point)
          break; // Only one decimal point allowed in numeric literal
        decimal_point = true;
      }
      ++next_;
    }

    if (next_ == from)
      return ParseError("Value expected but not found.");

    // Optional exponent syntax: (e|E)(+|-)?[0-9]+
    if ((next_ < end_) && ((*next_ == 'e') || (*next_ == 'E'))) {
      decimal_point = true; // treat as float-literal
      ++next_;
      if ((next_ < end_) && ((*next_ == '+') || (*next_ == '-')))
        ++next_;
      while ((next_ < end_) && (std::isdigit(static_cast<unsigned char>(*next_))))
        ++next_;
    }

    result.value = std::string(from, static_cast<size_t>(next_ - from));
    result.type = decimal_point ? LiteralType::FLOAT_LITERAL : LiteralType::INT_LITERAL;
  }
  return Common::Status::OK();
}

bool ParserBase::NextIsValidFloatString() {
  auto nextch = NextChar();
  const auto *const from = next_;
  constexpr int INFINITY_LENGTH = 8;

  if (std::isalpha(nextch)) {
    while (next_ < end_ && std::isalpha(static_cast<unsigned char>(*next_)) &&
           (next_ - from) <= INFINITY_LENGTH) {
      ++next_;
    }

    if (next_ < end_ && std::isdigit(static_cast<unsigned char>(*next_))) { // No trailing digits
      next_ = from;
      return false;
    }

    std::string candidate = std::string(from, static_cast<size_t>(next_ - from));

    // Reset parser location before continuing.
    next_ = from;

    std::transform(candidate.begin(), candidate.end(), candidate.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (candidate == std::string_view("inf") || candidate == std::string_view("infinity") ||
        candidate == std::string_view("nan")) {
      return true;
    }
  }
  return false;
}

Common::Status OnnxParser::Parse(IdList &idlist) {
  idlist.clear();
  std::string id;
  bool found = false;
  CHECK_PARSER_STATUS(ParseOptionalQuotableIdentifier(id, found));
  if (!found)
    return Common::Status::OK();
  idlist.push_back(id);
  while (Matches(',')) {
    CHECK_PARSER_STATUS(ParseOptionalQuotableIdentifier(id, found));
    if (!found)
      break;
    idlist.push_back(id);
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(char open, IdList &idlist, char close) {
  idlist.clear();
  if (Matches(open)) {
    PARSE(idlist);
    MATCH(close);
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(IdList &idlist, AttrList &attrlist) {
  idlist.clear();
  attrlist.clear();
  do {
    std::string id;
    CHECK_PARSER_STATUS(ParseQuotableIdentifier(id));
    auto next = NextChar();
    if (next == ':' || next == '=') {
      attrlist.emplace_back();
      CHECK_PARSER_STATUS(Parse(attrlist.back(), id));
    } else {
      idlist.push_back(id);
    }
  } while (Matches(','));
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(char open, IdList &idlist, AttrList &attrlist, char close) {
  if (Matches(open)) {
    PARSE(idlist, attrlist);
    MATCH(close);
  } else {
    idlist.clear();
    attrlist.clear();
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(TensorShapeProto &shape) {
  shape.clr_dim();
  do {
    if (Matches('?')) {
      shape.add_dim();
    } else if (NextChar() == '"') {
      // Check for a quoted string as symbolic dim ...
      std::string id;
      CHECK_PARSER_STATUS(ParserBase::Parse(id));
      shape.add_dim().set_dim_param(id);
    } else {
      // Check for a symbolic identifier ...
      auto id = ParseOptionalIdentifier();
      if (!id.empty()) {
        shape.add_dim().set_dim_param(id);
      } else {
        // ...or an integer value
        int64_t dimval = 0;
        PARSE_TOKEN(dimval);
        shape.add_dim().set_dim_value(dimval);
      }
    }
  } while (Matches(','));
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(TypeProto &typeProto) {
  std::string id;
  CHECK_PARSER_STATUS(ParseIdentifier(id));
  int dtype = PrimitiveTypeNameMap::Lookup(id);
  if (dtype != 0) {
    auto &tensortype = typeProto.ref_tensor_type();
    tensortype.set_elem_type(dtype);
    tensortype.reset_shape();
    // Grammar:
    // float indicates scalar (rank 0)
    // float [] indicates unknown rank tensor (not a zero rank tensor)
    // float [one-or-more-dimensions] indicates tensor of known rank > 0.
    if (Matches('[')) {
      if (!Matches(']')) {
        PARSE(tensortype.ref_shape());
        MATCH(']');
      }
    } else {
      // Create shape with zero dimensions for scalar
      tensortype.ref_shape();
    }
  } else {
    switch (KeyWordMap::Lookup(id)) {
    case KeyWordMap::KeyWord::SEQ_TYPE: {
      // Grammar: seq ( type )
      MATCH('(');
      auto &seqtype = typeProto.ref_sequence_type();
      PARSE(seqtype.ref_elem_type());
      MATCH(')');
      break;
    }
    case KeyWordMap::KeyWord::MAP_TYPE: {
      // Grammar: map ( prim-type , type )
      MATCH('(');
      auto &maptype = typeProto.ref_map_type();
      CHECK_PARSER_STATUS(ParseIdentifier(id));
      dtype = PrimitiveTypeNameMap::Lookup(id);
      if (dtype == 0) {
        return ParseError("Expecting primitive type as map key type.");
      }
      maptype.set_key_type(dtype);
      MATCH(',');
      PARSE(maptype.ref_value_type());
      MATCH(')');
      break;
    }
    case KeyWordMap::KeyWord::OPTIONAL_TYPE: {
      // Grammar: optional ( type )
      MATCH('(');
      auto &opttype = typeProto.ref_optional_type();
      PARSE(opttype.ref_elem_type());
      MATCH(')');
      break;
    }
    case KeyWordMap::KeyWord::SPARSE_TENSOR_TYPE: {
      // Grammar: sparse_tensor ( tensor-type )
      MATCH('(');
      CHECK_PARSER_STATUS(ParseIdentifier(id));
      dtype = PrimitiveTypeNameMap::Lookup(id);
      if (dtype != 0) {
        auto &sparsetype = typeProto.ref_sparse_tensor_type();
        sparsetype.set_elem_type(static_cast<TensorProto::DataType>(dtype));
        sparsetype.reset_shape();
        // Grammar:
        // float indicates scalar (rank 0)
        // float [] indicates unknown rank tensor (not a zero rank tensor)
        // float [one-or-more-dimensions] indicates tensor of known rank > 0.
        if (Matches('[')) {
          if (!Matches(']')) {
            PARSE(sparsetype.ref_shape());
            MATCH(']');
          }
        } else {
          // Create shape with zero dimensions for scalar
          sparsetype.ref_shape();
        }
      } else {
        return ParseError("Unexpected type in sparse-tensor element type.");
      }
      MATCH(')');
      break;
    }
    default:
      return ParseError("Unexpected type.");
    }
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(ValueInfoProto &valueinfo) {
  if (NextIsType())
    PARSE(valueinfo.ref_type());
  std::string name;
  CHECK_PARSER_STATUS(ParseQuotableIdentifier(name));
  valueinfo.set_name(name);
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(char open, ValueInfoList &vilist, char close) {
  MATCH(open);
  if (!Matches(close)) {
    do {
      vilist.emplace_back();
      PARSE(vilist.back());
    } while (Matches(','));
    MATCH(close);
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::ParseGraphInputOutput(ValueInfoList &vilist) {
  vilist.clear();
  PARSE('(', vilist, ')');
  return Common::Status::OK();
}

Common::Status OnnxParser::ParseFunctionInputOutput(IdList &idlist, ValueInfoList &vilist) {
  // Do not clear vilist, as it accumulates values over inputs and outputs.
  idlist.clear();
  MATCH('(');
  if (!Matches(')')) {
    do {
      // Function inputs/outputs can be optionally typed.
      // Syntax: Name | Type Name
      // The name is added to idlist. If the optional type is present, an entry is
      // added to vilist.
      idlist.push_back(std::string{});
      std::string &name = idlist.back();
      ValueInfoProto *vi = nullptr;

      if (NextIsType()) {
        vilist.emplace_back();
        vi = &vilist.back();
        PARSE(vi->ref_type());
      }
      CHECK_PARSER_STATUS(ParseQuotableIdentifier(name));
      if (vi != nullptr)
        vi->set_name(name);
    } while (Matches(','));
    MATCH(')');
  }
  return Common::Status::OK();
}

// Each input element is a value-info with an optional initializer of the form "= initial-value".
// The value-info is added to the "inputs", while the initializer is added to initializers.
Common::Status OnnxParser::ParseInput(ValueInfoList &inputs, TensorList &initializers) {
  inputs.clear();
  if (Matches('(')) {
    if (!Matches(')')) {
      do {
        inputs.emplace_back();
        ValueInfoProto &vi = inputs.back();
        PARSE(vi);
        if (Matches('=')) {
          // default value for input
          initializers.emplace_back();
          TensorProto &tp = initializers.back();
          tp.set_name(vi.ref_name().as_string());
          CHECK_PARSER_STATUS(Parse(tp, vi.ref_type()));
        }
      } while (Matches(','));
      MATCH(')');
    }
  }
  return Common::Status::OK();
}

// This is handled slightly differently from the inputs.
// Each element is either a value-info or an initializer.
// A value-info is added to the "value_infos", while an initializer is added to initializers.
Common::Status OnnxParser::ParseValueInfo(ValueInfoList &value_infos, TensorList &initializers) {
  value_infos.clear();
  if (Matches('<')) {
    if (!Matches('>')) {
      do {
        ValueInfoProto vi;
        PARSE(vi);
        if (Matches('=')) {
          // initializer
          initializers.emplace_back();
          TensorProto &tp = initializers.back();
          tp.set_name(vi.ref_name().as_string());
          CHECK_PARSER_STATUS(Parse(tp, vi.ref_type()));
        } else {
          // valueinfo
          value_infos.push_back(vi);
        }
      } while (Matches(','));
      MATCH('>');
    }
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(StringStringList &stringStringList) {
  std::string strval;
  do {
    stringStringList.emplace_back();
    auto &metadata = stringStringList.back();
    PARSE_TOKEN(strval);
    metadata.set_key(strval);
    MATCH(':');
    PARSE_TOKEN(strval);
    metadata.set_value(strval);
  } while (Matches(','));
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(TensorProto &tensorProto) {
  tensorProto = TensorProto{};
  // Parse the concrete tensor-type with numeric dimensions:
  TypeProto typeProto;
  PARSE(typeProto);
  std::string tensor_name;
  CHECK_PARSER_STATUS(ParseOptionalQuotableIdentifier(tensor_name));
  tensorProto.set_name(tensor_name);
  (void)Matches('='); // Optional, to unify handling of initializers
  return Parse(tensorProto, typeProto);
}

// Parse TensorProto data given its type:
Common::Status OnnxParser::Parse(TensorProto &tensorProto, const TypeProto &tensorTypeProto) {
  if (!tensorTypeProto.has_tensor_type())
    return ParseError("Error parsing TensorProto (expected a tensor type).");
  const auto &ttype = tensorTypeProto.ref_tensor_type();
  int32_t elem_type = static_cast<int32_t>(ttype.ref_elem_type());
  tensorProto.set_data_type(elem_type);
  if (!ttype.has_shape())
    return ParseError("Error parsing TensorProto (expected a tensor shape).");
  const auto &shape = ttype.ref_shape();
  for (size_t i = 0; i < shape.ref_dim().size(); ++i) {
    const auto &dim = shape.ref_dim()[i];
    if (!dim.has_dim_value())
      return ParseError("Error parsing TensorProto shape (expected numeric dimension).");
    auto dimval = dim.ref_dim_value();
    tensorProto.ref_dims().push_back(static_cast<uint64_t>(dimval));
  }

  int64_t intval = 0;
  uint64_t uintval = 0;
  float floatval = 0.0f;
  double dblval = 0.0;
  std::string strval;
  if (Matches('{')) {
    if (!Matches('}')) {
      do {
        switch (static_cast<TensorProto::DataType>(elem_type)) {
        case TensorProto::DataType::INT2:
        case TensorProto::DataType::INT4:
        case TensorProto::DataType::INT8:
        case TensorProto::DataType::INT16:
        case TensorProto::DataType::INT32:
        case TensorProto::DataType::UINT2:
        case TensorProto::DataType::UINT4:
        case TensorProto::DataType::UINT8:
        case TensorProto::DataType::UINT16:
        case TensorProto::DataType::FLOAT16:
        case TensorProto::DataType::BFLOAT16:
        case TensorProto::DataType::FLOAT8E4M3FN:
        case TensorProto::DataType::FLOAT8E4M3FNUZ:
        case TensorProto::DataType::FLOAT8E5M2:
        case TensorProto::DataType::FLOAT8E5M2FNUZ:
        case TensorProto::DataType::FLOAT8E8M0:
        case TensorProto::DataType::BOOL:
        case TensorProto::DataType::FLOAT4E2M1:
          PARSE_TOKEN(intval);
          if (intval > std::numeric_limits<int32_t>::max() ||
              intval < std::numeric_limits<int32_t>::min()) {
            return ParseError("Mismatch between data type and value: %d, %d", elem_type, intval);
          }
          // NOLINTNEXTLINE(bugprone-narrowing-conversions)
          tensorProto.ref_int32_data().push_back(static_cast<int32_t>(intval));
          break;
        case TensorProto::DataType::INT64:
          PARSE_TOKEN(intval);
          tensorProto.ref_int64_data().push_back(intval);
          break;
        case TensorProto::DataType::UINT32:
        case TensorProto::DataType::UINT64:
          PARSE_TOKEN(uintval);
          tensorProto.ref_uint64_data().push_back(uintval);
          break;
        case TensorProto::DataType::COMPLEX64:
        case TensorProto::DataType::FLOAT:
          PARSE_TOKEN(floatval);
          tensorProto.ref_float_data().push_back(floatval);
          break;
        case TensorProto::DataType::COMPLEX128:
        case TensorProto::DataType::DOUBLE:
          PARSE_TOKEN(dblval);
          tensorProto.ref_double_data().push_back(dblval);
          break;
        case TensorProto::DataType::STRING:
          PARSE_TOKEN(strval);
          tensorProto.add_string_data() = strval;
          break;
        default:
          return ParseError("Unhandled type: %d", elem_type);
        }
      } while (Matches(','));
      MATCH('}');
    }
  } else if (Matches('[')) {
    tensorProto.set_data_location(TensorProto::EXTERNAL);
    StringStringList externalData;
    PARSE(externalData);
    for (auto &entry : externalData) {
      auto &ed = tensorProto.add_external_data();
      ed.set_key(entry.ref_key().as_string());
      ed.set_value(entry.ref_value().as_string());
    }
    MATCH(']');
  }
  return Common::Status::OK();
}

bool OnnxParser::NextIsIdentifier() {
  auto id = PeekIdentifier();
  return !id.empty();
}

bool OnnxParser::NextIsType() {
  auto id = PeekIdentifier();
  if (PrimitiveTypeNameMap::IsTypeName(id))
    return true;
  switch (KeyWordMap::Lookup(id)) {
  case KeyWordMap::KeyWord::SEQ_TYPE:
  case KeyWordMap::KeyWord::MAP_TYPE:
  case KeyWordMap::KeyWord::OPTIONAL_TYPE:
  case KeyWordMap::KeyWord::SPARSE_TENSOR_TYPE:
    return true;
  default:
    return false;
  }
}

Common::Status OnnxParser::ParseSingleAttributeValue(AttributeProto &attr,
                                                     AttributeProto::AttributeType expected) {
  // Parse a single value
  auto next = NextChar();
  if (std::isalpha(static_cast<unsigned char>(next)) || next == '_') {
    if (NextIsType()) {
      TypeProto typeProto;
      CHECK_PARSER_STATUS(Parse(typeProto));
      next = NextChar();
      if ((next == '{') || (next == '=') || (NextIsIdentifier())) {
        attr.set_type(AttributeProto::AttributeType::TENSOR);
        auto &tensorProto = attr.ref_t();
        std::string tensor_name;
        CHECK_PARSER_STATUS(ParseOptionalQuotableIdentifier(tensor_name));
        tensorProto.set_name(tensor_name);
        (void)Matches('='); // Optional, to unify handling of initializers
        CHECK_PARSER_STATUS(Parse(tensorProto, typeProto));
      } else {
        attr.set_type(AttributeProto::AttributeType::TYPE_PROTO);
        attr.ref_tp().CopyFrom(typeProto);
      }
    } else {
      if (NextIsValidFloatString()) {
        Literal literal;
        PARSE_TOKEN(literal);
        attr.set_type(AttributeProto::AttributeType::FLOAT);
        attr.set_f(std::stof(literal.value));
      } else {
        attr.set_type(AttributeProto::AttributeType::GRAPH);
        PARSE(attr.ref_g());
      }
    }
  } else if (Matches('@')) {
    std::string name;
    CHECK_PARSER_STATUS(ParseQuotableIdentifier(name));
    attr.set_ref_attr_name(name);
  } else {
    Literal literal;
    PARSE_TOKEN(literal);
    switch (literal.type) {
    case LiteralType::UNDEFINED:
      return ParseError("Internal error");
    case LiteralType::INT_LITERAL:
      attr.set_type(AttributeProto::AttributeType::INT);
      attr.set_i(std::stol(literal.value));
      break;
    case LiteralType::FLOAT_LITERAL:
      attr.set_type(AttributeProto::AttributeType::FLOAT);
      attr.set_f(std::stof(literal.value));
      break;
    case LiteralType::STRING_LITERAL:
      attr.set_type(AttributeProto::AttributeType::STRING);
      attr.set_s(literal.value);
      break;
    }
  }
  if ((expected != AttributeProto::AttributeType::UNDEFINED) && (expected != attr.ref_type())) {
    // Mismatch between type-annotation and attribute-value. We do an implicit cast
    // only in the special case of FLOAT type and integral value like 2
    if ((expected == AttributeProto::AttributeType::FLOAT) &&
        (attr.ref_type() == AttributeProto::AttributeType::INT)) {
      attr.set_type(AttributeProto::AttributeType::FLOAT);
      attr.set_f(static_cast<float>(attr.ref_i()));
    } else {
      return ParseError("Mismatch between expected attribute type and specified value type.");
    }
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(AttributeProto &attr) {
  attr = AttributeProto{};
  std::string name;
  CHECK_PARSER_STATUS(ParseIdentifier(name));
  return Parse(attr, name);
}

static bool IsSingletonAttribute(AttributeProto::AttributeType type) {
  switch (type) {
  case AttributeProto::AttributeType::FLOAT:
  case AttributeProto::AttributeType::INT:
  case AttributeProto::AttributeType::STRING:
  case AttributeProto::AttributeType::TENSOR:
  case AttributeProto::AttributeType::GRAPH:
  case AttributeProto::AttributeType::SPARSE_TENSOR:
  case AttributeProto::AttributeType::TYPE_PROTO:
    return true;
  default:
    return false;
  }
}

static AttributeProto::AttributeType ToSingletonType(AttributeProto::AttributeType type) {
  switch (type) {
  case AttributeProto::AttributeType::FLOATS:
    return AttributeProto::AttributeType::FLOAT;
  case AttributeProto::AttributeType::INTS:
    return AttributeProto::AttributeType::INT;
  case AttributeProto::AttributeType::STRINGS:
    return AttributeProto::AttributeType::STRING;
  case AttributeProto::AttributeType::TENSORS:
    return AttributeProto::AttributeType::TENSOR;
  case AttributeProto::AttributeType::GRAPHS:
    return AttributeProto::AttributeType::GRAPH;
  case AttributeProto::AttributeType::SPARSE_TENSORS:
    return AttributeProto::AttributeType::SPARSE_TENSOR;
  case AttributeProto::AttributeType::TYPE_PROTOS:
    return AttributeProto::AttributeType::TYPE_PROTO;
  default:
    return type;
  }
}

Common::Status OnnxParser::Parse(AttributeProto &attr, std::string &name) {
  attr.set_name(name);
  if (Matches(':')) {
    CHECK_PARSER_STATUS(ParseIdentifier(name));
    int attrtype = AttributeTypeNameMap::Lookup(name);
    if (attrtype != 0) {
      attr.set_type(static_cast<AttributeProto::AttributeType>(attrtype));
    } else {
      return ParseError("Unexpected attribute type.");
    }
  }
  MATCH('=');
  if (NextChar() == '[') {
    // Parse a list of values. For an empty list, the type MUST be specified
    // using the type-annotation syntax of ": type".
    MATCH('[');
    if (NextChar() != ']') {
      do {
        AttributeProto nextval;
        auto expected_type = ToSingletonType(attr.ref_type());
        CHECK_PARSER_STATUS(ParseSingleAttributeValue(nextval, expected_type));
        switch (nextval.ref_type()) {
        case AttributeProto::AttributeType::INT:
          attr.set_type(AttributeProto::AttributeType::INTS);
          attr.add_ints() = nextval.ref_i();
          break;
        case AttributeProto::AttributeType::FLOAT:
          attr.set_type(AttributeProto::AttributeType::FLOATS);
          attr.add_floats() = nextval.ref_f();
          break;
        case AttributeProto::AttributeType::STRING:
          attr.add_strings() = nextval.ref_s();
          attr.set_type(AttributeProto::AttributeType::STRINGS);
          break;
        default:
          break;
        }
      } while (Matches(','));
    } else {
      if (attr.ref_type() == AttributeProto::AttributeType::UNDEFINED)
        return ParseError("Empty list attribute value requires type annotation.");
      if (IsSingletonAttribute(attr.ref_type()))
        return ParseError("Singleton attribute value cannot be specified as a list.");
    }
    MATCH(']');
  } else {
    CHECK_PARSER_STATUS(ParseSingleAttributeValue(attr, attr.ref_type()));
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(AttrList &attrlist) {
  attrlist.clear();
  if (Matches('<')) {
    do {
      attrlist.emplace_back();
      PARSE(attrlist.back());
    } while (Matches(','));
    MATCH('>');
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(NodeProto &node) {
  if (Matches('[')) {
    std::string node_name;
    CHECK_PARSER_STATUS(ParseOptionalQuotableIdentifier(node_name));
    node.set_name(node_name);
    MATCH(']');
  }
  IdList outputs;
  PARSE(outputs);
  for (const auto &id : outputs)
    node.add_output() = id;
  MATCH('=');
  std::string domain;
  std::string id = ParseOptionalIdentifier();
  while (Matches('.')) {
    if (!domain.empty())
      domain += ".";
    domain += id;
    CHECK_PARSER_STATUS(ParseIdentifier(id));
  }
  node.set_domain(domain);
  node.set_op_type(id);

  if (Matches(':')) {
    std::string overload;
    CHECK_PARSER_STATUS(ParseIdentifier(overload));
    node.set_overload(overload);
  }
  AttrList attrs;
  PARSE(attrs);
  for (const auto &a : attrs)
    node.add_attribute(a);
  MATCH('(');
  IdList inputs;
  PARSE(inputs);
  for (const auto &inp : inputs)
    node.add_input() = inp;
  MATCH(')');
  if (node.ref_attribute().size() == 0) {
    // Permit attributes to be specified before or after parameters.
    AttrList attrs2;
    PARSE(attrs2);
    for (const auto &a : attrs2)
      node.add_attribute(a);
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(NodeList &nodelist) {
  nodelist.clear();
  MATCH('{');
  while (!Matches('}')) {
    nodelist.emplace_back();
    PARSE(nodelist.back());
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(GraphProto &graph) {
  std::string id;
  CHECK_PARSER_STATUS(ParseQuotableIdentifier(id));
  return Parse(id, graph);
}

Common::Status OnnxParser::Parse(std::string name, GraphProto &graph) {
  graph.set_name(name);
  graph.ref_initializer().clear();

  ValueInfoList inputs;
  TensorList initializers;
  CHECK_PARSER_STATUS(ParseInput(inputs, initializers));
  for (const auto &vi : inputs)
    graph.add_input(vi);
  for (const auto &t : initializers)
    graph.add_initializer(t);

  MATCH('=');
  MATCH('>', false);

  ValueInfoList outputs;
  CHECK_PARSER_STATUS(ParseGraphInputOutput(outputs));
  for (const auto &vi : outputs)
    graph.add_output(vi);

  ValueInfoList value_infos;
  TensorList more_initializers;
  CHECK_PARSER_STATUS(ParseValueInfo(value_infos, more_initializers));
  for (const auto &vi : value_infos)
    graph.add_value_info(vi);
  for (const auto &t : more_initializers)
    graph.add_initializer(t);

  NodeList nodes;
  PARSE(nodes);
  for (const auto &node : nodes)
    graph.add_node(node);
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(FunctionProto &fn) {
  fn = FunctionProto{};
  std::string strval;
  if (Matches('<')) {
    do {
      KeyWordMap::KeyWord keyword = KeyWordMap::KeyWord::NONE;
      PARSE_TOKEN(keyword);
      MATCH(':');
      switch (keyword) {
      case KeyWordMap::KeyWord::OPSET_IMPORT: {
        OpsetIdList opsets;
        PARSE(opsets);
        for (const auto &op : opsets)
          fn.ref_opset_import().push_back(op);
        break;
      }
      case KeyWordMap::KeyWord::DOC_STRING:
        PARSE_TOKEN(strval);
        fn.set_doc_string(strval);
        break;
      case KeyWordMap::KeyWord::DOMAIN_KW:
        PARSE_TOKEN(strval);
        fn.set_domain(strval);
        break;
      case KeyWordMap::KeyWord::OVERLOAD_KW:
        PARSE_TOKEN(strval);
        fn.set_overload(strval);
        break;
      default:
        return ParseError("Unhandled keyword.");
      }
    } while (Matches(','));
    MATCH('>');
  }
  std::string id;
  CHECK_PARSER_STATUS(ParseQuotableIdentifier(id));
  fn.set_name(id);

  IdList fn_attrs;
  AttrList fn_attr_protos;
  PARSE('<', fn_attrs, fn_attr_protos, '>');
  for (const auto &a : fn_attrs)
    fn.add_attribute() = a;
  for (const auto &ap : fn_attr_protos)
    fn.add_attribute_proto(ap);

  fn.ref_value_info().clear();

  IdList fn_inputs;
  ValueInfoList fn_input_vis;
  CHECK_PARSER_STATUS(ParseFunctionInputOutput(fn_inputs, fn_input_vis));
  for (const auto &inp : fn_inputs)
    fn.add_input() = inp;
  for (const auto &vi : fn_input_vis)
    fn.add_value_info(vi);

  MATCH('=');
  MATCH('>', false);

  IdList fn_outputs;
  ValueInfoList fn_output_vis;
  CHECK_PARSER_STATUS(ParseFunctionInputOutput(fn_outputs, fn_output_vis));
  for (const auto &out : fn_outputs)
    fn.add_output() = out;
  for (const auto &vi : fn_output_vis)
    fn.add_value_info(vi);

  if (NextChar() == '<') {
    ValueInfoList extra_vis;
    PARSE('<', extra_vis, '>');
    for (const auto &vi : extra_vis)
      fn.add_value_info(vi);
  }

  NodeList nodes;
  PARSE(nodes);
  for (const auto &node : nodes)
    fn.add_node(node);
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(OpsetIdList &opsets) {
  std::string strval;
  int64_t intval = 0;
  MATCH('[');
  if (!Matches(']')) {
    do {
      opsets.emplace_back();
      auto &import = opsets.back();
      PARSE_TOKEN(strval);
      import.set_domain(strval);
      MATCH(':');
      PARSE_TOKEN(intval);
      import.set_version(intval);
    } while (Matches(','));
    MATCH(']');
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(ModelProto &model) {
  model = ModelProto{};
  std::string strval;
  int64_t intval = 0;
  if (Matches('<')) {
    do {
      KeyWordMap::KeyWord keyword = KeyWordMap::KeyWord::NONE;
      PARSE_TOKEN(keyword);
      MATCH(':');
      switch (keyword) {
      case KeyWordMap::KeyWord::IR_VERSION:
        PARSE_TOKEN(intval);
        model.set_ir_version(intval);
        break;
      case KeyWordMap::KeyWord::OPSET_IMPORT: {
        OpsetIdList opsets;
        PARSE(opsets);
        for (const auto &op : opsets)
          model.ref_opset_import().push_back(op);
        break;
      }
      case KeyWordMap::KeyWord::PRODUCER_NAME:
        PARSE_TOKEN(strval);
        model.set_producer_name(strval);
        break;
      case KeyWordMap::KeyWord::PRODUCER_VERSION:
        PARSE_TOKEN(strval);
        model.set_producer_version(strval);
        break;
      case KeyWordMap::KeyWord::DOMAIN_KW:
        PARSE_TOKEN(strval);
        model.set_domain(strval);
        break;
      case KeyWordMap::KeyWord::MODEL_VERSION:
        PARSE_TOKEN(intval);
        model.set_model_version(intval);
        break;
      case KeyWordMap::KeyWord::DOC_STRING:
        PARSE_TOKEN(strval);
        model.set_doc_string(strval);
        break;
      case KeyWordMap::KeyWord::METADATA_PROPS: {
        StringStringList metadata_props;
        MATCH('[');
        if (!Matches(']')) {
          PARSE(metadata_props);
          MATCH(']');
        }
        for (auto &entry : metadata_props) {
          auto &prop = model.add_metadata_props();
          prop.set_key(entry.ref_key().as_string());
          prop.set_value(entry.ref_value().as_string());
        }
        break;
      }
      default:
        return ParseError("Unhandled keyword.");
      }
    } while (Matches(','));
    MATCH('>');
  }
  PARSE(model.ref_graph());

  auto &functions = model.ref_functions();
  while (!EndOfInput()) {
    FunctionProto fn;
    PARSE(fn);
    functions.push_back(fn);
  }
  return Common::Status::OK();
}

const std::unordered_map<std::string, KeyWordMap::KeyWord> &KeyWordMap::Instance() {
  static KeyWordMap instance;
  return instance.map_;
}

const std::string &KeyWordMap::ToString(KeyWord kw) {
  static std::string undefined("undefined");
  for (const auto &pair : Instance()) {
    if (pair.second == kw)
      return pair.first;
  }
  return undefined;
}

} // namespace ONNX_NAMESPACE
