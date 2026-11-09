// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
//
// libFuzzer harness for the ONNX textual-format parser.
//
// Parse errors are returned as Status; unexpected exceptions are findings.

#include "onnx_manipulations/parser.h"
#include "onnx_proto/onnx.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <type_traits>

using ONNX_LIGHT_NAMESPACE::FunctionProto;
using ONNX_LIGHT_NAMESPACE::GraphProto;
using ONNX_LIGHT_NAMESPACE::ModelProto;
using ONNX_LIGHT_NAMESPACE::NodeProto;
using ONNX_LIGHT_NAMESPACE::OnnxParser;

namespace {

template <typename T> void CheckFloat(std::string_view text) {
  OnnxParser parser(text);
  T expected = 0;
  const auto expected_status = parser.Parse(expected);
  const std::string body = std::is_same_v<T, float> ? "value_float: float=" + std::string(text)
                                                    : "value=double[1] {" + std::string(text) + "}";
  NodeProto node;
  const auto status = OnnxParser::Parse(node, "Y = Constant<" + body + ">()");
  if (status.IsOK() != expected_status.IsOK())
    std::abort();
  if (status.IsOK()) {
    const auto &attr = node.ref_attribute()[0];
    const T actual = [&]() -> T {
      if constexpr (std::is_same_v<T, float>)
        return attr.ref_f();
      else
        return attr.ref_t().ref_double_data()[0];
    }();
    if (std::isnan(expected) ? !std::isnan(actual)
                             : actual != expected || std::signbit(actual) != std::signbit(expected))
      std::abort();
  }
}

void CheckInteger(std::string_view text) {
  if (text.empty())
    return;
  const auto digits = text.substr(text.front() == '-' ? 1 : 0);
  if (digits.empty() || digits.find_first_not_of("0123456789") != std::string_view::npos)
    return;
  const std::string literal(text);
  for (const auto *kind : {"int", "int64", "uint64"}) {
    int64_t signed_value = 0;
    uint64_t unsigned_value = 0;
    const bool is_unsigned = std::string_view(kind) == "uint64";
    const auto result =
        is_unsigned ? std::from_chars(text.data(), text.data() + text.size(), unsigned_value)
                    : std::from_chars(text.data(), text.data() + text.size(), signed_value);
    const bool valid = result.ec == std::errc{} && result.ptr == text.data() + text.size();
    const std::string body = std::string_view(kind) == "int"
                                 ? "value_int: int=" + literal
                                 : std::string("value=") + kind + "[1] {" + literal + "}";
    NodeProto node;
    const auto status = OnnxParser::Parse(node, "Y = Constant<" + body + ">()");
    if (status.IsOK() != valid)
      std::abort();
    if (valid) {
      const auto &attr = node.ref_attribute()[0];
      if (is_unsigned) {
        if (attr.ref_t().ref_uint64_data()[0] != unsigned_value)
          std::abort();
      } else {
        const auto actual =
            std::string_view(kind) == "int" ? attr.ref_i() : attr.ref_t().ref_int64_data()[0];
        if (actual != signed_value)
          std::abort();
      }
    }
  }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  const std::string_view text(reinterpret_cast<const char *>(data), size);
  ModelProto model;
  GraphProto graph;
  FunctionProto function;
  NodeProto node;
  (void)OnnxParser::Parse(model, text);
  (void)OnnxParser::Parse(graph, text);
  (void)OnnxParser::Parse(function, text);
  (void)OnnxParser::Parse(node, text);
  CheckInteger(text);
  OnnxParser literal_parser(text);
  OnnxParser::Literal literal;
  if (literal_parser.Parse(literal).IsOK() && literal_parser.EndOfInput() &&
      literal.value == text &&
      (literal.type == OnnxParser::LiteralType::INT_LITERAL ||
       literal.type == OnnxParser::LiteralType::FLOAT_LITERAL)) {
    CheckFloat<float>(text);
    CheckFloat<double>(text);
  }

  std::string quoted;
  quoted.reserve(text.size());
  for (char c : text) {
    if (c == '\\' || c == '"')
      quoted += '\\';
    quoted += c;
  }
  NodeProto string_node;
  const auto status =
      OnnxParser::Parse(string_node, "Y = Constant<value_string=\"" + quoted + "\">()");
  if (!status.IsOK() || string_node.ref_attribute()[0].ref_s() != text)
    std::abort();
  return 0;
}
