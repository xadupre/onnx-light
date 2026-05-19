// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/// @file proto_utils.h
/// @brief General-purpose protobuf utility templates: debug formatting,
///        byte-buffer parsing, and attribute-value extraction.

#pragma once

#include "onnx_pb.h"
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

/**
 * @brief Returns a human-readable debug representation of a protobuf message.
 *
 * Calls @c PrintToVectorString() on @p proto with default print options and
 * joins the resulting lines with newline characters.
 *
 * @tparam Proto A protobuf-like message type that exposes a
 *               @c PrintToVectorString(utils::PrintOptions&) method.
 * @param proto  The message to format.
 * @returns A multi-line string representation of @p proto.
 */
template <typename Proto> inline std::string ProtoDebugString(const Proto &proto) {
  utils::PrintOptions options;
  std::vector<std::string> rows = proto.PrintToVectorString(options);
  std::string text;
  for (size_t i = 0; i < rows.size(); ++i) {
    if (i > 0) {
      text.push_back('\n');
    }
    text.append(rows[i]);
  }
  return text;
}

/**
 * @brief Parses a protobuf message from a raw byte buffer.
 *
 * Constructs a @c StringStream over the supplied buffer and calls
 * @c ParseFromStream() on @p proto.  Returns @c false without modifying
 * @p proto when @p proto is null, when @p buffer is null but @p length is
 * positive, when parsing raises a @c std::runtime_error, or when bytes remain
 * unconsumed after parsing completes.
 *
 * @tparam Proto A protobuf-like message type that exposes a
 *               @c ParseFromStream(utils::StringStream&, ParseOptions&) method.
 * @param proto  Destination message that receives the parsed content.  Must not
 *               be @c nullptr.
 * @param buffer Pointer to the serialized byte data.  May be @c nullptr only
 *               when @p length is @c 0.
 * @param length Number of bytes in @p buffer.
 * @returns @c true if the entire buffer was consumed and parsed successfully;
 *          @c false otherwise.
 */
template <typename Proto>
inline bool ParseProtoFromBytes(Proto *proto, const char *buffer, size_t length) {
  if (proto == nullptr || (buffer == nullptr && length > 0)) {
    return false;
  }
  ParseOptions options;
  utils::StringStream stream(reinterpret_cast<const uint8_t *>(buffer),
                             static_cast<int64_t>(length));
  try {
    proto->ParseFromStream(stream, options);
  } catch (const std::runtime_error &) {
    return false;
  }
  return !stream.NotEnd();
}

/**
 * @brief Extracts a typed list of values from an @c AttributeProto.
 *
 * The primary template is declared but not defined; callers must use one of
 * the explicit specializations for @c int64_t, @c std::string, or @c float.
 *
 * @tparam T Element type to retrieve.  Must be @c int64_t, @c std::string,
 *            or @c float.
 * @param attr Source attribute whose repeated field is extracted.
 * @returns A @c std::vector<T> containing the attribute values.
 */
template <typename T> inline std::vector<T> RetrieveValues(const AttributeProto &attr);

/**
 * @brief Extracts a list of @c int64_t values from an @c AttributeProto.
 *
 * @param attr Source attribute (must have type @c INT or @c INTS).
 * @returns A @c std::vector<int64_t> copied from the @c ints repeated field.
 */
template <> inline std::vector<int64_t> RetrieveValues(const AttributeProto &attr) {
  return {attr.ref_ints().begin(), attr.ref_ints().end()};
}

/**
 * @brief Extracts a list of @c std::string values from an @c AttributeProto.
 *
 * @param attr Source attribute (must have type @c STRING or @c STRINGS).
 * @returns A @c std::vector<std::string> copied from the @c strings repeated
 *          field.
 */
template <> inline std::vector<std::string> RetrieveValues(const AttributeProto &attr) {
  std::vector<std::string> result;
  result.reserve(attr.ref_strings().size());
  for (const utils::String &value : attr.ref_strings()) {
    result.push_back(value.as_string());
  }
  return result;
}

/**
 * @brief Extracts a list of @c float values from an @c AttributeProto.
 *
 * @param attr Source attribute (must have type @c FLOAT or @c FLOATS).
 * @returns A @c std::vector<float> copied from the @c floats repeated field.
 */
template <> inline std::vector<float> RetrieveValues(const AttributeProto &attr) {
  return {attr.ref_floats().begin(), attr.ref_floats().end()};
}

} // namespace ONNX_LIGHT_NAMESPACE
