#pragma once

/**
 * @file string_utils.h
 * @brief String conversion and concatenation helpers used by ONNX utilities.
 */

#include "onnx_pb.h"
#include <sstream>
#include <string>

namespace ONNX_LIGHT_NAMESPACE {

using std::stoi;
using std::to_string;

inline void MakeStringInternal(std::stringstream & /*ss*/) {}

template <typename T> inline void MakeStringInternal(std::stringstream &ss, const T &t) { ss << t; }

template <typename T, typename... Args>
inline void MakeStringInternal(std::stringstream &ss, const T &t, const Args &...args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

/**
 * @brief Concatenates values into a single string using stream insertion.
 *
 * This is the primary template. This header also provides an inline
 * specialization for @c std::string and an overload for @c const char*.
 *
 * @tparam Args Argument types intended to support stream insertion via
 * operator<<; compilation fails for unsupported types.
 * @param args Variadic parameter pack whose values are appended in order by
 * stream insertion.
 * @return Concatenated string representation of all inputs.
 */
template <typename... Args> std::string MakeString(const Args &...args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return ss.str();
}

template <> inline std::string MakeString(const std::string &str) { return str; }

/**
 * @brief Returns a @c std::string from a C string pointer.
 * @param c_str Null-terminated C string. This overload intentionally performs
 * no null check; passing null is undefined behavior.
 * @return Converted @c std::string value.
 */
inline std::string MakeString(const char *c_str) { return std::string(c_str); }

} // namespace ONNX_LIGHT_NAMESPACE
