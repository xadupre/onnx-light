#pragma once

#include "../../common/onnx_pb.h"
#include <sstream>
#include <string>

namespace ONNX_NAMESPACE {

using std::stoi;
using std::to_string;

inline void MakeStringInternal(std::stringstream & /*ss*/) {}

template <typename T> inline void MakeStringInternal(std::stringstream &ss, const T &t) { ss << t; }

template <typename T, typename... Args>
inline void MakeStringInternal(std::stringstream &ss, const T &t, const Args &...args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args> std::string MakeString(const Args &...args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return ss.str();
}

template <> inline std::string MakeString(const std::string &str) { return str; }

inline std::string MakeString(const char *c_str) { return std::string(c_str); }

} // namespace ONNX_NAMESPACE
