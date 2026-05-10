#pragma once

#include "onnx_pb.h"
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

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

template <typename T> inline std::vector<T> RetrieveValues(const AttributeProto &attr);

template <> inline std::vector<int64_t> RetrieveValues(const AttributeProto &attr) {
  return {attr.ref_ints().begin(), attr.ref_ints().end()};
}

template <> inline std::vector<std::string> RetrieveValues(const AttributeProto &attr) {
  std::vector<std::string> result;
  result.reserve(attr.ref_strings().size());
  for (const utils::String &value : attr.ref_strings()) {
    result.push_back(value.as_string());
  }
  return result;
}

template <> inline std::vector<float> RetrieveValues(const AttributeProto &attr) {
  return {attr.ref_floats().begin(), attr.ref_floats().end()};
}

} // namespace ONNX_LIGHT_NAMESPACE
