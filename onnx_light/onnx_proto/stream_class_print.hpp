#pragma once

#include "stream_class.h"
#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace onnx_light_helpers;

namespace ONNX_LIGHT_NAMESPACE {

template <typename T> struct name_exist_value {
  const char *name;
  bool exist;
  const T *value;
  inline name_exist_value(const char *n, bool e, const T *v) : name(n), exist(e), value(v) {}
};

template <typename T> std::string write_as_string(utils::PrintOptions &, const T &field) {
  return MakeString(field);
}

template <> inline std::string write_as_string(utils::PrintOptions &, const utils::String &field) {
  return utils::quote_string(field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &, const utils::OptionalString &field) {
  if (!field.has_value())
    return "null";
  return utils::quote_string(field.value());
}

template <>
inline std::string write_as_string(utils::PrintOptions &, const std::vector<uint8_t> &field) {
  const char *hex_chars = "0123456789ABCDEF";
  std::string result;
  result.reserve(field.size() * 2);
  for (const auto &b : field) {
    result += hex_chars[b / 16];
    result += hex_chars[b % 16];
  }
  return result;
}

template <>
inline std::string write_as_string(utils::PrintOptions &, const utils::ByteSpan &field) {
  const char *hex_chars = "0123456789ABCDEF";
  std::string result;
  result.reserve(field.size() * 2);
  const uint8_t *p = field.data();
  for (size_t i = 0; i < field.size(); ++i) {
    result += hex_chars[p[i] / 16];
    result += hex_chars[p[i] % 16];
  }
  return result;
}

template <typename T>
std::string write_as_string_vector(utils::PrintOptions &, const std::vector<T> &field) {
  std::string result;
  result += "[";
  for (size_t i = 0; i < field.size(); ++i) {
    result += MakeString(field[i]);
    if (i + 1 != field.size())
      result += ", ";
  }
  result += "]";
  return result;
}

template <typename T>
std::string write_as_repeated_field(utils::PrintOptions &, const utils::RepeatedField<T> &field) {
  std::string result;
  result += "[";
  for (size_t i = 0; i < field.size(); ++i) {
    result += MakeString(field[i]);
    if (i + 1 != field.size())
      result += ", ";
  }
  result += "]";
  return result;
}

template <>
inline std::string write_as_repeated_field(utils::PrintOptions &,
                                           const utils::RepeatedField<utils::String> &field) {
  std::string result;
  result += "[";
  for (size_t i = 0; i < field.size(); ++i) {
    result += utils::quote_string(field[i]);
    if (i + 1 != field.size())
      result += ", ";
  }
  result += "]";
  return result;
}

template <typename T>
std::string write_as_string_optional(utils::PrintOptions &options, const std::optional<T> &field) {
  if (!field)
    return "null";
  return write_as_string(options, *field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options, const std::vector<float> &field) {
  return write_as_string_vector(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const std::vector<int64_t> &field) {
  return write_as_string_vector(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const std::vector<uint64_t> &field) {
  return write_as_string_vector(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options, const std::vector<double> &field) {
  return write_as_string_vector(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const std::vector<int32_t> &field) {
  return write_as_string_vector(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const std::optional<float> &field) {
  return write_as_string_optional(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const std::optional<int64_t> &field) {
  return write_as_string_optional(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const std::optional<uint64_t> &field) {
  return write_as_string_optional(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const std::optional<double> &field) {
  return write_as_string_optional(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const std::optional<int32_t> &field) {
  return write_as_string_optional(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const utils::RepeatedField<float> &field) {
  return write_as_repeated_field(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const utils::RepeatedField<int64_t> &field) {
  return write_as_repeated_field(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const utils::RepeatedField<uint64_t> &field) {
  return write_as_repeated_field(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const utils::RepeatedField<int32_t> &field) {
  return write_as_repeated_field(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const utils::RepeatedField<double> &field) {
  return write_as_repeated_field(options, field);
}

template <>
inline std::string write_as_string(utils::PrintOptions &options,
                                   const utils::RepeatedField<utils::String> &field) {
  return write_as_repeated_field(options, field);
}

inline std::string write_as_string(utils::PrintOptions &options,
                                   const utils::RepeatedStringField &field) {
  return write_as_repeated_field(options,
                                 static_cast<const utils::RepeatedField<utils::String> &>(field));
}

template <typename... Args>
std::string write_as_string(utils::PrintOptions &options, const Args &...args) {
  std::string result;
  result += "{";

  auto append_arg = [&options, &result, first = true](const auto &arg) mutable {
    if (arg.exist) {
      if (!first) {
        result += ", ";
      }
      first = false;
      result += arg.name;
      result += ": ";
      auto s = write_as_string(options, *arg.value);
      if (!s.empty()) {
        if (s[s.size() - 1] == ',') {
          s.pop_back();
        }
        result += s;
      }
    }
  };

  (append_arg(args), ...);
  result += "}";
  return result;
}

template <typename T>
void write_into_stream(std::stringstream &ss, utils::PrintOptions &options, const char *field_name,
                       const T &field) {
  ss << field_name << ": ";
  field.PrintToStringStream(ss, options);
  ss << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const utils::String &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const utils::OptionalString &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const int64_t &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const float &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const uint64_t &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const int32_t &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const TensorProto::DataType &field) {
  ss << field_name << ": " << write_as_string(options, static_cast<int32_t>(field)) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const SequenceProto::DataType &field) {
  ss << field_name << ": " << write_as_string(options, static_cast<int32_t>(field)) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const OptionalProto::DataType &field) {
  ss << field_name << ": " << write_as_string(options, static_cast<int32_t>(field)) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const TensorProto::DataLocation &field) {
  ss << field_name << ": " << write_as_string(options, static_cast<int32_t>(field)) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const AttributeProto::AttributeType &field) {
  ss << field_name << ": " << write_as_string(options, static_cast<int32_t>(field)) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const std::vector<uint8_t> &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const utils::ByteSpan &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name,
                              const utils::RepeatedField<utils::String> &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const utils::RepeatedField<uint64_t> &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const utils::RepeatedField<int64_t> &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const utils::RepeatedField<int32_t> &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const utils::RepeatedField<float> &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const utils::RepeatedField<double> &field) {
  ss << field_name << ": " << write_as_string(options, field) << " ";
}

template <typename T>
void write_into_stream_optional(std::stringstream &ss, utils::PrintOptions &options,
                                const char *field_name, const utils::OptionalField<T> &field) {
  if (field.has_value()) {
    ss << field_name << ": " << write_as_string(options, *field) << " ";
  } else {
    ss << field_name << ": null ";
  }
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const utils::OptionalField<int64_t> &field) {
  write_into_stream_optional(ss, options, field_name, field);
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const utils::OptionalField<uint64_t> &field) {
  write_into_stream_optional(ss, options, field_name, field);
}

template <>
inline void write_into_stream(std::stringstream &ss, utils::PrintOptions &options,
                              const char *field_name, const utils::OptionalField<int32_t> &field) {
  write_into_stream_optional(ss, options, field_name, field);
}

template <typename... Args>
void write_proto_into_vector_string(std::stringstream &ss, utils::PrintOptions &options,
                                    const Args &...args) {
  ss << "{ ";
  auto append_arg = [&ss, &options](const auto &arg) mutable {
    if (arg.exist) {
      write_into_stream(ss, options, arg.name, *arg.value);
    }
  };
  (append_arg(args), ...);
  ss << "}";
}

} // namespace ONNX_LIGHT_NAMESPACE
