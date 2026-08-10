#pragma once

#include "stream_class.h"
#include "stream_class_print.hpp"
#include "stream_class_read.hpp"
#include "stream_class_size.hpp"
#include "stream_class_write.hpp"
#include <cstring>
#include <istream>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

////////////////
// macro helpers
////////////////

#define NAME_EXIST_VALUE(name) name_exist_value(_name_##name, has_##name(), ptr_##name())

#define IMPLEMENT_PROTO(cls)                                                                       \
  void cls::CopyFrom(const cls &proto) { CopyProtoFrom(*this, proto); }

///////////////////////
// macro serialize size
///////////////////////

#define SIZE_FIELD(size, options, stream, name)                                                    \
  if (has_##name()) {                                                                              \
    size += size_field(stream, order_##name(), ref_##name(), options);                             \
  }

// Variant for fields whose presence is guaranteed by construction. Skips the
// has_##name() branch entirely (one less load + branch in the size pass).
#define SIZE_FIELD_REQUIRED(size, options, stream, name)                                           \
  size += size_field(stream, order_##name(), ref_##name(), options);

#define SIZE_FIELD_NULL(size, options, stream, name)                                               \
  if (!name##_.null()) {                                                                           \
    size += size_field(stream, order_##name(), ref_##name(), options);                             \
  }

#define SIZE_FIELD_EMPTY(size, options, stream, name)                                              \
  size += size_field(stream, order_##name(), ref_##name(), options);

#define SIZE_FIELD_LIMIT(size, options, stream, name)                                              \
  if (has_##name()) {                                                                              \
    size += size_field_limit(stream, order_##name(), ref_##name(), options);                       \
  }

#define SIZE_ENUM_FIELD(size, options, stream, name)                                               \
  if (has_##name()) {                                                                              \
    size += size_enum_field(stream, order_##name(), ref_##name(), options);                        \
  }

// Variant for required enum fields: skips the has_##name() branch.
#define SIZE_ENUM_FIELD_REQUIRED(size, options, stream, name)                                      \
  size += size_enum_field(stream, order_##name(), ref_##name(), options);

#define SIZE_REPEATED_FIELD(size, options, stream, name)                                           \
  if (has_##name()) {                                                                              \
    size += size_repeated_field(stream, order_##name(), name##_, packed_##name(), options);        \
  }

#define SIZE_OPTIONAL_PROTO_FIELD(size, options, stream, name)                                     \
  if (has_##name()) {                                                                              \
    size += size_optional_proto_field(stream, order_##name(), name##_optional(), options);         \
  }

//////////////
// macro write
//////////////

#define WRITE_FIELD(options, stream, name)                                                         \
  if (has_##name()) {                                                                              \
    write_field(stream, order_##name(), ref_##name(), options);                                    \
  }

// Variant for required fields: skips the has_##name() branch.
#define WRITE_FIELD_REQUIRED(options, stream, name)                                                \
  write_field(stream, order_##name(), ref_##name(), options);

#define WRITE_FIELD_NULL(options, stream, name)                                                    \
  if (!name##_.null()) {                                                                           \
    write_field(stream, order_##name(), ref_##name(), options);                                    \
  }

#define WRITE_FIELD_EMPTY(options, stream, name)                                                   \
  write_field(stream, order_##name(), ref_##name(), options);

#define WRITE_FIELD_LIMIT(options, stream, name)                                                   \
  if (has_##name()) {                                                                              \
    write_field_limit(stream, order_##name(), ref_##name(), options);                              \
  }

#define WRITE_ENUM_FIELD(options, stream, name)                                                    \
  if (has_##name()) {                                                                              \
    write_enum_field(stream, order_##name(), ref_##name(), options);                               \
  }

// Variant for required enum fields: skips the has_##name() branch.
#define WRITE_ENUM_FIELD_REQUIRED(options, stream, name)                                           \
  write_enum_field(stream, order_##name(), ref_##name(), options);

#define WRITE_REPEATED_FIELD(options, stream, name)                                                \
  if (has_##name()) {                                                                              \
    write_repeated_field(stream, order_##name(), name##_, packed_##name(), options);               \
  }

#define WRITE_OPTIONAL_PROTO_FIELD(options, stream, name)                                          \
  if (has_##name()) {                                                                              \
    write_optional_proto_field(stream, order_##name(), name##_optional(), options);                \
  }

/////////////
// macro read
/////////////

#define READ_BEGIN(options, stream, cls)                                                           \
  DEBUG_PRINT("+ read begin " #cls)                                                                \
  while (stream.NotEnd()) {                                                                        \
    utils::FieldNumber field_number = stream.next_field();                                         \
    DEBUG_PRINT2("  = field number ", field_number.string().c_str())                               \
    if (field_number.field_number == 0) {                                                          \
      EXT_THROW("unexpected field_number=", field_number.string(), " in class ", #cls);            \
    }

#define READ_END(options, stream, cls)                                                             \
  else {                                                                                           \
    /* A field number not present in this message's schema is an unknown field.  Per the           \
     * protobuf wire-format compatibility rules, it must be skipped (based on its wire type)       \
     * rather than treated as a parse error, so forward/backward-compatible and adversarially      \
     * crafted messages alike can still be parsed. */                                              \
    SkipFieldByWireType(stream, field_number.wire_type, #cls);                                     \
  }                                                                                                \
  }                                                                                                \
  DEBUG_PRINT("+ read end " #cls)

#define READ_FIELD(options, stream, name)                                                          \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                        \
    DEBUG_PRINT("  + field " #name)                                                                \
    read_field(stream, static_cast<int>(field_number.wire_type), name##_, #name, options);         \
    DEBUG_PRINT("  - field " #name)                                                                \
  }

#define READ_FIELD_LIMIT_PARALLEL(options, stream, name)                                           \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                        \
    DEBUG_PRINT("  + field " #name)                                                                \
    read_field_limit_parallel(stream, static_cast<int>(field_number.wire_type), name##_, #name,    \
                              options);                                                            \
    DEBUG_PRINT("  - field " #name)                                                                \
  }

#define READ_OPTIONAL_PROTO_FIELD(options, stream, name)                                           \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                        \
    DEBUG_PRINT("  + optional field " #name)                                                       \
    read_optional_proto_field(stream, static_cast<int>(field_number.wire_type), name##_, #name,    \
                              options);                                                            \
    DEBUG_PRINT("  - optional field " #name)                                                       \
  }

#define READ_ENUM_FIELD(options, stream, name)                                                     \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                        \
    DEBUG_PRINT("  + enum " #name)                                                                 \
    read_enum_field(stream, static_cast<int>(field_number.wire_type), name##_, #name, options);    \
    DEBUG_PRINT("  - enum " #name)                                                                 \
  }

#define READ_OPTIONAL_ENUM_FIELD(options, stream, name)                                            \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                        \
    DEBUG_PRINT("  + enum " #name)                                                                 \
    read_optional_enum_field(stream, static_cast<int>(field_number.wire_type), name##_, #name,     \
                             options);                                                             \
    DEBUG_PRINT("  - enum " #name)                                                                 \
  }

#define READ_REPEATED_FIELD(options, stream, name)                                                 \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                        \
    DEBUG_PRINT("  + repeat " #name)                                                               \
    read_repeated_field(stream, static_cast<int>(field_number.wire_type), name##_, #name,          \
                        packed_##name(), options);                                                 \
    DEBUG_PRINT("  - repeat " #name)                                                               \
  }

namespace ONNX_LIGHT_NAMESPACE {

using CopyProtoSizeFunction = SerializeSizeResult (*)(const void *, utils::BinaryWriteStream &,
                                                      SerializeOptions &);
using CopyProtoWriteFunction = void (*)(const void *, utils::BinaryWriteStream &,
                                        SerializeOptions &);
using CopyProtoParseFunction = bool (*)(void *, utils::BinaryStream &, ParseOptions &);

void CopyProtoThroughWire(void *destination, const void *source,
                          CopyProtoSizeFunction size_function,
                          CopyProtoWriteFunction write_function,
                          CopyProtoParseFunction parse_function) {
  utils::StringWriteStream stream;
  SerializeOptions opts;
  SerializeSizeResult total_size = size_function(source, stream, opts);
  stream.pre_allocate(total_size.size());
  write_function(source, stream, opts);
  utils::StringStream read_stream(stream.data(), stream.size());
  ParseOptions parse_options;
  parse_function(destination, read_stream, parse_options);
}

template <typename T> void CopyProtoFrom(T &destination, const T &source) {
  CopyProtoThroughWire(
      &destination, &source,
      [](const void *proto, utils::BinaryWriteStream &stream, SerializeOptions &options) {
        return static_cast<const T *>(proto)->SerializeSize(stream, options);
      },
      [](const void *proto, utils::BinaryWriteStream &stream, SerializeOptions &options) {
        static_cast<const T *>(proto)->SerializeToStream(stream, options);
      },
      [](void *proto, utils::BinaryStream &stream, ParseOptions &options) {
        return static_cast<T *>(proto)->ParseFromStream(stream, options);
      });
}

} // namespace ONNX_LIGHT_NAMESPACE
