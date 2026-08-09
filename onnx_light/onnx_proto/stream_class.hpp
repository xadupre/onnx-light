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

#define IMPLEMENT_PROTO(cls)

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
