#pragma once

#include "stream_class.h"
#include "stream_class_print.hpp"
#include "stream_class_read.hpp"
#include "stream_class_size.hpp"
#include "stream_class_write.hpp"
#include <cstring>

////////////////
// macro helpers
////////////////

#define NAME_EXIST_VALUE(name) name_exist_value(_name_##name, has_##name(), ptr_##name())

#define IMPLEMENT_PROTO(cls)                                                                       \
  void cls::CopyFrom(const cls &proto) { _CopyFrom(*this, proto); }                                \
  SerializeSizeResult cls::SerializeSize() const { return _SerializeSize(*this); }                 \
  void cls::ParseFromString(const std::string &raw) { _ParseFromString(*this, raw); }              \
  void cls::ParseFromString(const std::string &raw, ParseOptions &opts) {                          \
    _ParseFromString(*this, raw, opts);                                                            \
  }                                                                                                \
  void cls::SerializeToString(std::string &out) const { _SerializeToString(*this, out); }          \
  void cls::SerializeToString(std::string &out, SerializeOptions &opts) const {                    \
    _SerializeToString(*this, out, opts);                                                          \
  }

///////////////////////
// macro serialize size
///////////////////////

#define SIZE_FIELD(size, options, stream, name)                                                    \
  if (has_##name()) {                                                                              \
    size += size_field(stream, order_##name(), ref_##name(), options);                             \
  }

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
    EXT_THROW("unable to parse field_number=", field_number.string(), " in class ", #cls);         \
  }                                                                                                \
  }                                                                                                \
  DEBUG_PRINT("+ read end " #cls)

#define READ_FIELD(options, stream, name)                                                          \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                        \
    DEBUG_PRINT("  + field " #name)                                                                \
    read_field(stream, field_number.wire_type, name##_, #name, options);                           \
    DEBUG_PRINT("  - field " #name)                                                                \
  }

#define READ_FIELD_LIMIT_PARALLEL(options, stream, name)                                           \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                        \
    DEBUG_PRINT("  + field " #name)                                                                \
    read_field_limit_parallel(stream, field_number.wire_type, name##_, #name, options);            \
    DEBUG_PRINT("  - field " #name)                                                                \
  }

#define READ_OPTIONAL_PROTO_FIELD(options, stream, name)                                           \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                        \
    DEBUG_PRINT("  + optional field " #name)                                                       \
    read_optional_proto_field(stream, field_number.wire_type, name##_, #name, options);            \
    DEBUG_PRINT("  - optional field " #name)                                                       \
  }

#define READ_ENUM_FIELD(options, stream, name)                                                     \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                        \
    DEBUG_PRINT("  + enum " #name)                                                                 \
    read_enum_field(stream, field_number.wire_type, name##_, #name, options);                      \
    DEBUG_PRINT("  - enum " #name)                                                                 \
  }

#define READ_OPTIONAL_ENUM_FIELD(options, stream, name)                                            \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                        \
    DEBUG_PRINT("  + enum " #name)                                                                 \
    read_optional_enum_field(stream, field_number.wire_type, name##_, #name, options);             \
    DEBUG_PRINT("  - enum " #name)                                                                 \
  }

#define READ_REPEATED_FIELD(options, stream, name)                                                 \
  else if (static_cast<int>(field_number.field_number) == order_##name()) {                        \
    DEBUG_PRINT("  + repeat " #name)                                                               \
    read_repeated_field(stream, field_number.wire_type, name##_, #name, packed_##name(), options); \
    DEBUG_PRINT("  - repeat " #name)                                                               \
  }

using namespace onnx_light_helpers;

namespace ONNX_LIGHT_NAMESPACE {

template <typename cls> void _CopyFrom(cls &self, const cls &proto) {
  utils::StringWriteStream stream;
  SerializeOptions opts;
  SerializeSizeResult total_size = proto.SerializeSize(stream, opts);
  stream.pre_allocate(total_size.size());
  proto.SerializeToStream(stream, opts);
  utils::StringStream read_stream(stream.data(), stream.size());
  ParseOptions ropts;
  self.ParseFromStream(read_stream, ropts);
}

template <typename cls> SerializeSizeResult _SerializeSize(cls &self) {
  SerializeOptions opts;
  utils::StringWriteStream stream;
  return self.SerializeSize(stream, opts);
}

template <typename cls> void _ParseFromString(cls &self, const std::string &raw) {
  ParseOptions opts;
  self.ParseFromString(raw, opts);
}

template <typename cls>
void _ParseFromString(cls &self, const std::string &raw, ParseOptions &opts) {
  const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
  ONNX_LIGHT_NAMESPACE::utils::StringStream st(ptr, raw.size());
  if (opts.parallel)
    st.StartThreadPool(opts.num_threads);
  self.ParseFromStream(st, opts);
  if (opts.parallel)
    st.WaitForDelayedBlock();
}

template <typename cls> void _SerializeToString(cls &self, std::string &out) {
  SerializeOptions opts;
  self.SerializeToString(out, opts);
}

template <typename cls>
void _SerializeToString(cls &self, std::string &out, SerializeOptions &opts) {
  ONNX_LIGHT_NAMESPACE::utils::StringWriteStream size_buf;
  // Two-pass approach: compute the total serialized size first so we can
  // resize the output string exactly once, then write directly into it via
  // BorrowedStringWriteStream — eliminating any intermediate copy.
  // The size pass also populates the stream's size cache so the write pass
  // reuses cached sub-message sizes without recomputing them.
  SerializeSizeResult total_size = self.SerializeSize(size_buf, opts);
  out.resize(static_cast<size_t>(total_size.size()));
  ONNX_LIGHT_NAMESPACE::utils::BorrowedStringWriteStream buf(
      reinterpret_cast<uint8_t *>(out.data()), total_size.size());
  size_buf.swap_size_cache(buf);
  if (opts.parallel) {
    buf.StartThreadPool(opts.num_threads);
  }
  self.SerializeToStream(buf, opts);
  if (buf.HasParallelizationStarted()) {
    buf.WaitForDelayedBlock();
  }
}

} // namespace ONNX_LIGHT_NAMESPACE
