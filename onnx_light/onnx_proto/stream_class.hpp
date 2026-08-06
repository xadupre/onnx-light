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
  void cls::CopyFrom(const cls &proto) { _CopyFrom(*this, proto); }                                \
  SerializeSizeResult cls::SerializeSize() const { return _SerializeSize(*this); }                 \
  size_t cls::ByteSizeLong() const { return static_cast<size_t>(SerializeSize().size()); }         \
  bool cls::ParseFromString(const std::string &raw) { return _ParseFromString(*this, raw); }       \
  bool cls::ParseFromString(const std::string &raw, ParseOptions &opts) {                          \
    return _ParseFromString(*this, raw, opts);                                                     \
  }                                                                                                \
  bool cls::ParseFromZeroCopyStream(utils::BinaryStream *stream) {                                 \
    return _ParseFromZeroCopyStream(*this, stream);                                                \
  }                                                                                                \
  bool cls::ParseFromZeroCopyStream(utils::BinaryStream *stream, ParseOptions &opts) {             \
    return _ParseFromZeroCopyStream(*this, stream, opts);                                          \
  }                                                                                                \
  bool cls::ParseFromIstream(std::istream *input) { return _ParseFromIstream(*this, input); }      \
  bool cls::ParseFromFileDescriptor(int fd) { return _ParseFromFileDescriptor(*this, fd); }        \
  std::string cls::SerializeAsString() const { return _SerializeAsString(*this); }                 \
  bool cls::SerializeToArray(void *data, int size) const {                                         \
    return _SerializeToArray(*this, data, size);                                                   \
  }                                                                                                \
  bool cls::SerializeToOstream(std::ostream *output) const {                                       \
    return _SerializeToOstream(*this, output);                                                     \
  }                                                                                                \
  bool cls::SerializeToOStream(std::ostream *output) const {                                       \
    return _SerializeToOstream(*this, output);                                                     \
  }                                                                                                \
  bool cls::SerializeToString(std::string &out) const { return _SerializeToString(*this, out); }   \
  bool cls::SerializeToString(std::string &out, SerializeOptions &opts) const {                    \
    return _SerializeToString(*this, out, opts);                                                   \
  }                                                                                                \
  bool cls::SerializeToFileDescriptor(int fd) const {                                              \
    return _SerializeToFileDescriptor(*this, fd);                                                  \
  }                                                                                                \
  bool cls::SerializeToFileDescriptor(int fd, SerializeOptions &opts) const {                      \
    return _SerializeToFileDescriptor(*this, fd, opts);                                            \
  }                                                                                                \
  bool cls::SaveToFileDescriptor(int fd) const { return _SerializeToFileDescriptor(*this, fd); }   \
  bool cls::SaveToFileDescriptor(int fd, SerializeOptions &opts) const {                           \
    return _SerializeToFileDescriptor(*this, fd, opts);                                            \
  }

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

template <typename cls> bool _ParseFromString(cls &self, const std::string &raw) {
  ParseOptions opts;
  return self.ParseFromString(raw, opts);
}

template <typename cls>
bool _ParseFromString(cls &self, const std::string &raw, ParseOptions &opts) {
  if (opts.format == SerializeFormat::kOrtFlatbuffers) {
    // Recursion-OOM guard: validate the depth limit before any parsing begins.
    // When the flatbuffer reader is fully implemented this limit will be
    // threaded through the recursive table traversal at each nesting level so
    // that a maliciously crafted .ort file cannot exhaust the call stack.
    EXT_ENFORCE(opts.max_recursion_depth > 0,
                "ParseFromString: ParseOptions::max_recursion_depth must be > 0 "
                "(got ",
                opts.max_recursion_depth,
                "). "
                "The ORT flatbuffer parser uses this limit to reject models "
                "nested more deeply than the configured value, preventing stack "
                "overflow on adversarially deep inputs.");
    // Tensor-size OOM guard: max_tensor_size_bytes must be >= 0.
    EXT_ENFORCE(opts.max_tensor_size_bytes >= 0,
                "ParseFromString: ParseOptions::max_tensor_size_bytes must be >= 0 "
                "(got ",
                opts.max_tensor_size_bytes,
                "). Use 0 to disable the limit or a positive value to cap tensor allocations.");
    EXT_THROW("ParseFromString: SerializeFormat::kOrtFlatbuffers is not implemented yet. "
              "Use SerializeFormat::kOnnx for now.");
  } else {
    EXT_ENFORCE(opts.format == SerializeFormat::kOnnx,
                "ParseFromString: unrecognised SerializeFormat value ",
                static_cast<int>(opts.format), "; only kOnnx is currently supported.");
    // Per the protobuf API contract, ParseFromString/ParseFromArray never throw: any
    // failure while decoding *raw* (malformed varints, truncated fields, corrupted
    // tags, etc.) must be reported by returning false, not by letting an exception
    // escape. Callers throughout onnxruntime (e.g. Model::LoadFromBytes) rely on this
    // to turn adversarial/corrupted model bytes into a normal Status rather than an
    // uncaught C++ exception. Deliberate, configurable resource guards
    // (ParseLimitExceeded: max_recursion_depth / max_tensor_size_bytes / alignment)
    // are re-thrown -- the caller configured that limit and needs to know it was hit,
    // unlike generic wire-format corruption.
    try {
      const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
      ONNX_LIGHT_NAMESPACE::utils::StringStream st(ptr, raw.size());
      if (opts.is_parallel())
        st.StartThreadPool(opts.num_threads);
      self.ParseFromStream(st, opts);
      if (opts.is_parallel())
        st.WaitForDelayedBlock();
    } catch (const onnx_light_helpers::ParseLimitExceeded &) {
      throw;
    } catch (const std::exception &) {
      return false;
    }
  }
  return true;
}

template <typename cls>
bool _ParseFromZeroCopyStream(cls &self, ONNX_LIGHT_NAMESPACE::utils::BinaryStream *stream) {
  EXT_ENFORCE(stream != nullptr, "ParseFromZeroCopyStream: stream pointer must not be null.");
  ParseOptions opts;
  self.ParseFromStream(*stream, opts);
  // Parsing errors are signalled via exceptions (EXT_THROW/EXT_ENFORCE). The
  // bool return follows the Google Protobuf API contract; it is always true
  // when the function returns normally.
  return true;
}

template <typename cls>
bool _ParseFromZeroCopyStream(cls &self, ONNX_LIGHT_NAMESPACE::utils::BinaryStream *stream,
                              ParseOptions &opts) {
  EXT_ENFORCE(stream != nullptr, "ParseFromZeroCopyStream: stream pointer must not be null.");
  if (opts.format == SerializeFormat::kOrtFlatbuffers) {
    // Recursion-OOM guard: validate the depth limit before any parsing begins.
    EXT_ENFORCE(opts.max_recursion_depth > 0,
                "ParseFromZeroCopyStream: ParseOptions::max_recursion_depth must be > 0 "
                "(got ",
                opts.max_recursion_depth,
                "). "
                "The ORT flatbuffer parser uses this limit to reject models "
                "nested more deeply than the configured value, preventing stack "
                "overflow on adversarially deep inputs.");
    // Tensor-size OOM guard: max_tensor_size_bytes must be >= 0.
    EXT_ENFORCE(opts.max_tensor_size_bytes >= 0,
                "ParseFromZeroCopyStream: ParseOptions::max_tensor_size_bytes must be >= 0 "
                "(got ",
                opts.max_tensor_size_bytes,
                "). Use 0 to disable the limit or a positive value to cap tensor allocations.");
    EXT_THROW("ParseFromZeroCopyStream: SerializeFormat::kOrtFlatbuffers is not implemented yet. "
              "Use SerializeFormat::kOnnx for now.");
  }
  EXT_ENFORCE(opts.format == SerializeFormat::kOnnx,
              "ParseFromZeroCopyStream: unrecognised SerializeFormat value ",
              static_cast<int>(opts.format), "; only kOnnx is currently supported.");
  if (opts.is_parallel())
    stream->StartThreadPool(opts.num_threads);
  self.ParseFromStream(*stream, opts);
  if (opts.is_parallel())
    stream->WaitForDelayedBlock();
  // Parsing errors are signalled via exceptions (EXT_THROW/EXT_ENFORCE). The
  // bool return follows the Google Protobuf API contract; it is always true
  // when the function returns normally.
  return true;
}

template <typename cls> bool _ParseFromIstream(cls &self, std::istream *input) {
  EXT_ENFORCE(input != nullptr, "ParseFromIstream: input stream pointer must not be null.");
  // For seekable streams, determine the size up front so the buffer is
  // allocated exactly once.  For non-seekable streams (pipes, network sockets,
  // etc.) fall back to the iterator-based read.
  std::string buffer;
  bool seekable_read_done = false;
  const std::streampos start = input->tellg();
  if (start != std::streampos(-1)) {
    if (input->seekg(0, std::ios::end)) {
      const std::streampos end_pos = input->tellg();
      input->seekg(start);
      if (end_pos != std::streampos(-1) && end_pos >= start) {
        const auto size = static_cast<std::streamsize>(end_pos - start);
        buffer.resize(static_cast<size_t>(size));
        input->read(buffer.data(), size);
        buffer.resize(static_cast<size_t>(input->gcount()));
        seekable_read_done = true;
      }
    }
  }
  if (!seekable_read_done) {
    buffer.assign(std::istreambuf_iterator<char>(*input), std::istreambuf_iterator<char>());
  }
  // Reaching EOF is expected after reading all bytes; only report failure for
  // genuine I/O errors (fail() without eof()).
  if (input->fail() && !input->eof()) {
    return false;
  }
  ParseOptions opts;
  const uint8_t *ptr = reinterpret_cast<const uint8_t *>(buffer.data());
  ONNX_LIGHT_NAMESPACE::utils::StringStream st(ptr, static_cast<int64_t>(buffer.size()));
  self.ParseFromStream(st, opts);
  return true;
}

template <typename cls> bool _ParseFromFileDescriptor(cls &self, int fd) {
  // Read the entire file descriptor into memory, then parse.
  std::string buffer;
  constexpr size_t kChunkSize = 4096;
  char chunk[kChunkSize];
  for (;;) {
    auto n = ::read(fd, chunk, kChunkSize);
    if (n < 0)
      return false;
    if (n == 0)
      break;
    buffer.append(chunk, static_cast<size_t>(n));
  }
  return self.ParseFromString(buffer);
}

template <typename cls> bool _SerializeToString(cls &self, std::string &out) {
  SerializeOptions opts;
  return self.SerializeToString(out, opts);
}

template <typename cls>
bool _SerializeToString(cls &self, std::string &out, SerializeOptions &opts) {
  if constexpr (std::is_same_v<std::remove_cv_t<cls>, ModelProto>) {
    if (opts.raw_data_callback || opts.node_callback) {
      // Apply the callbacks in place and restore the model once the bytes are produced, avoiding
      // a full-model copy. Serialization is logically const: the restorer puts every tensor/node a
      // callback touched back to its original state before returning, so the caller's model is
      // observably unchanged despite the transient in-place edits.
      ModelProto &mutable_self = const_cast<ModelProto &>(self);
      SerializeCallbackRestorer restorer = ApplySerializeRawDataCallback(mutable_self, opts);
      SerializeOptions local_opts = opts;
      local_opts.raw_data_callback = {};
      local_opts.node_callback = {};
      return _SerializeToString(self, out, local_opts);
    }
  }
  EXT_ENFORCE(opts.format == SerializeFormat::kOnnx,
              "SerializeToString: SerializeFormat::kOrtFlatbuffers is not implemented yet. "
              "Use SerializeFormat::kOnnx for now.");
  ONNX_LIGHT_NAMESPACE::utils::StringWriteStream size_buf;
  // Two-pass approach: compute the total serialized size first so we can
  // resize the output string exactly once, then write directly into it via
  // BorrowedStringWriteStream — eliminating any intermediate copy.
  // The size pass also populates the stream's size cache (via size_with_cache
  // in size_field/size_optional_proto_field/size_repeated_field) so the
  // write pass reuses cached sub-message sizes without recomputing them.
  SerializeSizeResult total_size = self.SerializeSize(size_buf, opts);
  if (!EnforceMaxSerializedSize(total_size, opts, "SerializeToString")) {
    out.clear();
    return false;
  }
  out.resize(static_cast<size_t>(total_size.size()));
  ONNX_LIGHT_NAMESPACE::utils::BorrowedStringWriteStream buf(
      reinterpret_cast<uint8_t *>(out.data()), total_size.size());
  size_buf.swap_size_cache(buf);
  if (opts.is_parallel()) {
    buf.StartThreadPool(opts.num_threads);
  }
  self.SerializeToStream(buf, opts);
  if (buf.HasParallelizationStarted()) {
    buf.WaitForDelayedBlock();
  }
  return true;
}

template <typename cls> bool _SerializeToFileDescriptor(cls &self, int fd) {
  SerializeOptions opts;
  return self.SerializeToFileDescriptor(fd, opts);
}

template <typename cls> bool _SerializeToFileDescriptor(cls &self, int fd, SerializeOptions &opts) {
  if constexpr (std::is_same_v<std::remove_cv_t<cls>, ModelProto>) {
    if (opts.raw_data_callback || opts.node_callback) {
      // Apply the callbacks in place and restore the model once the bytes are produced, avoiding
      // a full-model copy. See _SerializeToString for why the const_cast is safe here.
      ModelProto &mutable_self = const_cast<ModelProto &>(self);
      SerializeCallbackRestorer restorer = ApplySerializeRawDataCallback(mutable_self, opts);
      SerializeOptions local_opts = opts;
      local_opts.raw_data_callback = {};
      local_opts.node_callback = {};
      return _SerializeToFileDescriptor(self, fd, local_opts);
    }
  }
  EXT_ENFORCE(opts.format == SerializeFormat::kOnnx,
              "SerializeToFileDescriptor: SerializeFormat::kOrtFlatbuffers is not implemented "
              "yet. Use SerializeFormat::kOnnx for now.");
  SerializeOptions local_opts = opts;
  local_opts.num_threads = 1;
  ONNX_LIGHT_NAMESPACE::utils::StringWriteStream size_buf;
  SerializeSizeResult total_size = self.SerializeSize(size_buf, local_opts);
  if (!EnforceMaxSerializedSize(total_size, local_opts, "SerializeToFileDescriptor")) {
    return false;
  }
  ONNX_LIGHT_NAMESPACE::utils::FdWriteStream stream(fd);
  size_buf.swap_size_cache(stream);
  self.SerializeToStream(stream, local_opts);
  return stream.Flush();
}

template <typename cls> std::string _SerializeAsString(const cls &self) {
  std::string out;
  SerializeOptions opts;
  _SerializeToString(self, out, opts);
  return out;
}

template <typename cls> bool _SerializeToArray(const cls &self, void *data, int size) {
  EXT_ENFORCE(data != nullptr, "SerializeToArray: data pointer must not be null.");
  EXT_ENFORCE(size >= 0, "SerializeToArray: size must be non-negative.");
  std::string out;
  SerializeOptions opts;
  if (!_SerializeToString(self, out, opts)) {
    return false;
  }
  EXT_ENFORCE(static_cast<size_t>(size) >= out.size(), "SerializeToArray: buffer too small (need ",
              out.size(), " bytes, got ", size, ").");
  std::memcpy(data, out.data(), out.size());
  return true;
}

template <typename cls> bool _SerializeToOstream(const cls &self, std::ostream *output) {
  EXT_ENFORCE(output != nullptr, "SerializeToOstream: output stream pointer must not be null.");
  std::string out;
  SerializeOptions opts;
  if (!_SerializeToString(self, out, opts)) {
    return false;
  }
  output->write(out.data(), static_cast<std::streamsize>(out.size()));
  return output->good();
}

} // namespace ONNX_LIGHT_NAMESPACE
