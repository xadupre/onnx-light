#pragma once

#include <cstring>
#include <istream>
#include <iterator>
#include <ostream>
#include <string>
#include <type_traits>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace ONNX_LIGHT_NAMESPACE {

template <typename Derived> void ProtoMessageAdapter<Derived>::CopyFrom(const Derived &proto) {
  utils::StringWriteStream stream;
  SerializeOptions opts;
  SerializeSizeResult total_size = proto.SerializeSize(stream, opts);
  stream.pre_allocate(total_size.size());
  proto.SerializeToStream(stream, opts);
  utils::StringStream read_stream(stream.data(), stream.size());
  ParseOptions parse_options;
  derived().ParseFromStream(read_stream, parse_options);
}

template <typename Derived>
SerializeSizeResult ProtoMessageAdapter<Derived>::SerializeSize() const {
  SerializeOptions opts;
  utils::StringWriteStream stream;
  return derived().SerializeSize(stream, opts);
}

template <typename Derived> size_t ProtoMessageAdapter<Derived>::ByteSizeLong() const {
  return static_cast<size_t>(SerializeSize().size());
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::ParseFromString(const std::string &raw) {
  ParseOptions opts;
  return ParseFromString(raw, opts);
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::ParseFromString(const std::string &raw, ParseOptions &opts) {
  if (opts.format == SerializeFormat::kOrtFlatbuffers) {
    EXT_ENFORCE(opts.max_recursion_depth > 0,
                "ParseFromString: ParseOptions::max_recursion_depth must be > 0 (got ",
                opts.max_recursion_depth, ").");
    EXT_ENFORCE(opts.max_tensor_size_bytes >= 0,
                "ParseFromString: ParseOptions::max_tensor_size_bytes must be >= 0 (got ",
                opts.max_tensor_size_bytes, ").");
    EXT_THROW("ParseFromString: SerializeFormat::kOrtFlatbuffers is not implemented yet. "
              "Use SerializeFormat::kOnnx for now.");
  }
  EXT_ENFORCE(opts.format == SerializeFormat::kOnnx,
              "ParseFromString: unrecognised SerializeFormat value ", static_cast<int>(opts.format),
              "; only kOnnx is currently supported.");
  try {
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(raw.data());
    utils::StringStream stream(ptr, raw.size());
    if (opts.is_parallel())
      stream.StartThreadPool(opts.num_threads);
    derived().ParseFromStream(stream, opts);
    if (opts.is_parallel())
      stream.WaitForDelayedBlock();
  } catch (const onnx_light_helpers::ParseLimitExceeded &) {
    throw;
  } catch (const std::exception &) {
    return false;
  }
  return true;
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::ParseFromArray(const void *data, int size) {
  EXT_ENFORCE(data != nullptr || size == 0, "ParseFromArray: data pointer must not be null.");
  EXT_ENFORCE(size >= 0, "ParseFromArray: size must be non-negative.");
  return ParseFromString(std::string(static_cast<const char *>(data), static_cast<size_t>(size)));
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::ParseFromZeroCopyStream(utils::BinaryStream *stream) {
  ParseOptions opts;
  return ParseFromZeroCopyStream(stream, opts);
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::ParseFromZeroCopyStream(utils::BinaryStream *stream,
                                                           ParseOptions &opts) {
  EXT_ENFORCE(stream != nullptr, "ParseFromZeroCopyStream: stream pointer must not be null.");
  if (opts.format == SerializeFormat::kOrtFlatbuffers) {
    EXT_ENFORCE(opts.max_recursion_depth > 0,
                "ParseFromZeroCopyStream: ParseOptions::max_recursion_depth must be > 0 (got ",
                opts.max_recursion_depth, ").");
    EXT_ENFORCE(opts.max_tensor_size_bytes >= 0,
                "ParseFromZeroCopyStream: ParseOptions::max_tensor_size_bytes must be >= 0 (got ",
                opts.max_tensor_size_bytes, ").");
    EXT_THROW("ParseFromZeroCopyStream: SerializeFormat::kOrtFlatbuffers is not implemented yet. "
              "Use SerializeFormat::kOnnx for now.");
  }
  EXT_ENFORCE(opts.format == SerializeFormat::kOnnx,
              "ParseFromZeroCopyStream: unrecognised SerializeFormat value ",
              static_cast<int>(opts.format), "; only kOnnx is currently supported.");
  if (opts.is_parallel())
    stream->StartThreadPool(opts.num_threads);
  derived().ParseFromStream(*stream, opts);
  if (opts.is_parallel())
    stream->WaitForDelayedBlock();
  return true;
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::ParseFromIstream(std::istream *input) {
  EXT_ENFORCE(input != nullptr, "ParseFromIstream: input stream pointer must not be null.");
  std::string buffer;
  bool seekable_read_done = false;
  const std::streampos start = input->tellg();
  if (start != std::streampos(-1) && input->seekg(0, std::ios::end)) {
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
  if (!seekable_read_done)
    buffer.assign(std::istreambuf_iterator<char>(*input), std::istreambuf_iterator<char>());
  if (input->fail() && !input->eof())
    return false;
  ParseOptions opts;
  const uint8_t *ptr = reinterpret_cast<const uint8_t *>(buffer.data());
  utils::StringStream stream(ptr, static_cast<int64_t>(buffer.size()));
  derived().ParseFromStream(stream, opts);
  return true;
}

template <typename Derived> bool ProtoMessageAdapter<Derived>::ParseFromFileDescriptor(int fd) {
  std::string buffer;
  constexpr size_t chunk_size = 4096;
  char chunk[chunk_size];
  for (;;) {
    auto count = ::read(fd, chunk, chunk_size);
    if (count < 0)
      return false;
    if (count == 0)
      break;
    buffer.append(chunk, static_cast<size_t>(count));
  }
  return ParseFromString(buffer);
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::SerializeToString(std::string &out) const {
  SerializeOptions opts;
  return SerializeToString(out, opts);
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::SerializeToString(std::string *out) const {
  EXT_ENFORCE(out != nullptr, "SerializeToString: output pointer must not be null.");
  return SerializeToString(*out);
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::SerializeToString(std::string &out,
                                                     SerializeOptions &opts) const {
  if constexpr (std::is_same_v<Derived, ModelProto>) {
    if (opts.raw_data_callback || opts.node_callback) {
      ModelProto &mutable_model = const_cast<ModelProto &>(derived());
      SerializeCallbackRestorer restorer = ApplySerializeRawDataCallback(mutable_model, opts);
      SerializeOptions local_opts = opts;
      local_opts.raw_data_callback = {};
      local_opts.node_callback = {};
      return SerializeToString(out, local_opts);
    }
  }
  EXT_ENFORCE(opts.format == SerializeFormat::kOnnx,
              "SerializeToString: SerializeFormat::kOrtFlatbuffers is not implemented yet. "
              "Use SerializeFormat::kOnnx for now.");
  utils::StringWriteStream size_stream;
  SerializeSizeResult total_size = derived().SerializeSize(size_stream, opts);
  if (!EnforceMaxSerializedSize(total_size, opts, "SerializeToString")) {
    out.clear();
    return false;
  }
  out.resize(static_cast<size_t>(total_size.size()));
  utils::BorrowedStringWriteStream stream(reinterpret_cast<uint8_t *>(out.data()),
                                          total_size.size());
  size_stream.swap_size_cache(stream);
  if (opts.is_parallel())
    stream.StartThreadPool(opts.num_threads);
  derived().SerializeToStream(stream, opts);
  if (stream.HasParallelizationStarted())
    stream.WaitForDelayedBlock();
  return true;
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::SerializeToFileDescriptor(int fd) const {
  SerializeOptions opts;
  return SerializeToFileDescriptor(fd, opts);
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::SerializeToFileDescriptor(int fd, SerializeOptions &opts) const {
  if constexpr (std::is_same_v<Derived, ModelProto>) {
    if (opts.raw_data_callback || opts.node_callback) {
      ModelProto &mutable_model = const_cast<ModelProto &>(derived());
      SerializeCallbackRestorer restorer = ApplySerializeRawDataCallback(mutable_model, opts);
      SerializeOptions local_opts = opts;
      local_opts.raw_data_callback = {};
      local_opts.node_callback = {};
      return SerializeToFileDescriptor(fd, local_opts);
    }
  }
  EXT_ENFORCE(opts.format == SerializeFormat::kOnnx,
              "SerializeToFileDescriptor: SerializeFormat::kOrtFlatbuffers is not implemented "
              "yet. Use SerializeFormat::kOnnx for now.");
  SerializeOptions local_opts = opts;
  local_opts.num_threads = 1;
  utils::StringWriteStream size_stream;
  SerializeSizeResult total_size = derived().SerializeSize(size_stream, local_opts);
  if (!EnforceMaxSerializedSize(total_size, local_opts, "SerializeToFileDescriptor"))
    return false;
  utils::FdWriteStream stream(fd);
  size_stream.swap_size_cache(stream);
  derived().SerializeToStream(stream, local_opts);
  return stream.Flush();
}

template <typename Derived> bool ProtoMessageAdapter<Derived>::SaveToFileDescriptor(int fd) const {
  return SerializeToFileDescriptor(fd);
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::SaveToFileDescriptor(int fd, SerializeOptions &opts) const {
  return SerializeToFileDescriptor(fd, opts);
}

template <typename Derived> std::string ProtoMessageAdapter<Derived>::SerializeAsString() const {
  std::string out;
  SerializeToString(out);
  return out;
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::SerializeToArray(void *data, int size) const {
  EXT_ENFORCE(data != nullptr, "SerializeToArray: data pointer must not be null.");
  EXT_ENFORCE(size >= 0, "SerializeToArray: size must be non-negative.");
  std::string out;
  if (!SerializeToString(out))
    return false;
  EXT_ENFORCE(static_cast<size_t>(size) >= out.size(), "SerializeToArray: buffer too small (need ",
              out.size(), " bytes, got ", size, ").");
  std::memcpy(data, out.data(), out.size());
  return true;
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::SerializeToOstream(std::ostream *output) const {
  EXT_ENFORCE(output != nullptr, "SerializeToOstream: output stream pointer must not be null.");
  std::string out;
  if (!SerializeToString(out))
    return false;
  output->write(out.data(), static_cast<std::streamsize>(out.size()));
  return output->good();
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::SerializeToOStream(std::ostream *output) const {
  return SerializeToOstream(output);
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::SerializeToZeroCopyStream(
    utils::BinaryWriteStream *output) const {
  EXT_ENFORCE(output != nullptr, "SerializeToZeroCopyStream: output pointer must not be null.");
  std::string buffer;
  if (!SerializeToString(buffer))
    return false;
  output->write_raw_bytes(reinterpret_cast<const uint8_t *>(buffer.data()), buffer.size());
  return true;
}

template <typename Derived>
bool ProtoMessageAdapter<Derived>::ParseFromStream(utils::BinaryStream &stream) {
  ParseOptions opts;
  return derived().ParseFromStream(stream, opts);
}

template <typename Derived>
void ProtoMessageAdapter<Derived>::SerializeToStream(utils::BinaryWriteStream &stream) const {
  SerializeOptions opts;
  derived().SerializeToStream(stream, opts);
}

} // namespace ONNX_LIGHT_NAMESPACE
