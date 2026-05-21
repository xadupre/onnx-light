#include "stream.h"
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#else
#include <malloc.h>
#define NOMINMAX
#include <windows.h>
#endif
#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace utils {

namespace {

std::filesystem::path normalized_model_parent(const std::string &model_path) {
  std::filesystem::path parent = std::filesystem::path(model_path).parent_path();
  if (parent.empty()) {
    return std::filesystem::path(".");
  }
  return parent.lexically_normal();
}

std::string validate_weights_file_is_next_to_model(const std::string &model_path,
                                                   const std::string &weights_file) {
  std::filesystem::path weights_path(weights_file);
  EXT_ENFORCE(!weights_path.empty(), "External weights file cannot be empty.");

  std::filesystem::path normalized_parent = normalized_model_parent(model_path);
  std::filesystem::path normalized_weights = weights_path.lexically_normal();
  std::filesystem::path weights_parent = normalized_weights.parent_path();
  if (weights_parent.empty() || weights_parent == std::filesystem::path(".")) {
    weights_parent = normalized_parent;
  }
  const bool same_dir = (weights_parent == normalized_parent);
  EXT_ENFORCE(same_dir, "External weights file must be next to model file. model=", model_path,
              ", weights=", weights_file);
  EXT_ENFORCE(!normalized_weights.filename().empty(),
              "External weights file must include a filename. model=", model_path,
              ", weights=", weights_file);

  return (weights_parent / normalized_weights.filename()).string();
}

std::filesystem::path validate_external_location_is_next_to_model(const std::string &model_path,
                                                                  const std::string &location) {
  std::filesystem::path location_path(location);
  EXT_ENFORCE(!location_path.empty(), "External data location cannot be empty.");
  EXT_ENFORCE(
      !location_path.is_absolute(),
      "External data location must be a file name next to the model file. location=", location);

  std::filesystem::path normalized_location = location_path.lexically_normal();
  const bool same_dir = normalized_location.parent_path().empty() ||
                        normalized_location.parent_path() == std::filesystem::path(".");
  EXT_ENFORCE(
      same_dir,
      "External data location must be a file name next to the model file. location=", location);
  EXT_ENFORCE(!normalized_location.filename().empty(),
              "External data location must include a filename. location=", location);

  return normalized_model_parent(model_path) / normalized_location.filename();
}

std::shared_ptr<uint8_t> make_shared_file_buffer(size_t size, size_t alignment) {
  if (size == 0) {
    return std::shared_ptr<uint8_t>();
  }
  if (alignment <= 1) {
    return std::shared_ptr<uint8_t>(new uint8_t[size], std::default_delete<uint8_t[]>());
  }
#if !defined(_WIN32)
  void *raw = nullptr;
  const int err = posix_memalign(&raw, alignment, size);
  EXT_ENFORCE(err == 0 && raw != nullptr, "posix_memalign failed for size=", size,
              ", alignment=", alignment, ", err=", err);
  return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t *>(raw),
                                  [](uint8_t *ptr) { std::free(ptr); });
#else
  void *raw = _aligned_malloc(size, alignment);
  EXT_ENFORCE(raw != nullptr, "_aligned_malloc failed for size=", size, ", alignment=", alignment);
  return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t *>(raw),
                                  [](uint8_t *ptr) { _aligned_free(ptr); });
#endif
}

void read_file_into_buffer_in_chunks(const std::string &file_path, uint8_t *buffer, int64_t size) {
  std::ifstream stream(file_path, std::ios::binary);
  EXT_ENFORCE(stream.is_open(), "Unable to open external weights file: ", file_path);
  // Adaptive chunk size: small files are read in a single call, medium files use
  // 16 MB chunks, and very large files use 64 MB chunks. This keeps individual
  // reads comfortably below large std::streamsize limits while issuing only a
  // small number of I/O calls for multi-GB weight files, and avoids unnecessary
  // chunking overhead for smaller files.
  constexpr int64_t kSmallFileThreshold = 10LL * 1024 * 1024;   // 10 MB
  constexpr int64_t kMediumFileThreshold = 100LL * 1024 * 1024; // 100 MB
  constexpr std::streamsize kMediumFileChunkSize = 16LL * 1024 * 1024;
  constexpr std::streamsize kLargeFileChunkSize = 64LL * 1024 * 1024;
  const std::streamsize chunk_size =
      size < kSmallFileThreshold
          ? static_cast<std::streamsize>(size)
          : (size < kMediumFileThreshold ? kMediumFileChunkSize : kLargeFileChunkSize);
  int64_t done = 0;
  while (done < size) {
    const int64_t remaining = size - done;
    const std::streamsize chunk =
        static_cast<std::streamsize>(std::min<int64_t>(remaining, chunk_size));
    stream.read(reinterpret_cast<char *>(buffer + done), chunk);
    const std::streamsize got = stream.gcount();
    EXT_ENFORCE(got == chunk, "Unable to read external weights file fully: ", file_path,
                ", expected chunk=", chunk, ", got=", got, ", offset=", done);
    done += static_cast<int64_t>(got);
  }
}

// Maps an entire file into read-only virtual memory and returns a shared_ptr<uint8_t>
// whose deleter unmaps the region.  On POSIX, mmap(MAP_PRIVATE|PROT_READ) is used;
// on Windows, CreateFileMapping + MapViewOfFile.
// Returns an empty shared_ptr when file_size == 0.
// The mapped base address is always page-aligned, which satisfies any typical tensor
// alignment requirement (16 / 32 / 64 bytes) when combined with an aligned file offset.
std::shared_ptr<uint8_t> mmap_file_as_shared_ptr(const std::string &file_path, int64_t file_size) {
  if (file_size <= 0) {
    return std::shared_ptr<uint8_t>();
  }
  const size_t sz = static_cast<size_t>(file_size);
#if !defined(_WIN32)
  const int fd = ::open(file_path.c_str(), O_RDONLY);
  EXT_ENFORCE(fd >= 0, "mmap_file: Unable to open file: ", file_path);
  void *mapped = ::mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  EXT_ENFORCE(mapped != MAP_FAILED, "mmap_file: mmap failed for file: ", file_path);
  return std::shared_ptr<uint8_t>(static_cast<uint8_t *>(mapped),
                                  [sz](uint8_t *ptr) { ::munmap(ptr, sz); });
#else
  HANDLE hFile = ::CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  EXT_ENFORCE(hFile != INVALID_HANDLE_VALUE, "mmap_file: Unable to open file: ", file_path);
  HANDLE hMapping = ::CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
  ::CloseHandle(hFile);
  EXT_ENFORCE(hMapping != nullptr, "mmap_file: CreateFileMapping failed for file: ", file_path);
  void *mapped = ::MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
  ::CloseHandle(hMapping);
  EXT_ENFORCE(mapped != nullptr, "mmap_file: MapViewOfFile failed for file: ", file_path);
  return std::shared_ptr<uint8_t>(static_cast<uint8_t *>(mapped),
                                  [](uint8_t *ptr) { ::UnmapViewOfFile(ptr); });
#endif
}

} // namespace

#if !defined(_WIN32)
namespace {

// Reads a delayed block from a shared file descriptor using positional reads.
// It retries on EINTR and enforces full reads to avoid truncated tensor payloads.
void ReadBlockFromFd(int fd, const DelayedBlock &block, const char *context) {
  size_t done = 0;
  while (done < block.size) {
    ssize_t bytes_read =
        pread(fd, block.data + done, block.size - done, static_cast<off_t>(block.offset + done));
    if (bytes_read < 0) {
      if (errno == EINTR) {
        continue;
      }
      const int err = errno;
      EXT_THROW(context, " failed to read delayed block at offset=", block.offset, ", errno=", err,
                " (", strerror(err), ")");
    }
    EXT_ENFORCE(bytes_read > 0, context,
                " reached end of file while reading delayed block at offset=", block.offset,
                ", expected=", block.size, ", read=", done);
    done += static_cast<size_t>(bytes_read);
  }
}

} // namespace
#endif

///////////////
// BinaryStream
///////////////

std::string FieldNumber::string() const {
  return onnx_light_helpers::MakeString("[field_number=", field_number, ", wire_type=", wire_type,
                                        "]");
}

void BinaryStream::_check() {
  EXT_ENFORCE(limits_.empty(), "BinaryStream destructor called with non-empty limits stack.");
}

BinaryStream::~BinaryStream() { _check(); }

RefString BinaryStream::next_string() {
  // Depending on the stream implementation, the string may be disappear after reading another item.
  uint64_t length = next_uint64();
  this->CanRead(length, "[BinaryStream::next_string]");
  return RefString(reinterpret_cast<const char *>(read_bytes(length)), static_cast<size_t>(length));
}

int64_t BinaryStream::next_int64() {
  uint64_t value = next_uint64();
  // return decodeZigZag64(value);
  return static_cast<int64_t>(value);
}

int32_t BinaryStream::next_int32() {
  uint64_t value = next_uint64();
  // return decodeZigZag64(value);
  return static_cast<int32_t>(value);
}

float BinaryStream::next_float() {
  float value;
  read_bytes(sizeof(float), reinterpret_cast<uint8_t *>(&value));
  return value;
}

double BinaryStream::next_double() {
  double value;
  read_bytes(sizeof(double), reinterpret_cast<uint8_t *>(&value));
  return value;
}

FieldNumber BinaryStream::next_field() {
  FieldNumber n;
  n.wire_type = next_uint64();
  n.field_number = n.wire_type >> 3;
  n.wire_type = n.wire_type & 0x07;
  return n;
}

void BinaryStream::ReadDelayedBlock(DelayedBlock &) {
  EXT_THROW("ReadDelayedBlock is not implemented for this stream.");
}

void BinaryStream::WaitForDelayedBlock() {
  EXT_THROW("WaitForDelayedBlock is not implemented for this stream.");
}

void BinaryStream::StartThreadPool(size_t) {
  EXT_THROW("StartThreadPool is not implemented for this stream.");
}

void BinaryStream::LimitToNext(uint64_t length) {
  CanRead(length, "Too many bytes requested in LimitToNext.");
  limits_.push_back(size());
  LimitTo(tell() + length);
}

void BinaryStream::Restore() {
  EXT_ENFORCE(!limits_.empty(), "Cannot restore, no limits set");
  uint64_t last_limit = limits_.back();
  LimitTo(last_limit);
  limits_.pop_back();
}

///////////////
// StringStream
///////////////

void StringStream::Setup(const uint8_t *data, int64_t size) {
  EXT_ENFORCE(!thread_pool_.IsStarted(), "ThreadPool is still running.");
  pos_ = 0;
  size_ = size;
  data_ = data;
  thread_pool_.Clear();
  blocks_.clear();
}

void StringStream::CanRead(uint64_t len, const char *msg) {
  EXT_ENFORCE(pos_ + static_cast<int64_t>(len) <= size_, msg, " unable to read ", len,
              " bytes, pos_=", pos_, ", size_=", size_);
}

const uint8_t *StringStream::read_bytes(offset_t n_bytes, uint8_t *pre_allocated_buffer) {
  if (pre_allocated_buffer != nullptr) {
    memcpy(pre_allocated_buffer, data_ + pos_, n_bytes);
    pos_ += n_bytes;
    return pre_allocated_buffer;
  } else {
    const uint8_t *res = data_ + pos_;
    pos_ += n_bytes;
    return res;
  }
}

void StringStream::skip_bytes(offset_t n_bytes) { pos_ += n_bytes; }

uint64_t StringStream::next_uint64() {
  // Fast path: single-byte varint (covers field numbers 1-15 and other small values).
  if (pos_ < size_ && (data_[pos_] & 0x80) == 0) {
    return static_cast<uint64_t>(data_[pos_++]);
  }

  uint64_t result = 0;
  int shift = 0;

  for (int i = 0; i < 10 && pos_ < size_; ++i) {
    uint8_t byte = data_[pos_++];
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;

    if ((byte & 0x80) == 0)
      return result;

    shift += 7;
  }
  EXT_THROW("[StringStream::next_uint64] unable to read an uint64 at pos=", pos_, ", size=", size_);
}

std::string StringStream::tell_around() const {
  offset_t begin = pos_;
  offset_t end =
      pos_ + 10 < static_cast<offset_t>(size()) ? pos_ + 10 : static_cast<offset_t>(size());
  RefString ref(reinterpret_cast<const char *>(data_) + begin, end - begin);
  return ref.as_string();
}

void StringStream::LimitTo(uint64_t len) {
  EXT_ENFORCE(limits_.size() > 0, "No limit was stored.");
  size_ = len;
}

void StringStream::ReadDelayedBlock(DelayedBlock &block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot read delayed block.");
  EXT_ENFORCE(block.stream_id == 0,
              "Only one stream is allowed to read delayed blocks, but stream_id=", block.stream_id);
  blocks_.push_back(block);
  thread_pool_.SubmitTask(
      [this, block]() { memcpy(block.data, this->data_ + block.offset, block.size); });
  pos_ += block.size;
}

void StringStream::WaitForDelayedBlock() { thread_pool_.Wait(); }

void StringStream::StartThreadPool(size_t n_threads) { thread_pool_.Start(n_threads); }

////////////////////
// BinaryWriteStream
////////////////////

void BinaryWriteStream::write_variant_uint64(uint64_t value) {
  // A 64-bit varint encodes at most 10 bytes (ceil(64/7)).
  uint8_t buf[10];
  int len = 0;
  while (value > 127) {
    buf[len++] = static_cast<uint8_t>((value & 0x7F) | 0x80);
    value >>= 7;
  }
  buf[len++] = static_cast<uint8_t>(value);
  write_raw_bytes(buf, len);
}

uint64_t BinaryWriteStream::size_variant_uint64(uint64_t value) { return VarintSize(value); }

void BinaryWriteStream::write_int64(int64_t value) {
  write_variant_uint64(static_cast<uint64_t>(value));
}

uint64_t BinaryWriteStream::size_int64(int64_t value) {
  return VarintSize(static_cast<uint64_t>(value));
}

void BinaryWriteStream::write_int32(int32_t value) {
  write_variant_uint64(static_cast<uint64_t>(value));
}

uint64_t BinaryWriteStream::size_int32(int32_t value) {
  return VarintSize(static_cast<uint64_t>(value));
}

void BinaryWriteStream::write_float(float value) {
  write_raw_bytes(reinterpret_cast<uint8_t *>(&value), sizeof(float));
}

uint64_t BinaryWriteStream::size_float(float) { return sizeof(float); }

void BinaryWriteStream::write_double(double value) {
  write_raw_bytes(reinterpret_cast<uint8_t *>(&value), sizeof(double));
}

uint64_t BinaryWriteStream::size_double(double) { return sizeof(double); }

void BinaryWriteStream::write_field_header(uint32_t field_number, uint8_t wire_type) {
  write_variant_uint64((field_number << 3) | wire_type);
}

uint64_t BinaryWriteStream::VarintSize(uint64_t value) {
  size_t size = 0;
  do {
    size++;
    value >>= 7;
  } while (value != 0);
  return size;
}

uint64_t BinaryWriteStream::size_field_header(uint32_t field_number, uint8_t wire_type) {
  return VarintSize((field_number << 3) | wire_type);
}

void BinaryWriteStream::write_string(const std::string &value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

uint64_t BinaryWriteStream::size_string(const std::string &value) {
  return VarintSize(value.size()) + value.size();
}

void BinaryWriteStream::write_string(const String &value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

uint64_t BinaryWriteStream::size_string(const String &value) {
  return VarintSize(value.size()) + value.size();
}

void BinaryWriteStream::write_string(const RefString &value) {
  write_variant_uint64(value.size());
  write_raw_bytes(reinterpret_cast<const uint8_t *>(value.data()), value.size());
}

uint64_t BinaryWriteStream::size_string(const RefString &value) {
  return VarintSize(value.size()) + value.size();
}

void BinaryWriteStream::write_string_stream(const StringWriteStream &stream) {
  write_variant_uint64(stream.size());
  write_raw_bytes(stream.data(), stream.size());
}

uint64_t BinaryWriteStream::size_string_stream(const StringWriteStream &stream) {
  return VarintSize(stream.size()) + stream.size();
}

void BinaryWriteStream::write_string_stream(const BorrowedWriteStream &stream) {
  write_variant_uint64(stream.size());
  write_raw_bytes(stream.data(), stream.size());
}

uint64_t BinaryWriteStream::size_string_stream(const BorrowedWriteStream &stream) {
  return VarintSize(stream.size()) + stream.size();
}

void BinaryWriteStream::CacheSize(const void *ptr, SerializeSizeResult size) {
  size_cache_[ptr] = size;
}

bool BinaryWriteStream::GetCachedSize(const void *ptr, SerializeSizeResult &size) {
  auto it = size_cache_.find(ptr);
  if (it != size_cache_.end()) {
    size = it->second;
    return true;
  }
  return false;
}

void BinaryWriteStream::StartThreadPool(size_t) {}

void BinaryWriteStream::WriteDelayedBlock(DelayedWriteBlock &) {
  EXT_THROW("WriteDelayedBlock is not implemented for this stream.");
}

void BinaryWriteStream::WaitForDelayedBlock() {}

////////////////////
// StringWriteStream
////////////////////

// Note: The serialization thread exclusively calls write_raw_bytes,
// WriteDelayedBlock, and all write_pos_ updates.  Worker threads only
// execute the memcpy inside the submitted tasks and never touch write_pos_
// or resize the buffer.

void StringWriteStream::write_raw_bytes(const uint8_t *ptr, offset_t n_bytes) {
  if (write_pos_ + n_bytes > static_cast<offset_t>(buffer_.size())) {
    buffer_.resize(static_cast<size_t>(write_pos_ + n_bytes));
  }
  std::memcpy(buffer_.data() + write_pos_, ptr, static_cast<size_t>(n_bytes));
  write_pos_ += n_bytes;
}

int64_t StringWriteStream::size() const { return write_pos_; }
const uint8_t *StringWriteStream::data() const { return buffer_.data(); }

void StringWriteStream::pre_allocate(int64_t total_bytes) {
  buffer_.assign(static_cast<size_t>(total_bytes), 0);
  write_pos_ = 0;
}

void StringWriteStream::StartThreadPool(size_t n_threads) { thread_pool_.Start(n_threads); }

void StringWriteStream::WriteDelayedBlock(DelayedWriteBlock &block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot write delayed block.");
  if (block.offset == -1) {
    block.offset = static_cast<offset_t>(write_pos_);
  }
  EXT_ENFORCE(block.offset == static_cast<offset_t>(write_pos_),
              "Only append-mode delayed writes are supported but block.offset=", block.offset,
              " and write_pos_=", write_pos_);
  // The buffer must have been pre-allocated (via pre_allocate) before parallel
  // writes begin so that buffer_.data() remains stable during concurrent tasks.
  EXT_ENFORCE(block.offset + static_cast<int64_t>(block.size) <=
                  static_cast<int64_t>(buffer_.size()),
              "Buffer not pre-allocated: delayed write at offset=", block.offset,
              " size=", block.size, " exceeds buffer size=", buffer_.size());
  write_pos_ += static_cast<offset_t>(block.size);
  uint8_t *dest = reinterpret_cast<uint8_t *>(buffer_.data()) + block.offset;
  // block.data must remain valid (point to live proto data) until WaitForDelayedBlock() returns.
  // This is the caller's responsibility, identical to the FileWriteStream contract.
  thread_pool_.SubmitTask([dest, block]() { std::memcpy(dest, block.data, block.size); });
}

void StringWriteStream::WaitForDelayedBlock() { thread_pool_.Wait(); }

//////////////////////////
// BorrowedStringWriteStream
//////////////////////////

void BorrowedStringWriteStream::write_raw_bytes(const uint8_t *ptr, offset_t n_bytes) {
  EXT_ENFORCE(write_pos_ + n_bytes <= capacity_, "Buffer too small: write at offset=", write_pos_,
              " size=", n_bytes, " exceeds capacity=", capacity_);
  std::memcpy(data_ + write_pos_, ptr, static_cast<size_t>(n_bytes));
  write_pos_ += n_bytes;
}

void BorrowedStringWriteStream::WriteDelayedBlock(DelayedWriteBlock &block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot write delayed block.");
  if (block.offset == -1) {
    block.offset = static_cast<offset_t>(write_pos_);
  }
  EXT_ENFORCE(block.offset == static_cast<offset_t>(write_pos_),
              "Only append-mode delayed writes are supported but block.offset=", block.offset,
              " and write_pos_=", write_pos_);
  EXT_ENFORCE(block.offset + static_cast<int64_t>(block.size) <= capacity_,
              "Buffer too small: delayed write at offset=", block.offset, " size=", block.size,
              " exceeds capacity=", capacity_);
  write_pos_ += static_cast<offset_t>(block.size);
  uint8_t *dest = data_ + block.offset;
  thread_pool_.SubmitTask([dest, block]() { std::memcpy(dest, block.data, block.size); });
}

//////////////////////
// BorrowedWriteStream
//////////////////////

void BorrowedWriteStream::write_raw_bytes(const uint8_t *, offset_t) {
  EXT_THROW("This method cannot be called on this class (BorrowedWriteStream).");
}

//////////////////
// FileWriteStream
//////////////////

FileWriteStream::FileWriteStream(const std::string &file_path)
    : BinaryWriteStream(), file_path_(file_path), file_stream_(file_path, std::ios::binary) {
  written_bytes_ = 0;
}

void FileWriteStream::write_raw_bytes(const uint8_t *data, offset_t n_bytes) {
  file_stream_.write(reinterpret_cast<const char *>(data), n_bytes);
  written_bytes_ += static_cast<uint64_t>(n_bytes);
}

int64_t FileWriteStream::size() const { return static_cast<int64_t>(written_bytes_); }

const uint8_t *FileWriteStream::data() const {
  EXT_THROW("This method cannot be called on this class (FileWriteStream).");
}

void FileWriteStream::StartThreadPool(size_t n_threads) { thread_pool_.Start(n_threads); }

void FileWriteStream::WriteDelayedBlock(DelayedWriteBlock &block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot write delayed block.");
  EXT_ENFORCE(
      block.stream_id == 0,
      "Only one stream is allowed to write delayed blocks, but stream_id=", block.stream_id);
  if (block.offset == -1) {
    block.offset = static_cast<offset_t>(written_bytes_);
  }
  EXT_ENFORCE(block.offset == static_cast<offset_t>(written_bytes_),
              "Only append-mode delayed writes are supported but block.offset=", block.offset,
              " and written_bytes_=", written_bytes_);
  file_stream_.seekp(static_cast<std::streamoff>(block.size), std::ios::cur);
  written_bytes_ += static_cast<uint64_t>(block.size);
  std::string file_path = file_path_;
  thread_pool_.SubmitTask([file_path, block]() {
    std::fstream file_stream(file_path, std::ios::in | std::ios::out | std::ios::binary);
    file_stream.seekp(block.offset);
    file_stream.write(reinterpret_cast<const char *>(block.data), block.size);
  });
}

void FileWriteStream::WaitForDelayedBlock() { thread_pool_.Wait(); }

void FileWriteStream::pre_allocate(int64_t total_bytes) {
  EXT_ENFORCE(total_bytes > 0, "pre_allocate requires total_bytes > 0, got ", total_bytes);
  // Seek to the last byte and write a zero to establish the file size.
  // Flush immediately so the ofstream buffer is empty; otherwise the destructor
  // would re-flush this sentinel byte and overwrite the last byte written by a
  // parallel task that fills the file concurrently.
  file_stream_.seekp(total_bytes - 1);
  const uint8_t zero = 0;
  file_stream_.write(reinterpret_cast<const char *>(&zero), 1);
  file_stream_.flush();
  written_bytes_ = static_cast<uint64_t>(total_bytes);
}

/////////////
// FileStream
/////////////

FileStream::FileStream(const std::string &file_path)
    : lock_(false), file_path_(file_path), file_stream_(file_path, std::ios::binary) {
  if (!file_stream_.is_open()) {
    EXT_THROW("Unable to open file: ", file_path);
  }
#if !defined(_WIN32)
  file_descriptor_ = open(file_path.c_str(), O_RDONLY);
  if (file_descriptor_ < 0) {
    const int err = errno;
    EXT_THROW("Unable to open file descriptor for: ", file_path, ", errno=", err, " (",
              strerror(err), ")");
  }
#endif
  file_stream_.seekg(0, std::ios::end);
  std::streampos end = file_stream_.tellg();
  file_stream_.seekg(0);
  size_ = static_cast<offset_t>(end);
  read_buf_.resize(READ_BUF_SIZE);
}

bool FileStream::is_open() const { return file_stream_.is_open(); }

void FileStream::LimitTo(uint64_t len) {
  EXT_ENFORCE(limits_.size() > 0, "No limit was stored.");
  size_ = len;
}

void FileStream::CanRead(uint64_t len, const char *msg) {
  EXT_ENFORCE(static_cast<int64_t>(tell()) + static_cast<int64_t>(len) <= size_, msg,
              " unable to read ", len, " bytes, pos_=", tell(), ", size_=", size_);
}

void FileStream::_fill_read_buffer() {
  int64_t avail = size_ - tell();
  if (avail <= 0)
    return;
  int64_t to_read = std::min(static_cast<int64_t>(READ_BUF_SIZE), avail);
  file_stream_.read(reinterpret_cast<char *>(read_buf_.data()), to_read);
  read_buf_end_ = static_cast<size_t>(file_stream_.gcount());
  read_buf_pos_ = 0;
}

void FileStream::_invalidate_read_buffer() {
  if (read_buf_pos_ < read_buf_end_) {
    auto unread = static_cast<std::streamoff>(read_buf_end_ - read_buf_pos_);
    file_stream_.seekg(-unread, std::ios::cur);
    read_buf_pos_ = read_buf_end_ = 0;
  }
}

uint64_t FileStream::next_uint64() {
  // Fast path: single-byte varint already buffered (covers field numbers 1-15).
  if (read_buf_pos_ < read_buf_end_ && (read_buf_[read_buf_pos_] & 0x80) == 0) {
    return static_cast<uint64_t>(read_buf_[read_buf_pos_++]);
  }

  uint64_t result = 0;
  int shift = 0;

  for (int i = 0; i < 10; ++i) {
    if (read_buf_pos_ >= read_buf_end_) {
      _fill_read_buffer();
      EXT_ENFORCE(read_buf_pos_ < read_buf_end_,
                  "[FileStream::next_uint64] unable to read an int64 at pos=", tell(),
                  ", size=", size_);
    }
    uint8_t byte = read_buf_[read_buf_pos_++];
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;

    if ((byte & 0x80) == 0)
      return result;

    shift += 7;
  }
  EXT_THROW("[FileStream::next_uint64] unable to read an int64 at pos=", tell(), ", size=", size_);
}

const uint8_t *FileStream::read_bytes(offset_t n_bytes, uint8_t *pre_allocated_buffer) {
  if (pre_allocated_buffer) {
    // Drain the read-ahead buffer first, then pull any remaining bytes from file.
    size_t buffered = read_buf_end_ - read_buf_pos_;
    size_t from_buf =
        (buffered < static_cast<size_t>(n_bytes)) ? buffered : static_cast<size_t>(n_bytes);
    if (from_buf > 0) {
      memcpy(pre_allocated_buffer, read_buf_.data() + read_buf_pos_, from_buf);
      read_buf_pos_ += from_buf;
    }
    auto remaining = static_cast<offset_t>(n_bytes - from_buf);
    if (remaining > 0) {
      file_stream_.read(reinterpret_cast<char *>(pre_allocated_buffer + from_buf), remaining);
    }
    return pre_allocated_buffer;
  }
  // No pre_allocated_buffer: invalidate the read-ahead buffer so that
  // file_stream_.tellg() equals tell(), then read into the internal buffer_.
  _invalidate_read_buffer();
  if (n_bytes > static_cast<offset_t>(buffer_.size()))
    buffer_.resize(n_bytes);
  file_stream_.read(reinterpret_cast<char *>(buffer_.data()), n_bytes);
  return buffer_.data();
}

void FileStream::skip_bytes(offset_t n_bytes) {
  size_t buffered = read_buf_end_ - read_buf_pos_;
  if (static_cast<size_t>(n_bytes) <= buffered) {
    read_buf_pos_ += static_cast<size_t>(n_bytes);
  } else {
    auto remaining = static_cast<std::streamoff>(n_bytes - buffered);
    read_buf_pos_ = read_buf_end_ = 0;
    file_stream_.seekg(remaining, std::ios::cur);
  }
}

bool FileStream::NotEnd() const { return static_cast<int64_t>(tell()) < size_; }

offset_t FileStream::tell() const {
  return static_cast<offset_t>(const_cast<std::ifstream &>(file_stream_).tellg()) -
         static_cast<offset_t>(read_buf_end_ - read_buf_pos_);
}

std::string FileStream::tell_around() const {
  RefString ref(reinterpret_cast<const char *>(buffer_.data()),
                buffer_.size() < 10 ? buffer_.size() : 10);
  return ref.as_string();
}

FileStream::~FileStream() {
#if !defined(_WIN32)
  if (file_descriptor_ >= 0) {
    close(file_descriptor_);
  }
#endif
}

void FileStream::ReadDelayedBlock(DelayedBlock &block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot read delayed block.");
  EXT_ENFORCE(block.stream_id == 0,
              "Only one stream is allowed to read delayed blocks, but stream_id=", block.stream_id);
  blocks_.push_back(block);
  thread_pool_.SubmitTask([this, block]() {
#if !defined(_WIN32)
    ReadBlockFromFd(this->file_descriptor_, block, "[FileStream::ReadDelayedBlock]");
#else
    std::ifstream file_stream(this->file_path_, std::ios::binary);
    file_stream.seekg(block.offset);
    file_stream.read(reinterpret_cast<char *>(block.data), block.size);
#endif
  });
  // Advance the stream past the block data, draining the read-ahead buffer first.
  skip_bytes(block.size);
}

void FileStream::WaitForDelayedBlock() { thread_pool_.Wait(); }

void FileStream::StartThreadPool(size_t n_threads) { thread_pool_.Start(n_threads); }

//////////////////////
// TwoFilesWriteStream
//////////////////////

TwoFilesWriteStream::TwoFilesWriteStream(const std::string &file_path,
                                         const std::string &weights_file)
    : FileWriteStream(file_path),
      weights_stream_(validate_weights_file_is_next_to_model(file_path, weights_file)),
      active_weights_location_(
          std::filesystem::path(weights_stream_.file_path()).filename().string()),
      default_weights_location_(
          std::filesystem::path(weights_stream_.file_path()).filename().string()) {
  std::filesystem::path parent = normalized_model_parent(file_path);
  std::filesystem::path weights = std::filesystem::path(weights_stream_.file_path());
  std::filesystem::path rel = std::filesystem::relative(weights, parent);
  if (!rel.empty() && rel.string() != ".") {
    default_weights_location_ = rel.string();
  }
}

void TwoFilesWriteStream::set_active_weights_location(const std::string &location) {
  if (location.empty()) {
    active_weights_location_ = weights_stream_.file_path();
    return;
  }
  if (location == active_weights_location_) {
    return;
  }
  if (location == weights_stream_.file_path() || location == default_weights_location_) {
    active_weights_location_ = location;
    return;
  }
  auto it = extra_weights_streams_.find(location);
  if (it == extra_weights_streams_.end()) {
    std::filesystem::path path = validate_external_location_is_next_to_model(file_path_, location);
    auto stream = std::make_unique<FileWriteStream>(path.string());
    extra_weights_streams_.emplace(location, std::move(stream));
  }
  active_weights_location_ = location;
}

void TwoFilesWriteStream::write_raw_bytes(const uint8_t *data, offset_t n_bytes) {
  main_buf_.write_raw_bytes(data, n_bytes);
}

int64_t TwoFilesWriteStream::size() const { return main_buf_.size(); }

void TwoFilesWriteStream::FlushMainToFile() {
  int64_t sz = main_buf_.size();
  if (sz > 0) {
    file_stream_.write(reinterpret_cast<const char *>(main_buf_.data()), sz);
    written_bytes_ = static_cast<uint64_t>(sz);
  }
}

void TwoFilesWriteStream::WriteDelayedBlock(DelayedWriteBlock &) {
  EXT_THROW("WriteDelayedBlock is not supported for TwoFilesWriteStream main content; "
            "all large tensor writes are routed to the weights stream.");
}

int64_t TwoFilesWriteStream::weights_size() const {
  if (active_weights_location_ == weights_stream_.file_path() ||
      active_weights_location_ == default_weights_location_) {
    return parallel_write_ ? virtual_write_pos_ : weights_stream_.size();
  }
  auto it = extra_weights_streams_.find(active_weights_location_);
  EXT_ENFORCE(it != extra_weights_streams_.end(),
              "Unknown active weights location: ", active_weights_location_);
  return it->second->size();
}

int64_t TwoFilesWriteStream::weights_size(const std::string &location) const {
  if (location.empty() || location == weights_stream_.file_path() ||
      location == default_weights_location_) {
    return parallel_write_ ? virtual_write_pos_ : weights_stream_.size();
  }
  auto it = extra_weights_streams_.find(location);
  if (it == extra_weights_streams_.end()) {
    return 0;
  }
  return it->second->size();
}

int64_t TwoFilesWriteStream::weights_size_for_location(const std::string &location) const {
  return weights_size(location);
}

void TwoFilesWriteStream::pre_allocate_weights(int64_t total_bytes) {
  EXT_ENFORCE(total_bytes >= 0, "total_bytes must be non-negative, got ", total_bytes);
  if (total_bytes == 0)
    return;
  weights_stream_.pre_allocate(total_bytes);
}

void TwoFilesWriteStream::StartWriteThreadPool(int32_t n_threads) {
  EXT_ENFORCE(!parallel_write_, "StartWriteThreadPool already called.");
  parallel_write_ = true;
  virtual_write_pos_ = 0;
  write_thread_pool_.Start(n_threads);
}

void TwoFilesWriteStream::WaitForWriteCompletion() {
  if (parallel_write_) {
    write_thread_pool_.Wait();
    parallel_write_ = false;
  }
}

void TwoFilesWriteStream::write_raw_bytes_in_second_stream(const uint8_t *ptr, offset_t n_bytes) {
  if (active_weights_location_ == weights_stream_.file_path() ||
      active_weights_location_ == default_weights_location_) {
    if (parallel_write_) {
      // `virtual_write_pos_` is only ever read and written on the serialization (calling) thread —
      // worker threads capture `offset` by value and never touch `virtual_write_pos_` — so no
      // synchronization is needed here.
      int64_t offset = virtual_write_pos_;
      virtual_write_pos_ += n_bytes;
      // `ptr` points into a TensorProto::raw_data_ vector owned by the ModelProto that was passed
      // to SerializeModelProtoToStream.  WaitForWriteCompletion() is called before that function
      // returns, so the pointed-to memory is guaranteed to outlive every task.
      const std::string &wpath = weights_stream_.file_path();
      write_thread_pool_.SubmitTask([wpath, ptr, n_bytes, offset]() {
        std::fstream f(wpath, std::ios::binary | std::ios::in | std::ios::out);
        EXT_ENFORCE(f.is_open(), "Failed to open weights file for parallel write: ", wpath);
        f.seekp(offset);
        f.write(reinterpret_cast<const char *>(ptr), n_bytes);
        EXT_ENFORCE(!f.fail(), "Write failed for weights file: ", wpath, " at offset=", offset,
                    " n_bytes=", n_bytes);
      });
    } else {
      weights_stream_.write_raw_bytes(ptr, n_bytes);
    }
    return;
  }

  EXT_ENFORCE(!parallel_write_,
              "Parallel writes are only supported for the default external weights file.");
  auto it = extra_weights_streams_.find(active_weights_location_);
  EXT_ENFORCE(it != extra_weights_streams_.end(),
              "Unknown active weights location: ", active_weights_location_);
  it->second->write_raw_bytes(ptr, n_bytes);
}

void TwoFilesWriteStream::write_raw_bytes_in_second_stream(const uint8_t *ptr, offset_t n_bytes,
                                                           const std::string &location) {
  set_active_weights_location(location);
  write_raw_bytes_in_second_stream(ptr, n_bytes);
}

TwoFilesStream::TwoFilesStream(const std::string &file_path, const std::string &weights_file)
    : FileStream(file_path), weights_stream_(weights_file), active_weights_location_(weights_file),
      default_weights_location_(weights_file) {
  std::filesystem::path parent = std::filesystem::path(file_path).parent_path();
  std::filesystem::path weights = std::filesystem::path(weights_file);
  std::filesystem::path rel = std::filesystem::relative(weights, parent);
  if (!rel.empty() && rel.string() != ".") {
    default_weights_location_ = rel.string();
  }
}

void TwoFilesStream::set_active_weights_location(const std::string &location) {
  if (location.empty()) {
    active_weights_location_ = weights_stream_.file_path();
    return;
  }
  if (location == active_weights_location_) {
    return;
  }
  if (location == weights_stream_.file_path() || location == default_weights_location_) {
    active_weights_location_ = location;
    return;
  }
  auto it = extra_weights_streams_.find(location);
  if (it == extra_weights_streams_.end()) {
    std::filesystem::path path(location);
    if (!path.is_absolute()) {
      std::filesystem::path parent = std::filesystem::path(file_path_).parent_path();
      path = parent / path;
    }
    auto stream = std::make_unique<FileStream>(path.string());
    extra_weights_streams_.emplace(location, std::move(stream));
  }
  active_weights_location_ = location;
}

bool TwoFilesStream::using_default_weights_location() const {
  return active_weights_location_ == weights_stream_.file_path() ||
         active_weights_location_ == default_weights_location_;
}

FileStream &TwoFilesStream::active_weights_stream() {
  if (using_default_weights_location()) {
    return weights_stream_;
  }
  auto it = extra_weights_streams_.find(active_weights_location_);
  EXT_ENFORCE(it != extra_weights_streams_.end(),
              "Unknown active weights location: ", active_weights_location_);
  return *it->second;
}

const FileStream &TwoFilesStream::active_weights_stream() const {
  if (using_default_weights_location()) {
    return weights_stream_;
  }
  auto it = extra_weights_streams_.find(active_weights_location_);
  EXT_ENFORCE(it != extra_weights_streams_.end(),
              "Unknown active weights location: ", active_weights_location_);
  return *it->second;
}

int64_t TwoFilesStream::weights_size(const std::string &location) const {
  if (location.empty() || location == weights_stream_.file_path() ||
      location == default_weights_location_) {
    return weights_stream_.size();
  }
  auto it = extra_weights_streams_.find(location);
  if (it == extra_weights_streams_.end()) {
    return 0;
  }
  return it->second->size();
}

TwoFilesStream::SharedWeightsBuffer &
TwoFilesStream::ensure_shared_weights_buffer(const std::string &location, size_t alignment) {
  set_active_weights_location(location);
  FileStream &wstream = active_weights_stream();
  const std::string &canonical_path = wstream.file_path();
  auto it = shared_weights_buffers_.find(canonical_path);
  if (it != shared_weights_buffers_.end()) {
    if (alignment > 1) {
      EXT_ENFORCE(it->second.alignment == 0 || it->second.alignment == alignment,
                  "Shared external buffer already loaded with incompatible alignment for file ",
                  canonical_path, ": existing=", it->second.alignment, ", requested=", alignment);
    }
    return it->second;
  }

  SharedWeightsBuffer loaded;
  loaded.size = wstream.size();
  // mmap returns page-aligned memory (typically 4096-byte aligned), which satisfies
  // any tensor alignment up to the page size.  Store alignment=0 to signal that the
  // buffer is compatible with any requested alignment so the compatibility check above
  // always passes on a cache hit.
  loaded.alignment = 0;
  loaded.data = mmap_file_as_shared_ptr(canonical_path, loaded.size);
  auto inserted = shared_weights_buffers_.emplace(canonical_path, std::move(loaded));
  return inserted.first->second;
}

void TwoFilesStream::read_bytes_from_weights_stream(offset_t n_bytes, uint8_t *pre_allocated_buffer,
                                                    offset_t offset) {
  FileStream &wstream = active_weights_stream();
  if (offset >= 0) {
    // Discard any pre-fetched bytes before seeking so that the read-ahead buffer
    // stays consistent with the file position.
    wstream._invalidate_read_buffer();
    wstream.file_stream_.seekg(offset);
  }
  wstream.read_bytes(n_bytes, pre_allocated_buffer);
}

const uint8_t *TwoFilesStream::borrow_weights_bytes(const std::string &location, offset_t offset,
                                                    offset_t n_bytes, size_t alignment,
                                                    std::shared_ptr<void> &owner) {
  EXT_ENFORCE(offset >= 0, "External weights offset must be >= 0, got ", offset);
  EXT_ENFORCE(n_bytes >= 0, "External weights size must be >= 0, got ", n_bytes);
  SharedWeightsBuffer &buffer = ensure_shared_weights_buffer(location, alignment);
  EXT_ENFORCE(offset + n_bytes <= buffer.size, "External weights slice is out of bounds for file ",
              active_weights_stream().file_path(), ": offset=", offset, ", size=", n_bytes,
              ", file_size=", buffer.size);
  owner = std::static_pointer_cast<void>(buffer.data);
  return buffer.data.get() + offset;
}

void TwoFilesStream::ReadDelayedBlock(DelayedBlock &block) {
  EXT_ENFORCE(thread_pool_.IsStarted(), "Thread pool is not started, cannot read delayed block.");
  EXT_ENFORCE(
      block.stream_id == 0 || block.stream_id == 1,
      "Only two streams are allowed to read delayed blocks, but stream_id=", block.stream_id);
  blocks_.push_back(block);
  if (block.stream_id == 0) {
    thread_pool_.SubmitTask([this, block]() {
      EXT_ENFORCE(block.offset < static_cast<offset_t>(size()),
                  "Offset for weights stream is out of bounds: ", block.offset,
                  " >= ", static_cast<offset_t>(size()));
#if !defined(_WIN32)
      ReadBlockFromFd(this->file_descriptor_, block, "[TwoFilesStream::ReadDelayedBlock#main]");
#else
      std::ifstream file_stream(this->file_path(), std::ios::binary);
      file_stream.seekg(block.offset);
      file_stream.read(reinterpret_cast<char *>(block.data), block.size);
#endif
    });
    // Advance past the block, draining the read-ahead buffer first.
    skip_bytes(block.size);
  } else {
    EXT_ENFORCE(using_default_weights_location(),
                "Parallel delayed reads are not supported for non-default external locations.");
    thread_pool_.SubmitTask([this, block]() {
      EXT_ENFORCE(block.offset < static_cast<offset_t>(weights_stream_.size()),
                  "Offset for weights stream is out of bounds: ", block.offset,
                  " >= ", weights_stream_.size());
#if !defined(_WIN32)
      ReadBlockFromFd(weights_stream_.file_descriptor_, block,
                      "[TwoFilesStream::ReadDelayedBlock#weights]");
#else
      std::ifstream file_stream(this->weights_file_path(), std::ios::binary);
      file_stream.seekg(block.offset);
      file_stream.read(reinterpret_cast<char *>(block.data), block.size);
#endif
    });
    // Advance the weights stream past the block; the weights stream's read
    // buffer is not used for structure parsing, so a direct seek is fine.
    weights_stream_.file_stream_.seekg(block.size, std::ios::cur);
  }
}

} // namespace utils
} // namespace ONNX_LIGHT_NAMESPACE
