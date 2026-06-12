#pragma once

// Compatibility shim that provides google::protobuf namespace aliases
// mapped to onnx-light's own implementations. This allows code written
// against the protobuf API (e.g., onnxruntime) to compile without
// linking libprotobuf.

#include "fields.h"
#include "stream.h"
#include <cstdint>
#include <istream>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

namespace google {
namespace protobuf {

// --- Repeated field types ---

template <typename T> using RepeatedField = ONNX_LIGHT_NAMESPACE::utils::RepeatedField<T>;

template <typename T> using RepeatedPtrField = ONNX_LIGHT_NAMESPACE::utils::RepeatedProtoField<T>;

// --- RepeatedFieldBackInserter ---

/** Output iterator that appends to a RepeatedField via push_back. */
template <typename T> class RepeatedFieldBackInsertIterator {
public:
  using iterator_category = std::output_iterator_tag;
  using value_type = void;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = void;

  explicit RepeatedFieldBackInsertIterator(RepeatedField<T> *field) : field_(field) {}

  RepeatedFieldBackInsertIterator &operator=(const T &value) {
    field_->push_back(value);
    return *this;
  }

  RepeatedFieldBackInsertIterator &operator=(T &&value) {
    field_->push_back(std::move(value));
    return *this;
  }

  RepeatedFieldBackInsertIterator &operator*() { return *this; }
  RepeatedFieldBackInsertIterator &operator++() { return *this; }
  RepeatedFieldBackInsertIterator operator++(int) { return *this; }

private:
  RepeatedField<T> *field_;
};

/** Creates a back-insert iterator for a RepeatedField. */
template <typename T>
RepeatedFieldBackInsertIterator<T> RepeatedFieldBackInserter(RepeatedField<T> *field) {
  return RepeatedFieldBackInsertIterator<T>(field);
}

// --- Lifecycle ---

/** No-op: onnx-light has no global protobuf state to shut down. */
inline void ShutdownProtobufLibrary() {}

// --- I/O streams (minimal stubs) ---

namespace io {

/** Minimal input stream wrapping a string. */
class ArrayInputStream {
public:
  ArrayInputStream(const void *data, int size)
      : data_(static_cast<const char *>(data)), size_(size), pos_(0) {}

  bool Next(const void **data, int *size) {
    if (pos_ >= size_)
      return false;
    *data = data_ + pos_;
    *size = size_ - pos_;
    pos_ = size_;
    return true;
  }

  void BackUp(int count) { pos_ -= count; }
  int64_t ByteCount() const { return pos_; }

private:
  const char *data_;
  int size_;
  int pos_;
};

/** Minimal coded input stream wrapping an ArrayInputStream. */
class CodedInputStream {
public:
  explicit CodedInputStream(ArrayInputStream *input) : input_(input), limit_(0x7FFFFFFF) {}

  void SetTotalBytesLimit(int total_bytes_limit) { limit_ = total_bytes_limit; }
  int TotalBytesLimit() const { return limit_; }

private:
  ArrayInputStream *input_;
  int limit_;
};

/** Minimal output stream wrapping a std::string. */
class StringOutputStream {
public:
  explicit StringOutputStream(std::string *target) : target_(target) {}

  bool Next(void **data, int *size) {
    size_t old_size = target_->size();
    size_t new_size = old_size + 1024;
    target_->resize(new_size);
    *data = &(*target_)[old_size];
    *size = static_cast<int>(new_size - old_size);
    return true;
  }

  void BackUp(int count) { target_->resize(target_->size() - static_cast<size_t>(count)); }
  int64_t ByteCount() const { return static_cast<int64_t>(target_->size()); }

private:
  std::string *target_;
};

/** Minimal zero-copy stream wrapping a file descriptor. */
class FileOutputStream {
public:
  explicit FileOutputStream(int fd) : fd_(fd), buffer_size_(0) {}

  bool Next(void **data, int *size) {
    *data = buffer_;
    *size = static_cast<int>(sizeof(buffer_));
    buffer_size_ = sizeof(buffer_);
    return true;
  }

  void BackUp(int count) { buffer_size_ -= static_cast<size_t>(count); }

  bool Flush() {
    if (buffer_size_ > 0) {
      // Write is handled by caller flushing the fd
      buffer_size_ = 0;
    }
    return true;
  }

  bool Close() { return Flush(); }
  int64_t ByteCount() const { return 0; }

private:
  int fd_;
  char buffer_[4096];
  size_t buffer_size_;
};

/** Minimal zero-copy stream wrapping a std::istream. */
class IstreamInputStream {
public:
  explicit IstreamInputStream(std::istream *stream, int block_size = 4096)
      : stream_(stream), block_size_(block_size), buffer_(static_cast<size_t>(block_size)), pos_(0),
        size_(0) {}

  bool Next(const void **data, int *size) {
    stream_->read(buffer_.data(), block_size_);
    size_ = static_cast<int>(stream_->gcount());
    if (size_ == 0)
      return false;
    *data = buffer_.data();
    *size = size_;
    pos_ += size_;
    return true;
  }

  void BackUp(int count) {
    stream_->seekg(-count, std::ios_base::cur);
    pos_ -= count;
  }

  int64_t ByteCount() const { return pos_; }

private:
  std::istream *stream_;
  int block_size_;
  std::vector<char> buffer_;
  int64_t pos_;
  int size_;
};

/** Minimal zero-copy stream wrapping a std::ostream. */
class OstreamOutputStream {
public:
  explicit OstreamOutputStream(std::ostream *stream, int block_size = 4096)
      : stream_(stream), block_size_(block_size), buffer_(static_cast<size_t>(block_size)),
        used_(0) {}

  ~OstreamOutputStream() { Flush(); }

  bool Next(void **data, int *size) {
    Flush();
    *data = buffer_.data();
    *size = block_size_;
    used_ = block_size_;
    return true;
  }

  void BackUp(int count) { used_ -= count; }

  bool Flush() {
    if (used_ > 0) {
      stream_->write(buffer_.data(), used_);
      used_ = 0;
    }
    return stream_->good();
  }

  int64_t ByteCount() const { return 0; }

private:
  std::ostream *stream_;
  int block_size_;
  std::vector<char> buffer_;
  int used_;
};

} // namespace io
} // namespace protobuf
} // namespace google
