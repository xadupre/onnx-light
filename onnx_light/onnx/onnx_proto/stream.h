#pragma once

#include "onnx_extended_helpers.h"
#include "simple_string.h"
#include "thread_pool.h"
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace onnx {
namespace utils {

typedef int64_t offset_t;

inline int64_t decodeZigZag64(uint64_t n) { return (n >> 1) ^ -(n & 1); }
inline uint64_t encodeZigZag64(int64_t n) {
  return (static_cast<uint64_t>(n) << 1) ^ static_cast<uint64_t>(n >> 63);
}

class StringStream;

struct FieldNumber {
  uint64_t field_number;
  uint64_t wire_type;
  std::string string() const;
};

struct DelayedBlock {
  uint64_t size;
  uint8_t *data;
  offset_t offset;
  uint8_t stream_id = 0; // this is used to identify the substream the data should be coming from
};

struct DelayedWriteBlock {
  uint64_t size;
  const uint8_t *data;
  offset_t offset = -1;
  uint8_t stream_id = 0; // this is used to identify the substream the data should be going to
};

/** Base class for binary input streams. */
class BinaryStream {
public:
  explicit inline BinaryStream() {}
  virtual ~BinaryStream();
  virtual bool ExternalWeights() const { return false; }
  // to overwrite
  virtual uint64_t next_uint64() = 0;
  virtual void CanRead(uint64_t len, const char *msg) = 0;
  virtual bool NotEnd() const = 0;
  virtual offset_t tell() const = 0;
  virtual std::string tell_around() const = 0;
  virtual const uint8_t *read_bytes(offset_t n_bytes, uint8_t *pre_allocated_buffer = nullptr) = 0;
  virtual void skip_bytes(offset_t n_bytes) = 0;
  virtual int64_t size() const = 0;
  // defines from the previous ones
  virtual RefString next_string();
  virtual int64_t next_int64();
  virtual int32_t next_int32();
  virtual float next_float();
  virtual double next_double();
  virtual FieldNumber next_field();
  template <typename T> void next_packed_element(T &value) {
    value = *reinterpret_cast<const T *>(read_bytes(sizeof(T)));
  }
  // Reading substream
  virtual void LimitToNext(uint64_t len);
  virtual void Restore();

  // parallelization of big blocks.
  virtual bool HasParallelizationStarted() const { return false; }
  virtual void StartThreadPool(size_t n_threads);
  virtual void ReadDelayedBlock(DelayedBlock &block);
  virtual void WaitForDelayedBlock();

protected:
  virtual void LimitTo(uint64_t len) = 0;
  virtual void _check();
  std::vector<uint64_t> limits_;
};

class StringWriteStream;
class BorrowedWriteStream;
class FileStream;

/** Base class for binary output streams. */
class BinaryWriteStream {
public:
  explicit inline BinaryWriteStream() {}
  virtual ~BinaryWriteStream() {}
  // to overwrite
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) = 0;
  virtual int64_t size() const = 0;
  virtual const uint8_t *data() const = 0;
  // defined from the previous ones
  virtual void write_variant_uint64(uint64_t value);
  virtual void write_int64(int64_t value);
  virtual void write_int32(int32_t value);
  virtual void write_float(float value);
  virtual void write_double(double value);
  virtual void write_string(const std::string &value);
  virtual void write_string(const String &value);
  virtual void write_string(const RefString &value);
  virtual void write_string_stream(const StringWriteStream &stream);
  virtual void write_string_stream(const BorrowedWriteStream &stream);
  virtual void write_field_header(uint32_t field_number, uint8_t wire_type);
  template <typename T> void write_packed_element(const T &value) {
    write_raw_bytes(reinterpret_cast<const uint8_t *>(&value), sizeof(T));
  }
  // size
  virtual uint64_t size_field_header(uint32_t field_number, uint8_t wire_type);
  virtual uint64_t VarintSize(uint64_t value);
  virtual uint64_t size_variant_uint64(uint64_t value);
  virtual uint64_t size_int64(int64_t value);
  virtual uint64_t size_int32(int32_t value);
  virtual uint64_t size_float(float value);
  virtual uint64_t size_double(double value);
  virtual uint64_t size_string(const std::string &value);
  virtual uint64_t size_string(const String &value);
  virtual uint64_t size_string(const RefString &value);
  virtual uint64_t size_string_stream(const StringWriteStream &stream);
  virtual uint64_t size_string_stream(const BorrowedWriteStream &stream);
  // weights
  virtual bool ExternalWeights() const { return false; }

  // cache
  virtual void CacheSize(const void *ptr, uint64_t size);
  virtual bool GetCachedSize(const void *ptr, uint64_t &size);
  // parallelization of big blocks.
  virtual bool HasParallelizationStarted() const { return false; }
  virtual void StartThreadPool(size_t n_threads);
  virtual void WriteDelayedBlock(DelayedWriteBlock &block);
  virtual void WaitForDelayedBlock();

protected:
  std::unordered_map<const void *, uint64_t> size_cache_;
};

///////////
/// strings
///////////

/** Binary reader backed by an in-memory buffer. */
class StringStream : public BinaryStream {
  friend class FileStream;

public:
  explicit inline StringStream() : BinaryStream(), pos_(0), size_(0), data_(nullptr) {}
  explicit inline StringStream(const uint8_t *data, int64_t size)
      : BinaryStream(), pos_(0), size_(size), data_(data) {}
  void Setup(const uint8_t *data, int64_t size);
  virtual void CanRead(uint64_t len, const char *msg) override;
  virtual uint64_t next_uint64() override;
  virtual const uint8_t *read_bytes(offset_t n_bytes,
                                    uint8_t *pre_allocated_buffer = nullptr) override;
  virtual void skip_bytes(offset_t n_bytes) override;
  virtual bool NotEnd() const override { return pos_ < size_; }
  virtual offset_t tell() const override { return static_cast<offset_t>(pos_); }
  virtual std::string tell_around() const override;
  virtual inline int64_t size() const override { return size_; }

  // parallelization of big blocks.
  virtual bool HasParallelizationStarted() const override { return thread_pool_.IsStarted(); }
  virtual void StartThreadPool(size_t n_threads) override;
  virtual void ReadDelayedBlock(DelayedBlock &block) override;
  virtual void WaitForDelayedBlock() override;

protected:
  virtual void LimitTo(uint64_t len) override;

protected:
  offset_t pos_;
  offset_t size_;
  const uint8_t *data_;

protected:
  // parallelization
  std::vector<DelayedBlock> blocks_;
  ThreadPool thread_pool_;
};

/** Binary writer backed by an owned memory buffer. */
class StringWriteStream : public BinaryWriteStream {
public:
  explicit inline StringWriteStream() : BinaryWriteStream(), buffer_(), write_pos_(0) {}
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override;
  virtual int64_t size() const override;
  virtual const uint8_t *data() const override;

  /** Pre-allocates the buffer to *total_bytes* bytes (zero-filled).
   *  Requires calling before StartThreadPool; ensures buffer_.data() remains
   *  stable so no reallocation occurs during concurrent writes. */
  void pre_allocate(int64_t total_bytes);

  // parallelization of big blocks.
  virtual bool HasParallelizationStarted() const override { return thread_pool_.IsStarted(); }
  virtual void StartThreadPool(size_t n_threads) override;
  virtual void WriteDelayedBlock(DelayedWriteBlock &block) override;
  virtual void WaitForDelayedBlock() override;

protected:
  std::vector<uint8_t> buffer_;
  offset_t write_pos_;

  // parallelization
  ThreadPool thread_pool_;
};

/** Binary writer backed by externally provided memory. */
class BorrowedWriteStream : public BinaryWriteStream {
public:
  explicit inline BorrowedWriteStream(const uint8_t *data, int64_t size)
      : BinaryWriteStream(), data_(data), size_(size) {}
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override;
  virtual int64_t size() const override { return size_; }
  virtual const uint8_t *data() const override { return data_; }

protected:
  const uint8_t *data_;
  int64_t size_;
};

////////
// files
////////

/** Binary writer that persists bytes to a file. */
class FileWriteStream : public BinaryWriteStream {
public:
  explicit FileWriteStream(const std::string &file_path);
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override;
  virtual int64_t size() const override;
  virtual const uint8_t *data() const override;
  inline const std::string &file_path() const { return file_path_; }
  virtual bool HasParallelizationStarted() const override { return thread_pool_.IsStarted(); }
  virtual void StartThreadPool(size_t n_threads) override;
  virtual void WriteDelayedBlock(DelayedWriteBlock &block) override;
  virtual void WaitForDelayedBlock() override;
  /** Pre-allocates the file to *total_bytes* by seeking and writing a zero at the last position.
   *  Flushes immediately so the ofstream buffer is clear before parallel tasks write concurrently.
   */
  void pre_allocate(int64_t total_bytes);

protected:
  std::string file_path_;
  std::ofstream file_stream_;
  uint64_t written_bytes_;
  ThreadPool thread_pool_;
};

class TwoFilesStream;

/** Binary reader that streams bytes from a file. */
class FileStream : public BinaryStream {
  friend class TwoFilesStream;

public:
  explicit FileStream(const std::string &file_path);
  virtual ~FileStream();
  inline const std::string &file_path() const { return file_path_; }
  virtual void CanRead(uint64_t len, const char *msg) override;
  virtual uint64_t next_uint64() override;
  virtual const uint8_t *read_bytes(offset_t n_bytes,
                                    uint8_t *pre_allocated_buffer = nullptr) override;
  virtual void skip_bytes(offset_t n_bytes) override;
  /**
   * This is a dangerous zone. StreamStream points to the buffer_.data().
   * buffer_ changes every time new bytes are read from the file.
   * So unlock() must be called or this class raises an exception.
   */
  virtual bool NotEnd() const override;
  virtual offset_t tell() const override;
  virtual std::string tell_around() const override;
  virtual bool is_open() const;
  virtual int64_t size() const override { return size_; }

  // parallelization of big blocks.
  virtual bool HasParallelizationStarted() const override { return thread_pool_.IsStarted(); }
  virtual void StartThreadPool(size_t n_threads) override;
  virtual void ReadDelayedBlock(DelayedBlock &block) override;
  virtual void WaitForDelayedBlock() override;

protected:
  virtual void LimitTo(uint64_t len) override;

  /** Fills the read-ahead buffer from the current file position. */
  void _fill_read_buffer();
  /** Seeks the underlying file stream back to tell() and clears the buffer.
   *  Must be called before any direct file_stream_ seek or non-buffered read. */
  void _invalidate_read_buffer();

protected:
  bool lock_;
  std::string file_path_;
  std::ifstream file_stream_;
#if !defined(_WIN32)
  int file_descriptor_ = -1;
#endif
  int64_t size_;
  std::vector<uint8_t> buffer_;
  // parallelization
  std::vector<DelayedBlock> blocks_;
  ThreadPool thread_pool_;

  // Read-ahead buffer for fast sequential varint/byte parsing.
  // Buffers up to READ_BUF_SIZE bytes so that next_uint64() can
  // read single bytes without calling file_stream_.read() each time.
  static constexpr size_t READ_BUF_SIZE = 4096;
  std::vector<uint8_t> read_buf_;
  size_t read_buf_pos_ = 0;
  size_t read_buf_end_ = 0;
};

//////////////////////////////
// Stream for external weights
//////////////////////////////

/** Two-file writer for external ONNX tensor data. */
class TwoFilesWriteStream : public FileWriteStream {
public:
  explicit TwoFilesWriteStream(const std::string &file_path, const std::string &weights_file);
  inline const std::string &weights_file_path() const { return weights_stream_.file_path(); }
  virtual bool ExternalWeights() const override { return true; }
  virtual void write_raw_bytes_in_second_stream(const uint8_t *data, offset_t n_bytes);
  virtual int64_t weights_size() const;

  /** Pre-allocates the weights file to *total_bytes* by writing a zero at the last position.
   *  Must be called before StartWriteThreadPool. */
  void pre_allocate_weights(int64_t total_bytes);

  /** Starts a thread pool of *n_threads* workers for parallel offset-based writes.
   *  After this call write_raw_bytes_in_second_stream submits writes asynchronously. */
  void StartWriteThreadPool(int32_t n_threads);

  /** Blocks until all pending write tasks have completed and stops the thread pool. */
  void WaitForWriteCompletion();

protected:
  FileWriteStream weights_stream_;
  std::unordered_map<const void *, uint64_t> position_cache_;

  // Parallel-write state
  bool parallel_write_ = false;
  int64_t virtual_write_pos_ = 0; // tracks sequential position for offset validation
  ThreadPool write_thread_pool_;
};

/** Two-file reader for ONNX models with external tensor data. */
class TwoFilesStream : public FileStream {
public:
  explicit TwoFilesStream(const std::string &file_path, const std::string &weights_file);
  inline const std::string &weights_file_path() const { return weights_stream_.file_path(); }
  inline uint64_t weights_tell() const { return weights_stream_.tell(); }
  virtual bool ExternalWeights() const override { return true; }
  virtual void read_bytes_from_weights_stream(offset_t n_bytes,
                                              uint8_t *pre_allocated_buffer = nullptr,
                                              offset_t offset = -1);
  virtual void ReadDelayedBlock(DelayedBlock &block) override;
  virtual int64_t weights_size() const { return weights_stream_.size(); }

protected:
  FileStream weights_stream_;
  std::unordered_map<const void *, uint64_t> position_cache_;
};

} // namespace utils
} // namespace onnx
