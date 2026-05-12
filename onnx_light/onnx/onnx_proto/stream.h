#pragma once

#include "onnx_light_helpers.h"
#include "simple_string.h"
#include "thread_pool.h"
#include <cstddef>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

constexpr int64_t kSmallTensorDataThresholdBytes = 8 * static_cast<int64_t>(sizeof(int64_t));

/** Splits serialized bytes between the protobuf payload and separate tensor data. */
struct SerializeSizeResult {
  /** Stores the number of bytes written to small external tensor data blocks. */
  int64_t small_data_size = 0;
  /** Stores the number of bytes written to big external tensor data blocks. */
  int64_t big_data_size = 0;
  /** Stores the number of bytes kept in the protobuf payload. */
  int64_t proto_size = 0;

  /** Initializes an empty size split. */
  constexpr SerializeSizeResult() = default;
  /** Initializes the size split from small data, big data, and protobuf byte counts. */
  constexpr SerializeSizeResult(int64_t small_data_size, int64_t big_data_size, int64_t proto_size)
      : small_data_size(small_data_size), big_data_size(big_data_size), proto_size(proto_size) {}

  /** Adds tensor bytes to the small or big bucket depending on threshold. */
  constexpr void add_tensor_data_size(int64_t tensor_data_size, int64_t threshold) {
    if (tensor_data_size < threshold) {
      small_data_size += tensor_data_size;
    } else {
      big_data_size += tensor_data_size;
    }
  }

  /** Accumulates another serialized size split into this result. */
  constexpr SerializeSizeResult &operator+=(const SerializeSizeResult &other) {
    small_data_size += other.small_data_size;
    big_data_size += other.big_data_size;
    proto_size += other.proto_size;
    return *this;
  }

  /** Returns the total serialized size across protobuf and external data. */
  constexpr int64_t size() const { return small_data_size + big_data_size + proto_size; }
};

/** Returns the sum of two serialized size splits. */
inline constexpr SerializeSizeResult operator+(SerializeSizeResult left,
                                               const SerializeSizeResult &right) {
  left += right;
  return left;
}

namespace utils {

/** Signed byte-offset type used by stream seek and length operations. */
typedef int64_t offset_t;

/** Decodes a 64-bit integer from its ZigZag-encoded unsigned representation.
 *  ZigZag maps signed integers to unsigned values so that small negative numbers
 *  have a short varint encoding: 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, ... */
inline int64_t decodeZigZag64(uint64_t n) { return (n >> 1) ^ -(n & 1); }

/** Encodes a signed 64-bit integer using ZigZag encoding.
 *  The resulting unsigned value is suitable for efficient varint serialization
 *  of values that are likely to be small in magnitude. */
inline uint64_t encodeZigZag64(int64_t n) {
  return (static_cast<uint64_t>(n) << 1) ^ static_cast<uint64_t>(n >> 63);
}

class StringStream;

/** Decoded protobuf tag containing the field number and wire type. */
struct FieldNumber {
  /** Identifies which field in the message this tag belongs to. */
  uint64_t field_number;
  /** Wire type that describes the on-wire encoding of the field value. */
  uint64_t wire_type;
  /** Returns a human-readable representation of the field number and wire type. */
  std::string string() const;
};

/** Descriptor for a block of data that should be processed asynchronously by a thread pool. */
struct DelayedBlock {
  /** Number of bytes in the block. */
  uint64_t size;
  /** Pointer to the destination buffer that will receive the decoded bytes. */
  uint8_t *data;
  /** Byte offset within the source stream where the block starts. */
  offset_t offset;
  /** Identifies the substream the data should be read from (0 = primary stream). */
  uint8_t stream_id = 0;
};

/** Descriptor for a block of data that should be written asynchronously by a thread pool. */
struct DelayedWriteBlock {
  /** Number of bytes to write. */
  uint64_t size;
  /** Pointer to the source data. The caller must keep this buffer alive until
   *  WaitForDelayedBlock() returns. */
  const uint8_t *data;
  /** Byte offset within the destination stream where the block should be written.
   *  A value of -1 means the current sequential write position is used. */
  offset_t offset = -1;
  /** Identifies the substream the data should be written to (0 = primary stream). */
  uint8_t stream_id = 0;
};

/** Base class for binary input streams.
 *  Concrete subclasses provide byte-level reading from different sources
 *  (memory buffers, files) and expose higher-level helpers for varint,
 *  float, string, and protobuf tag decoding built on top of the primitives. */
class BinaryStream {
public:
  /** Initializes a base binary stream with no active limits. */
  explicit inline BinaryStream() {}
  virtual ~BinaryStream();
  /** Returns true if this stream delivers tensor weights from a separate storage backend. */
  virtual bool ExternalWeights() const { return false; }
  /** Returns true if read_bytes(n, nullptr) returns a stable pointer into the backing buffer.
   *  When true, the returned pointer remains valid as long as the underlying data buffer lives.
   *  In-memory streams (StringStream) return true; file-backed streams return false. */
  virtual bool CanNoCopy() const { return false; }
  // to overwrite
  /** Reads and decodes the next base-128 varint as an unsigned 64-bit integer. */
  virtual uint64_t next_uint64() = 0;
  /** Raises an exception if fewer than *len* bytes remain before the current limit.
   *  The *msg* parameter is included in the error message for diagnostics. */
  virtual void CanRead(uint64_t len, const char *msg) = 0;
  /** Returns true if the current read position has not yet reached the active limit. */
  virtual bool NotEnd() const = 0;
  /** Returns the current byte offset from the beginning of the stream. */
  virtual offset_t tell() const = 0;
  /** Returns a short human-readable excerpt of bytes around the current position.
   *  Useful for error messages and debugging. */
  virtual std::string tell_around() const = 0;
  /** Reads *n_bytes* bytes and returns a pointer to the data.
   *  If *pre_allocated_buffer* is non-null the bytes are copied into it;
   *  otherwise the stream may return a pointer into its internal buffer. */
  virtual const uint8_t *read_bytes(offset_t n_bytes, uint8_t *pre_allocated_buffer = nullptr) = 0;
  /** Advances the read position by *n_bytes* without returning the data. */
  virtual void skip_bytes(offset_t n_bytes) = 0;
  /** Returns the total number of bytes in the stream. */
  virtual int64_t size() const = 0;
  // defines from the previous ones
  /** Reads the next length-prefixed byte sequence and returns a non-owning view. */
  virtual RefString next_string();
  /** Reads the next varint and interprets it as a ZigZag-encoded signed 64-bit integer. */
  virtual int64_t next_int64();
  /** Reads the next varint and interprets it as a ZigZag-encoded signed 32-bit integer. */
  virtual int32_t next_int32();
  /** Reads the next 4 bytes and interprets them as a little-endian IEEE 754 float. */
  virtual float next_float();
  /** Reads the next 8 bytes and interprets them as a little-endian IEEE 754 double. */
  virtual double next_double();
  /** Reads the next protobuf tag and returns a FieldNumber struct. */
  virtual FieldNumber next_field();
  /** Reads a single packed element of type *T* from the stream into *value*. */
  template <typename T> void next_packed_element(T &value) {
    value = *reinterpret_cast<const T *>(read_bytes(sizeof(T)));
  }
  // Reading substream
  /** Pushes a new read limit of *len* bytes relative to the current position.
   *  NotEnd() will return false once this many bytes have been consumed. */
  virtual void LimitToNext(uint64_t len);
  /** Pops the most recently pushed read limit, restoring the previous limit. */
  virtual void Restore();

  // parallelization of big blocks.
  /** Returns true once StartThreadPool() has been called and is still active. */
  virtual bool HasParallelizationStarted() const { return false; }
  /** Starts an internal thread pool with *n_threads* worker threads for parallel block reads. */
  virtual void StartThreadPool(size_t n_threads);
  /** Submits *block* to the thread pool so its data is read asynchronously. */
  virtual void ReadDelayedBlock(DelayedBlock &block);
  /** Blocks until all pending asynchronous read blocks have completed. */
  virtual void WaitForDelayedBlock();

protected:
  /** Sets the internal read limit to *len* bytes from the current position. */
  virtual void LimitTo(uint64_t len) = 0;
  /** Validates the internal limit stack; throws if the state is inconsistent. */
  virtual void _check();
  /** Stack of absolute byte offsets used to implement nested LimitToNext/Restore pairs. */
  std::vector<uint64_t> limits_;
};

class StringWriteStream;
class BorrowedWriteStream;
class FileStream;

/** Base class for binary output streams.
 *  Concrete subclasses persist bytes to different backends (memory buffers, files).
 *  Higher-level helpers for varint, float, string, and protobuf tag encoding
 *  are implemented on top of the write_raw_bytes() primitive. */
class BinaryWriteStream {
public:
  /** Initializes an empty binary write stream. */
  explicit inline BinaryWriteStream() {}
  virtual ~BinaryWriteStream() {}
  // to overwrite
  /** Appends *n_bytes* bytes starting at *data* to the stream. */
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) = 0;
  /** Returns the number of bytes written so far. */
  virtual int64_t size() const = 0;
  /** Returns a pointer to the beginning of the stream's internal buffer.
   *  Only valid for in-memory streams; file-backed streams return nullptr. */
  virtual const uint8_t *data() const = 0;
  // defined from the previous ones
  /** Encodes *value* as a base-128 varint and appends it to the stream. */
  virtual void write_variant_uint64(uint64_t value);
  /** Encodes *value* as a ZigZag varint and appends it to the stream. */
  virtual void write_int64(int64_t value);
  /** Encodes *value* as a ZigZag varint and appends it to the stream. */
  virtual void write_int32(int32_t value);
  /** Appends the 4-byte little-endian IEEE 754 representation of *value*. */
  virtual void write_float(float value);
  /** Appends the 8-byte little-endian IEEE 754 representation of *value*. */
  virtual void write_double(double value);
  /** Encodes *value* as a length-prefixed byte sequence and appends it. */
  virtual void write_string(const std::string &value);
  /** Encodes *value* as a length-prefixed byte sequence and appends it. */
  virtual void write_string(const String &value);
  /** Encodes *value* as a length-prefixed byte sequence and appends it. */
  virtual void write_string(const RefString &value);
  /** Encodes the contents of *stream* as a length-prefixed byte sequence and appends it. */
  virtual void write_string_stream(const StringWriteStream &stream);
  /** Encodes the contents of *stream* as a length-prefixed byte sequence and appends it. */
  virtual void write_string_stream(const BorrowedWriteStream &stream);
  /** Encodes a protobuf field tag from *field_number* and *wire_type* and appends it. */
  virtual void write_field_header(uint32_t field_number, uint8_t wire_type);
  /** Appends a single packed element of type *T* to the stream. */
  template <typename T> void write_packed_element(const T &value) {
    write_raw_bytes(reinterpret_cast<const uint8_t *>(&value), sizeof(T));
  }
  // size
  /** Returns the serialized byte size of the protobuf field tag for the given field number and
   * wire type. */
  virtual uint64_t size_field_header(uint32_t field_number, uint8_t wire_type);
  /** Returns the number of bytes required to encode *value* as a base-128 varint. */
  virtual uint64_t VarintSize(uint64_t value);
  /** Returns the encoded byte size of *value* as a varint. */
  virtual uint64_t size_variant_uint64(uint64_t value);
  /** Returns the encoded byte size of *value* as a ZigZag varint. */
  virtual uint64_t size_int64(int64_t value);
  /** Returns the encoded byte size of *value* as a ZigZag varint. */
  virtual uint64_t size_int32(int32_t value);
  /** Returns the encoded byte size of *value* (always 4). */
  virtual uint64_t size_float(float value);
  /** Returns the encoded byte size of *value* (always 8). */
  virtual uint64_t size_double(double value);
  /** Returns the encoded byte size of the length-prefixed string *value*. */
  virtual uint64_t size_string(const std::string &value);
  /** Returns the encoded byte size of the length-prefixed string *value*. */
  virtual uint64_t size_string(const String &value);
  /** Returns the encoded byte size of the length-prefixed string *value*. */
  virtual uint64_t size_string(const RefString &value);
  /** Returns the encoded byte size of the contents of *stream* as a length-prefixed sequence. */
  virtual uint64_t size_string_stream(const StringWriteStream &stream);
  /** Returns the encoded byte size of the contents of *stream* as a length-prefixed sequence. */
  virtual uint64_t size_string_stream(const BorrowedWriteStream &stream);
  // weights
  /** Returns true if this stream routes tensor weight data to a separate backend. */
  virtual bool ExternalWeights() const { return false; }
  /** Appends *n_bytes* from *data* to the external weights backend. */
  virtual void write_raw_bytes_in_second_stream(const uint8_t *, offset_t) {
    EXT_THROW("write_raw_bytes_in_second_stream is not implemented.");
  }
  /** Appends *n_bytes* from *data* to the external weights backend selected by *location*. */
  virtual void write_raw_bytes_in_second_stream(const uint8_t *data, offset_t n_bytes,
                                                const std::string &) {
    write_raw_bytes_in_second_stream(data, n_bytes);
  }
  /** Returns number of bytes written to the default external weights backend. */
  virtual int64_t weights_size() const { return 0; }
  /** Returns number of bytes written to the external weights backend selected by *location*. */
  virtual int64_t weights_size_for_location(const std::string &) const { return weights_size(); }

  // cache
  /** Associates serialized size information with the object at *ptr* in the size cache. */
  virtual void CacheSize(const void *ptr, SerializeSizeResult size);
  /** Looks up the cached serialized size for the object at *ptr*.
   *  Returns true and writes the result into *size* if found. */
  virtual bool GetCachedSize(const void *ptr, SerializeSizeResult &size);
  /** Swaps the size cache with *other*, transferring cached sizes between streams
   *  in O(1) so the write pass can reuse sizes computed by a separate size pass. */
  void swap_size_cache(BinaryWriteStream &other) { std::swap(size_cache_, other.size_cache_); }
  // parallelization of big blocks.
  /** Returns true once StartThreadPool() has been called and is still active. */
  virtual bool HasParallelizationStarted() const { return false; }
  /** Starts an internal thread pool with *n_threads* worker threads for parallel block writes. */
  virtual void StartThreadPool(size_t n_threads);
  /** Submits *block* to the thread pool so its data is written asynchronously. */
  virtual void WriteDelayedBlock(DelayedWriteBlock &block);
  /** Blocks until all pending asynchronous write blocks have completed. */
  virtual void WaitForDelayedBlock();

protected:
  /** Per-object serialized-size cache used to avoid redundant recomputation. */
  std::unordered_map<const void *, SerializeSizeResult> size_cache_;
};

///////////
/// strings
///////////

/** Binary reader backed by an in-memory buffer.
 *  Provides fast sequential access to a caller-owned byte range without
 *  copying the underlying data. */
class StringStream : public BinaryStream {
  friend class FileStream;

public:
  /** Initializes an empty stream pointing to no data. */
  explicit inline StringStream() : BinaryStream(), pos_(0), size_(0), data_(nullptr) {}
  /** Initializes a stream that reads *size* bytes starting at *data*. */
  explicit inline StringStream(const uint8_t *data, int64_t size)
      : BinaryStream(), pos_(0), size_(size), data_(data) {}
  /** Resets the stream to read *size* bytes starting at *data*. */
  void Setup(const uint8_t *data, int64_t size);
  virtual void CanRead(uint64_t len, const char *msg) override;
  virtual uint64_t next_uint64() override;
  virtual const uint8_t *read_bytes(offset_t n_bytes,
                                    uint8_t *pre_allocated_buffer = nullptr) override;
  virtual void skip_bytes(offset_t n_bytes) override;
  /** Returns true while the read position is before the end of the active limit. */
  virtual bool NotEnd() const override { return pos_ < size_; }
  /** Returns true: read_bytes(n, nullptr) yields a stable pointer into data_. */
  virtual bool CanNoCopy() const override { return true; }
  /** Returns the current byte offset from the beginning of the buffer. */
  virtual offset_t tell() const override { return static_cast<offset_t>(pos_); }
  virtual std::string tell_around() const override;
  /** Returns the total number of bytes in the buffer. */
  virtual inline int64_t size() const override { return size_; }

  // parallelization of big blocks.
  /** Returns true once StartThreadPool() has been called and is still active. */
  virtual bool HasParallelizationStarted() const override { return thread_pool_.IsStarted(); }
  virtual void StartThreadPool(size_t n_threads) override;
  virtual void ReadDelayedBlock(DelayedBlock &block) override;
  virtual void WaitForDelayedBlock() override;

protected:
  virtual void LimitTo(uint64_t len) override;

protected:
  /** Current read offset within data_. */
  offset_t pos_;
  /** Active read limit; equals the buffer size when no sub-limit is active. */
  offset_t size_;
  /** Non-owning pointer to the backing byte buffer. */
  const uint8_t *data_;

protected:
  // parallelization
  /** List of blocks scheduled for parallel asynchronous processing. */
  std::vector<DelayedBlock> blocks_;
  /** Thread pool used for parallel block reads. */
  ThreadPool thread_pool_;
};

/** Binary writer backed by an owned memory buffer.
 *  All bytes are accumulated in an internal std::vector<uint8_t>.
 *  Supports optional parallel writes via an internal thread pool after the
 *  buffer has been pre-allocated with pre_allocate(). */
class StringWriteStream : public BinaryWriteStream {
public:
  /** Initializes an empty write stream with no pre-allocated storage. */
  explicit inline StringWriteStream() : BinaryWriteStream(), buffer_(), write_pos_(0) {}
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override;
  /** Returns the number of bytes written so far. */
  virtual int64_t size() const override;
  /** Returns a pointer to the beginning of the internal buffer. */
  virtual const uint8_t *data() const override;

  /** Pre-allocates the buffer to *total_bytes* bytes (zero-filled).
   *  Requires calling before StartThreadPool; ensures buffer_.data() remains
   *  stable so no reallocation occurs during concurrent writes. */
  void pre_allocate(int64_t total_bytes);

  // parallelization of big blocks.
  /** Returns true once StartThreadPool() has been called and is still active. */
  virtual bool HasParallelizationStarted() const override { return thread_pool_.IsStarted(); }
  virtual void StartThreadPool(size_t n_threads) override;
  virtual void WriteDelayedBlock(DelayedWriteBlock &block) override;
  virtual void WaitForDelayedBlock() override;

protected:
  /** Owned byte buffer that accumulates written data. */
  std::vector<uint8_t> buffer_;
  /** Current sequential write offset within buffer_. */
  offset_t write_pos_;

  // parallelization
  /** Thread pool used for parallel block writes. */
  ThreadPool thread_pool_;
};

/** Binary writer backed by a caller-provided fixed-capacity memory buffer.
 *  Inherits string-writing helpers from StringWriteStream but never reallocates.
 *  Throws std::runtime_error if a write would exceed the initial capacity. */
class BorrowedStringWriteStream : public StringWriteStream {
public:
  /** Initializes a write stream that borrows `size` bytes starting at `data`.
   *  The caller must ensure the buffer outlives this stream. */
  explicit inline BorrowedStringWriteStream(uint8_t *data, int64_t size)
      : StringWriteStream(), data_(data), capacity_(size) {
    write_pos_ = 0;
  }
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override;
  /** Submits *block* to the thread pool to write directly into the borrowed buffer. */
  virtual void WriteDelayedBlock(DelayedWriteBlock &block) override;
  /** Returns the number of bytes written so far. */
  virtual int64_t size() const override { return write_pos_; }
  /** Returns a pointer to the beginning of the borrowed buffer. */
  virtual const uint8_t *data() const override { return data_; }

protected:
  /** Non-owning pointer to writable backing storage. */
  uint8_t *data_;
  /** Maximum number of writable bytes in data_. */
  int64_t capacity_;
};

/** Binary writer backed by externally provided memory.
 *  Wraps a caller-owned byte range and exposes the BinaryWriteStream interface
 *  without owning or copying the underlying storage. */
class BorrowedWriteStream : public BinaryWriteStream {
public:
  /** Initializes a write stream that borrows *size* bytes starting at *data*.
   *  The caller must ensure the buffer outlives this stream. */
  explicit inline BorrowedWriteStream(const uint8_t *data, int64_t size)
      : BinaryWriteStream(), data_(data), size_(size) {}
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override;
  /** Returns the number of bytes in the borrowed buffer. */
  virtual int64_t size() const override { return size_; }
  /** Returns a pointer to the beginning of the borrowed buffer. */
  virtual const uint8_t *data() const override { return data_; }

protected:
  /** Non-owning pointer to the backing byte buffer. */
  const uint8_t *data_;
  /** Total number of bytes in the borrowed buffer. */
  int64_t size_;
};

////////
// files
////////

/** Binary writer that persists bytes to a file.
 *  Uses an internal std::ofstream with an optional 4096-byte write buffer.
 *  Also supports parallel offset-based writes via an internal thread pool
 *  after the file has been pre-allocated with pre_allocate(). */
class FileWriteStream : public BinaryWriteStream {
public:
  /** Opens the file at *file_path* for writing (creates or truncates). */
  explicit FileWriteStream(const std::string &file_path);
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override;
  /** Returns the number of bytes written so far. */
  virtual int64_t size() const override;
  /** Returns nullptr; file-backed streams do not expose an in-memory buffer. */
  virtual const uint8_t *data() const override;
  /** Returns the path of the file this stream writes to. */
  inline const std::string &file_path() const { return file_path_; }
  /** Returns true once StartThreadPool() has been called and is still active. */
  virtual bool HasParallelizationStarted() const override { return thread_pool_.IsStarted(); }
  virtual void StartThreadPool(size_t n_threads) override;
  virtual void WriteDelayedBlock(DelayedWriteBlock &block) override;
  virtual void WaitForDelayedBlock() override;
  /** Pre-allocates the file to *total_bytes* by seeking and writing a zero at the last position.
   *  Flushes immediately so the ofstream buffer is clear before parallel tasks write concurrently.
   */
  void pre_allocate(int64_t total_bytes);

protected:
  /** Absolute path of the destination file. */
  std::string file_path_;
  /** Underlying output stream used for sequential writes. */
  std::ofstream file_stream_;
  /** Running count of bytes written to the file. */
  uint64_t written_bytes_;
  /** Thread pool used for parallel block writes. */
  ThreadPool thread_pool_;
};

class TwoFilesStream;

/** Binary reader that streams bytes from a file.
 *  Uses a 4096-byte read-ahead buffer (read_buf_) so that sequential varint
 *  decoding does not issue a system call per byte.  Supports optional parallel
 *  block reads via an internal thread pool. */
class FileStream : public BinaryStream {
  friend class TwoFilesStream;

public:
  /** Opens the file at *file_path* for reading. */
  explicit FileStream(const std::string &file_path);
  virtual ~FileStream();
  /** Returns the path of the file this stream reads from. */
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
  /** Returns the current byte offset from the beginning of the file. */
  virtual offset_t tell() const override;
  virtual std::string tell_around() const override;
  /** Returns true if the underlying file stream is open. */
  virtual bool is_open() const;
  /** Returns the total number of bytes in the file. */
  virtual int64_t size() const override { return size_; }

  // parallelization of big blocks.
  /** Returns true once StartThreadPool() has been called and is still active. */
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
  /** Prevents concurrent access while a sub-stream is locked to a region. */
  bool lock_;
  /** Absolute path of the source file. */
  std::string file_path_;
  /** Underlying input stream used for file I/O. */
  std::ifstream file_stream_;
#if !defined(_WIN32)
  /** POSIX file descriptor used for pread-based parallel reads on non-Windows platforms. */
  int file_descriptor_ = -1;
#endif
  /** Total size of the file in bytes. */
  int64_t size_;
  /** Scratch buffer used to hold bytes read for a single non-buffered read_bytes() call. */
  std::vector<uint8_t> buffer_;
  // parallelization
  /** List of blocks scheduled for parallel asynchronous processing. */
  std::vector<DelayedBlock> blocks_;
  /** Thread pool used for parallel block reads. */
  ThreadPool thread_pool_;

  // Read-ahead buffer for fast sequential varint/byte parsing.
  // Buffers up to READ_BUF_SIZE bytes so that next_uint64() can
  // read single bytes without calling file_stream_.read() each time.
  static constexpr size_t READ_BUF_SIZE = 4096;
  /** Circular read-ahead buffer of up to READ_BUF_SIZE bytes. */
  std::vector<uint8_t> read_buf_;
  /** Current read index within read_buf_. */
  size_t read_buf_pos_ = 0;
  /** One-past-last valid index within read_buf_. */
  size_t read_buf_end_ = 0;
};

//////////////////////////////
// Stream for external weights
//////////////////////////////

/** Two-file writer for external ONNX tensor data.
 *  Buffers the model protobuf structure in memory (main_buf_) and flushes it
 *  to the primary file in a single write via FlushMainToFile().  Tensor weight
 *  data is streamed directly to a separate weights file.  This separates the
 *  two I/O streams so the OS can coalesce writes efficiently and avoids the
 *  overhead of many small writes to the main file.
 *  Supports parallel offset-based writes to the weights file via
 *  StartWriteThreadPool(). */
class TwoFilesWriteStream : public FileWriteStream {
public:
  /** Opens *file_path* for protobuf data and *weights_file* for weight data. */
  explicit TwoFilesWriteStream(const std::string &file_path, const std::string &weights_file);
  /** Returns the path of the separate weights file. */
  inline const std::string &weights_file_path() const { return weights_stream_.file_path(); }
  /** Selects the active external weights location for subsequent tensor raw-data writes. */
  void set_active_weights_location(const std::string &location);
  /** Returns true; this stream routes tensor data to a separate weights file. */
  virtual bool ExternalWeights() const override { return true; }
  /** Appends *n_bytes* bytes starting at *data* to the weights file. */
  virtual void write_raw_bytes_in_second_stream(const uint8_t *data, offset_t n_bytes) override;
  /** Appends *n_bytes* bytes to the weights file designated by *location*. */
  virtual void write_raw_bytes_in_second_stream(const uint8_t *data, offset_t n_bytes,
                                                const std::string &location) override;
  /** Returns the number of bytes written to the weights file so far. */
  virtual int64_t weights_size() const override;
  /** Returns the number of bytes written so far for one weights location. */
  int64_t weights_size(const std::string &location) const;
  /** Returns the number of bytes written so far for one weights location. */
  virtual int64_t weights_size_for_location(const std::string &location) const override;

  /** Pre-allocates the weights file to *total_bytes* by writing a zero at the last position.
   *  Must be called before StartWriteThreadPool. */
  void pre_allocate_weights(int64_t total_bytes);

  /** Starts a thread pool of *n_threads* workers for parallel offset-based writes.
   *  After this call write_raw_bytes_in_second_stream submits writes asynchronously. */
  void StartWriteThreadPool(int32_t n_threads);

  /** Blocks until all pending write tasks have completed and stops the thread pool. */
  void WaitForWriteCompletion();

  /** Flushes the in-memory main-file buffer to disk in a single write.
   *  Must be called after all serialization is complete (after WaitForWriteCompletion). */
  void FlushMainToFile();

  // Redirect main-content writes to the in-memory buffer so the two file
  // streams are kept separate and can be flushed independently.

  /** Writes *n_bytes* bytes from *data* into the in-memory main-content buffer.
   *  Overrides the base file write so that the two output files are kept
   *  separate and can be flushed independently. */
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override;
  /** Returns the number of bytes accumulated in the main-content buffer so far. */
  virtual int64_t size() const override;
  /** Does nothing: the main-content buffer grows dynamically and does not need
   *  thread-pool parallelism because all large tensor writes go to the weights
   *  stream. */
  virtual void StartThreadPool(size_t) override {}
  /** Returns false because no parallel writes are submitted for the main-content buffer. */
  virtual bool HasParallelizationStarted() const override { return false; }
  /** Throws unconditionally: main-content writes are sequential and this path is unreachable. */
  virtual void WriteDelayedBlock(DelayedWriteBlock &block) override;
  /** Does nothing because no delayed writes are outstanding for the main-content buffer. */
  virtual void WaitForDelayedBlock() override {}

protected:
  /** In-memory buffer that accumulates the main .onnx structure bytes.
   *  Flushed to the primary file in one shot by FlushMainToFile(). */
  StringWriteStream main_buf_;
  /** Writer for the separate weights file. */
  FileWriteStream weights_stream_;
  /** Additional writers when external_data.location points to multiple files. */
  std::unordered_map<std::string, std::unique_ptr<FileWriteStream>> extra_weights_streams_;
  /** Active external location used by the current tensor raw-data write. */
  std::string active_weights_location_;
  /** Relative location key associated with the default weights stream. */
  std::string default_weights_location_;

  // Parallel-write state
  /** Set to true once StartWriteThreadPool() has been called. */
  bool parallel_write_ = false;
  /** Tracks the sequential write position for offset validation during parallel writes. */
  int64_t virtual_write_pos_ = 0;
  /** Thread pool used for parallel writes to the weights file. */
  ThreadPool write_thread_pool_;
};

/** Two-file reader for ONNX models with external tensor data.
 *  Reads the model protobuf from a primary file (inherited from FileStream)
 *  and tensor weight data from a separate weights file on demand. */
class TwoFilesStream : public FileStream {
public:
  /** Opens *file_path* for protobuf data and *weights_file* for weight data. */
  explicit TwoFilesStream(const std::string &file_path, const std::string &weights_file);
  /** Returns the path of the separate weights file. */
  inline const std::string &weights_file_path() const { return weights_stream_.file_path(); }
  /** Selects the active external weights location for subsequent reads. */
  void set_active_weights_location(const std::string &location);
  /** Returns true when the active location is the default weights file. */
  bool using_default_weights_location() const;
  /** Returns the current byte offset within the weights file. */
  inline uint64_t weights_tell() const { return weights_stream_.tell(); }
  /** Returns true; this stream reads tensor data from a separate weights file. */
  virtual bool ExternalWeights() const override { return true; }
  /** Reads *n_bytes* from the weights file at the given *offset* (or sequentially if -1)
   *  into *pre_allocated_buffer*. */
  virtual void read_bytes_from_weights_stream(offset_t n_bytes,
                                              uint8_t *pre_allocated_buffer = nullptr,
                                              offset_t offset = -1);
  virtual void ReadDelayedBlock(DelayedBlock &block) override;
  /** Returns the total number of bytes in the weights file. */
  virtual int64_t weights_size() const { return weights_stream_.size(); }
  /** Returns the total number of bytes in one weights file. */
  int64_t weights_size(const std::string &location) const;

protected:
  FileStream &active_weights_stream();
  const FileStream &active_weights_stream() const;
  /** Reader for the separate weights file. */
  FileStream weights_stream_;
  /** Additional readers when external_data.location points to multiple files. */
  std::unordered_map<std::string, std::unique_ptr<FileStream>> extra_weights_streams_;
  /** Active external location used by the current tensor read. */
  std::string active_weights_location_;
  /** Relative location key associated with the default weights stream. */
  std::string default_weights_location_;
  /** Maps object pointers to their byte offsets in the weights file. */
  std::unordered_map<const void *, uint64_t> position_cache_;
};

} // namespace utils
} // namespace ONNX_LIGHT_NAMESPACE
