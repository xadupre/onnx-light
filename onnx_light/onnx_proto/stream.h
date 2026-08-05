#pragma once

#include "onnx_light_helpers.h"
#include "simple_string.h"
#include "thread_pool.h"
#include <cstddef>
#include <cstring>
#include <fstream>
#include <istream>
#include <iterator>
#include <memory>
#include <ostream>
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
inline int64_t decodeZigZag64(uint64_t n) {
  return static_cast<int64_t>(n >> 1) ^ -static_cast<int64_t>(n & 1);
}

/** Encodes a signed 64-bit integer using ZigZag encoding.
 *  The resulting unsigned value is suitable for efficient varint serialization
 *  of values that are likely to be small in magnitude. */
inline uint64_t encodeZigZag64(int64_t n) {
  return (static_cast<uint64_t>(n) << 1) ^ static_cast<uint64_t>(n >> 63);
}

/** Maps an entire file into read-only virtual memory and returns a shared_ptr<uint8_t>
 *  whose deleter unmaps the region. On POSIX, mmap(MAP_PRIVATE|PROT_READ) is used;
 *  on Windows, CreateFileMapping + MapViewOfFile.
 *  Returns an empty shared_ptr when *file_size* is 0.
 *  The mapped base address is page-aligned and therefore satisfies any typical
 *  tensor alignment requirement (16 / 32 / 64 bytes) when combined with an
 *  aligned file offset. */
std::shared_ptr<uint8_t> mmap_file_as_shared_ptr(const std::string &file_path, int64_t file_size);

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

  // protobuf ZeroCopyInputStream-compatible interface. These mirror the methods
  // of google::protobuf::io::ArrayInputStream (see ArrayInputStream in
  // google_protobuf_compat.h) so a BinaryStream can be consumed anywhere a
  // ZeroCopyInputStream is expected. They are implemented on top of the
  // primitives above (tell/size/read_bytes) and require a no-copy stream
  // (CanNoCopy() == true) because Next() hands out a pointer into the buffer.
  /** Returns the next contiguous block of data and advances past it.
   *  Sets the data pointer and size to the block and returns true, or returns
   *  false at end. */
  virtual bool Next(const void **data, int *size);
  /** Pushes the last *count* bytes returned by Next() back so the following
   *  Next() call serves them again. *count* must not exceed the last block. */
  virtual void BackUp(int count);
  /** Returns the number of bytes consumed so far (logical read position). */
  virtual int64_t ByteCount() const;

protected:
  /** Sets the internal read limit to *len* bytes from the current position. */
  virtual void LimitTo(uint64_t len) = 0;
  /** Validates the internal limit stack; throws if the state is inconsistent. */
  virtual void _check();
  /** Stack of absolute byte offsets used to implement nested LimitToNext/Restore pairs. */
  std::vector<uint64_t> limits_;
  /** Pointer to the block last returned by Next(); used to implement BackUp(). */
  const uint8_t *last_next_data_ = nullptr;
  /** Size of the block last returned by Next(); used to bound BackUp(). */
  int64_t last_next_size_ = 0;
  /** Number of bytes pushed back by BackUp() and pending re-delivery by Next(). */
  int64_t backed_up_ = 0;
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

  // protobuf ZeroCopyOutputStream-compatible interface. These mirror the methods
  // of google::protobuf::io::StringOutputStream (see StringOutputStream in
  // google_protobuf_compat.h) so a BinaryWriteStream can be used anywhere a
  // ZeroCopyOutputStream is expected. Only growable in-memory streams
  // (StringWriteStream / BorrowedStringWriteStream) implement Next()/BackUp();
  // other backends throw.
  /** Hands out a writable block and advances the write position past it.
   *  Sets the data pointer and size to the block and returns true, or returns
   *  false when no space is available. Bytes left unwritten must be returned via
   *  BackUp(). */
  virtual bool Next(void **data, int *size) {
    (void)data;
    (void)size;
    EXT_THROW("BinaryWriteStream::Next is only supported by in-memory write streams.");
  }
  /** Returns the last *count* bytes handed out by Next() that were not used. */
  virtual void BackUp(int count) {
    (void)count;
    EXT_THROW("BinaryWriteStream::BackUp is only supported by in-memory write streams.");
  }
  /** Returns the number of bytes produced so far (logical write position). */
  virtual int64_t ByteCount() const { return size(); }

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
  /** Initializes a stream that reads *size* bytes starting at *data*.
   *  Accepts protobuf-style const void* buffers so StringStream can serve as the
   *  google::protobuf::io::ArrayInputStream alias. */
  explicit inline StringStream(const void *data, int64_t size)
      : BinaryStream(), pos_(0), size_(size), data_(static_cast<const uint8_t *>(data)) {}
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

/** Binary reader backed by a memory-mapped file.
 *  Inherits all in-memory parsing fast paths from StringStream and adds
 *  ownership of the mmap region via a shared_ptr so the mapping stays alive
 *  as long as this stream (or any borrowed ByteSpan that captured the owner)
 *  is in use.  Use this in place of FileStream for single-file loads to avoid
 *  the double-buffered ifstream-then-read_buf path: the OS page cache is
 *  exposed directly as contiguous memory, eliminating per-byte std::ifstream
 *  bookkeeping and the seek-to-invalidate sequences triggered by large
 *  payload reads. */
class MmapFileStream : public StringStream {
public:
  /** Maps the file at *file_path* and exposes its contents as a binary stream. */
  explicit MmapFileStream(const std::string &file_path);
  /** Returns the path of the mapped file. */
  inline const std::string &file_path() const { return file_path_; }
  /** Returns the shared_ptr that keeps the mmap region alive.
   *  Borrowed ByteSpans can capture this pointer so the mapping survives the
   *  stream's destruction, mirroring how TwoFilesStream tracks external
   *  weights buffers. */
  inline const std::shared_ptr<uint8_t> &mmap_owner() const { return mmap_; }

protected:
  /** Absolute path of the mapped file. */
  std::string file_path_;
  /** Shared ownership of the mmap region; munmap fires when the last reference dies. */
  std::shared_ptr<uint8_t> mmap_;
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

  // ZeroCopyOutputStream-compatible interface (see BinaryWriteStream).
  /** Grows the buffer as needed and returns a writable block past the current
   *  write position; advances the write position to the end of that block. */
  virtual bool Next(void **data, int *size) override;
  /** Rewinds the write position by *count* bytes handed out by the last Next(). */
  virtual void BackUp(int count) override;

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

  /** Returns a writable block from the remaining borrowed capacity without
   *  growing; returns false once the fixed buffer is exhausted. */
  virtual bool Next(void **data, int *size) override;

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

  /** Pre-allocates the weights file associated with *location* to *total_bytes*.
   *  If *location* refers to the default weights file, behaves like
   *  ``pre_allocate_weights(total_bytes)``.  Otherwise creates the matching extra
   *  ``FileWriteStream`` (if not already created) and pre-allocates it.
   *  Must be called before StartWriteThreadPool. */
  void pre_allocate_weights(const std::string &location, int64_t total_bytes);

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
  /** Per-location virtual write positions for parallel writes to extra weights files.
   *  Key is the relative location string (the same value used in external_data.location). */
  std::unordered_map<std::string, int64_t> extra_virtual_write_pos_;
  /** Thread pool used for parallel writes to the weights file. */
  ThreadPool write_thread_pool_;
};

/** Two-file reader for ONNX models with external tensor data.
 *  Reads the model protobuf from a primary file (inherited from FileStream)
 *  and tensor weight data from a separate weights file on demand. */
class TwoFilesStream : public FileStream {
public:
  struct SharedWeightsBuffer {
    std::shared_ptr<uint8_t> data;
    int64_t size = 0;
    size_t alignment = 0;
  };

  /** Opens *file_path* for protobuf data and *weights_file* for weight data. */
  explicit TwoFilesStream(const std::string &file_path, const std::string &weights_file);
  /** Returns the path of the separate weights file. */
  inline const std::string &weights_file_path() const { return weights_stream_.file_path(); }
  /** Selects the active external weights location for subsequent reads. */
  void set_active_weights_location(const std::string &location);
  /** Requests that copying reads of the weights file (the ``no_copy=false`` path) source their
   *  bytes from a memory-mapped view of each weights file instead of a buffered ``std::ifstream``.
   *  This lets ``file_load_mode=MMAP`` memory-map the weights file even when ``no_copy`` is not
   *  set, in which case the mapped bytes are copied into owned per-tensor buffers. */
  inline void set_use_mmap_weights(bool value) { use_mmap_weights_ = value; }
  /** Returns true when copying reads source their bytes from a memory-mapped weights view. */
  inline bool use_mmap_weights() const { return use_mmap_weights_; }
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
  /** Loads one external weights file once, then returns a shared borrowed view into it. */
  const uint8_t *borrow_weights_bytes(const std::string &location, offset_t offset,
                                      offset_t n_bytes, size_t alignment,
                                      std::shared_ptr<void> &owner);
  virtual void ReadDelayedBlock(DelayedBlock &block) override;
  /** Returns the total number of bytes in the weights file. */
  virtual int64_t weights_size() const { return weights_stream_.size(); }
  /** Returns the total number of bytes in one weights file. */
  int64_t weights_size(const std::string &location) const;

protected:
  FileStream &active_weights_stream();
  const FileStream &active_weights_stream() const;
  SharedWeightsBuffer &ensure_shared_weights_buffer(const std::string &location, size_t alignment);
  /** Reader for the separate weights file. */
  FileStream weights_stream_;
  /** Additional readers when external_data.location points to multiple files. */
  std::unordered_map<std::string, std::unique_ptr<FileStream>> extra_weights_streams_;
  /** Shared file buffers used by no-copy external-data loading. */
  std::unordered_map<std::string, SharedWeightsBuffer> shared_weights_buffers_;
  /** Active external location used by the current tensor read. */
  std::string active_weights_location_;
  /** Relative location key associated with the default weights stream. */
  std::string default_weights_location_;
  /** Maps object pointers to their byte offsets in the weights file. */
  std::unordered_map<const void *, uint64_t> position_cache_;
  /** When true, copying reads of the weights file source their bytes from a memory-mapped
   *  view (see set_use_mmap_weights). Set once during construction; not meant to change
   *  after reading has started. */
  bool use_mmap_weights_;
};

///////////////////////////////////////////////////////////////////////////////
// protobuf ZeroCopy adapter streams
///////////////////////////////////////////////////////////////////////////////

/** In-memory write stream whose bytes are appended to a caller-owned std::string.
 *  Implements the full BinaryWriteStream interface plus the protobuf
 *  ZeroCopyOutputStream methods (Next / BackUp / ByteCount), so it can back
 *  google::protobuf::io::StringOutputStream as a pure alias. */
class StdStringWriteStream : public BinaryWriteStream {
public:
  /** Initializes a write stream that appends to *target*.
   *  The caller must ensure *target* outlives this stream. */
  explicit inline StdStringWriteStream(std::string *target)
      : BinaryWriteStream(), target_(target) {}
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override {
    target_->append(reinterpret_cast<const char *>(data), static_cast<size_t>(n_bytes));
  }
  /** Returns the number of bytes currently held by the target string. */
  virtual int64_t size() const override { return static_cast<int64_t>(target_->size()); }
  /** Returns a pointer to the beginning of the target string's storage. */
  virtual const uint8_t *data() const override {
    return reinterpret_cast<const uint8_t *>(target_->data());
  }

  // ZeroCopyOutputStream-compatible interface (see BinaryWriteStream).
  /** Grows the target string by a chunk and returns a writable block at its
   *  former end; the string size is advanced past that block. */
  virtual bool Next(void **data, int *size) override {
    size_t old_size = target_->size();
    size_t new_size = old_size + 1024;
    target_->resize(new_size);
    *data = &(*target_)[old_size];
    *size = static_cast<int>(new_size - old_size);
    return true;
  }
  /** Trims the last *count* bytes handed out by the previous Next(). */
  virtual void BackUp(int count) override {
    target_->resize(target_->size() - static_cast<size_t>(count));
  }
  /** Returns the number of bytes produced so far (target string size). */
  virtual int64_t ByteCount() const override { return static_cast<int64_t>(target_->size()); }

protected:
  /** Non-owning pointer to the destination string. */
  std::string *target_;
};

/** In-memory reader that owns a full copy of a std::istream's contents.
 *  The whole stream is drained into an owned std::string in the constructor and
 *  exposed through StringStream's zero-copy read interface, so it can back
 *  google::protobuf::io::IstreamInputStream as a pure alias. */
class IstreamStream : public StringStream {
public:
  /** Reads the entire contents of *stream* into an owned buffer.
   *  *block_size* is accepted for protobuf API compatibility but ignored. */
  explicit inline IstreamStream(std::istream *stream, int block_size = 4096) : StringStream() {
    (void)block_size;
    owned_.assign(std::istreambuf_iterator<char>(*stream), std::istreambuf_iterator<char>());
    Setup(reinterpret_cast<const uint8_t *>(owned_.data()), static_cast<int64_t>(owned_.size()));
  }

protected:
  /** Owned copy of the source stream's bytes; keeps data_ valid for the stream's life. */
  std::string owned_;
};

/** Write stream that forwards its bytes to a std::ostream.
 *  Implements the BinaryWriteStream interface plus the protobuf
 *  ZeroCopyOutputStream methods (Next / BackUp / ByteCount / Flush), so it can
 *  back google::protobuf::io::OstreamOutputStream as a pure alias. */
class OstreamWriteStream : public BinaryWriteStream {
public:
  /** Initializes a write stream that writes to *stream* using *block_size* chunks. */
  explicit inline OstreamWriteStream(std::ostream *stream, int block_size = 4096)
      : BinaryWriteStream(), stream_(stream), block_size_(block_size),
        buffer_(static_cast<size_t>(block_size)), used_(0), written_(0) {}
  inline ~OstreamWriteStream() override { Flush(); }
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override {
    Flush();
    stream_->write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(n_bytes));
    written_ += static_cast<int64_t>(n_bytes);
  }
  /** Returns the number of bytes produced so far (flushed plus pending). */
  virtual int64_t size() const override { return written_ + static_cast<int64_t>(used_); }
  /** Returns nullptr; ostream-backed streams do not expose an in-memory buffer. */
  virtual const uint8_t *data() const override { return nullptr; }

  // ZeroCopyOutputStream-compatible interface (see BinaryWriteStream).
  /** Flushes any pending bytes and hands out the internal block buffer. */
  virtual bool Next(void **data, int *size) override {
    Flush();
    *data = buffer_.data();
    *size = block_size_;
    used_ = block_size_;
    return true;
  }
  /** Returns the last *count* bytes handed out by Next() that were not written. */
  virtual void BackUp(int count) override { used_ -= count; }
  /** Returns the number of bytes produced so far (flushed plus pending). */
  virtual int64_t ByteCount() const override { return written_ + static_cast<int64_t>(used_); }
  /** Writes any pending buffered bytes to the underlying ostream. */
  inline bool Flush() {
    if (used_ > 0) {
      stream_->write(buffer_.data(), used_);
      written_ += static_cast<int64_t>(used_);
      used_ = 0;
    }
    return stream_->good();
  }

protected:
  /** Non-owning pointer to the destination stream. */
  std::ostream *stream_;
  /** Size of each block handed out by Next(). */
  int block_size_;
  /** Scratch buffer backing the current Next() block. */
  std::vector<char> buffer_;
  /** Number of valid bytes currently pending in buffer_. */
  int used_;
  /** Number of bytes already flushed to the ostream. */
  int64_t written_;
};

/** Write stream that forwards its bytes to a raw file descriptor.
 *  Implements the BinaryWriteStream interface plus the protobuf
 *  ZeroCopyOutputStream methods (Next / BackUp / ByteCount / Flush / Close), so
 *  it can back google::protobuf::io::FileOutputStream as a pure alias.
 *  Unlike the previous compat stub this performs real platform writes. */
class FdWriteStream : public BinaryWriteStream {
public:
  /** Initializes a write stream over the open file descriptor *fd*. */
  explicit inline FdWriteStream(int fd) : BinaryWriteStream(), fd_(fd), used_(0), written_(0) {}
  /** Initializes a write stream over *fd* with an ignored block-size hint
   *  (kept for protobuf API compatibility with FileOutputStream). */
  inline FdWriteStream(int fd, int /*block_size*/) : FdWriteStream(fd) {}
  inline ~FdWriteStream() override { Flush(); }
  virtual void write_raw_bytes(const uint8_t *data, offset_t n_bytes) override;
  /** Returns the number of bytes produced so far (written plus pending). */
  virtual int64_t size() const override { return written_ + static_cast<int64_t>(used_); }
  /** Returns nullptr; fd-backed streams do not expose an in-memory buffer. */
  virtual const uint8_t *data() const override { return nullptr; }

  // ZeroCopyOutputStream-compatible interface (see BinaryWriteStream).
  /** Flushes any pending bytes and hands out the internal block buffer. */
  virtual bool Next(void **data, int *size) override {
    Flush();
    *data = buffer_;
    *size = static_cast<int>(sizeof(buffer_));
    used_ = sizeof(buffer_);
    return true;
  }
  /** Returns the last *count* bytes handed out by Next() that were not written. */
  virtual void BackUp(int count) override { used_ -= static_cast<size_t>(count); }
  /** Returns the number of bytes produced so far (written plus pending). */
  virtual int64_t ByteCount() const override { return written_ + static_cast<int64_t>(used_); }
  /** Writes any pending buffered bytes to the file descriptor. */
  bool Flush();
  /** Flushes and returns true; the descriptor itself is not closed. */
  inline bool Close() { return Flush(); }
  /** No-op for protobuf FileOutputStream compatibility. */
  inline void SetCloseOnDelete(bool) {}
  /** Returns 0; onnx-light throws on I/O errors rather than setting errno. */
  inline int GetErrno() const { return 0; }

protected:
  /** File descriptor the bytes are written to (not owned). */
  int fd_;
  /** Scratch buffer backing the current Next() block. */
  char buffer_[4096];
  /** Number of valid bytes currently pending in buffer_. */
  size_t used_;
  /** Number of bytes already written to the descriptor. */
  int64_t written_;
};

/** Read stream that reads bytes from a raw file descriptor.
 *  Provides the ZeroCopyInputStream-compatible interface so it can back
 *  google::protobuf::io::FileInputStream as a pure alias. */
/** Zero-copy input stream over a seekable file descriptor that also implements
 *  the full BinaryStream interface.  This allows it to be passed directly to
 *  ParseFromZeroCopyStream(BinaryStream*) without loading the entire file into
 *  memory first.  The file descriptor must support lseek (regular files);
 *  non-seekable descriptors fall back to an INT64_MAX limit and rely on the
 *  eof_ flag for end-of-stream detection. */
class FdReadStream : public BinaryStream {
public:
  /** Initializes a read stream over the open file descriptor *fd*.
   *  \p block_size is the internal buffer size used for reads. */
  explicit FdReadStream(int fd, int block_size = 4096);
  ~FdReadStream() override;

  // --- BinaryStream pure virtual overrides ---

  /** Decodes the next base-128 varint as an unsigned 64-bit integer. */
  virtual uint64_t next_uint64() override;
  /** Raises an exception if fewer than *len* bytes remain before the current limit. */
  virtual void CanRead(uint64_t len, const char *msg) override;
  /** Returns true while the logical read position is before the active limit. */
  virtual bool NotEnd() const override { return !eof_ && pos_ < limit_; }
  /** Returns the current byte offset from stream construction (bytes consumed). */
  virtual offset_t tell() const override { return pos_; }
  /** Returns the effective end of the stream (file size or active sub-limit). */
  virtual int64_t size() const override { return limit_; }
  /** Returns a short human-readable excerpt around the current position. */
  virtual std::string tell_around() const override;
  /** Reads *n_bytes* into *pre_allocated_buffer*.
   *  Zero-copy mode (pre_allocated_buffer == nullptr) is not supported for fd streams;
   *  always provide a destination buffer. */
  virtual const uint8_t *read_bytes(offset_t n_bytes, uint8_t *pre_allocated_buffer) override;
  /** Advances the read position by *n_bytes* using lseek. */
  virtual void skip_bytes(offset_t n_bytes) override;
  /** Returns false: fd streams do not support zero-copy pointer-into-buffer mode. */
  virtual bool CanNoCopy() const override { return false; }
  /** Returns the number of bytes consumed up to the start of the current block. */
  virtual int64_t ByteCount() const override {
    return total_read_ - static_cast<int64_t>(available_);
  }

  // --- ZeroCopyInputStream (override BinaryStream::Next / BackUp) ---

  /** Hands out a pointer into the next chunk of unread data.
   *  Returns false when EOF or an error is reached. */
  virtual bool Next(const void **data, int *size) override;
  /** Returns the last *count* bytes handed out by Next() that were not consumed. */
  virtual void BackUp(int count) override;

  // --- protobuf compat helpers ---

  /** Skips forward by *count* bytes.  Returns false if EOF is reached first. */
  bool Skip(int count);
  /** No-op: onnx-light does not close the fd on delete. */
  inline void SetCloseOnDelete(bool) {}
  /** No-op placeholder; the fd is managed by the caller. */
  inline bool Close() { return true; }
  /** Returns 0; onnx-light throws on I/O errors rather than setting errno. */
  inline int GetErrno() const { return 0; }

protected:
  virtual void LimitTo(uint64_t len) override { limit_ = static_cast<int64_t>(len); }

  /** Computes the remaining byte count from the current fd position to the end
   *  of the file via lseek.  Returns INT64_MAX when lseek is unavailable
   *  (non-regular files such as pipes or sockets). */
  static int64_t _InitLimit(int fd) noexcept;

  int fd_;
  int block_size_;
  char *buffer_;
  int available_;
  int position_;
  int64_t total_read_;
  /** Logical read position for the BinaryStream interface (updated by read_bytes/skip_bytes). */
  int64_t pos_;
  /** Effective end position: file-remaining-bytes at construction, or an active sub-limit. */
  int64_t limit_;
  /** Set to true by Next() when ::read() returns ≤ 0 (end-of-file or error). */
  bool eof_;
};

/** Minimal coded input stream wrapping a BinaryStream, providing the protobuf
 *  total-bytes-limit accessors so it can back google::protobuf::io::CodedInputStream.
 *  Accepts any BinaryStream subclass (StringStream, FdReadStream, etc.). */
class CodedInputStream {
public:
  /** Opaque type returned by PushLimit / consumed by PopLimit. */
  using Limit = int;

  /** Wraps *input* with a default total-bytes limit of INT32_MAX. */
  explicit inline CodedInputStream(BinaryStream *input)
      : input_(input), limit_(0x7FFFFFFF), position_(0) {}
  /** Sets the maximum number of bytes that may be read. */
  inline void SetTotalBytesLimit(int total_bytes_limit) { limit_ = total_bytes_limit; }
  /** Returns the configured total-bytes limit. */
  inline int TotalBytesLimit() const { return limit_; }
  /** Returns the current read position (bytes consumed so far). */
  inline int CurrentPosition() const { return position_; }

  /** Reads a varint32 from the underlying stream.  Returns true on success. */
  inline bool ReadVarint32(uint32_t *value) {
    uint32_t result = 0;
    int shift = 0;
    for (;;) {
      uint8_t byte = 0;
      if (!ReadRaw(&byte, 1))
        return false;
      result |= static_cast<uint32_t>(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0)
        break;
      shift += 7;
      if (shift >= 35)
        return false;
    }
    *value = result;
    return true;
  }

  /** Reads exactly *size* bytes into *buffer*.  Returns true on success. */
  inline bool ReadRaw(void *buffer, int size) {
    if (size <= 0)
      return size == 0;
    if (!input_)
      return false;
    const void *data = nullptr;
    int avail = 0;
    auto *dst = static_cast<char *>(buffer);
    int remaining = size;
    while (remaining > 0) {
      if (!input_->Next(&data, &avail))
        return false;
      int to_copy = (avail < remaining) ? avail : remaining;
      std::memcpy(dst, data, static_cast<size_t>(to_copy));
      dst += to_copy;
      remaining -= to_copy;
      if (to_copy < avail)
        input_->BackUp(avail - to_copy);
    }
    position_ += size;
    return true;
  }

  /** Skips *size* bytes.  Returns true on success. */
  inline bool Skip(int size) {
    if (size <= 0)
      return size == 0;
    if (!input_)
      return false;
    // Consume via Next/BackUp to stay compatible with all BinaryStream subclasses.
    int remaining = size;
    while (remaining > 0) {
      const void *data = nullptr;
      int avail = 0;
      if (!input_->Next(&data, &avail))
        return false;
      int consumed = (avail < remaining) ? avail : remaining;
      remaining -= consumed;
      if (consumed < avail)
        input_->BackUp(avail - consumed);
    }
    position_ += size;
    return true;
  }

  /** Saves the current limit and installs a new one.
   *  The new limit is CurrentPosition() + *byte_limit*.  Returns the previous limit. */
  inline Limit PushLimit(int byte_limit) {
    int old = limit_;
    limit_ = position_ + byte_limit;
    return old;
  }

  /** Restores a previous limit saved by PushLimit(). */
  inline void PopLimit(Limit old_limit) { limit_ = old_limit; }

  /** Returns true if the entire stream has been consumed (stub: always true). */
  inline bool ConsumedEntireMessage() const { return true; }

protected:
  /** Non-owning pointer to the wrapped input stream. */
  BinaryStream *input_;
  /** Configured maximum number of readable bytes. */
  int limit_;
  /** Current read position (bytes consumed). */
  int position_;
};

} // namespace utils
} // namespace ONNX_LIGHT_NAMESPACE
