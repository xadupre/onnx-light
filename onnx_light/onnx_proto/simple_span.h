#pragma once

#include "onnx_light_helpers.h"
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdint.h>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace utils {

/**
 * Non-owning, read-only view of a contiguous byte sequence.
 * It references existing memory and never allocates or frees storage.
 * Analogous to RefString but for raw bytes.
 */
class Span {
protected:
  const uint8_t *ptr_ = nullptr;
  size_t size_ = 0;

public:
  /** Initializes an empty span. */
  inline Span() = default;
  /** Initializes a span from a pointer and a size. */
  inline Span(const uint8_t *ptr, size_t size) : ptr_(ptr), size_(size) {}

  /** Returns the number of bytes in the span. */
  inline size_t size() const { return size_; }
  /** Returns a const pointer to the byte data. */
  inline const uint8_t *data() const { return ptr_; }
  /** Returns true when the span covers zero bytes. */
  inline bool empty() const { return size_ == 0; }
  /** Returns a const reference to the byte at index i (no bounds check). */
  inline const uint8_t &operator[](size_t i) const { return ptr_[i]; }

  /** Returns true when both spans have the same size and byte content. */
  inline bool operator==(const Span &other) const {
    if (size_ != other.size_)
      return false;
    return size_ == 0 || std::memcmp(ptr_, other.ptr_, size_) == 0;
  }
  /** Returns true when the spans differ in size or content. */
  inline bool operator!=(const Span &other) const { return !(*this == other); }
};

/**
 * Owns a contiguous byte buffer without value-initializing newly allocated bytes.
 * It preserves the existing prefix on resize and grows geometrically for append-heavy use.
 */
class OwnedByteBuffer {
public:
  /** Initializes an empty buffer. */
  OwnedByteBuffer() = default;

  /** Creates a deep copy of the stored bytes. */
  inline OwnedByteBuffer(const OwnedByteBuffer &other) { assign(other.data(), other.size()); }

  /** Replaces the buffer with a deep copy of the stored bytes. */
  inline OwnedByteBuffer &operator=(const OwnedByteBuffer &other) {
    if (this != &other)
      assign(other.data(), other.size());
    return *this;
  }

  /** Transfers ownership of the stored bytes and empties the source buffer. */
  inline OwnedByteBuffer(OwnedByteBuffer &&other) noexcept
      : storage_(std::move(other.storage_)), size_(other.size_), capacity_(other.capacity_) {
    other.size_ = 0;
    other.capacity_ = 0;
  }

  /** Transfers ownership of the stored bytes and empties the source buffer. */
  inline OwnedByteBuffer &operator=(OwnedByteBuffer &&other) noexcept {
    if (this != &other) {
      storage_ = std::move(other.storage_);
      size_ = other.size_;
      capacity_ = other.capacity_;
      other.size_ = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  /** Creates a buffer by copying bytes from a vector. */
  inline OwnedByteBuffer(const std::vector<uint8_t> &v) { assign(v.data(), v.size()); }

  /** Replaces the buffer content by copying bytes from a vector. */
  inline OwnedByteBuffer &operator=(const std::vector<uint8_t> &v) {
    assign(v.data(), v.size());
    return *this;
  }

  /** Returns the number of stored bytes. */
  inline size_t size() const { return size_; }

  /** Returns a mutable pointer to the stored bytes. */
  inline uint8_t *data() { return storage_.get(); }

  /** Returns a const pointer to the stored bytes. */
  inline const uint8_t *data() const { return storage_.get(); }

  /** Clears the logical content while preserving capacity. */
  inline void clear() { size_ = 0; }

  /** Resizes the buffer without value-initializing newly exposed bytes. */
  inline void resize(size_t n) {
    if (n <= capacity_) {
      size_ = n;
      return;
    }
    reserve(n);
    size_ = n;
  }

  /** Replaces the content by copying raw bytes. */
  inline void assign(const uint8_t *src, size_t n) {
    if (n == 0) {
      clear();
      return;
    }
    std::unique_ptr<uint8_t[]> next(new uint8_t[n]);
    std::memcpy(next.get(), src, n);
    storage_ = std::move(next);
    size_ = n;
    capacity_ = n;
  }

  /** Replaces the content by copying a byte range. */
  inline void assign(const uint8_t *begin, const uint8_t *end) { assign(begin, end - begin); }

  /** Appends one byte, growing geometrically when needed. */
  inline void push_back(uint8_t v) {
    if (size_ == capacity_) {
      size_t next =
          capacity_ == 0 ? 1 : (capacity_ < 4096 ? capacity_ * 2 : capacity_ + capacity_ / 2);
      reserve(next);
    }
    storage_[size_++] = v;
  }

private:
  /** Ensures the buffer can store at least n bytes. */
  inline void reserve(size_t n) {
    if (n <= capacity_)
      return;
    std::unique_ptr<uint8_t[]> next(new uint8_t[n]);
    if (size_ > 0)
      std::memcpy(next.get(), storage_.get(), size_);
    storage_ = std::move(next);
    capacity_ = n;
  }

  std::unique_ptr<uint8_t[]> storage_;
  size_t size_ = 0;
  size_t capacity_ = 0;
};

/**
 * A byte buffer that can either own its data or borrow a non-owning view into
 * an external buffer.  Inherits the non-owning-view interface from Span and
 * overrides all read accessors so they always return correct data in both modes.
 *
 * The borrowed mode is used for zero-copy parsing: when ParseOptions::no_copy
 * is true and the stream supports it, tensor raw data is not copied — instead
 * assign_borrowed() sets the base-class ptr_/size_ to point directly into the
 * source bytes buffer.  The caller MUST keep that buffer alive for as long as
 * the ByteSpan is in borrowed mode.
 *
 * The aligned-owned mode is used when ParseOptions::alignment > 0: the buffer
 * is over-allocated by (alignment - 1) bytes and ptr_/size_ are set to the
 * aligned region within owned_.  In this mode mutable data() returns the
 * aligned pointer and size() returns the logical (not allocated) size.
 *
 * An explicit borrowed_ flag is used to unambiguously track the storage mode,
 * including degenerate edge cases such as a zero-length borrowed span.
 *
 * In plain owned mode the class behaves like a growable byte buffer; all read
 * accessors delegate to owned_.data() / owned_.size() so there is no risk of
 * stale cached pointers when owned_ reallocates.
 */
class ByteSpan : public Span {
public:
  /** Constructs an empty buffer (owned mode, no allocation). */
  ByteSpan() = default;

  /** Constructs an owned buffer by copying from a std::vector<uint8_t>. */
  inline ByteSpan(const std::vector<uint8_t> &v) : owned_(v) {}

  /** Copy constructor: handles aligned-owned mode by recomputing the internal pointer. */
  inline ByteSpan(const ByteSpan &other)
      : Span(nullptr, 0), borrowed_(other.borrowed_), aligned_owned_(other.aligned_owned_),
        align_(other.align_), owner_(other.owner_) {
    if (other.aligned_owned_) {
      // Re-allocate and re-align, then copy only the logical data bytes.
      const size_t n = other.size_;
      owned_.resize(n + align_ - 1);
      void *vptr = owned_.data();
      size_t space = owned_.size();
      void *aligned = std::align(align_, n, vptr, space);
      EXT_ENFORCE(aligned != nullptr, "ByteSpan: copy re-alignment failed.");
      ptr_ = static_cast<uint8_t *>(aligned);
      size_ = n;
      if (n > 0)
        std::memcpy(const_cast<uint8_t *>(ptr_), other.ptr_, n);
    } else {
      owned_ = other.owned_;
      // plain owned: ptr_/size_ unused; borrowed: intentional shallow copy.
      ptr_ = other.ptr_;
      size_ = other.size_;
    }
  }

  /** Copy assignment: handles aligned-owned mode by recomputing the internal pointer. */
  inline ByteSpan &operator=(const ByteSpan &other) {
    if (this != &other) {
      borrowed_ = other.borrowed_;
      aligned_owned_ = other.aligned_owned_;
      align_ = other.align_;
      owner_ = other.owner_;
      if (other.aligned_owned_) {
        const size_t n = other.size_;
        owned_.resize(n + align_ - 1);
        void *vptr = owned_.data();
        size_t space = owned_.size();
        void *aligned = std::align(align_, n, vptr, space);
        EXT_ENFORCE(aligned != nullptr, "ByteSpan: copy assignment re-alignment failed.");
        ptr_ = static_cast<uint8_t *>(aligned);
        size_ = n;
        if (n > 0)
          std::memcpy(const_cast<uint8_t *>(ptr_), other.ptr_, n);
      } else {
        owned_ = other.owned_;
        ptr_ = other.ptr_;
        size_ = other.size_;
      }
    }
    return *this;
  }

  /** Move constructor: for aligned-owned mode the pointer remains valid after the move. */
  inline ByteSpan(ByteSpan &&other) noexcept
      : owned_(std::move(other.owned_)), borrowed_(other.borrowed_),
        aligned_owned_(other.aligned_owned_), align_(other.align_), owner_(std::move(other.owner_)),
        Span(other.ptr_, other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.borrowed_ = false;
    other.aligned_owned_ = false;
    other.align_ = 0;
    other.owner_.reset();
  }

  /** Move assignment: for aligned-owned mode the pointer remains valid after the move. */
  inline ByteSpan &operator=(ByteSpan &&other) noexcept {
    if (this != &other) {
      owned_ = std::move(other.owned_);
      borrowed_ = other.borrowed_;
      aligned_owned_ = other.aligned_owned_;
      align_ = other.align_;
      owner_ = std::move(other.owner_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
      other.borrowed_ = false;
      other.aligned_owned_ = false;
      other.align_ = 0;
      other.owner_.reset();
    }
    return *this;
  }

  /** Assigns owned data by copying from a std::vector<uint8_t>; clears all special modes. */
  inline ByteSpan &operator=(const std::vector<uint8_t> &v) {
    owned_ = v;
    ptr_ = nullptr;
    size_ = 0;
    borrowed_ = false;
    aligned_owned_ = false;
    align_ = 0;
    owner_.reset();
    return *this;
  }

  /** Returns true when the data is borrowed (non-owning). */
  inline bool is_borrowed() const { return borrowed_; }

  /** Returns true when the data is in aligned-owned mode. */
  inline bool is_aligned_owned() const { return aligned_owned_; }

  // --- Overridden read accessors (always use the correct storage source) ---

  /** Returns the number of bytes in either storage mode. */
  inline size_t size() const {
    if (aligned_owned_)
      return size_;
    return borrowed_ ? size_ : owned_.size();
  }
  /** Returns true when no data is stored (zero bytes in either mode). */
  inline bool empty() const { return size() == 0; }
  /** Returns a const pointer to the byte data in either storage mode. */
  inline const uint8_t *data() const {
    if (aligned_owned_)
      return ptr_;
    return borrowed_ ? ptr_ : owned_.data();
  }
  /** Returns a const reference to the byte at index i (no bounds check). */
  inline const uint8_t &operator[](size_t i) const { return data()[i]; }

  /** Returns true when both ByteSpans have the same size and byte content. */
  inline bool operator==(const ByteSpan &other) const {
    const size_t sz = size();
    if (sz != other.size())
      return false;
    return sz == 0 || std::memcmp(data(), other.data(), sz) == 0;
  }
  /** Returns true when the ByteSpans differ in size or content. */
  inline bool operator!=(const ByteSpan &other) const { return !(*this == other); }

  // --- Mutable operations ---

  /** Returns a mutable pointer to the owned data.
   *  Calling this in borrowed mode raises an error at runtime. */
  inline uint8_t *data() {
    EXT_ENFORCE(!borrowed_, "ByteSpan: mutable data() called on a borrowed (zero-copy) buffer; "
                            "use const data() or assign owned data first.");
    if (aligned_owned_)
      return const_cast<uint8_t *>(ptr_);
    return owned_.data();
  }

  /** Returns a mutable reference to the byte at index i; valid only in owned mode.
   *  No bounds check is performed; behaviour is undefined for i >= size(). */
  inline uint8_t &operator[](size_t i) { return data()[i]; }

  /** Resizes the owned buffer to n bytes and switches to plain owned mode. */
  inline void resize(size_t n) {
    ptr_ = nullptr;
    size_ = 0;
    borrowed_ = false;
    aligned_owned_ = false;
    align_ = 0;
    owner_.reset();
    owned_.resize(n);
  }

  /** Resizes the owned buffer ensuring data() is aligned to align bytes.
   *  Over-allocates by (align - 1) bytes so std::align can find a suitable start.
   *  If align <= 1 or n == 0, falls back to plain resize(n).
   *  The caller may read/write the data via data() as usual; size() returns n. */
  inline void resize_aligned(size_t n, size_t align) {
    ptr_ = nullptr;
    size_ = 0;
    borrowed_ = false;
    aligned_owned_ = false;
    align_ = 0;
    owner_.reset();
    if (align <= 1 || n == 0) {
      owned_.resize(n);
      return;
    }
    // Over-allocate so std::align always finds a suitable start.
    owned_.resize(n + align - 1);
    void *vptr = owned_.data();
    size_t space = owned_.size();
    void *aligned = std::align(align, n, vptr, space);
    EXT_ENFORCE(aligned != nullptr, "ByteSpan::resize_aligned: failed to align ", n, " bytes to ",
                align, ".");
    ptr_ = static_cast<uint8_t *>(aligned);
    size_ = n;
    aligned_owned_ = true;
    align_ = align;
  }

  /** Sets borrowed mode: stores ptr/size in the base-class Span fields without any copy.
   *  The pointed-to buffer MUST outlive this ByteSpan. */
  inline void assign_borrowed(const uint8_t *ptr, size_t sz, std::shared_ptr<void> owner = {}) {
    owned_.clear();
    ptr_ = ptr;
    size_ = sz;
    borrowed_ = true;
    aligned_owned_ = false;
    align_ = 0;
    owner_ = std::move(owner);
  }

  /** Clears all data and resets to the empty owned state. */
  inline void clear() {
    owned_.clear();
    ptr_ = nullptr;
    size_ = 0;
    borrowed_ = false;
    aligned_owned_ = false;
    align_ = 0;
    owner_.reset();
  }

  /** Appends a single byte; switches to owned mode (copying any borrowed/aligned data first). */
  inline void push_back(uint8_t v) {
    if (borrowed_) {
      owned_.assign(ptr_, ptr_ + size_);
      ptr_ = nullptr;
      size_ = 0;
      borrowed_ = false;
      owner_.reset();
    } else if (aligned_owned_) {
      owned_.assign(ptr_, ptr_ + size_);
      ptr_ = nullptr;
      size_ = 0;
      aligned_owned_ = false;
      align_ = 0;
      owner_.reset();
    }
    owned_.push_back(v);
  }

private:
  OwnedByteBuffer owned_;
  bool borrowed_ = false;
  bool aligned_owned_ = false;
  /** Stored alignment for aligned-owned mode; used to re-align on copy. */
  size_t align_ = 0;
  /** Keeps borrowed backing storage alive when the model owns the shared buffer. */
  std::shared_ptr<void> owner_;
};

} // namespace utils
} // namespace ONNX_LIGHT_NAMESPACE
