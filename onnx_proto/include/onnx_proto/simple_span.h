#pragma once

#include "onnx_light_helpers.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::utils {

namespace detail {
/** Copies a complete proto's fields with an explicitly selected payload-sharing policy. */
ONNX_LIGHT_PROTO_API void CopyProtoWithSharedPayloads(void *destination, const void *source,
                                                      uint16_t proto_id);
} // namespace detail

template <typename T> T CopyWithSharedPayloads(const T &source);
template <typename T, typename Allocator>
std::vector<T, Allocator> CopyWithSharedPayloads(const std::vector<T, Allocator> &source);
template <typename Key, typename Value, typename Compare, typename Allocator>
std::map<Key, Value, Compare, Allocator>
CopyWithSharedPayloads(const std::map<Key, Value, Compare, Allocator> &source);

/**
 * Copies message structure independently while sharing eligible payload allocations.
 *
 * Subsequent mutable container accesses detach owned payloads. Mutable payload pointers, element
 * references, iterators and underlying vector references obtained before this call must not be used
 * afterward; reacquire them from the appropriate copy. Later mutable accesses can invalidate
 * payload views as storage detaches. References to message and field objects remain valid.
 * Borrowed storage with an owner remains read-only and retains its
 * owner; ownerless borrowed storage and actively exported storage are copied instead. The caller
 * must synchronize access to the source during copying, including its logically const storage.
 */
template <typename T> T CopyWithSharedPayloads(const T &source) {
  if constexpr (requires { source.SharedCopy(); }) {
    return source.SharedCopy();
  } else {
    return T(source);
  }
}

template <typename T, typename Allocator>
std::vector<T, Allocator> CopyWithSharedPayloads(const std::vector<T, Allocator> &source) {
  std::vector<T, Allocator> result(source.get_allocator());
  result.reserve(source.size());
  for (const auto &value : source)
    result.push_back(CopyWithSharedPayloads(value));
  return result;
}

template <typename Key, typename Value, typename Compare, typename Allocator>
std::map<Key, Value, Compare, Allocator>
CopyWithSharedPayloads(const std::map<Key, Value, Compare, Allocator> &source) {
  std::map<Key, Value, Compare, Allocator> result(source.key_comp(), source.get_allocator());
  for (const auto &[key, value] : source)
    result.emplace(key, CopyWithSharedPayloads(value));
  return result;
}

/**
 * Non-owning, read-only view of a contiguous byte sequence.
 * It references existing memory and never allocates or frees storage.
 * Analogous to RefString but for raw bytes.
 */
class ONNX_LIGHT_PROTO_API Span {
protected:
  mutable const uint8_t *ptr_ = nullptr;
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
class ONNX_LIGHT_PROTO_API OwnedByteBuffer {
public:
  /** Initializes an empty buffer. */
  OwnedByteBuffer() = default;

  /** Creates a deep copy of the stored bytes. */
  inline OwnedByteBuffer(const OwnedByteBuffer &other) {
    assign(other.data(), other.size());
    if (other.data() != nullptr && other.size() == 0) {
      resize(1);
      clear();
    }
  }

  /** Replaces the buffer with a deep copy of the stored bytes. */
  inline OwnedByteBuffer &operator=(const OwnedByteBuffer &other) {
    if (this != &other) {
      OwnedByteBuffer copied(other);
      *this = std::move(copied);
    }
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
  inline uint8_t *data() {
    detach();
    return storage_.get();
  }

  /** Returns a const pointer to the stored bytes. */
  inline const uint8_t *data() const { return storage_.get(); }

  /** Clears the logical content while preserving capacity. */
  inline void clear() {
    if (is_shared()) {
      reset();
      return;
    }
    size_ = 0;
  }

  /** Shares an allocation only for an explicitly requested payload copy. */
  inline void share_from(const OwnedByteBuffer &other) {
    storage_ = other.storage_;
    size_ = other.size_;
    capacity_ = other.capacity_;
  }

  /** Returns whether another buffer retains the allocation. */
  inline bool is_shared() const { return storage_.use_count() > 1; }

  /** Returns a token retaining the allocation independently of this buffer. */
  inline std::shared_ptr<void> storage_owner() const { return storage_; }

  /** Detaches shared storage before mutation. */
  inline void detach() {
    if (is_shared()) {
      std::shared_ptr<uint8_t[]> next(new uint8_t[capacity_]);
      if (size_ > 0)
        std::memcpy(next.get(), storage_.get(), size_);
      storage_ = std::move(next);
    }
  }

  /** Releases all storage. */
  inline void reset() {
    storage_.reset();
    size_ = 0;
    capacity_ = 0;
  }

  /** Resizes the buffer without value-initializing newly exposed bytes. */
  inline void resize(size_t n) {
    if (n <= capacity_) {
      detach();
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
    std::shared_ptr<uint8_t[]> next(new uint8_t[n]);
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
    } else
      detach();
    storage_[size_++] = v;
  }

private:
  /** Ensures the buffer can store at least n bytes. */
  inline void reserve(size_t n) {
    if (n <= capacity_)
      return;
    std::shared_ptr<uint8_t[]> next(new uint8_t[n]);
    if (size_ > 0)
      std::memcpy(next.get(), storage_.get(), size_);
    storage_ = std::move(next);
    capacity_ = n;
  }

  std::shared_ptr<uint8_t[]> storage_;
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
class ONNX_LIGHT_PROTO_API ByteSpan : public Span {
public:
  /** Constructs an empty buffer (owned mode, no allocation). */
  ByteSpan() = default;

  /** Constructs an owned buffer by copying from a std::vector<uint8_t>. */
  inline ByteSpan(const std::vector<uint8_t> &v) { *this = v; }

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

  /** Copies metadata and explicitly shares eligible storage without changing ordinary copies. */
  inline ByteSpan SharedCopy() const {
    if (has_active_exports())
      return ByteSpan(*this);
    ByteSpan result;
    if (borrowed_ && owner_.use_count() == 0) {
      result.owned_.assign(data(), size());
      if (data() != nullptr && empty())
        result.resize(0);
      return result;
    }
    result.borrowed_ = borrowed_;
    result.aligned_owned_ = aligned_owned_;
    result.align_ = align_;
    result.owner_ = owner_;
    result.ptr_ = ptr_;
    result.size_ = size_;
    if (!borrowed_)
      result.owned_.share_from(owned_);
    return result;
  }

  /** Copy assignment: handles aligned-owned mode by recomputing the internal pointer. */
  inline ByteSpan &operator=(const ByteSpan &other) {
    if (this != &other) {
      ByteSpan copied(other);
      *this = std::move(copied);
    }
    return *this;
  }

  /** Move constructor: for aligned-owned mode the pointer remains valid after the move. */
  inline ByteSpan(ByteSpan &&other) noexcept
      : Span(other.ptr_, other.size_), owned_(std::move(other.owned_)), borrowed_(other.borrowed_),
        aligned_owned_(other.aligned_owned_), align_(other.align_), owner_(std::move(other.owner_)),
        export_guard_(std::move(other.export_guard_)) {
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
      export_guard_ = std::move(other.export_guard_);
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
    check_no_active_exports();
    export_guard_.reset();
    owned_ = v;
    if (v.empty()) {
      owned_.resize(1);
      owned_.clear();
    }
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

  /** Returns the token retaining borrowed storage, or an empty token when caller-owned. */
  inline const std::shared_ptr<void> &owner() const { return owner_; }

  /** Retains an owned-storage export lease, or the lifetime owner of borrowed storage. */
  std::shared_ptr<void> acquire_export_guard() const;

  /** Returns whether consumers still hold an export lease on this storage. */
  inline bool has_active_exports() const {
    auto guard = export_guard_.lock();
    return !borrowed_ && guard &&
           guard->first.get() == static_cast<const OwnedByteBuffer &>(owned_).data();
  }

  /** Implicit conversion to a standard string so the type is a drop-in for
   *  protobuf ``bytes`` fields (which are std::string) in consuming code. */
  inline operator std::string() const {
    const uint8_t *p = data();
    return p == nullptr ? std::string() : std::string(reinterpret_cast<const char *>(p), size());
  }

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
    return borrowed_ ? ptr_ : static_cast<const OwnedByteBuffer &>(owned_).data();
  }
  /** Returns a const reference to the byte at index i (no bounds check). */
  inline const uint8_t &operator[](size_t i) const { return data()[i]; }
  /** Returns a const char pointer to the data (std::string compatibility). */
  inline const char *c_str() const { return reinterpret_cast<const char *>(data()); }

  /** Returns true when both ByteSpans have the same size and byte content. */
  inline bool operator==(const ByteSpan &other) const {
    const size_t sz = size();
    if (sz != other.size())
      return false;
    return sz == 0 || std::memcmp(data(), other.data(), sz) == 0;
  }
  /** Returns true when the ByteSpans differ in size or content. */
  inline bool operator!=(const ByteSpan &other) const { return !(*this == other); }

  /** Compare with std::string (byte-level comparison). */
  inline bool operator==(const std::string &other) const {
    return size() == other.size() &&
           (size() == 0 || std::memcmp(data(), other.data(), size()) == 0);
  }
  inline bool operator!=(const std::string &other) const { return !(*this == other); }

  /** Allow std::string == ByteSpan. */
  friend inline bool operator==(const std::string &lhs, const ByteSpan &rhs) { return rhs == lhs; }
  friend inline bool operator!=(const std::string &lhs, const ByteSpan &rhs) { return rhs != lhs; }

  // --- Mutable operations ---

  /** Returns a mutable pointer to the owned data.
   *  Calling this in borrowed mode raises an error at runtime.
   *  The pointer must be reacquired after a shared snapshot or a later mutable buffer access. */
  inline uint8_t *data() {
    EXT_ENFORCE(!borrowed_, "ByteSpan: mutable data() called on a borrowed (zero-copy) buffer; "
                            "use const data() or assign owned data first.");
    check_no_active_exports();
    detach_owned();
    if (aligned_owned_)
      return const_cast<uint8_t *>(ptr_);
    return owned_.data();
  }

  /** Returns a mutable char pointer to the data (std::string compatibility). */
  inline char *data_as_char() { return reinterpret_cast<char *>(data()); }

  /** Returns a mutable reference to the byte at index i; valid only in owned mode.
   *  No bounds check is performed; behaviour is undefined for i >= size(). */
  inline uint8_t &operator[](size_t i) { return data()[i]; }

  /** Resizes to plain owned mode, preserving the previous state if allocation fails. */
  inline void resize(size_t n) {
    check_no_active_exports();
    if (aligned_owned_) {
      OwnedByteBuffer next;
      next.resize(n == 0 ? 1 : n);
      if (n > 0)
        std::memcpy(next.data(), ptr_, std::min(n, size_));
      else
        next.clear();
      owned_ = std::move(next);
    } else if (n == 0) {
      owned_.resize(1);
      owned_.clear();
    } else {
      owned_.resize(n);
    }
    export_guard_.reset();
    ptr_ = nullptr;
    size_ = 0;
    borrowed_ = false;
    aligned_owned_ = false;
    align_ = 0;
    owner_.reset();
  }

  /** Represents an explicitly present empty byte sequence. */
  inline void set_empty() {
    if (empty() && data() == nullptr)
      resize(0);
  }

  inline void assign(const char *data, size_t len) {
    // Keeps an aliased source alive when resize replaces aligned or borrowed storage.
    const auto source_owner = aligned_owned_ ? owned_.storage_owner() : owner_;
    resize(len);
    if (len > 0)
      std::memmove(this->data(), data, len);
  }

  inline void assign(const void *data, size_t len) { assign(static_cast<const char *>(data), len); }

  /** Resizes the owned buffer ensuring data() is aligned to align bytes.
   *  Over-allocates by (align - 1) bytes so std::align can find a suitable start.
   *  If align <= 1 or n == 0, falls back to plain resize(n).
   *  The caller may read/write the data via data() as usual; size() returns n.
   *  Preserves the logical prefix even when the allocation's alignment offset changes.
   *  Allocation or validation failure preserves the previous bytes and storage mode. */
  inline void resize_aligned(size_t n, size_t align) {
    check_no_active_exports();
    if (align <= 1 || n == 0) {
      resize(n);
      return;
    }
    EXT_ENFORCE_INVALID((align & (align - 1)) == 0,
                        "ByteSpan::resize_aligned: alignment must be a power of two.");
    EXT_ENFORCE_INVALID(n <= std::numeric_limits<size_t>::max() - (align - 1),
                        "ByteSpan::resize_aligned: aligned allocation size overflows.");
    // Over-allocate so std::align always finds a suitable start.
    OwnedByteBuffer next;
    next.resize(n + align - 1);
    void *vptr = next.data();
    size_t space = next.size();
    void *aligned = std::align(align, n, vptr, space);
    EXT_ENFORCE(aligned != nullptr, "ByteSpan::resize_aligned: failed to align ", n, " bytes to ",
                align, ".");
    const size_t preserved = std::min(n, size());
    if (preserved > 0)
      std::memcpy(aligned, static_cast<const ByteSpan &>(*this).data(), preserved);
    owned_ = std::move(next);
    export_guard_.reset();
    ptr_ = static_cast<uint8_t *>(aligned);
    size_ = n;
    borrowed_ = false;
    aligned_owned_ = true;
    align_ = align;
    owner_.reset();
  }

  /** Sets borrowed mode: stores ptr/size in the base-class Span fields without any copy.
   *  The pointed-to buffer MUST outlive this ByteSpan. */
  inline void assign_borrowed(const uint8_t *ptr, size_t sz, std::shared_ptr<void> owner = {}) {
    if (ptr == nullptr && sz == 0) {
      clear();
      return;
    }
    owned_.clear();
    export_guard_.reset();
    ptr_ = ptr;
    size_ = sz;
    borrowed_ = true;
    aligned_owned_ = false;
    align_ = 0;
    owner_ = std::move(owner);
  }

  /** Sets borrowed mode with a custom deleter.  Stores ptr/size without any copy and arranges for
   *  deleter() to be called when all copies of this ByteSpan are destroyed (or the span is
   *  overwritten/cleared).  The deleter receives no arguments and returns void.
   *
   *  More flexible than assign_borrowed(ptr, sz, owner) because any callable — lambda,
   *  function pointer, or functor — can be supplied directly instead of wrapping it in a
   *  shared_ptr first. */
  template <typename Deleter>
  inline void assign_with_deleter(const uint8_t *ptr, size_t sz, Deleter &&deleter) {
    // Wrap the callable in a shared_ptr whose custom deleter invokes it on destruction.
    // A null managed pointer is intentional: the shared_ptr exists solely to call the
    // deleter at the right time, not to manage the raw byte pointer.
    std::shared_ptr<void> owner(static_cast<void *>(nullptr),
                                [d = std::forward<Deleter>(deleter)](void *) { d(); });
    assign_borrowed(ptr, sz, std::move(owner));
  }

  /** Attaches a custom deleter to the data without changing the stored bytes or the storage
   *  mode (owned, aligned-owned or borrowed).  deleter() is called exactly once when the last
   *  copy of this ByteSpan that shares the owner token is destroyed (or the span is
   *  overwritten/cleared).  The deleter receives no arguments and returns void.
   *
   *  Unlike assign_with_deleter, this method keeps the current pointer/size in place, so it can
   *  register cleanup for data that has already been parsed (for example a tensor's raw_data
   *  relocated by a parse callback).  Any previously stored owner token is replaced. */
  template <typename Deleter> inline void attach_deleter(Deleter &&deleter) {
    check_no_active_exports();
    export_guard_.reset();
    owner_ = std::shared_ptr<void>(static_cast<void *>(nullptr),
                                   [d = std::forward<Deleter>(deleter)](void *) { d(); });
  }

  /** Clears all data and resets to the empty owned state. */
  inline void clear() {
    export_guard_.reset();
    owned_.reset();
    ptr_ = nullptr;
    size_ = 0;
    borrowed_ = false;
    aligned_owned_ = false;
    align_ = 0;
    owner_.reset();
  }

  /** Appends a single byte; switches to owned mode (copying any borrowed/aligned data first). */
  inline void push_back(uint8_t v) {
    check_no_active_exports();
    export_guard_.reset();
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

  /** BREAKING CHANGE: value() is no longer supported and always raises.
   *
   *  It used to return a `const std::string&` by lazily materializing an internal
   *  cache.  That is fundamentally incompatible with the ByteSpan storage model:
   *  a ByteSpan may hold data in borrowed (zero-copy) or aligned-owned mode, where
   *  the bytes are NOT stored as a std::string.  Returning them as a std::string
   *  therefore required a hidden full copy on every call (defeating the zero-copy
   *  and alignment guarantees), and the returned reference could dangle as soon as
   *  the underlying buffer was mutated or reassigned.
   *
   *  Access the bytes through the ByteSpan API instead: data()/size(), operator[],
   *  the explicit `operator std::string()` conversion (when a real copy is wanted),
   *  or the field's ref_<name>() accessor. */
  inline const std::string &value() const {
    EXT_THROW("ByteSpan::value() is a removed API (breaking change): a ByteSpan may store borrowed "
              "or aligned-owned bytes that are not backed by a std::string, so exposing them as a "
              "`const std::string&` requires a hidden copy and can dangle. Use data()/size(), the "
              "explicit std::string conversion, or ref_<field>() instead.");
  }

  /** BREAKING CHANGE: mutable_value() is no longer supported and always raises.
   *
   *  It used to return a `std::string*` into an internal cache that had to be
   *  written back with sync_from_cache() after mutation — an error-prone step that
   *  was easy to forget — and it could not preserve the buffer's alignment or its
   *  borrowed semantics.  Because a ByteSpan does not own its bytes as a std::string
   *  in borrowed/aligned-owned mode, there is no std::string to hand out safely.
   *
   *  Mutate the bytes through the ByteSpan API instead: data()/data_as_char() and
   *  operator[] for in-place edits, or assign()/resize()/resize_aligned() to replace
   *  the contents. */
  inline std::string *mutable_value() {
    EXT_THROW("ByteSpan::mutable_value() is a removed API (breaking change): it returned a pointer "
              "into a cache that required an explicit sync_from_cache() write-back and could not "
              "preserve alignment or borrowed semantics. Mutate via data()/data_as_char() or "
              "replace via assign()/resize()/resize_aligned() instead.");
  }

private:
  inline void check_no_active_exports() const {
    EXT_ENFORCE_INVALID(borrowed_ || !has_active_exports(),
                        "ByteSpan: cannot modify owned storage with active exported views.");
  }

  /** Preserves logical bytes and alignment when detaching an owned allocation. */
  inline void detach_owned() const {
    if (!owned_.is_shared())
      return;
    if (!aligned_owned_) {
      owned_.detach();
      export_guard_.reset();
      return;
    }
    OwnedByteBuffer next;
    next.resize(size_ + align_ - 1);
    void *vptr = next.data();
    size_t space = next.size();
    void *aligned = std::align(align_, size_, vptr, space);
    EXT_ENFORCE(aligned != nullptr, "ByteSpan: detach re-alignment failed.");
    if (size_ > 0)
      std::memcpy(aligned, ptr_, size_);
    owned_ = std::move(next);
    ptr_ = static_cast<uint8_t *>(aligned);
    export_guard_.reset();
  }

  mutable OwnedByteBuffer owned_;
  bool borrowed_ = false;
  bool aligned_owned_ = false;
  /** Stored alignment for aligned-owned mode; used to re-align on copy. */
  size_t align_ = 0;
  /** Keeps borrowed backing storage alive when the model owns the shared buffer. */
  std::shared_ptr<void> owner_;
  using ExportGuard = std::pair<std::shared_ptr<void>, std::shared_ptr<void>>;
  mutable std::weak_ptr<ExportGuard> export_guard_;
};

} // namespace ONNX_LIGHT_NAMESPACE::utils
