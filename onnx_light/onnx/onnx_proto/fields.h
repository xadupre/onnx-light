#pragma once

#include "onnx_light_helpers.h"
#include "simple_string.h"
#include <cstddef>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace onnx {
namespace utils {

/** Options that control how a proto message is printed. */
struct PrintOptions {
  /** If true, raw data will not be printed but skipped; tensors are not valid in that case but the
   * model structure is still available. */
  bool skip_raw_data = false;
  /** If skip_raw_data is true, raw data will be printed only if it is larger than the threshold. */
  int64_t raw_data_threshold = 1024;
};

/** Minimal unique_ptr-like holder used by generated proto containers. */
template <typename T> class simple_unique_ptr {
public:
  /** Constructs from a raw pointer (defaults to null); takes ownership if non-null. */
  explicit inline simple_unique_ptr(T *ptr = nullptr) : ptr_(ptr) {}
  /** Releases owned memory. */
  inline ~simple_unique_ptr() { delete ptr_; }
  /** Constructs by taking ownership from another instance. */
  inline simple_unique_ptr(simple_unique_ptr &&other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  /** Transfers ownership from another instance. */
  inline simple_unique_ptr &operator=(simple_unique_ptr &&other) noexcept {
    if (this != &other) {
      delete ptr_;
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  /** Returns true if the held pointer is null. */
  inline bool isnull() const { return ptr_ == nullptr; }
  /** Returns true if both instances hold the same pointer. */
  inline bool operator==(const simple_unique_ptr &other) const { return ptr_ == other.ptr_; }
  /** Returns true if the instances hold different pointers. */
  inline bool operator!=(const simple_unique_ptr &other) const { return ptr_ != other.ptr_; }
  /** Deleted copy constructor; use move semantics instead. */
  simple_unique_ptr(const simple_unique_ptr &) {
    EXT_THROW("simple_unique_ptr cannot be copied, only moved (1).");
  }
  /** Deleted copy assignment; use move semantics instead. */
  simple_unique_ptr &operator=(const simple_unique_ptr &) {
    EXT_THROW("simple_unique_ptr cannot be copied, only moved (2).");
  }
  /** Returns the raw pointer without releasing ownership. */
  inline T *get() const { return ptr_; }
  /** Dereferences the held pointer. */
  inline T &operator*() const { return *ptr_; }
  /** Provides member access through the held pointer. */
  inline T *operator->() const { return ptr_; }
  // inline T *release() { T *tmp = ptr_; ptr_ = nullptr; return tmp; }
  /** Replaces the held pointer and returns a reference to the new object. */
  inline T &reset_and(T *new_ptr) {
    EXT_ENFORCE(new_ptr != nullptr, "cannot you simple_unique_ptr::reset_and with a null pointer.");
    reset(new_ptr);
    return *this;
  }
  /** Deletes the held object and optionally takes a new pointer. */
  inline void reset(T *new_ptr = nullptr) {
    delete ptr_;
    ptr_ = new_ptr;
  }
  // inline void swap(simple_unique_ptr &other) noexcept { std::swap(ptr_, other.ptr_); }
private:
  T *ptr_;
};

/** Repeated primitive field storage. */
template <typename T> class RepeatedField {
public:
  /** Constructs an empty field. */
  explicit inline RepeatedField() {}
  /** Reserves storage for at least n elements. */
  inline void reserve(size_t n) { values_.reserve(n); }
  /** Removes all elements. */
  inline void clear() { values_.clear(); }
  /** Returns true if the field contains no elements. */
  inline bool empty() const { return values_.empty(); }
  /** Returns the number of elements. */
  inline size_t size() const { return values_.size(); }
  /** Returns a mutable reference to the element at the given index. */
  inline T &operator[](size_t index) { return values_[index]; }
  /** Returns a const reference to the underlying vector. */
  inline const std::vector<T> &values() const { return values_; }
  /** Returns a mutable reference to the underlying vector. */
  inline std::vector<T> &mutable_values() { return values_; }
  /** Returns a const reference to the element at the given index. */
  inline const T &operator[](size_t index) const { return values_[index]; }
  /** Removes a contiguous range; currently only start=0, step=1, and stop=size() are supported. */
  inline void remove_range(size_t start, size_t stop, size_t step) {
    EXT_ENFORCE(step == 1, "remove_range not implemented for step=", static_cast<int>(step));
    EXT_ENFORCE(start == 0, "remove_range not implemented for start=", static_cast<int>(start));
    EXT_ENFORCE(stop == size(), "remove_range not implemented for stop=", static_cast<int>(stop),
                " and size=", static_cast<int>(size()));
    clear();
  }
  /** Appends a copy of v at the end. */
  inline void push_back(const T &v) { values_.push_back(v); }
  /** Appends all elements from a vector. */
  inline void extend(const std::vector<T> &v) { values_.insert(values_.end(), v.begin(), v.end()); }
  /** Appends all elements from another RepeatedField. */
  inline void extend(const RepeatedField<T> &v) {
    values_.insert(values_.end(), v.begin(), v.end());
  }
  /** Appends a default-constructed element and returns a reference to it. */
  inline T &add() {
    values_.emplace_back(T());
    return values_.back();
  }
  /** Returns a reference to the last element. */
  inline T &back() { return values_.back(); }
  /** Returns a mutable iterator to the first element. */
  inline typename std::vector<T>::iterator begin() { return values_.begin(); }
  /** Returns a mutable iterator past the last element. */
  inline typename std::vector<T>::iterator end() { return values_.end(); }
  /** Returns a const iterator to the first element. */
  inline typename std::vector<T>::const_iterator begin() const { return values_.begin(); }
  /** Returns a const iterator past the last element. */
  inline typename std::vector<T>::const_iterator end() const { return values_.end(); }
  /** Constructs a new element in-place at the end. */
  template <class... Args> inline void emplace_back(Args &&...args) {
    values_.emplace_back(std::forward<Args>(args)...);
  }
  /** Returns a vector of string representations of the contained values. */
  std::vector<std::string> PrintToVectorString(PrintOptions &options) const;

private:
  std::vector<T> values_;
};

/** Repeated message field storage with owning pointers. */
template <typename T> class RepeatedProtoField {
public:
  /** Constructs an empty field. */
  explicit inline RepeatedProtoField() {}
  /** Reserves storage for at least n elements. */
  inline void reserve(size_t n) { values_.reserve(n); }
  /** Returns true if the field contains no elements. */
  inline bool empty() const { return values_.empty(); }
  /** Returns the number of elements. */
  inline size_t size() const { return values_.size(); }
  /** Returns a mutable reference to the element at the given index. */
  inline T &operator[](size_t index);
  /** Returns a const reference to the element at the given index. */
  inline const T &operator[](size_t index) const;
  /** Returns a mutable reference to the owning pointer at the given index. */
  inline simple_unique_ptr<T> &get(size_t index) { return values_[index]; }
  /** Removes a contiguous range; currently only start=0, step=1, and stop=size() are supported. */
  inline void remove_range(size_t start, size_t stop, size_t step) {
    EXT_ENFORCE(step == 1, "remove_range not implemented for step=", static_cast<int>(step));
    EXT_ENFORCE(start == 0, "remove_range not implemented for start=", static_cast<int>(start));
    EXT_ENFORCE(stop == size(), "remove_range not implemented for stop=", static_cast<int>(stop),
                " and size=", static_cast<int>(size()));
    clear();
  }

  /** Removes all elements. */
  void clear();
  /** Appends a copy of v at the end. */
  void push_back(const T &v);
  /** Appends all elements from a vector. */
  void extend(const std::vector<T> &v);
  /** Appends all elements from another RepeatedProtoField by copy. */
  void extend(const RepeatedProtoField<T> &v);
  /** Appends all elements from another RepeatedProtoField by move. */
  void extend(const RepeatedProtoField<T> &&v);
  /** Appends a default-constructed element and returns a reference to it. */
  T &add();
  /** Returns a reference to the last element. */
  T &back();
  /** Returns a vector of string representations of the contained values. */
  std::vector<std::string> PrintToVectorString(PrintOptions &options) const;

  /** Mutable iterator for repeated proto fields. */
  class iterator {
  private:
    RepeatedProtoField<T> *parent_;
    size_t pos_;

  public:
    /** Constructs an iterator for parent starting at pos. */
    iterator(RepeatedProtoField<T> *parent, size_t pos = 0) : parent_(parent), pos_(pos) {}
    /** Advances the iterator to the next element. */
    iterator &operator++() {
      ++pos_;
      return *this;
    }
    /** Returns true if both iterators point to the same position in the same field. */
    bool operator==(const iterator &other) const {
      return pos_ == other.pos_ && parent_ == other.parent_;
    }
    /** Returns true if the iterators differ. */
    bool operator!=(const iterator &other) const { return !(*this == other); }
    /** Dereferences to the current element. */
    T &operator*() { return (*parent_)[pos_]; }
  };
  /** Returns a mutable iterator to the first element. */
  inline iterator begin() { return iterator(this, 0); }
  /** Returns a mutable iterator past the last element. */
  inline iterator end() { return iterator(this, size()); }

  /** Const iterator for repeated proto fields. */
  class const_iterator {
  private:
    const RepeatedProtoField<T> *parent_;
    size_t pos_;

  public:
    /** Constructs a const iterator for parent starting at pos. */
    explicit const_iterator(const RepeatedProtoField<T> *parent, size_t pos = 0)
        : parent_(parent), pos_(pos) {}
    /** Advances the iterator to the next element. */
    const_iterator &operator++() {
      ++pos_;
      return *this;
    }
    /** Returns true if both iterators point to the same position in the same field. */
    bool operator==(const const_iterator &other) const {
      return pos_ == other.pos_ && parent_ == other.parent_;
    }
    /** Returns true if the iterators differ. */
    bool operator!=(const const_iterator &other) const { return !(*this == other); }
    /** Dereferences to the current element. */
    const T &operator*() const { return (*parent_)[pos_]; }
  };
  /** Returns a const iterator to the first element. */
  inline const_iterator begin() const { return const_iterator(this, 0); }
  /** Returns a const iterator past the last element. */
  inline const_iterator end() const { return const_iterator(this, size()); }

private:
  std::vector<simple_unique_ptr<T>> values_;
};

/** Optional field wrapper for message-like values. */
template <typename T> class OptionalField {
public:
  /** Constructs an empty optional field. */
  explicit inline OptionalField() : value_(nullptr) {}
  /** Constructs by copying another optional field. */
  explicit inline OptionalField(const OptionalField<T> &copy) : value_(nullptr) { *this = copy; }
  /** Constructs by moving from another optional field. */
  inline OptionalField(OptionalField<T> &&move) noexcept : value_(std::move(move.value_)) {
    move.reset();
  }
  /** Returns true if the field holds a value. */
  inline bool has_value() const { return !value_.isnull(); }
  /** Releases the held value, leaving the field empty. */
  inline void reset();
  /** Dereferences to the held value. */
  T &operator*();
  /** Dereferences to the held value (const). */
  const T &operator*() const;
  /** Assigns a copy of the given value. */
  OptionalField<T> &operator=(const T &other);
  /** Assigns from another optional field. */
  OptionalField<T> &operator=(const OptionalField<T> &other);
  /** Moves from another optional field. */
  inline OptionalField<T> &operator=(OptionalField<T> &&other) noexcept {
    if (this != &other) {
      value_ = std::move(other.value_);
    }
    return *this;
  }
  /** Initializes the field with a default-constructed value. */
  void set_empty_value();

private:
  simple_unique_ptr<T> value_;
};

/** Optional field wrapper for scalar values. */
template <typename T> class _OptionalField {
public:
  /** Constructs an empty optional field. */
  explicit inline _OptionalField() {}
  /** Returns true if the field holds a value. */
  inline bool has_value() const { return value_.has_value(); }
  /** Releases the held value, leaving the field empty. */
  inline void reset() { value_.reset(); }
  /** Dereferences to the held value (const). */
  inline const T &operator*() const { return *value_; }
  /** Dereferences to the held value. */
  inline T &operator*() { return *value_; }
  /** Returns true if the held value equals another optional field's value. */
  inline bool operator==(const _OptionalField<T> &v) const { return value_ == v; }
  /** Returns true if the held value equals v. */
  inline bool operator==(const T &v) const { return value_ == v; }
  /** Assigns a scalar value to the field. */
  inline _OptionalField<T> &operator=(const T &other) {
    value_ = other;
    return *this;
  }
  /** Initializes the field with zero-cast value. */
  inline void set_empty_value() { value_ = static_cast<T>(0); }

protected:
  std::optional<T> value_;
};

/** Optional field specialization for int64_t. */
template <> class OptionalField<int64_t> : public _OptionalField<int64_t> {
public:
  /** Constructs an empty optional int64_t field. */
  explicit inline OptionalField() : _OptionalField<int64_t>() {}
  /** Assigns an int64_t value to the field. */
  inline OptionalField<int64_t> &operator=(const int64_t &other) {
    value_ = other;
    return *this;
  }
};

/** Optional field specialization for int32_t. */
template <> class OptionalField<int32_t> : public _OptionalField<int32_t> {
public:
  /** Constructs an empty optional int32_t field. */
  explicit inline OptionalField() : _OptionalField<int32_t>() {}
  /** Assigns an int32_t value to the field. */
  inline OptionalField<int32_t> &operator=(const int32_t &other) {
    value_ = other;
    return *this;
  }
};

/** Optional field specialization for float. */
template <> class OptionalField<float> : public _OptionalField<float> {
public:
  /** Constructs an empty optional float field. */
  explicit inline OptionalField() : _OptionalField<float>() {}
  /** Assigns a float value to the field. */
  inline OptionalField<float> &operator=(const float &other) {
    value_ = other;
    return *this;
  }
};

/** Optional field wrapper for enum values. */
template <typename T> class OptionalEnumField {
public:
  /** Constructs an empty optional enum field. */
  explicit inline OptionalEnumField() {}
  /** Returns true if the field holds a value. */
  inline bool has_value() const { return value_.has_value(); }
  /** Releases the held value, leaving the field empty. */
  inline void reset() { value_.reset(); }
  /** Dereferences to the held value (const). */
  inline const T &operator*() const { return *value_; }
  /** Dereferences to the held value. */
  inline T &operator*() { return *value_; }
  /** Returns true if the held value equals another optional enum field's value. */
  inline bool operator==(const OptionalEnumField<T> &v) const { return value_ == v; }
  /** Returns true if the held value equals v. */
  inline bool operator==(const T &v) const { return value_ == v; }
  /** Assigns an enum value to the field. */
  inline OptionalEnumField<T> &operator=(const T &other) {
    value_ = other;
    return *this;
  }
  /** Initializes the field with zero-cast enum value. */
  inline void set_empty_value() { value_ = static_cast<T>(0); }

protected:
  std::optional<T> value_;
};

/** A byte buffer that can either own its data or borrow a non-owning view into an external buffer.
 *  The borrowed mode is used for zero-copy parsing: when ParseOptions::no_copy is true and the
 *  stream supports it, tensor raw data is not copied — instead assign_borrowed() stores a pointer
 *  directly into the source bytes buffer.  The caller MUST keep that buffer alive for as long as
 *  the ByteSpan is in borrowed mode.  In owned mode the class behaves like std::vector<uint8_t>. */
class ByteSpan {
public:
  /** Constructs an empty buffer (owned mode, no allocation). */
  ByteSpan() = default;

  /** Constructs an owned buffer by copying from a std::vector<uint8_t>. */
  inline ByteSpan(const std::vector<uint8_t> &v) : owned_(v) {}

  /** Assigns owned data by copying from a std::vector<uint8_t>; clears any borrowed state. */
  inline ByteSpan &operator=(const std::vector<uint8_t> &v) {
    owned_ = v;
    borrowed_ptr_ = nullptr;
    borrowed_size_ = 0;
    return *this;
  }

  /** Returns true when no data is stored in either mode. */
  inline bool empty() const { return borrowed_ptr_ == nullptr && owned_.empty(); }

  /** Returns the number of bytes available from either storage mode. */
  inline size_t size() const { return borrowed_ptr_ != nullptr ? borrowed_size_ : owned_.size(); }

  /** Returns a const pointer to the byte data from either storage mode. */
  inline const uint8_t *data() const {
    return borrowed_ptr_ != nullptr ? borrowed_ptr_ : owned_.data();
  }

  /** Returns a mutable pointer; valid only in owned mode (i.e. after resize()).
   *  Calling this in borrowed mode raises an error at runtime. */
  inline uint8_t *data() {
    EXT_ENFORCE(borrowed_ptr_ == nullptr,
                "ByteSpan: mutable data() called on a borrowed (zero-copy) buffer; "
                "use const data() or assign owned data first.");
    return owned_.data();
  }

  /** Resizes the owned buffer to n bytes and switches to owned mode. */
  inline void resize(size_t n) {
    borrowed_ptr_ = nullptr;
    borrowed_size_ = 0;
    owned_.resize(n);
  }

  /** Sets borrowed mode: stores ptr/size without any copy.
   *  The pointed-to buffer MUST outlive this ByteSpan. */
  inline void assign_borrowed(const uint8_t *ptr, size_t size) {
    owned_.clear();
    borrowed_ptr_ = ptr;
    borrowed_size_ = size;
  }

  /** Returns true when the data is borrowed (non-owning). */
  inline bool is_borrowed() const { return borrowed_ptr_ != nullptr; }

  /** Clears all data and resets to the empty owned state. */
  inline void clear() {
    owned_.clear();
    borrowed_ptr_ = nullptr;
    borrowed_size_ = 0;
  }

  /** Returns true when both spans have the same size and identical byte content. */
  inline bool operator==(const ByteSpan &other) const {
    const size_t sz = size();
    if (sz != other.size())
      return false;
    return sz == 0 || std::memcmp(data(), other.data(), sz) == 0;
  }

  /** Returns true when the spans differ in size or content. */
  inline bool operator!=(const ByteSpan &other) const { return !(*this == other); }

  /** Appends a single byte; switches to owned mode (copying any borrowed data first). */
  inline void push_back(uint8_t v) {
    if (borrowed_ptr_ != nullptr) {
      owned_.assign(borrowed_ptr_, borrowed_ptr_ + borrowed_size_);
      borrowed_ptr_ = nullptr;
      borrowed_size_ = 0;
    }
    owned_.push_back(v);
  }

  /** Returns a const reference to the byte at index i.
   *  No bounds check is performed; behaviour is undefined for i >= size(). */
  inline const uint8_t &operator[](size_t i) const { return data()[i]; }

  /** Returns a mutable reference to the byte at index i; valid only in owned mode.
   *  No bounds check is performed; behaviour is undefined for i >= size(). */
  inline uint8_t &operator[](size_t i) { return data()[i]; }

private:
  std::vector<uint8_t> owned_;
  const uint8_t *borrowed_ptr_ = nullptr;
  size_t borrowed_size_ = 0;
};

} // namespace utils
} // namespace onnx
