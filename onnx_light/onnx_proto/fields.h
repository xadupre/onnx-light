#pragma once

#include "onnx_light_helpers.h"
#include "simple_span.h"
#include "simple_string.h"
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::utils {

/** Options that control how a proto message is printed. */
struct PrintOptions {
  /** If true, raw data will not be printed but skipped; tensors are not valid in that case but the
   * model structure is still available. */
  bool skip_raw_data = false;
  /** If skip_raw_data is true, raw data will be printed only if it is larger than the threshold. */
  int64_t raw_data_threshold = 1024;
  /** Repeated fields with at most this many elements are printed as a bracketed list; all output
   * is always flat (no newlines). */
  int64_t inline_threshold = 9;
};

/** Returns true if a repeated field of the given size should be printed inline on a single row. */
inline bool is_inline_size(const PrintOptions &options, size_t size) {
  return static_cast<int64_t>(size) <= options.inline_threshold;
}

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
  /** Constructs from an iterator range. */
  template <typename Iter> inline RepeatedField(Iter first, Iter last) : values_(first, last) {}
  inline RepeatedField &operator=(std::initializer_list<T> init) {
    values_.assign(init.begin(), init.end());
    return *this;
  }
  /** Reserves storage for at least n elements. */
  inline void reserve(size_t n) { values_.reserve(n); }
  /** Reserves storage for at least n elements (protobuf compat). */
  inline void Reserve(size_t n) { values_.reserve(static_cast<size_t>(n)); }
  /** Removes all elements. */
  inline void clear() { values_.clear(); }
  /** Removes all elements. */
  inline void Clear() { values_.clear(); }
  /** Returns a const iterator to the first element (protobuf compat). */
  inline typename std::vector<T>::const_iterator cbegin() const { return values_.cbegin(); }
  /** Returns a const iterator past the last element (protobuf compat). */
  inline typename std::vector<T>::const_iterator cend() const { return values_.cend(); }
  /** Returns true if the field contains no elements. */
  inline bool empty() const { return values_.empty(); }
  /** Returns the number of elements. */
  inline size_t size() const { return values_.size(); }
  /** Returns a mutable reference to the element at the given index. */
  inline T &operator[](size_t index) { return values_[index]; }
  /** Returns a const reference to the element at the given index. */
  inline const T &Get(size_t index) const { return values_[index]; }
  /** Sets the element at the given index (protobuf compat). */
  inline void Set(size_t index, const T &value) { values_[index] = value; }
  /** Returns a mutable reference to the owning pointer at the given index. */
  inline T *Mutable(size_t index) { return &values_[index]; }
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
  inline void extend(const std::vector<T> &v) {
    values_.reserve(values_.size() + v.size());
    values_.insert(values_.end(), v.begin(), v.end());
  }
  /** Appends all elements from another RepeatedField. */
  inline void extend(const RepeatedField<T> &v) {
    values_.reserve(values_.size() + v.size());
    values_.insert(values_.end(), v.begin(), v.end());
  }
  inline T &add() { return values_.emplace_back(); }
  inline T *Add() { return &values_.emplace_back(); }
  /** Appends a single value (protobuf compat). Accepts anything convertible to T,
   *  including types with an explicit conversion (e.g. std::string -> utils::String). */
  template <class U> inline void Add(U &&v) { values_.emplace_back(std::forward<U>(v)); }
  /** Appends all elements in [first, last) (protobuf RepeatedField::Add range compat). */
  template <class InputIt> inline void Add(InputIt first, InputIt last) {
    values_.insert(values_.end(), first, last);
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
    if constexpr (sizeof...(Args) == 0) {
      add();
    } else {
      values_.emplace_back(std::forward<Args>(args)...);
    }
  }
  /** Returns a pointer to the underlying data array. */
  inline T *data() { return values_.data(); }
  /** Returns a mutable pointer to the underlying data array (protobuf compat). */
  inline T *mutable_data() { return values_.data(); }
  /** Returns a const pointer to the underlying data array. */
  inline const T *data() const { return values_.data(); }
  /** Resizes the field to n elements, filling new slots with value. */
  inline void Resize(size_t n, const T &value) { values_.resize(n, value); }
  /** Resizes the field to n elements, default-constructing new slots. */
  inline void resize(size_t n) { values_.resize(n); }
  /** Assigns the contents from an iterator range. */
  template <typename InputIt> inline void Assign(InputIt first, InputIt last) {
    values_.assign(first, last);
  }
  /** Copies all elements from another RepeatedField. */
  inline void CopyFrom(const RepeatedField<T> &other) { values_ = other.values_; }
  /** Returns the number of elements as int (protobuf compat). */
  inline int size_int() const { return static_cast<int>(values_.size()); }
  /** Writes string representations of the contained values to ss. */
  void PrintToStringStream(std::stringstream &ss, PrintOptions &options) const;

private:
  std::vector<T> values_;
};

/** Repeated field used by name-like string lists (for example NodeProto inputs/outputs). */
class RepeatedStringField : public RepeatedField<utils::String> {
public:
  using RepeatedField<utils::String>::RepeatedField;
};

/** Output iterator that appends to a RepeatedField via push_back.
 *  Mirrors std::back_insert_iterator so it can back
 *  google::protobuf::RepeatedFieldBackInsertIterator as a pure alias. */
template <typename T> class RepeatedFieldBackInsertIterator {
public:
  using iterator_category = std::output_iterator_tag;
  using value_type = void;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = void;

  /** Wraps *field*; appended values go to its end. */
  explicit RepeatedFieldBackInsertIterator(RepeatedField<T> *field) : field_(field) {}

  /** Appends a copy of *value*. */
  RepeatedFieldBackInsertIterator &operator=(const T &value) {
    field_->push_back(value);
    return *this;
  }

  /** Appends *value* (moved). */
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

/** Repeated message field storage with owning pointers. */
template <typename T> class RepeatedProtoField {
public:
  /** Constructs an empty field. */
  explicit inline RepeatedProtoField() {}
  /** Constructs by copying elements from a value-storage RepeatedField<T>.
   *  This makes onnx-light's value-based repeated fields (used for message
   *  entries such as StringStringEntryProto) usable wherever a protobuf
   *  RepeatedPtrField (== RepeatedProtoField) is expected by drop-in consumers
   *  such as onnxruntime, without changing the field's own storage. */
  inline RepeatedProtoField(const RepeatedField<T> &src) {
    values_.reserve(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
      values_.push_back(std::make_shared<T>(src[i]));
    }
  }
  /** Copies elements from a vector into a RepeatedProtoField. */
  inline RepeatedProtoField(const std::vector<T> &src) { extend(src); }
  /** Move-constructs by moving each element from a vector into the field (no deep copy). */
  inline RepeatedProtoField(std::vector<T> &&src) { extend(std::move(src)); }
  /** Deep-copies elements from another RepeatedProtoField (protobuf value semantics:
   *  each owned element is cloned rather than sharing ownership). */
  inline RepeatedProtoField(const RepeatedProtoField<T> &src) { extend(src); }
  /** Move-constructs by stealing ownership from another RepeatedProtoField. */
  inline RepeatedProtoField(RepeatedProtoField<T> &&src) noexcept
      : values_(std::move(src.values_)) {}
  /** Deep-copies elements from another RepeatedProtoField (protobuf value semantics). */
  inline RepeatedProtoField &operator=(const RepeatedProtoField<T> &src) {
    if (this != &src) {
      clear();
      extend(src);
    }
    return *this;
  }
  /** Move-assigns by stealing ownership from another RepeatedProtoField. */
  inline RepeatedProtoField &operator=(RepeatedProtoField<T> &&src) noexcept {
    if (this != &src) {
      values_ = std::move(src.values_);
    }
    return *this;
  }
  /** Reserves storage for at least n elements. */
  inline void reserve(size_t n) { values_.reserve(n); }
  /** Reserves storage for at least n elements (protobuf compat). */
  inline void Reserve(size_t n) { values_.reserve(static_cast<size_t>(n)); }
  /** Resizes the field to contain exactly n elements, appending
   *  default-constructed elements or removing trailing elements as needed. */
  void resize(size_t n);
  /** Returns true if the field contains no elements. */
  inline bool empty() const { return values_.empty(); }
  /** Returns the number of elements. */
  inline size_t size() const { return values_.size(); }
  /** Returns a mutable reference to the element at the given index. */
  inline T &operator[](size_t index);
  /** Returns a const reference to the element at the given index. */
  inline const T &operator[](size_t index) const;
  /** Returns a mutable reference to the owning pointer at the given index. */
  inline std::shared_ptr<T> &get(size_t index) { return values_[index]; }
  /** Returns the shared pointer at the given index (shared ownership copy). */
  inline std::shared_ptr<T> shared_at(size_t index) const { return values_[index]; }
  /** Returns a mutable reference to the owning pointer at the given index. */
  inline const T &Get(size_t index) const { return *values_[index]; }
  /** Returns a const reference to the element at the given index (bounds-checked, protobuf compat).
   */
  inline const T &at(size_t index) const { return *values_.at(index); }
  /** Returns a mutable reference to the element at the given index (bounds-checked, protobuf
   * compat). */
  inline T &at(size_t index) { return *values_.at(index); }
  /** Returns a mutable reference to the owning pointer at the given index. */
  inline T *Mutable(size_t index) { return values_[index].get(); }
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
  /** Removes all elements. */
  inline void Clear() { clear(); }
  inline void RemoveLast() {
    if (!values_.empty())
      values_.pop_back();
  }
  /** Appends a copy of v at the end. */
  void push_back(const T &v);
  /** Appends v at the end by stealing its contents (no copy). */
  void push_back(T &&v);
  /** Appends all elements from a vector. */
  void extend(const std::vector<T> &v);
  /** Appends all elements from a vector by move (no per-element copy). */
  void extend(std::vector<T> &&v);
  /** Appends all elements from another RepeatedProtoField by copy. */
  void extend(const RepeatedProtoField<T> &v);
  /** Appends all elements from another RepeatedProtoField by move (steals ownership). */
  void extend(RepeatedProtoField<T> &&v);
  /** Appends a default-constructed element and returns a reference to it. */
  T &add();
  /** Appends a default-constructed element and returns a pointer to it. */
  T *Add() { return &add(); }
  /** Swaps the elements at positions i and j (protobuf RepeatedPtrField::SwapElements compat). */
  inline void SwapElements(int i, int j) {
    std::swap(values_[static_cast<size_t>(i)], values_[static_cast<size_t>(j)]);
  }
  /** Constructs a new element in-place at the end. */
  template <class... Args> inline void emplace_back(Args &&...args) {
    if constexpr (sizeof...(Args) == 0) {
      add();
    } else {
      values_.emplace_back(std::forward<Args>(args)...);
    }
  }
  /** Returns a reference to the last element. */
  T &back();
  /** Writes string representations of the contained values to ss. */
  void PrintToStringStream(std::stringstream &ss, PrintOptions &options) const;

  /** Mutable iterator for repeated proto fields. */
  class iterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;

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
    /** Returns the distance between two iterators (random-access compat). */
    inline difference_type operator-(const iterator &other) const {
      return static_cast<difference_type>(pos_) - static_cast<difference_type>(other.pos_);
    }
    /** Returns an iterator advanced backwards by n positions (random-access compat). */
    iterator operator-(std::ptrdiff_t n) const {
      return iterator(parent_, static_cast<size_t>(static_cast<std::ptrdiff_t>(pos_) - n));
    }
    /** Returns an iterator advanced forwards by n positions (random-access compat). */
    iterator operator+(std::ptrdiff_t n) const {
      return iterator(parent_, static_cast<size_t>(static_cast<std::ptrdiff_t>(pos_) + n));
    }
    /** Dereferences to the current element. */
    T &operator*() const { return (*parent_)[pos_]; }
    T *operator->() const { return &(**this); }
    /** Returns the current position index. */
    size_t pos() const { return pos_; }
  };

  /** Returns a mutable iterator to the first element. */
  inline iterator begin() { return iterator(this, 0); }
  /** Returns a mutable iterator past the last element. */
  inline iterator end() { return iterator(this, size()); }

  /** Iterator over raw element pointers (protobuf pointer_begin/pointer_end compat). */
  class pointer_iterator {
  private:
    RepeatedProtoField<T> *parent_;
    size_t pos_;

  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T *;
    using difference_type = std::ptrdiff_t;
    using pointer = T **;
    using reference = T *;
    /** Constructs a pointer iterator for parent starting at pos. */
    pointer_iterator(RepeatedProtoField<T> *parent, size_t pos = 0) : parent_(parent), pos_(pos) {}
    /** Advances the iterator to the next element. */
    pointer_iterator &operator++() {
      ++pos_;
      return *this;
    }
    /** Advances the iterator to the next element (post-increment). */
    pointer_iterator operator++(int) {
      pointer_iterator tmp = *this;
      ++pos_;
      return tmp;
    }
    /** Returns true if both iterators point to the same position in the same field. */
    bool operator==(const pointer_iterator &other) const {
      return pos_ == other.pos_ && parent_ == other.parent_;
    }
    /** Returns true if the iterators differ. */
    bool operator!=(const pointer_iterator &other) const { return !(*this == other); }
    /** Dereferences to the raw pointer of the current element. */
    T *operator*() const { return parent_->values_[pos_].get(); }
    /** Returns the current position index. */
    size_t pos() const { return pos_; }
  };

  /** Returns a pointer iterator to the first element. */
  inline pointer_iterator pointer_begin() { return pointer_iterator(this, 0); }
  /** Returns a pointer iterator past the last element. */
  inline pointer_iterator pointer_end() { return pointer_iterator(this, size()); }
  /** Removes the element at the given iterator position and returns an iterator to the next
   * element. */
  inline iterator erase(iterator it) {
    values_.erase(values_.begin() + static_cast<ptrdiff_t>(it.pos()));
    return iterator(this, it.pos());
  }
  /** Removes the elements in [first, last) and returns an iterator to the element
   *  that followed the last removed element (protobuf RepeatedPtrField::erase compat). */
  inline iterator erase(iterator first, iterator last) {
    values_.erase(values_.begin() + static_cast<ptrdiff_t>(first.pos()),
                  values_.begin() + static_cast<ptrdiff_t>(last.pos()));
    return iterator(this, first.pos());
  }

  /** Const iterator for repeated proto fields. */
  class const_iterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T *;
    using reference = const T &;

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
    inline difference_type operator-(const const_iterator &other) const {
      return static_cast<difference_type>(pos_) - static_cast<difference_type>(other.pos_);
    }
    /** Dereferences to the current element. */
    const T &operator*() const { return (*parent_)[pos_]; }
    const T *operator->() const { return &(**this); }
  };
  /** Returns a const iterator to the first element. */
  inline const_iterator begin() const { return const_iterator(this, 0); }
  /** Returns a const iterator past the last element. */
  inline const_iterator end() const { return const_iterator(this, size()); }
  /** Returns a const iterator to the first element (protobuf compat). */
  inline const_iterator cbegin() const { return const_iterator(this, 0); }
  /** Returns a const iterator past the last element (protobuf compat). */
  inline const_iterator cend() const { return const_iterator(this, size()); }

private:
  std::vector<std::shared_ptr<T>> values_;
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
  /** Assigns by moving the given value into the field. */
  OptionalField<T> &operator=(T &&other);
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

} // namespace ONNX_LIGHT_NAMESPACE::utils
