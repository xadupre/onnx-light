// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * DataType — alias for ``TensorProto::DataType``.
 *
 * Provides a short, namespace-local name for the upstream ONNX
 * ``TensorProto::DataType`` enumeration so backend test code can write
 * ``DataType::INT64`` instead of fully qualifying every reference.
 */
using DataType = TensorProto::DataType;

/**
 * Shape — a fixed-capacity tensor shape storing up to 16 dimensions.
 *
 * Stores dimension values in an inline ``int64_t[kMaxRank]`` array so no heap
 * allocation is required for ordinary shapes.  The number of valid dimensions
 * is tracked by ``size_``.  An empty shape (``size_ == 0``) represents a
 * scalar.
 *
 * Implicit conversions to and from ``std::vector<int64_t>`` are provided for
 * backward compatibility with existing call sites that use
 * ``std::vector<int64_t>`` for shape parameters.
 */
struct Shape {
  /// Maximum supported tensor rank.
  static constexpr size_t kMaxRank = 16;

  /// Constructs an empty shape (scalar).
  Shape() noexcept : dims_{}, size_(0) {}

  /// Constructs from a brace-enclosed initializer list, e.g. ``Shape{2, 3}``.
  /// Throws ``std::invalid_argument`` when the list exceeds ``kMaxRank``.
  Shape(std::initializer_list<int64_t> il) : dims_{}, size_(il.size()) {
    EXT_ENFORCE_INVALID(il.size() <= kMaxRank, "Shape rank ", il.size(), " exceeds maximum of ",
                        kMaxRank, ".");
    size_t i = 0;
    for (int64_t v : il)
      dims_[i++] = v;
  }

  /// Constructs from a ``std::vector<int64_t>`` (copies all dimensions).
  /// Throws ``std::invalid_argument`` when the vector size exceeds ``kMaxRank``.
  Shape(const std::vector<int64_t> &v) : dims_{}, size_(v.size()) {
    EXT_ENFORCE_INVALID(v.size() <= kMaxRank, "Shape rank ", v.size(), " exceeds maximum of ",
                        kMaxRank, ".");
    std::memcpy(dims_, v.data(), v.size() * sizeof(int64_t));
  }

  /// Constructs from a moved ``std::vector<int64_t>`` (copies all dimensions).
  /// Throws ``std::invalid_argument`` when the vector size exceeds ``kMaxRank``.
  Shape(std::vector<int64_t> &&v) : dims_{}, size_(v.size()) {
    EXT_ENFORCE_INVALID(v.size() <= kMaxRank, "Shape rank ", v.size(), " exceeds maximum of ",
                        kMaxRank, ".");
    std::memcpy(dims_, v.data(), v.size() * sizeof(int64_t));
  }

  Shape(const Shape &) noexcept = default;
  Shape(Shape &&) noexcept = default;
  Shape &operator=(const Shape &) noexcept = default;
  Shape &operator=(Shape &&) noexcept = default;

  /// Assigns from a brace-enclosed initializer list.
  /// Throws ``std::invalid_argument`` when the list exceeds ``kMaxRank``.
  Shape &operator=(std::initializer_list<int64_t> il) {
    EXT_ENFORCE_INVALID(il.size() <= kMaxRank, "Shape rank ", il.size(), " exceeds maximum of ",
                        kMaxRank, ".");
    size_ = il.size();
    size_t i = 0;
    for (int64_t v : il)
      dims_[i++] = v;
    return *this;
  }

  /// Assigns from a ``std::vector<int64_t>``.
  /// Throws ``std::invalid_argument`` when the vector size exceeds ``kMaxRank``.
  Shape &operator=(const std::vector<int64_t> &v) {
    EXT_ENFORCE_INVALID(v.size() <= kMaxRank, "Shape rank ", v.size(), " exceeds maximum of ",
                        kMaxRank, ".");
    size_ = v.size();
    std::memcpy(dims_, v.data(), v.size() * sizeof(int64_t));
    return *this;
  }

  /// Assigns from a moved ``std::vector<int64_t>``.
  /// Throws ``std::invalid_argument`` when the vector size exceeds ``kMaxRank``.
  Shape &operator=(std::vector<int64_t> &&v) {
    EXT_ENFORCE_INVALID(v.size() <= kMaxRank, "Shape rank ", v.size(), " exceeds maximum of ",
                        kMaxRank, ".");
    size_ = v.size();
    std::memcpy(dims_, v.data(), v.size() * sizeof(int64_t));
    return *this;
  }

  /// Implicitly converts to ``std::vector<int64_t>`` for backward compatibility.
  operator std::vector<int64_t>() const { return std::vector<int64_t>(dims_, dims_ + size_); }

  /// Returns the number of valid dimensions.
  size_t size() const noexcept { return size_; }
  /// Returns ``true`` when there are no dimensions (scalar shape).
  bool empty() const noexcept { return size_ == 0; }

  /// Computes and returns the product of all dimensions; 1 for an empty (scalar) shape.
  int64_t product() const noexcept {
    int64_t n = 1;
    for (size_t i = 0; i < size_; ++i) {
      n *= dims_[i];
    }
    return n;
  }

  /**
   * Computes the product of dimensions in [begin, end).
   *
   * Validates the requested range and checks for negative dimensions and INT64
   * overflow while multiplying.
   *
   * Parameters:
   *   begin: Start index of the dimension range (inclusive).
   *   end: End index of the dimension range (exclusive).
   *   where: Caller-provided context describing the multiplication site.
   * Returns:
   *   The product of the selected dimensions, or 1 for an empty range.
   */
  int64_t product(size_t begin, size_t end, const std::string &where) const {
    EXT_ENFORCE_INVALID(begin <= end && end <= size_, "Shape::product: invalid range [", begin,
                        ", ", end, ") for rank ", size_, ".");
    int64_t n = 1;
    for (size_t i = begin; i < end; ++i) {
      EXT_ENFORCE_INVALID(!(dims_[i] < 0), "Shape::product: ", where, " encounters a negative ",
                          "dimension at index ", i, " with value ", dims_[i], ".");
      EXT_ENFORCE_INVALID(!(n != 0 && dims_[i] > std::numeric_limits<int64_t>::max() / n),
                          "Shape::product: ", where, " would overflow INT64 shape arithmetic.");
      n *= dims_[i];
    }
    return n;
  }

  int64_t *begin() noexcept { return dims_; }
  const int64_t *begin() const noexcept { return dims_; }
  int64_t *end() noexcept { return dims_ + size_; }
  const int64_t *end() const noexcept { return dims_ + size_; }

  /// Returns a pointer to the underlying dimension array.
  const int64_t *data() const noexcept { return dims_; }

  /// Element access without bounds checking (matches ``std::vector`` semantics).
  /// Behaviour is undefined when ``i >= size()``.
  int64_t &operator[](size_t i) noexcept { return dims_[i]; }
  /// Element access without bounds checking (matches ``std::vector`` semantics).
  /// Behaviour is undefined when ``i >= size()``.
  const int64_t &operator[](size_t i) const noexcept { return dims_[i]; }

  /// Appends a dimension. Throws ``std::invalid_argument`` when already at ``kMaxRank``.
  void push_back(int64_t v) {
    EXT_ENFORCE_INVALID(size_ < kMaxRank, "Shape rank ", size_, " already at maximum of ", kMaxRank,
                        ".");
    dims_[size_++] = v;
  }

  /// No-op: storage is always inline; provided for interface parity with ``std::vector``.
  void reserve(size_t) noexcept {}

  /// Replaces the contents with ``count`` copies of ``value``.
  /// Throws ``std::invalid_argument`` when ``count`` exceeds ``kMaxRank``.
  void assign(size_t count, int64_t value) {
    EXT_ENFORCE_INVALID(count <= kMaxRank, "Shape rank ", count, " exceeds maximum of ", kMaxRank,
                        ".");
    size_ = count;
    for (size_t i = 0; i < count; ++i)
      dims_[i] = value;
  }

  /// Replaces the contents with the elements in the range ``[first, last)``.
  /// Throws ``std::invalid_argument`` when the range size exceeds ``kMaxRank``.
  template <typename InputIt, typename = std::enable_if_t<!std::is_integral<InputIt>::value>>
  void assign(InputIt first, InputIt last) {
    size_t n = 0;
    for (InputIt it = first; it != last; ++it)
      ++n;
    EXT_ENFORCE_INVALID(n <= kMaxRank, "Shape rank ", n, " exceeds maximum of ", kMaxRank, ".");
    size_ = n;
    size_t i = 0;
    for (InputIt it = first; it != last; ++it)
      dims_[i++] = *it;
  }

  /// Returns a reference to the last dimension.
  /// Throws ``std::invalid_argument`` when the shape is empty.
  int64_t &back() {
    EXT_ENFORCE_INVALID(size_ > 0, "Cannot call back() on an empty Shape.");
    return dims_[size_ - 1];
  }
  const int64_t &back() const {
    EXT_ENFORCE_INVALID(size_ > 0, "Cannot call back() on an empty Shape.");
    return dims_[size_ - 1];
  }

  /// Inserts ``value`` before the element pointed to by ``pos``.
  /// Throws ``std::invalid_argument`` when already at ``kMaxRank``.
  /// Returns an iterator to the inserted element.
  int64_t *insert(int64_t *pos, int64_t value) {
    EXT_ENFORCE_INVALID(size_ < kMaxRank, "Shape rank ", size_, " already at maximum of ", kMaxRank,
                        ".");
    const size_t idx = static_cast<size_t>(pos - dims_);
    std::memmove(dims_ + idx + 1, dims_ + idx, (size_ - idx) * sizeof(int64_t));
    dims_[idx] = value;
    ++size_;
    return dims_ + idx;
  }

  /// Inserts elements from ``[first, last)`` before the element pointed to by ``pos``.
  /// Throws ``std::invalid_argument`` when the resulting rank would exceed ``kMaxRank``.
  /// Returns an iterator to the first inserted element.
  template <typename InputIt> int64_t *insert(int64_t *pos, InputIt first, InputIt last) {
    size_t n = 0;
    for (InputIt it = first; it != last; ++it)
      ++n;
    EXT_ENFORCE_INVALID(size_ + n <= kMaxRank, "Shape rank ", size_ + n, " exceeds maximum of ",
                        kMaxRank, ".");
    const size_t idx = static_cast<size_t>(pos - dims_);
    std::memmove(dims_ + idx + n, dims_ + idx, (size_ - idx) * sizeof(int64_t));
    size_t i = idx;
    for (InputIt it = first; it != last; ++it)
      dims_[i++] = *it;
    size_ += n;
    return dims_ + idx;
  }

  bool operator==(const Shape &other) const noexcept {
    if (size_ != other.size_)
      return false;
    for (size_t i = 0; i < size_; ++i)
      if (dims_[i] != other.dims_[i])
        return false;
    return true;
  }

  bool operator==(const std::vector<int64_t> &v) const noexcept {
    if (size_ != v.size())
      return false;
    for (size_t i = 0; i < size_; ++i)
      if (dims_[i] != v[i])
        return false;
    return true;
  }

  bool operator!=(const Shape &other) const noexcept { return !(*this == other); }
  bool operator!=(const std::vector<int64_t> &v) const noexcept { return !(*this == v); }

private:
  int64_t dims_[kMaxRank];
  size_t size_;
};

/// Symmetric comparison: ``std::vector<int64_t> == Shape``.
inline bool operator==(const std::vector<int64_t> &v, const Shape &s) noexcept { return s == v; }
/// Symmetric comparison: ``std::vector<int64_t> != Shape``.
inline bool operator!=(const std::vector<int64_t> &v, const Shape &s) noexcept { return s != v; }

/**
 * DefaultInitAllocator — allocator that default-initialises elements.
 *
 * A drop-in ``std::allocator`` replacement whose only difference is that a
 * value-initialisation request with no arguments (as issued by
 * ``std::vector::resize(n)`` or ``std::vector(n)``) performs *default*
 * initialisation instead. For a trivially constructible element type such as
 * ``uint8_t`` this leaves the newly created bytes uninitialised rather than
 * zero-filling them, which avoids a needless ``memset`` when the buffer will be
 * fully overwritten (for example a kernel result). All other construction (with
 * explicit arguments) behaves exactly like ``std::allocator``.
 */
template <typename T, typename Allocator = std::allocator<T>>
class DefaultInitAllocator : public Allocator {
  using traits = std::allocator_traits<Allocator>;

public:
  template <typename U> struct rebind {
    using other = DefaultInitAllocator<U, typename traits::template rebind_alloc<U>>;
  };

  using Allocator::Allocator;
  DefaultInitAllocator() noexcept = default;

  template <typename U, typename A>
  explicit DefaultInitAllocator(const DefaultInitAllocator<U, A> &other) noexcept
      : Allocator(static_cast<const A &>(other)) {}

  /// Default-initialises (leaves uninitialised for scalars) instead of
  /// value-initialising.
  template <typename U>
  void construct(U *ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
    ::new (static_cast<void *>(ptr)) U;
  }

  /// Forwards all other constructions to the wrapped allocator unchanged.
  template <typename U, typename... Args> void construct(U *ptr, Args &&...args) {
    traits::construct(static_cast<Allocator &>(*this), ptr, std::forward<Args>(args)...);
  }
};

/// Byte storage that skips zero-initialisation on resize/allocation.
using RawByteBuffer = std::vector<uint8_t, DefaultInitAllocator<uint8_t>>;

/**
 * RawBuffer — an owned byte buffer equivalent to ``std::vector<uint8_t>``.
 *
 * Wraps a ``std::vector<uint8_t>`` (backed by :cpp:class:`DefaultInitAllocator`)
 * under a dedicated type name to make the ownership semantics of raw element
 * bytes explicit in the ``Tensor`` struct and to provide a natural extension
 * point should the storage strategy change in the future. Because the backing
 * allocator default-initialises, :cpp:func:`resize` and the size constructor
 * leave the fresh bytes uninitialised rather than zero-filling them.
 *
 * The full ``std::vector<uint8_t>`` interface subset needed by the codebase
 * is exposed: ``size``, ``empty``, ``data``, ``begin``/``end``, indexed
 * access, ``assign``, and ``resize``.  Implicit conversions to and from
 * ``std::vector<uint8_t>`` are provided for backward compatibility.
 */
struct RawBuffer {
  RawBuffer() = default;

  /// Constructs a buffer of ``n`` bytes whose contents are left uninitialised.
  explicit RawBuffer(size_t n) { storage_.resize(n); }

  RawBuffer(const std::vector<uint8_t> &v) : storage_(v.begin(), v.end()) {}
  // ``std::vector<uint8_t>`` uses the default allocator, so its storage cannot
  // be adopted by ``RawByteBuffer`` (different allocator type); this rvalue
  // overload therefore copies element-by-element like the const-ref one. It is
  // kept only for source compatibility with callers that pass an rvalue. Prefer
  // the ``RawByteBuffer&&`` overload below when a genuine move is wanted.
  RawBuffer(std::vector<uint8_t> &&v) : storage_(v.begin(), v.end()) {}
  RawBuffer(RawByteBuffer &&v) noexcept : storage_(std::move(v)) {}

  RawBuffer(const RawBuffer &) = default;
  RawBuffer(RawBuffer &&) noexcept = default;
  RawBuffer &operator=(const RawBuffer &) = default;
  RawBuffer &operator=(RawBuffer &&) noexcept = default;

  RawBuffer &operator=(const std::vector<uint8_t> &v) {
    storage_.assign(v.begin(), v.end());
    return *this;
  }

  // Copies rather than moves: ``std::vector<uint8_t>`` uses a different
  // allocator than ``RawByteBuffer``, so its storage cannot be adopted. Kept
  // for source compatibility with callers that assign from an rvalue.
  RawBuffer &operator=(std::vector<uint8_t> &&v) {
    storage_.assign(v.begin(), v.end());
    return *this;
  }

  /// Implicitly converts to ``std::vector<uint8_t>`` for backward compatibility.
  operator std::vector<uint8_t>() const { return {storage_.begin(), storage_.end()}; }

  bool operator==(const RawBuffer &other) const noexcept { return storage_ == other.storage_; }
  bool operator!=(const RawBuffer &other) const noexcept { return storage_ != other.storage_; }
  bool operator==(const std::vector<uint8_t> &v) const noexcept {
    return storage_.size() == v.size() && std::equal(storage_.begin(), storage_.end(), v.begin());
  }
  bool operator!=(const std::vector<uint8_t> &v) const noexcept { return !(*this == v); }

  size_t size() const noexcept { return storage_.size(); }
  bool empty() const noexcept { return storage_.empty(); }

  uint8_t *data() noexcept { return storage_.data(); }
  const uint8_t *data() const noexcept { return storage_.data(); }

  uint8_t &operator[](size_t i) noexcept { return storage_[i]; }
  const uint8_t &operator[](size_t i) const noexcept { return storage_[i]; }

  auto begin() noexcept { return storage_.begin(); }
  auto end() noexcept { return storage_.end(); }
  auto begin() const noexcept { return storage_.begin(); }
  auto end() const noexcept { return storage_.end(); }

  /// Moves the underlying byte storage out of the buffer, leaving it empty.
  ///
  /// Returns the owned :cpp:type:`RawByteBuffer` by move so callers can take
  /// ownership of the bytes (for example to hand them to NumPy through a
  /// DLPack-style capsule) without copying. After the call the buffer is
  /// empty (``size() == 0``).
  RawByteBuffer release() noexcept { return std::move(storage_); }

  /// Fills the buffer with ``count`` copies of ``value``, resizing as needed.
  void assign(size_t count, uint8_t value) { storage_.assign(count, value); }

  /// Replaces the buffer contents with the bytes from ``[first, last)``.
  template <typename InputIt> void assign(InputIt first, InputIt last) {
    storage_.assign(first, last);
  }

  /// Resizes the buffer to ``count`` bytes. Newly added bytes are left
  /// uninitialised (the backing allocator default-initialises), so callers must
  /// fully overwrite the buffer before reading it.
  void resize(size_t count) { storage_.resize(count); }

private:
  RawByteBuffer storage_;
};

class RawBufferAllocator;

/**
 * Tensor — minimal runtime tensor used by backend test cases.
 *
 * This struct is intentionally distinct from ``TensorProto``: it carries no
 * protobuf wire dependency, owns its bytes in row-major little-endian layout,
 * and is meant to be consumed directly by a runtime exercising a single
 * backend test node case.
 */
struct Tensor {
  /// Optional name of the tensor (input/output name in the test model).
  std::string name;
  /// Element data type stored as a ``DataType`` integer value.
  int32_t data_type = 0;
  /// Tensor shape; an empty shape denotes a scalar (element_count == 1).
  Shape shape;
  /// Raw element bytes in row-major little-endian layout (owned storage).
  ///
  /// Empty when the tensor uses a borrowed (non-owning) view — use
  /// :cpp:func:`bytes` and :cpp:func:`size_bytes` to access element bytes
  /// regardless of storage mode.  Also empty when ``data_type`` is
  /// ``DataType::STRING``; in that case the element values are stored in
  /// ``string_data`` instead.
  RawBuffer data;

  /// String element values in row-major layout. Populated only when
  /// ``data_type`` is ``DataType::STRING`` and the tensor owns its string
  /// storage; empty for all other element types and for borrowed string
  /// views.
  std::vector<std::string> string_data;

  Tensor() = default;
  Tensor(std::string n, int32_t dt, Shape s, std::vector<uint8_t> d)
      : name(std::move(n)), data_type(dt), shape(std::move(s)), data(std::move(d)) {}

  /// Releases the allocator-backed allocation, if any (no-op for inline or
  /// borrowed storage). Makes ``Tensor`` self-owning so callers no longer
  /// need to manually free the allocation before a ``Tensor`` is destroyed
  /// or overwritten.
  ~Tensor();

  /// Deep-copies ``other``. When ``other`` is allocator-backed
  /// (``has_allocation()``), a fresh buffer is acquired from the *same*
  /// allocator and the bytes are duplicated, so the copy never aliases
  /// ``other``'s allocation — the two tensors can be freed independently
  /// without a double free. Borrowed (non-owning) views are copied by
  /// reference, as before.
  Tensor(const Tensor &other);
  Tensor &operator=(const Tensor &other);

  /// Transfers ownership of any allocator-backed allocation from ``other``
  /// to ``*this`` and resets ``other`` to a non-owning empty state, so
  /// ``other`` can be safely destroyed or overwritten afterwards (e.g. left
  /// behind in a map after ``std::move(it->second)``) without triggering a
  /// double free.
  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;

  /// Constructs a ``STRING`` tensor whose elements live in ``string_data``.
  /// Distinct from the bytes-based constructor so brace-enclosed
  /// ``{ ... }`` initializer lists at call sites are unambiguous.
  static Tensor MakeString(std::string n, Shape s, std::vector<std::string> sd) {
    Tensor t;
    t.name = std::move(n);
    t.data_type = static_cast<int32_t>(DataType::STRING);
    t.shape = std::move(s);
    t.string_data = std::move(sd);
    return t;
  }

  /**
   * Creates a non-owning (borrowed) ``Tensor`` that references an external
   * byte buffer without copying.
   *
   * The pointed-to buffer at ``ptr[0 .. sz-1]`` **MUST** outlive this
   * ``Tensor``.  Both the const and non-const ``As<T>()`` overloads (and
   * ``AsBool()``) are available on borrowed tensors; the non-const overloads
   * return a ``T *`` via ``const_cast``.  Callers must not write through that
   * pointer if the underlying storage is immutable — doing so is undefined
   * behaviour.
   *
   * @param name  Tensor name.
   * @param dtype Element data type (a ``DataType`` integer value).
   * @param shape Tensor shape.
   * @param ptr   Pointer to the first byte of element data.
   * @param sz    Total byte count of the element buffer.
   * @return      A ``Tensor`` backed by the external buffer.
   */
  static Tensor Borrow(std::string name, int32_t dtype, Shape shape, const uint8_t *ptr, size_t sz);

  /// Creates a non-owning (borrowed) ``STRING`` tensor that references an
  /// external string vector without copying.
  ///
  /// The referenced string vector **MUST** outlive this ``Tensor``.
  static Tensor BorrowStrings(std::string name, Shape shape,
                              const std::vector<std::string> &strings);

  /// Returns a pointer to the raw element bytes.
  /// Works for both owned (``data``) and borrowed (non-owning view) tensors.
  const uint8_t *bytes() const noexcept {
    if (allocation_ != nullptr) {
      return allocation_->data();
    }
    return borrow_ptr_ != nullptr ? borrow_ptr_ : data.data();
  }

  /// Returns a mutable pointer to the raw element bytes.
  /// Works for owned tensors and allocator-backed tensors. For borrowed
  /// tensors the underlying storage may be read-only; callers must not write
  /// through the returned pointer when the tensor was created via
  /// :cpp:func:`Borrow` with an immutable backing buffer.
  uint8_t *mutable_bytes() noexcept {
    if (allocation_ != nullptr) {
      return allocation_->data();
    }
    // For borrowed tensors borrow_ptr_ is const; cast away the const here.
    // Callers are responsible for ensuring the backing storage is writable.
    return borrow_ptr_ != nullptr ? const_cast<uint8_t *>(borrow_ptr_) : data.data();
  }

  /// Returns the total number of raw element bytes.
  /// Works for both owned and borrowed tensors.
  size_t size_bytes() const noexcept {
    if (allocation_ != nullptr) {
      return allocation_->size();
    }
    return borrow_ptr_ != nullptr ? borrow_size_ : data.size();
  }

  /// Marks the tensor storage as allocator-backed.
  /// Callers must release/clear any existing allocation first.
  void SetAllocation(RawBufferAllocator *allocator_owner, RawBuffer *allocation) {
    EXT_ENFORCE(allocation_ == nullptr, "Tensor::SetAllocation: allocation is already set.");
    EXT_ENFORCE(allocator_owner != nullptr,
                "Tensor::SetAllocation: allocator owner must not be null.");
    EXT_ENFORCE(allocation != nullptr, "Tensor::SetAllocation: allocation must not be null.");
    allocation_owner_ = allocator_owner;
    allocation_ = allocation;
    // Drop inline storage: bytes()/size_bytes() now resolve from allocation_.
    data = RawBuffer{};
    borrow_ptr_ = nullptr;
    borrow_size_ = 0;
  }

  bool has_allocation() const noexcept { return allocation_ != nullptr; }

  /// Returns whether the tensor is a non-owning (borrowed) view over external
  /// memory (created via :cpp:func:`Borrow` / :cpp:func:`BorrowStrings`, e.g. a
  /// zero-copy view into a ``TensorProto``'s ``raw_data``). Borrowed tensors do
  /// not own their bytes: the backing storage must outlive the tensor.
  bool is_borrowed() const noexcept {
    return borrow_ptr_ != nullptr || borrow_string_data_ != nullptr;
  }

  /// Returns an owned deep copy of this tensor that references no external
  /// memory: the bytes (or, for ``STRING`` tensors, the strings) are copied
  /// into inline storage the returned tensor owns. Use this to detach a
  /// borrowed view (see :cpp:func:`is_borrowed`) from its backing buffer — for
  /// example a graph output that borrows into a ``TensorProto``'s ``raw_data``
  /// — so it stays valid once that buffer is released. Allocator-backed and
  /// already-owned tensors are copied inline as well.
  Tensor ToOwned() const;

  RawBuffer *allocation() const {
    EXT_ENFORCE(allocation_ != nullptr, "Tensor::allocation: tensor is not allocator-backed.");
    return allocation_;
  }
  RawBufferAllocator *allocation_owner() const {
    EXT_ENFORCE(allocation_owner_ != nullptr,
                "Tensor::allocation_owner: tensor is not allocator-backed.");
    return allocation_owner_;
  }
  void ClearAllocation() noexcept {
    allocation_ = nullptr;
    allocation_owner_ = nullptr;
  }

  /// Returns the product of all shape dimensions; 1 for an empty shape.
  int64_t element_count() const;

  /// Returns the size in bytes of one element of ``data_type``.
  /// Throws ``std::invalid_argument`` for unsupported types.
  size_t element_size() const;

  /// Typed factories that construct a tensor of the given shape and copy the
  /// provided values into ``data``. They throw ``std::invalid_argument`` if
  /// any dimension in ``shape`` is negative or if ``values.size()`` does not
  /// match ``prod(shape)``.
  ///
  /// The templated ``From<T>`` factory is the generic version. The non-template
  /// ``FromFloat``/``FromDouble``/``FromInt32``/``FromInt64`` are thin wrappers
  /// kept for source compatibility.
  ///
  /// When ``allocator`` is non-null the element bytes are acquired from it (via
  /// :cpp:func:`MakeOutputTensor`) and the returned tensor is allocator-backed;
  /// when null the tensor uses inline ``std::vector`` storage (the legacy path).
  /// Kernels producing a result should pass the runtime context allocator so no
  /// output buffer is allocated outside it.
  template <typename T>
  static Tensor From(const std::string &name, const Shape &shape, const std::vector<T> &values,
                     RawBufferAllocator *allocator = nullptr);

  static Tensor FromFloat(const std::string &name, const Shape &shape,
                          const std::vector<float> &values,
                          RawBufferAllocator *allocator = nullptr);
  static Tensor FromDouble(const std::string &name, const Shape &shape,
                           const std::vector<double> &values,
                           RawBufferAllocator *allocator = nullptr);
  static Tensor FromInt32(const std::string &name, const Shape &shape,
                          const std::vector<int32_t> &values,
                          RawBufferAllocator *allocator = nullptr);
  static Tensor FromInt64(const std::string &name, const Shape &shape,
                          const std::vector<int64_t> &values,
                          RawBufferAllocator *allocator = nullptr);
  static Tensor FromInt8(const std::string &name, const Shape &shape,
                         const std::vector<int8_t> &values,
                         RawBufferAllocator *allocator = nullptr);
  static Tensor FromUint8(const std::string &name, const Shape &shape,
                          const std::vector<uint8_t> &values,
                          RawBufferAllocator *allocator = nullptr);
  static Tensor FromInt16(const std::string &name, const Shape &shape,
                          const std::vector<int16_t> &values,
                          RawBufferAllocator *allocator = nullptr);
  static Tensor FromUint16(const std::string &name, const Shape &shape,
                           const std::vector<uint16_t> &values,
                           RawBufferAllocator *allocator = nullptr);
  static Tensor FromUint32(const std::string &name, const Shape &shape,
                           const std::vector<uint32_t> &values,
                           RawBufferAllocator *allocator = nullptr);
  static Tensor FromUint64(const std::string &name, const Shape &shape,
                           const std::vector<uint64_t> &values,
                           RawBufferAllocator *allocator = nullptr);
  /// Constructs a ``BOOL`` tensor; element values are stored as one byte each
  /// (0 == false, non-zero == true). Provided as a ``uint8_t`` vector so the
  /// usual ``std::vector<bool>`` packing pitfalls are avoided.
  static Tensor FromBool(const std::string &name, const Shape &shape,
                         const std::vector<uint8_t> &values,
                         RawBufferAllocator *allocator = nullptr);
  /// Constructs a ``STRING`` tensor whose elements are the provided UTF-8
  /// strings (stored in ``string_data``). Throws ``std::invalid_argument`` if
  /// any dimension in ``shape`` is negative or if ``values.size()`` does not
  /// match ``prod(shape)``.
  static Tensor FromStrings(const std::string &name, const Shape &shape,
                            const std::vector<std::string> &values);

  /// Typed views over the underlying ``data`` buffer. They throw if the
  /// requested type does not match ``data_type``.
  ///
  /// The templated ``As<T>()`` accessor is the generic version. The non-template
  /// ``AsFloat``/``AsDouble``/``AsInt32``/``AsInt64`` are thin wrappers kept for
  /// source compatibility.
  template <typename T> const T *As() const;
  template <typename T> T *As();

  const float *AsFloat() const;
  float *AsFloat();
  const double *AsDouble() const;
  double *AsDouble();
  const int32_t *AsInt32() const;
  int32_t *AsInt32();
  const int64_t *AsInt64() const;
  int64_t *AsInt64();
  const int8_t *AsInt8() const;
  int8_t *AsInt8();
  const uint8_t *AsUint8() const;
  uint8_t *AsUint8();
  const int16_t *AsInt16() const;
  int16_t *AsInt16();
  const uint16_t *AsUint16() const;
  uint16_t *AsUint16();
  const uint32_t *AsUint32() const;
  uint32_t *AsUint32();
  const uint64_t *AsUint64() const;
  uint64_t *AsUint64();
  /// Typed view over ``data`` for ``BOOL`` element type, stored one byte per
  /// element. The byte value is 0 for false and non-zero (canonically 1) for
  /// true.
  const uint8_t *AsBool() const;
  uint8_t *AsBool();

  /// Typed view over the underlying ``string_data`` buffer. Throws
  /// ``std::invalid_argument`` if ``data_type`` is not
  /// ``DataType::STRING``. Borrowed string tensors return a const reference to
  /// the external backing vector; requesting a non-const view of borrowed
  /// string storage throws ``std::invalid_argument``.
  const std::vector<std::string> &AsStrings() const;
  std::vector<std::string> &AsStrings();

private:
  /// Non-null when the tensor bytes are owned by a ``RawBufferAllocator``.
  RawBuffer *allocation_ = nullptr;
  /// Allocator owning ``allocation_``.
  RawBufferAllocator *allocation_owner_ = nullptr;
  /// Non-null only for borrowed (non-owning) tensors created via
  /// :cpp:func:`Borrow`.  When set, element bytes are read from
  /// ``borrow_ptr_[0 .. borrow_size_-1]`` rather than from ``data``.
  const uint8_t *borrow_ptr_ = nullptr;
  size_t borrow_size_ = 0;
  const std::vector<std::string> *borrow_string_data_ = nullptr;
};

/// ``Tensors`` — the runtime value produced by kernels that emit more than one
/// tensor (for example ``Split``, ``Loop`` or the training optimizers).
///
/// It is an ordered, owning list of :cpp:struct:`Tensor` values. Derives from
/// ``std::vector<Tensor>`` and inherits its constructors, behaving exactly like
/// the underlying vector while giving kernel signatures a named type to express
/// "a list of created tensors" instead of spelling out ``std::vector<Tensor>``
/// at every call site.
class Tensors : public std::vector<Tensor> {
public:
  using std::vector<Tensor>::vector;
  Tensors() = default;
  Tensors(const std::vector<Tensor> &values) : std::vector<Tensor>(values) {}
  Tensors(std::vector<Tensor> &&values) : std::vector<Tensor>(std::move(values)) {}
};

/// Trait mapping a C++ element type to its ``DataType`` value.
/// Specialize to support additional element types in ``Tensor::From``/``As``.
template <typename T> struct TensorElementType; // primary template intentionally undefined

#define ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(CPP_TYPE, ENUM_VALUE)                               \
  template <> struct TensorElementType<CPP_TYPE> {                                                 \
    static constexpr int32_t value = ENUM_VALUE;                                                   \
  }

ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(float, DataType::FLOAT);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(double, DataType::DOUBLE);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(int16_t, DataType::INT16);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(int32_t, DataType::INT32);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(int64_t, DataType::INT64);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(int8_t, DataType::INT8);
// Note: ``uint8_t`` aliases both ``UINT8`` and ``BOOL`` element storage; the
// trait maps it to ``UINT8`` and ``BOOL`` accessors go through ``AsBool``
// which uses the same byte layout but validates ``data_type == BOOL``.
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(uint8_t, DataType::UINT8);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(uint16_t, DataType::UINT16);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(uint32_t, DataType::UINT32);
ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE(uint64_t, DataType::UINT64);

#undef ONNX_LIGHT_DECLARE_TENSOR_ELEMENT_TYPE

// Forward declaration so the allocator-backed path of the ``From<T>`` template
// below resolves; the full declaration with documentation follows later.
Tensor MakeOutputTensor(int32_t data_type, const Shape &shape, size_t n_bytes,
                        RawBufferAllocator *allocator);

template <typename T>
Tensor Tensor::From(const std::string &name, const Shape &shape, const std::vector<T> &values,
                    RawBufferAllocator *allocator) {
  for (int64_t d : shape) {
    EXT_ENFORCE_INVALID(d >= 0, "Tensor shape dimensions must be non-negative.");
  }
  const int64_t expected = shape.product();
  EXT_ENFORCE_INVALID(static_cast<int64_t>(values.size()) == expected,
                      "Tensor values size does not match the product of shape.");
  const size_t n_bytes = values.size() * sizeof(T);
  if (allocator != nullptr) {
    Tensor t = MakeOutputTensor(TensorElementType<T>::value, shape, n_bytes, allocator);
    t.name = name;
    if (!values.empty()) {
      std::memcpy(t.mutable_bytes(), values.data(), n_bytes);
    }
    return t;
  }
  std::vector<uint8_t> bytes(n_bytes);
  if (!values.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return Tensor(name, TensorElementType<T>::value, shape, std::move(bytes));
}

template <typename T> const T *Tensor::As() const {
  EXT_ENFORCE_INVALID(data_type == TensorElementType<T>::value,
                      "Tensor data_type does not match the requested view type.");
  return reinterpret_cast<const T *>(bytes());
}

template <typename T> T *Tensor::As() {
  EXT_ENFORCE_INVALID(data_type == TensorElementType<T>::value,
                      "Tensor data_type does not match the requested view type.");
  // For borrowed tensors borrow_ptr_ is const uint8_t*; const_cast is used so
  // both owned and borrowed tensors can be accessed through this overload.
  // Callers must not write through the returned pointer when the tensor is
  // borrowed — doing so is undefined behaviour if the underlying storage is
  // immutable (e.g. a string literal or a read-only mapping).
  return reinterpret_cast<T *>(const_cast<uint8_t *>(bytes()));
}

/// Returns the size in bytes of one element of ``dtype``
/// (a ``DataType`` integer). Throws ``std::invalid_argument``
/// for unsupported types. Sub-byte packed dtypes (INT4/UINT4/INT2/UINT2)
/// are not supported by this helper because they do not have a whole-byte
/// per-element size; use ``PackedByteSize`` instead.
size_t ElementSize(int32_t dtype);

/// Returns the storage size in bytes for ``element_count`` elements of
/// ``dtype``. Whole-byte dtypes return ``element_count * ElementSize(dtype)``;
/// sub-byte packed dtypes pack 2 (INT4/UINT4) or 4 (INT2/UINT2) elements per
/// byte and ``element_count`` is rounded up to fill the trailing byte. Throws
/// ``std::invalid_argument`` for unsupported types.
size_t PackedByteSize(int32_t dtype, int64_t element_count);

/// Fills ``vi`` with the type/shape information described by ``tensor``.
/// ``vi.name`` is set to ``tensor.name``.
void FillValueInfo(const Tensor &tensor, ValueInfoProto &vi);

/**
 * Converts a ``TensorProto`` to a :cpp:class:`Tensor`.
 *
 * Supports all numeric data types stored either in the typed repeated
 * fields (``float_data``, ``int32_data``, ``int64_data``, ``double_data``,
 * ``uint64_data``) or in the raw little-endian ``raw_data`` field.
 * ``STRING`` tensors are read from ``string_data``.
 *
 * The resulting ``Tensor::name`` is set from ``tp.name()``; the shape is
 * taken from ``tp.dims()``.
 *
 * For the typed-field path the byte buffer is acquired from ``allocator``
 * when it is non-null (the returned tensor is then allocator-backed); when
 * ``allocator`` is null an inline ``std::vector<uint8_t>`` is used. The
 * ``raw_data`` path always returns a borrowed (zero-copy) view and ignores
 * ``allocator``.
 *
 * @param tp        The source ``TensorProto``.
 * @param allocator Optional allocator for the typed-field byte buffer; may
 *                  be ``nullptr``.
 * @return          A ``Tensor`` whose data matches the content of ``tp``.
 *
 * @throws std::invalid_argument for unsupported ``data_type`` values.
 */
Tensor TensorFromProto(const TensorProto &tp, RawBufferAllocator *allocator = nullptr);

/**
 * Creates an empty output tensor of ``n_bytes`` bytes with the given
 * ``data_type`` and ``shape``.
 *
 * When ``allocator`` is non-null the byte buffer is acquired from it via
 * :cpp:func:`RawBufferAllocator::Allocate` and the returned tensor is
 * allocator-backed (``has_allocation()`` returns ``true``). When ``allocator``
 * is null the tensor uses an inline ``std::vector<uint8_t>`` of ``n_bytes``
 * bytes (the legacy path). In both cases the buffer contents are left
 * uninitialised: the caller is expected to fully overwrite the result, so no
 * time is spent zero-filling the memory.
 *
 * @param data_type   ONNX element type (a ``TensorProto::DataType`` integer).
 * @param shape       Output shape.
 * @param n_bytes     Total byte size of the output buffer
 *                    (``element_count × element_size``).
 * @param allocator   Optional allocator; may be ``nullptr``.
 */
Tensor MakeOutputTensor(int32_t data_type, const Shape &shape, size_t n_bytes,
                        RawBufferAllocator *allocator);

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
