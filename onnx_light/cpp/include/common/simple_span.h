#pragma once

#include <cstddef>

/// Namespace that provides lightweight standard-library-like utilities.
namespace std_ {

/**
 * Lightweight non-owning view over a contiguous sequence of elements.
 *
 * Provides a subset of the C++20 `std::span` interface without depending on
 * C++20.  The span does not own the underlying storage; the caller is
 * responsible for ensuring the pointed-to data remains valid for the lifetime
 * of the span.
 *
 * @tparam T Element type of the sequence.
 */
template <typename T> class span {
public:
  /**
   * Constructs a span from a pointer and an element count.
   *
   * @param data Pointer to the first element of the sequence.
   * @param size Number of elements in the sequence.
   */
  span(T *data, std::size_t size) : data_(data), size_(size) {}

  /** Returns a pointer to the first element of the sequence. */
  inline T *data() const { return data_; }

  /** Returns the number of elements in the sequence. */
  inline std::size_t size() const { return size_; }

  /**
   * Returns a reference to the element at the given index.
   *
   * @param index Zero-based position of the element to access.
   * @return Reference to the element at *index*.
   */
  inline T &operator[](std::size_t index) const { return data_[index]; }

  /** Returns a pointer to the first element, suitable for range-based iteration. */
  inline T *begin() const { return data_; }

  /** Returns a pointer one past the last element, suitable for range-based iteration. */
  inline T *end() const { return data_ + size_; }

private:
  /// Pointer to the first element of the viewed sequence.
  T *data_;
  /// Number of elements in the viewed sequence.
  std::size_t size_;
};

} // namespace std_
