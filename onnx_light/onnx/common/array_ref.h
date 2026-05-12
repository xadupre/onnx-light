// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

//===--- ArrayRef.h - Array Reference Wrapper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// ONNX: modified from llvm::ArrayRef.
// removed llvm-specific functionality
// removed some implicit const -> non-const conversions that rely on
// complicated std::enable_if meta-programming
// removed a bunch of slice variants for simplicity...

#pragma once

#include "onnx_pb.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <iterator>
#include <type_traits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Represents a non-owning view over a contiguous sequence of elements.
 *
 * The referenced storage must outlive the ArrayRef instance.
 */
template <typename T> class ArrayRef {
public:
  using iterator = const T *;
  using const_iterator = const T *;
  using size_type = size_t;
  using reverse_iterator = std::reverse_iterator<iterator>;

private:
  const T *data_;
  size_type length_;

public:
  /** Constructs an empty reference. */
  /*implicit*/ ArrayRef() : data_(nullptr), length_(0) {}
  /** Constructs a single-element reference. */
  /*implicit*/ ArrayRef(const T &one_elt) : data_(&one_elt), length_(1) {}
  /** Constructs a reference from a pointer and explicit length. */
  /*implicit*/ ArrayRef(const T *data, size_t length) : data_(data), length_(length) {}
  /** Constructs a reference from a half-open pointer range [begin, end). */
  ArrayRef(const T *begin, const T *end) : data_(begin), length_(end - begin) {}

  template <typename A>
  /** Constructs an ArrayRef from a vector. */
  /*implicit*/ ArrayRef(const std::vector<T, A> &vec) : data_(vec.data()), length_(vec.size()) {}

  template <size_t N>
  /** Constructs an ArrayRef from a fixed-size std::array. */
  /*implicit*/ constexpr ArrayRef(const std::array<T, N> &arr) : data_(arr.data()), length_(N) {}

  template <size_t N>
  /** Constructs an ArrayRef from a C array. */
  /*implicit*/ constexpr ArrayRef(const T (&arr)[N]) : data_(arr), length_(N) {}

  /** Constructs an ArrayRef from an initializer list. */
  /*implicit*/ ArrayRef(const std::initializer_list<T> &vec)
      : data_(vec.begin() == vec.end() ? static_cast<T *>(nullptr) : vec.begin()),
        length_(vec.size()) {}

  /** Returns an iterator to the beginning. */
  iterator begin() const { return data_; }
  /** Returns an iterator to the end. */
  iterator end() const { return data_ + length_; }
  /** Returns a reverse iterator to the beginning of the reversed range. */
  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  /** Returns a reverse iterator to the end of the reversed range. */
  reverse_iterator rend() const { return reverse_iterator(begin()); }
  /** Returns true if the reference is empty. */
  bool empty() const { return length_ == 0; }
  /** Returns the underlying data pointer. */
  const T *data() const { return data_; }
  /** Returns the number of elements. */
  size_t size() const { return length_; }

  /** Returns the first element. Precondition: the reference is non-empty. */
  const T &front() const {
    assert(!empty());
    return data_[0];
  }

  /** Returns the last element. Precondition: the reference is non-empty. */
  const T &back() const {
    assert(!empty());
    return data_[length_ - 1];
  }

  /** Returns true if both references contain the same values. */
  bool equals(ArrayRef rhs) const {
    if (length_ != rhs.length_) {
      return false;
    }
    return std::equal(begin(), end(), rhs.begin());
  }

  /** Returns a sub-reference starting at n with length m. */
  ArrayRef<T> slice(size_t n, size_t m) const {
    assert(n + m <= size() && "Invalid specifier");
    return ArrayRef<T>(data() + n, m);
  }

  /** Returns a sub-reference starting at n until the end. */
  ArrayRef<T> slice(size_t n) const { return slice(n, size() - n); }

  /** Returns the element at index. Precondition: index is in range. */
  const T &operator[](size_t index) const {
    assert(index < length_ && "Invalid index!");
    return data_[index];
  }

  /** Returns the element at index. Precondition: index is in range. */
  const T &at(size_t index) const {
    assert(index < length_ && "Invalid index!");
    return data_[index];
  }

  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>> &operator=(U &&temporary) = delete;

  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>> &operator=(std::initializer_list<U>) = delete;

  /** Returns a vector copy of the referenced elements. */
  std::vector<T> vec() const { return std::vector<T>(data_, data_ + length_); }

  /** Converts to a vector copy of the referenced elements. */
  operator std::vector<T>() const { return std::vector<T>(data_, data_ + length_); }
};

} // namespace ONNX_LIGHT_NAMESPACE
