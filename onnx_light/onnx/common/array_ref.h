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
 * A non-owning view over a contiguous sequence of elements.
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
  /** An empty reference. */
  /*implicit*/ ArrayRef() : data_(nullptr), length_(0) {}
  /** A single-element reference. */
  /*implicit*/ ArrayRef(const T &one_elt) : data_(&one_elt), length_(1) {}
  /** A reference from a pointer and explicit length. */
  /*implicit*/ ArrayRef(const T *data, size_t length) : data_(data), length_(length) {}
  /** A reference from a half-open pointer range [begin, end). */
  ArrayRef(const T *begin, const T *end) : data_(begin), length_(end - begin) {}

  template <typename A>
  /*implicit*/ ArrayRef(const std::vector<T, A> &vec) : data_(vec.data()), length_(vec.size()) {}

  template <size_t N>
  /*implicit*/ constexpr ArrayRef(const std::array<T, N> &arr) : data_(arr.data()), length_(N) {}

  template <size_t N>
  /*implicit*/ constexpr ArrayRef(const T (&arr)[N]) : data_(arr), length_(N) {}

  /*implicit*/ ArrayRef(const std::initializer_list<T> &vec)
      : data_(vec.begin() == vec.end() ? static_cast<T *>(nullptr) : vec.begin()),
        length_(vec.size()) {}

  iterator begin() const { return data_; }
  iterator end() const { return data_ + length_; }
  reverse_iterator rbegin() const { return reverse_iterator(end()); }
  reverse_iterator rend() const { return reverse_iterator(begin()); }
  bool empty() const { return length_ == 0; }
  const T *data() const { return data_; }
  size_t size() const { return length_; }

  /** The first element. */
  const T &front() const {
    assert(!empty());
    return data_[0];
  }

  /** The last element. */
  const T &back() const {
    assert(!empty());
    return data_[length_ - 1];
  }

  bool equals(ArrayRef rhs) const {
    if (length_ != rhs.length_) {
      return false;
    }
    return std::equal(begin(), end(), rhs.begin());
  }

  ArrayRef<T> slice(size_t n, size_t m) const {
    assert(n + m <= size() && "Invalid specifier");
    return ArrayRef<T>(data() + n, m);
  }

  ArrayRef<T> slice(size_t n) const { return slice(n, size() - n); }

  const T &operator[](size_t index) const {
    assert(index < length_ && "Invalid index!");
    return data_[index];
  }

  const T &at(size_t index) const {
    assert(index < length_ && "Invalid index!");
    return data_[index];
  }

  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>> &operator=(U &&temporary) = delete;

  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>> &operator=(std::initializer_list<U>) = delete;

  std::vector<T> vec() const { return std::vector<T>(data_, data_ + length_); }

  operator std::vector<T>() const { return std::vector<T>(data_, data_ + length_); }
};

} // namespace ONNX_LIGHT_NAMESPACE
