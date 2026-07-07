// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_kernels/simple_tensor.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

/**
 * Abstract base class for RawBuffer allocators.
 *
 * Provides a virtual interface for allocating and freeing
 * :cpp:struct:`RawBuffer` instances, enabling custom memory management
 * strategies without changing call sites.
 *
 * All concrete allocators must implement :cpp:func:`Allocate`,
 * :cpp:func:`Free`, and :cpp:func:`TotalAllocatedSize`.
 */
class RawBufferAllocator {
public:
  virtual ~RawBufferAllocator() = default;

  /**
   * Allocates a :cpp:struct:`RawBuffer` of at least ``n_bytes`` bytes.
   *
   * @param n_bytes  Number of bytes to allocate.
   * @returns        Pointer to the allocated :cpp:struct:`RawBuffer`, owned
   *                 by the allocator. The caller must release it with
   *                 :cpp:func:`Free`.
   * @throws std::bad_alloc (or a derived exception) if no free slot is
   *         available.
   */
  virtual RawBuffer *Allocate(size_t n_bytes) = 0;

  /**
   * Releases a previously allocated :cpp:struct:`RawBuffer`.
   *
   * @param buf  Pointer previously returned by :cpp:func:`Allocate`. Passing a
   *             pointer that was not returned by this allocator, or that has
   *             already been freed, is undefined behaviour.
   */
  virtual void Free(RawBuffer *buf) = 0;

  /**
   * Returns the total number of bytes across all currently allocated buffers.
   */
  virtual size_t TotalAllocatedSize() const = 0;
};

/**
 * Fixed-capacity pool allocator for :cpp:struct:`RawBuffer` instances.
 *
 * Manages a pre-allocated array of ``capacity`` :cpp:struct:`RawBuffer`
 * slots together with a parallel boolean array tracking which slots are
 * in use.  :cpp:func:`Allocate` locates the first free slot, resizes it
 * to the requested number of bytes, and marks it as allocated.
 * :cpp:func:`Free` finds the matching slot by pointer, clears its
 * contents, and marks it as free.
 *
 * @note This class is not thread-safe.
 */
class SimpleRawBufferAllocator : public RawBufferAllocator {
public:
  /**
   * Constructs an allocator with the given slot capacity.
   *
   * @param capacity  Maximum number of :cpp:struct:`RawBuffer` instances that
   *                  can be alive at the same time.
   */
  explicit SimpleRawBufferAllocator(size_t capacity)
      : buffers_(capacity), allocated_(capacity, false) {}

  /**
   * Allocates the first free slot, resizes it to ``n_bytes``, and marks it
   * as allocated.
   *
   * @throws std::bad_alloc if all slots are already in use.
   */
  RawBuffer *Allocate(size_t n_bytes) override {
    for (size_t i = 0; i < buffers_.size(); ++i) {
      if (!allocated_[i]) {
        buffers_[i].assign(n_bytes, 0);
        allocated_[i] = true;
        return &buffers_[i];
      }
    }
    throw std::bad_alloc();
  }

  /**
   * Releases the slot pointed to by ``buf``, clears its storage, and marks
   * the slot as free.
   *
   * @throws std::invalid_argument if ``buf`` does not belong to this
   *         allocator.
   */
  void Free(RawBuffer *buf) override {
    for (size_t i = 0; i < buffers_.size(); ++i) {
      if (&buffers_[i] == buf) {
        buffers_[i] = RawBuffer{};
        allocated_[i] = false;
        return;
      }
    }
    throw std::invalid_argument("SimpleRawBufferAllocator::Free: buffer does not belong to this "
                                "allocator.");
  }

  /**
   * Returns the sum of the sizes of all currently allocated buffers.
   */
  size_t TotalAllocatedSize() const override {
    size_t total = 0;
    for (size_t i = 0; i < buffers_.size(); ++i) {
      if (allocated_[i]) {
        total += buffers_[i].size();
      }
    }
    return total;
  }

  /// Returns the total number of slots managed by this allocator.
  size_t capacity() const noexcept { return buffers_.size(); }

  /// Returns the number of slots currently in use.
  size_t allocated_count() const noexcept {
    size_t count = 0;
    for (bool b : allocated_) {
      if (b) {
        ++count;
      }
    }
    return count;
  }

private:
  std::vector<RawBuffer> buffers_;
  std::vector<bool> allocated_;
};

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
