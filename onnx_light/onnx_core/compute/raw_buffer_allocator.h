// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/simple_tensor.h"

#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

/**
 * Abstract base class for RawBuffer allocators.
 *
 * Provides a virtual interface for allocating and freeing
 * :cpp:struct:`RawBuffer` instances, enabling custom memory management
 * strategies without changing call sites.
 *
 * All concrete allocators must implement :cpp:func:`Allocate`,
 * :cpp:func:`Free`, :cpp:func:`TotalAllocatedSize`,
 * :cpp:func:`PeakAllocatedSize`, and :cpp:func:`ResetPeak`.
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

  /**
   * Returns the memory peak, i.e. the maximum value ever reached by
   * :cpp:func:`TotalAllocatedSize` since construction or since the last call to
   * :cpp:func:`ResetPeak`.
   */
  virtual size_t PeakAllocatedSize() const = 0;

  /**
   * Resets the memory peak to the current value of
   * :cpp:func:`TotalAllocatedSize`.
   */
  virtual void ResetPeak() = 0;
};

/**
 * Fixed-capacity pool allocator for :cpp:struct:`RawBuffer` instances.
 *
 * Manages a pre-allocated array of ``capacity`` :cpp:struct:`RawBuffer`
 * slots. A free-slot stack gives O(1) :cpp:func:`Allocate`, a pointer-to-index
 * map gives O(1) :cpp:func:`Free`, and running counters give O(1)
 * :cpp:func:`TotalAllocatedSize` and :cpp:func:`allocated_count`.
 *
 * @note This class is not thread-safe.
 */
class SimpleRawBufferAllocator : public RawBufferAllocator {
public:
  /**
   * Constructs an allocator with the given slot capacity.
   *
   * All ``capacity`` slot addresses are stable for the lifetime of this object
   * because ``buffers_`` is never resized after construction.
   *
   * @param capacity  Maximum number of :cpp:struct:`RawBuffer` instances that
   *                  can be alive at the same time.
   */
  explicit SimpleRawBufferAllocator(size_t capacity) : buffers_(capacity) {
    free_slots_.reserve(capacity);
    index_map_.reserve(capacity);
    // Push indices in reverse so slot 0 is popped first.
    for (size_t i = capacity; i-- > 0;) {
      free_slots_.push_back(i);
    }
  }

  /**
   * Pops a free slot from the stack, resizes it to ``n_bytes``, records the
   * pointer in the index map, and returns the slot pointer — O(1).
   *
   * @throws std::bad_alloc if all slots are already in use.
   */
  RawBuffer *Allocate(size_t n_bytes) override {
    if (free_slots_.empty()) {
      throw std::bad_alloc();
    }
    const size_t i = free_slots_.back();
    free_slots_.pop_back();
    buffers_[i].assign(n_bytes, 0);
    index_map_[&buffers_[i]] = i;
    total_allocated_size_ += n_bytes;
    if (total_allocated_size_ > peak_allocated_size_) {
      peak_allocated_size_ = total_allocated_size_;
    }
    ++allocated_count_;
    return &buffers_[i];
  }

  /**
   * Looks up the slot index in the pointer map (O(1)), clears the slot,
   * removes it from the map, and pushes the index back onto the free stack.
   *
   * @throws std::invalid_argument if ``buf`` does not belong to this
   *         allocator.
   */
  void Free(RawBuffer *buf) override {
    const auto it = index_map_.find(buf);
    if (it == index_map_.end()) {
      throw std::invalid_argument(
          "SimpleRawBufferAllocator::Free: buffer does not belong to this allocator.");
    }
    const size_t i = it->second;
    total_allocated_size_ -= buffers_[i].size();
    --allocated_count_;
    buffers_[i] = RawBuffer{};
    index_map_.erase(it);
    free_slots_.push_back(i);
  }

  /// Returns the sum of the sizes of all currently allocated buffers — O(1).
  size_t TotalAllocatedSize() const override { return total_allocated_size_; }

  /**
   * Returns the memory peak — the maximum value ever reached by
   * :cpp:func:`TotalAllocatedSize` since construction or the last
   * :cpp:func:`ResetPeak` — O(1).
   */
  size_t PeakAllocatedSize() const override { return peak_allocated_size_; }

  /// Resets the memory peak to the current :cpp:func:`TotalAllocatedSize` — O(1).
  void ResetPeak() override { peak_allocated_size_ = total_allocated_size_; }

  /// Returns the total number of slots managed by this allocator.
  size_t capacity() const noexcept { return buffers_.size(); }

  /// Returns the number of slots currently in use — O(1).
  size_t allocated_count() const noexcept { return allocated_count_; }

private:
  std::vector<RawBuffer> buffers_;
  /// Stack of free slot indices; popped by Allocate, pushed back by Free.
  std::vector<size_t> free_slots_;
  /// Maps each live slot's address to its index for O(1) Free lookup.
  std::unordered_map<RawBuffer *, size_t> index_map_;
  size_t total_allocated_size_ = 0;
  size_t peak_allocated_size_ = 0;
  size_t allocated_count_ = 0;
};

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
