// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/raw_buffer_allocator.h"

#include <new>
#include <stdexcept>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

SimpleRawBufferAllocator::SimpleRawBufferAllocator(size_t capacity) : buffers_(capacity) {
  free_slots_.reserve(capacity);
  index_map_.reserve(capacity);
  // Push indices in reverse so slot 0 is popped first.
  for (size_t i = capacity; i-- > 0;) {
    free_slots_.push_back(i);
  }
}

RawBuffer *SimpleRawBufferAllocator::Allocate(size_t n_bytes) {
  if (free_slots_.empty()) {
    throw std::bad_alloc();
  }
  const size_t i = free_slots_.back();
  free_slots_.pop_back();
  // Size the buffer without zero-filling it: the caller is expected to fully
  // overwrite the result, so clearing the memory first would be wasted work.
  buffers_[i].resize(n_bytes);
  index_map_[&buffers_[i]] = i;
  total_allocated_size_ += n_bytes;
  if (total_allocated_size_ > peak_allocated_size_) {
    peak_allocated_size_ = total_allocated_size_;
  }
  ++allocated_count_;
  return &buffers_[i];
}

void SimpleRawBufferAllocator::Free(RawBuffer *buf) {
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

size_t SimpleRawBufferAllocator::TotalAllocatedSize() const { return total_allocated_size_; }

size_t SimpleRawBufferAllocator::PeakAllocatedSize() const { return peak_allocated_size_; }

void SimpleRawBufferAllocator::ResetPeak() { peak_allocated_size_ = total_allocated_size_; }

size_t SimpleRawBufferAllocator::capacity() const noexcept { return buffers_.size(); }

size_t SimpleRawBufferAllocator::allocated_count() const noexcept { return allocated_count_; }

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
