// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/raw_buffer_allocator.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <new>
#include <stdexcept>
#include <utility>

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

ExecutionArena::ExecutionArena(size_t capacity) : buffers_(capacity), live_slots_(capacity, false) {
  unused_slots_.reserve(capacity);
  slot_indices_.reserve(capacity);
  for (size_t i = capacity; i-- > 0;) {
    unused_slots_.push_back(i);
    slot_indices_.emplace(&buffers_[i], i);
  }
}

RawBuffer *ExecutionArena::Allocate(size_t n_bytes) {
  size_t i;
  size_t previous_capacity = 0;
  auto bucket = free_buckets_.lower_bound(n_bytes);

  if (bucket != free_buckets_.end()) {
    i = bucket->second.back();
    previous_capacity = buffers_[i].capacity();
    buffers_[i].resize(n_bytes);
    bucket->second.pop_back();
    if (bucket->second.empty()) {
      free_buckets_.erase(bucket);
    }
    retained_size_ -= previous_capacity;
    --retained_count_;
  } else if (!unused_slots_.empty()) {
    i = unused_slots_.back();
    buffers_[i].resize(n_bytes);
    unused_slots_.pop_back();
  } else if (!free_buckets_.empty()) {
    bucket = std::prev(free_buckets_.end());
    i = bucket->second.back();
    previous_capacity = buffers_[i].capacity();
    buffers_[i].resize(n_bytes);
    bucket->second.pop_back();
    if (bucket->second.empty()) {
      free_buckets_.erase(bucket);
    }
    retained_size_ -= previous_capacity;
    --retained_count_;
  } else {
    throw std::bad_alloc();
  }

  RawBuffer *buffer = &buffers_[i];
  live_slots_[i] = true;
  total_allocated_size_ += n_bytes;
  peak_allocated_size_ = std::max(peak_allocated_size_, total_allocated_size_);
  ++allocated_count_;
  return buffer;
}

void ExecutionArena::Free(RawBuffer *buf) {
  const auto slot = slot_indices_.find(buf);
  if (slot == slot_indices_.end() || !live_slots_[slot->second]) {
    throw std::invalid_argument("ExecutionArena::Free: buffer is not live in this arena.");
  }

  const size_t i = slot->second;
  const size_t logical_size = buffers_[i].size();
  const size_t retained_capacity = buffers_[i].capacity();
  free_buckets_[retained_capacity].push_back(i);
  buffers_[i].resize(0);
  total_allocated_size_ -= logical_size;
  retained_size_ += retained_capacity;
  ++retained_count_;
  --allocated_count_;
  live_slots_[i] = false;
}

size_t ExecutionArena::TotalAllocatedSize() const { return total_allocated_size_; }

size_t ExecutionArena::PeakAllocatedSize() const { return peak_allocated_size_; }

void ExecutionArena::ResetPeak() { peak_allocated_size_ = total_allocated_size_; }

size_t ExecutionArena::capacity() const noexcept { return buffers_.size(); }

size_t ExecutionArena::allocated_count() const noexcept { return allocated_count_; }

size_t ExecutionArena::RetainedSize() const noexcept { return retained_size_; }

size_t ExecutionArena::retained_count() const noexcept { return retained_count_; }

// ---------------------------------------------------------------------------
// IOLease
// ---------------------------------------------------------------------------

IOLease::IOLease(std::shared_ptr<IOArena> arena, RawBuffer *buffer, size_t logical_size) noexcept
    : arena_(std::move(arena)), buffer_(buffer), logical_size_(logical_size) {}

IOLease::IOLease(IOLease &&other) noexcept
    : arena_(std::move(other.arena_)), buffer_(other.buffer_), logical_size_(other.logical_size_) {
  other.buffer_ = nullptr;
  other.logical_size_ = 0;
}

IOLease &IOLease::operator=(IOLease &&other) noexcept {
  if (this != &other) {
    Reset();
    arena_ = std::move(other.arena_);
    buffer_ = other.buffer_;
    logical_size_ = other.logical_size_;
    other.buffer_ = nullptr;
    other.logical_size_ = 0;
  }
  return *this;
}

IOLease::~IOLease() { Reset(); }

void IOLease::Reset() noexcept {
  if (buffer_ != nullptr) {
    arena_->ReturnLease(buffer_);
    buffer_ = nullptr;
    logical_size_ = 0;
    arena_.reset();
  }
}

// ---------------------------------------------------------------------------
// IOArena
// ---------------------------------------------------------------------------

IOArena::IOArena(size_t capacity)
    : buffers_(capacity), live_slots_(capacity, false), leased_slots_(capacity, false) {
  unused_slots_.reserve(capacity);
  slot_indices_.reserve(capacity);
  for (size_t i = capacity; i-- > 0;) {
    unused_slots_.push_back(i);
    slot_indices_.emplace(&buffers_[i], i);
  }
}

std::shared_ptr<IOArena> IOArena::Create(size_t capacity) {
  return std::shared_ptr<IOArena>(new IOArena(capacity));
}

RawBuffer *IOArena::Allocate(size_t n_bytes) {
  size_t i;
  size_t previous_capacity = 0;
  auto bucket = free_buckets_.lower_bound(n_bytes);

  if (bucket != free_buckets_.end()) {
    i = bucket->second.back();
    previous_capacity = buffers_[i].capacity();
    buffers_[i].resize(n_bytes);
    bucket->second.pop_back();
    if (bucket->second.empty()) {
      free_buckets_.erase(bucket);
    }
    retained_size_ -= previous_capacity;
    --retained_count_;
  } else if (!unused_slots_.empty()) {
    i = unused_slots_.back();
    buffers_[i].resize(n_bytes);
    unused_slots_.pop_back();
  } else if (!free_buckets_.empty()) {
    bucket = std::prev(free_buckets_.end());
    i = bucket->second.back();
    previous_capacity = buffers_[i].capacity();
    buffers_[i].resize(n_bytes);
    bucket->second.pop_back();
    if (bucket->second.empty()) {
      free_buckets_.erase(bucket);
    }
    retained_size_ -= previous_capacity;
    --retained_count_;
  } else {
    throw std::bad_alloc();
  }

  RawBuffer *buffer = &buffers_[i];
  live_slots_[i] = true;
  total_allocated_size_ += n_bytes;
  peak_allocated_size_ = std::max(peak_allocated_size_, total_allocated_size_);
  ++allocated_count_;
  return buffer;
}

void IOArena::Free(RawBuffer *buf) {
  const auto slot = slot_indices_.find(buf);
  if (slot == slot_indices_.end() || !live_slots_[slot->second] || leased_slots_[slot->second]) {
    throw std::invalid_argument("IOArena::Free: buffer is not live in this arena.");
  }

  const size_t i = slot->second;
  const size_t logical_size = buffers_[i].size();
  const size_t retained_capacity = buffers_[i].capacity();
  free_buckets_[retained_capacity].push_back(i);
  buffers_[i].resize(0);
  total_allocated_size_ -= logical_size;
  retained_size_ += retained_capacity;
  ++retained_count_;
  --allocated_count_;
  live_slots_[i] = false;
}

IOLease IOArena::Export(RawBuffer *buf) {
  const auto slot = slot_indices_.find(buf);
  if (slot == slot_indices_.end() || !live_slots_[slot->second] || leased_slots_[slot->second]) {
    throw std::invalid_argument("IOArena::Export: buffer is not live in this arena.");
  }

  const size_t i = slot->second;
  const size_t logical_size = buffers_[i].size();
  // The buffer stays live and counted; ownership moves from the caller to the
  // lease, which pins the storage until it is released.
  leased_slots_[i] = true;
  --allocated_count_;
  ++leased_count_;
  return IOLease(shared_from_this(), buf, logical_size);
}

AllocationHandle IOArena::ExportHandle(RawBuffer *buf) {
  return AllocationHandle(this, Export(buf));
}

AllocationHandle IOArena::ExportHandle(AllocationHandle &&handle) {
  if (!handle || handle.holds_lease()) {
    throw std::invalid_argument(
        "IOArena::ExportHandle: handle must own a plain allocator-backed buffer.");
  }
  return AllocationHandle(this, Export(handle.Release()));
}

void IOArena::ReturnLease(RawBuffer *buf) noexcept {
  const auto slot = slot_indices_.find(buf);
  assert(slot != slot_indices_.end());
  const size_t i = slot->second;
  const size_t logical_size = buffers_[i].size();
  const size_t retained_capacity = buffers_[i].capacity();
  free_buckets_[retained_capacity].push_back(i);
  buffers_[i].resize(0);
  total_allocated_size_ -= logical_size;
  retained_size_ += retained_capacity;
  ++retained_count_;
  --leased_count_;
  live_slots_[i] = false;
  leased_slots_[i] = false;
}

size_t IOArena::TotalAllocatedSize() const { return total_allocated_size_; }

size_t IOArena::PeakAllocatedSize() const { return peak_allocated_size_; }

void IOArena::ResetPeak() { peak_allocated_size_ = total_allocated_size_; }

size_t IOArena::capacity() const noexcept { return buffers_.size(); }

size_t IOArena::allocated_count() const noexcept { return allocated_count_; }

size_t IOArena::leased_count() const noexcept { return leased_count_; }

size_t IOArena::RetainedSize() const noexcept { return retained_size_; }

size_t IOArena::retained_count() const noexcept { return retained_count_; }

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
