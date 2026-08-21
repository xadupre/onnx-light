// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/parallel_region_collector.h"

#include <algorithm>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

ParallelRegionCollector *&CurrentCollectorSlot() noexcept {
  thread_local ParallelRegionCollector *collector = nullptr;
  return collector;
}

} // namespace

ParallelRegionCollector::ParallelRegionCollector(size_t capacity) : events_(capacity) {}

void ParallelRegionCollector::Record(ParallelRegionEvent event) noexcept {
  size_t index = next_event_.load(std::memory_order_relaxed);
  while (index < events_.size() &&
         !next_event_.compare_exchange_weak(index, index + 1, std::memory_order_relaxed)) {
  }
  if (index >= events_.size()) {
    dropped_events_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  events_[index] = std::move(event);
}

std::span<const ParallelRegionEvent> ParallelRegionCollector::events() const noexcept {
  return std::span<const ParallelRegionEvent>(
      events_.data(), std::min(next_event_.load(std::memory_order_relaxed), events_.size()));
}

ParallelRegionCollector *CurrentParallelRegionCollector() noexcept {
  return CurrentCollectorSlot();
}

ParallelRegionCollectorScope::ParallelRegionCollectorScope(
    ParallelRegionCollector *collector) noexcept
    : previous_(CurrentCollectorSlot()) {
  CurrentCollectorSlot() = collector;
}

ParallelRegionCollectorScope::~ParallelRegionCollectorScope() {
  CurrentCollectorSlot() = previous_;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
