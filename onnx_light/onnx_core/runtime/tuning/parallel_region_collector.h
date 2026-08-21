// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_helpers.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <span>
#include <string_view>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/** Describes one portable ParallelFor execution. */
struct ParallelRegionEvent {
  std::string_view label;
  std::source_location location;
  int64_t total_iterations = 0;
  int64_t grain_size = 0;
  int32_t requested_threads = 0;
  int32_t admitted_threads = 0;
  int32_t observed_threads = 0;
  uint64_t wall_time_ns = 0;
  uint64_t executor_instance_id = 0;
  bool nested_inline = false;
};

/**
 * Collects parallel-region events in fixed storage without locking inference.
 *
 * Storage is allocated once by the constructor. Concurrent writers reserve
 * distinct slots atomically; events beyond ``capacity`` increment the dropped
 * count. Readers must inspect events only when no inference is writing.
 */
class ParallelRegionCollector {
public:
  /// Allocates storage for exactly ``capacity`` events.
  explicit ParallelRegionCollector(size_t capacity);

  ParallelRegionCollector(const ParallelRegionCollector &) = delete;
  ParallelRegionCollector &operator=(const ParallelRegionCollector &) = delete;

  /// Records an event or increments the dropped count when storage is full.
  void Record(ParallelRegionEvent event) noexcept;

  /// Returns the configured event capacity.
  size_t capacity() const noexcept { return events_.size(); }

  /// Returns the completed event slots. No inference may be writing concurrently.
  std::span<const ParallelRegionEvent> events() const noexcept;

  /// Returns the number of events rejected after capacity was exhausted.
  uint64_t dropped_events() const noexcept {
    return dropped_events_.load(std::memory_order_relaxed);
  }

private:
  std::vector<ParallelRegionEvent> events_;
  std::atomic<size_t> next_event_{0};
  std::atomic<uint64_t> dropped_events_{0};
};

/// Returns the non-owning collector installed on the calling thread.
ParallelRegionCollector *CurrentParallelRegionCollector() noexcept;

/** Installs a non-owning collector view for the lifetime of this scope. */
class ParallelRegionCollectorScope {
public:
  explicit ParallelRegionCollectorScope(ParallelRegionCollector *collector) noexcept;

  ParallelRegionCollectorScope(const ParallelRegionCollectorScope &) = delete;
  ParallelRegionCollectorScope &operator=(const ParallelRegionCollectorScope &) = delete;

  ~ParallelRegionCollectorScope();

private:
  ParallelRegionCollector *previous_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
