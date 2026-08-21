// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_helpers.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/** Describes one portable ParallelFor execution. */
struct ParallelRegionEvent {
  uint64_t region_id = 0;
  uint64_t parent_region_id = 0;
  uint64_t run_id = 0;
  std::thread::id calling_thread_id;
  std::string_view label;
  std::source_location location;
  int64_t total_iterations = 0;
  int64_t grain_size = 0;
  int32_t requested_threads = 0;
  int32_t admitted_threads = 0;
  int32_t observed_threads = 0;
  std::optional<uint64_t> wall_time_ns;
  std::optional<uint64_t> process_cpu_time_ns;
  std::optional<double> cpu_utilization;
  uint64_t executor_instance_id = 0;
  bool nested_inline = false;
};

/** Owns one event copied from a bounded parallel-region collector. */
struct ParallelRegionReportEvent {
  uint64_t region_id = 0;
  uint64_t parent_region_id = 0;
  uint64_t run_id = 0;
  std::thread::id calling_thread_id;
  std::string label;
  std::string file_name;
  std::string function_name;
  uint_least32_t line = 0;
  uint_least32_t column = 0;
  int64_t total_iterations = 0;
  int64_t grain_size = 0;
  int32_t requested_threads = 0;
  int32_t admitted_threads = 0;
  int32_t observed_threads = 0;
  std::optional<uint64_t> wall_time_ns;
  std::optional<uint64_t> process_cpu_time_ns;
  std::optional<double> cpu_utilization;
  /// Remains unavailable until a hardware-counter backend supplies it.
  std::optional<double> ipc;
  /// Remains unavailable until a hardware-counter backend supplies it.
  std::optional<double> llc_miss_rate;
  uint64_t executor_instance_id = 0;
  bool nested_inline = false;
};

/** Immutable snapshot of bounded parallel-region profiling data. */
class ParallelRegionReport {
public:
  ParallelRegionReport() = default;

  /// Returns the copied events in collector order.
  const std::vector<ParallelRegionReportEvent> &events() const noexcept { return events_; }

  /// Returns the dropped-event count captured with the snapshot.
  uint64_t dropped_events() const noexcept { return dropped_events_; }

private:
  friend class ParallelRegionCollector;
  ParallelRegionReport(std::vector<ParallelRegionReportEvent> events, uint64_t dropped_events)
      : events_(std::move(events)), dropped_events_(dropped_events) {}

  std::vector<ParallelRegionReportEvent> events_;
  uint64_t dropped_events_ = 0;
};

/// Returns normalized process CPU utilization when all inputs are valid.
std::optional<double> ComputeCpuUtilization(std::optional<uint64_t> process_cpu_time_ns,
                                            std::optional<uint64_t> wall_time_ns,
                                            int32_t admitted_threads) noexcept;

/// Returns aggregate process CPU time, or no value when the platform cannot provide it.
std::optional<uint64_t> ReadProcessCpuTimeNs() noexcept;

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

  /// Returns an owning snapshot that is independent of the collector storage.
  ParallelRegionReport Report() const;

private:
  std::vector<ParallelRegionEvent> events_;
  std::atomic<size_t> next_event_{0};
  std::atomic<uint64_t> dropped_events_{0};
};

/// Returns the non-owning collector installed on the calling thread.
ParallelRegionCollector *CurrentParallelRegionCollector() noexcept;

/// Returns the run identifier installed on the calling thread.
uint64_t CurrentParallelRegionRunId() noexcept;

/// Returns the active region identifier installed on the calling thread.
uint64_t CurrentParallelRegionId() noexcept;

/// Returns a new process-unique region identifier.
uint64_t NextParallelRegionId() noexcept;

/** Installs a non-owning collector and profiling identity for this scope. */
class ParallelRegionCollectorScope {
public:
  explicit ParallelRegionCollectorScope(ParallelRegionCollector *collector, uint64_t run_id = 0,
                                        uint64_t region_id = 0) noexcept;

  ParallelRegionCollectorScope(const ParallelRegionCollectorScope &) = delete;
  ParallelRegionCollectorScope &operator=(const ParallelRegionCollectorScope &) = delete;

  ~ParallelRegionCollectorScope();

private:
  ParallelRegionCollector *previous_collector_;
  uint64_t previous_run_id_;
  uint64_t previous_region_id_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
