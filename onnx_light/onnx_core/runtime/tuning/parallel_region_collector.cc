// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/parallel_region_collector.h"

#include <algorithm>
#include <limits>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <time.h>
#endif

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

struct ParallelRegionContext {
  ParallelRegionCollector *collector = nullptr;
  uint64_t run_id = 0;
  uint64_t region_id = 0;
};

ParallelRegionContext &CurrentContext() noexcept {
  thread_local ParallelRegionContext context;
  return context;
}

std::atomic<uint64_t> next_run_id{1};
std::atomic<uint64_t> next_region_id{1};

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

ParallelRegionReport ParallelRegionCollector::Report() const {
  const std::span<const ParallelRegionEvent> current_events = events();
  std::vector<ParallelRegionReportEvent> report_events;
  report_events.reserve(current_events.size());
  for (const ParallelRegionEvent &event : current_events) {
    report_events.push_back(ParallelRegionReportEvent{
        .region_id = event.region_id,
        .parent_region_id = event.parent_region_id,
        .run_id = event.run_id,
        .calling_thread_id = event.calling_thread_id,
        .label = std::string(event.label),
        .file_name = event.location.file_name(),
        .function_name = event.location.function_name(),
        .line = event.location.line(),
        .column = event.location.column(),
        .total_iterations = event.total_iterations,
        .grain_size = event.grain_size,
        .requested_threads = event.requested_threads,
        .admitted_threads = event.admitted_threads,
        .observed_threads = event.observed_threads,
        .wall_time_ns = event.wall_time_ns,
        .process_cpu_time_ns = event.process_cpu_time_ns,
        .cpu_utilization = event.cpu_utilization,
        .ipc = std::nullopt,
        .llc_miss_rate = std::nullopt,
        .executor_instance_id = event.executor_instance_id,
        .nested_inline = event.nested_inline,
    });
  }
  return ParallelRegionReport(std::move(report_events), dropped_events());
}

ParallelRegionCollector *CurrentParallelRegionCollector() noexcept {
  return CurrentContext().collector;
}

uint64_t CurrentParallelRegionRunId() noexcept { return CurrentContext().run_id; }

uint64_t CurrentParallelRegionId() noexcept { return CurrentContext().region_id; }

uint64_t NextParallelRegionId() noexcept {
  return next_region_id.fetch_add(1, std::memory_order_relaxed);
}

std::optional<double> ComputeCpuUtilization(std::optional<uint64_t> process_cpu_time_ns,
                                            std::optional<uint64_t> wall_time_ns,
                                            int32_t admitted_threads) noexcept {
  if (!process_cpu_time_ns.has_value() || !wall_time_ns.has_value() || *wall_time_ns == 0 ||
      admitted_threads <= 0) {
    return std::nullopt;
  }
  return static_cast<double>(*process_cpu_time_ns) / static_cast<double>(*wall_time_ns) /
         static_cast<double>(admitted_threads);
}

std::optional<uint64_t> ReadProcessCpuTimeNs() noexcept {
#if defined(_WIN32)
  FILETIME creation;
  FILETIME exit;
  FILETIME kernel;
  FILETIME user;
  if (GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user) == 0) {
    return std::nullopt;
  }
  ULARGE_INTEGER kernel_ticks;
  kernel_ticks.LowPart = kernel.dwLowDateTime;
  kernel_ticks.HighPart = kernel.dwHighDateTime;
  ULARGE_INTEGER user_ticks;
  user_ticks.LowPart = user.dwLowDateTime;
  user_ticks.HighPart = user.dwHighDateTime;
  if (kernel_ticks.QuadPart > std::numeric_limits<uint64_t>::max() - user_ticks.QuadPart ||
      kernel_ticks.QuadPart + user_ticks.QuadPart > std::numeric_limits<uint64_t>::max() / 100) {
    return std::nullopt;
  }
  return (kernel_ticks.QuadPart + user_ticks.QuadPart) * 100;
#else
  timespec process_time;
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &process_time) != 0 || process_time.tv_sec < 0 ||
      process_time.tv_nsec < 0 || process_time.tv_nsec >= 1000000000L ||
      static_cast<uint64_t>(process_time.tv_sec) >
          (std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(process_time.tv_nsec)) /
              1000000000ULL) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(process_time.tv_sec) * 1000000000ULL +
         static_cast<uint64_t>(process_time.tv_nsec);
#endif
}

ParallelRegionCollectorScope::ParallelRegionCollectorScope(ParallelRegionCollector *collector,
                                                           uint64_t run_id,
                                                           uint64_t region_id) noexcept
    : previous_collector_(CurrentContext().collector), previous_run_id_(CurrentContext().run_id),
      previous_region_id_(CurrentContext().region_id) {
  ParallelRegionContext &context = CurrentContext();
  if (run_id == 0 && collector != nullptr) {
    if (context.collector == collector && context.run_id != 0) {
      run_id = context.run_id;
      region_id = region_id == 0 ? context.region_id : region_id;
    } else {
      run_id = next_run_id.fetch_add(1, std::memory_order_relaxed);
    }
  }
  context = ParallelRegionContext{collector, run_id, region_id};
}

ParallelRegionCollectorScope::~ParallelRegionCollectorScope() {
  CurrentContext() =
      ParallelRegionContext{previous_collector_, previous_run_id_, previous_region_id_};
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
