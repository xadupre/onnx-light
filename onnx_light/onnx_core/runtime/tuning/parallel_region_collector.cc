// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/parallel_region_collector.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <limits>
#include <utility>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

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

#if defined(__linux__)
HardwareCounterStatus StatusFromErrno(int error) noexcept {
  if (error == EACCES || error == EPERM) {
    return HardwareCounterStatus::kPermissionDenied;
  }
  if (error == EOVERFLOW) {
    return HardwareCounterStatus::kOverflowed;
  }
  return HardwareCounterStatus::kUnsupported;
}
#endif

} // namespace

const char *HardwareCounterStatusName(HardwareCounterStatus status) noexcept {
  switch (status) {
  case HardwareCounterStatus::kDisabled:
    return "disabled";
  case HardwareCounterStatus::kUnsupported:
    return "unsupported";
  case HardwareCounterStatus::kPermissionDenied:
    return "permission_denied";
  case HardwareCounterStatus::kMultiplexed:
    return "multiplexed";
  case HardwareCounterStatus::kOverflowed:
    return "overflowed";
  case HardwareCounterStatus::kValid:
    return "valid";
  }
  return "unsupported";
}

class ParallelRegionCollector::HardwareCounterBackend {
public:
  explicit HardwareCounterBackend(bool enabled) : enabled_(enabled) {}

  ~HardwareCounterBackend() { Close(); }

  HardwareCounterMeasurement Begin() noexcept {
    if (!enabled_) {
      return HardwareCounterMeasurement{HardwareCounterStatus::kDisabled, false};
    }
    if (busy_.test_and_set(std::memory_order_acquire)) {
      return HardwareCounterMeasurement{HardwareCounterStatus::kMultiplexed, false};
    }
#if defined(__linux__)
    const std::thread::id current_thread = std::this_thread::get_id();
    if (!open_attempted_ || current_thread != owner_thread_) {
      Close();
      Open();
      owner_thread_ = current_thread;
      open_attempted_ = true;
    }
    if (status_ != HardwareCounterStatus::kValid) {
      busy_.clear(std::memory_order_release);
      return HardwareCounterMeasurement{status_, false};
    }
    if (ioctl(file_descriptors_[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0 ||
        ioctl(file_descriptors_[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
      const HardwareCounterStatus status = StatusFromErrno(errno);
      busy_.clear(std::memory_order_release);
      return HardwareCounterMeasurement{status, false};
    }
    return HardwareCounterMeasurement{HardwareCounterStatus::kValid, true};
#else
    busy_.clear(std::memory_order_release);
    return HardwareCounterMeasurement{HardwareCounterStatus::kUnsupported, false};
#endif
  }

  HardwareCounterSample End(HardwareCounterMeasurement measurement) noexcept {
    HardwareCounterSample sample;
    sample.status = measurement.status;
    if (!measurement.active) {
      return sample;
    }
#if defined(__linux__)
    struct GroupRead {
      uint64_t count;
      uint64_t time_enabled;
      uint64_t time_running;
      std::array<uint64_t, 4> values;
    } result{};
    if (ioctl(file_descriptors_[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
      sample.status = StatusFromErrno(errno);
      busy_.clear(std::memory_order_release);
      return sample;
    }
    const ssize_t bytes = read(file_descriptors_[0], &result, sizeof(result));
    busy_.clear(std::memory_order_release);
    if (bytes < 0) {
      sample.status = StatusFromErrno(errno);
      return sample;
    }
    if (bytes != static_cast<ssize_t>(sizeof(result)) || result.count != result.values.size()) {
      sample.status = HardwareCounterStatus::kUnsupported;
      return sample;
    }
    sample.time_enabled = result.time_enabled;
    sample.time_running = result.time_running;
    if (result.time_running == 0 || result.time_running != result.time_enabled) {
      sample.status = HardwareCounterStatus::kMultiplexed;
      return sample;
    }
    sample.status = HardwareCounterStatus::kValid;
    sample.cpu_cycles = result.values[0];
    sample.retired_instructions = result.values[1];
    sample.llc_references = result.values[2];
    sample.llc_misses = result.values[3];
    return sample;
#else
    busy_.clear(std::memory_order_release);
    sample.status = HardwareCounterStatus::kUnsupported;
    return sample;
#endif
  }

private:
#if defined(__linux__)
  void Open() noexcept {
    constexpr std::array<uint64_t, 4> configs{
        PERF_COUNT_HW_CPU_CYCLES,
        PERF_COUNT_HW_INSTRUCTIONS,
        PERF_COUNT_HW_CACHE_REFERENCES,
        PERF_COUNT_HW_CACHE_MISSES,
    };
    for (size_t index = 0; index < configs.size(); ++index) {
      perf_event_attr attributes{};
      attributes.type = PERF_TYPE_HARDWARE;
      attributes.size = sizeof(attributes);
      attributes.config = configs[index];
      attributes.disabled = index == 0;
      attributes.read_format =
          PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
      const int group = index == 0 ? -1 : file_descriptors_[0];
      const long result =
          syscall(__NR_perf_event_open, &attributes, 0, -1, group, PERF_FLAG_FD_CLOEXEC);
      if (result < 0) {
        status_ = StatusFromErrno(errno);
        Close();
        return;
      }
      file_descriptors_[index] = static_cast<int>(result);
    }
    status_ = HardwareCounterStatus::kValid;
  }
#endif

  void Close() noexcept {
#if defined(__linux__)
    for (int &file_descriptor : file_descriptors_) {
      if (file_descriptor >= 0) {
        close(file_descriptor);
        file_descriptor = -1;
      }
    }
#endif
  }

  bool enabled_ = false;
  std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
#if defined(__linux__)
  HardwareCounterStatus status_ = HardwareCounterStatus::kDisabled;
  bool open_attempted_ = false;
  std::thread::id owner_thread_;
  std::array<int, 4> file_descriptors_ = {-1, -1, -1, -1};
#endif
};

ParallelRegionCollector::ParallelRegionCollector(size_t capacity, bool hardware_counters)
    : events_(capacity),
      hardware_counter_backend_(std::make_unique<HardwareCounterBackend>(hardware_counters)) {}

ParallelRegionCollector::~ParallelRegionCollector() = default;

HardwareCounterMeasurement ParallelRegionCollector::BeginHardwareCounters() noexcept {
  return hardware_counter_backend_->Begin();
}

HardwareCounterSample
ParallelRegionCollector::EndHardwareCounters(HardwareCounterMeasurement measurement,
                                             bool isolated) noexcept {
  HardwareCounterSample sample = hardware_counter_backend_->End(measurement);
  if (!isolated && sample.status == HardwareCounterStatus::kValid) {
    sample.status = HardwareCounterStatus::kMultiplexed;
    sample.cpu_cycles.reset();
    sample.retired_instructions.reset();
    sample.llc_references.reset();
    sample.llc_misses.reset();
  }
  return sample;
}

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
        .counter_status = event.counters.status,
        .cpu_cycles = event.counters.cpu_cycles,
        .retired_instructions = event.counters.retired_instructions,
        .llc_references = event.counters.llc_references,
        .llc_misses = event.counters.llc_misses,
        .counter_time_enabled = event.counters.time_enabled,
        .counter_time_running = event.counters.time_running,
        .ipc =
            event.counters.status == HardwareCounterStatus::kValid &&
                    event.counters.cpu_cycles.value_or(0) != 0 &&
                    event.counters.retired_instructions.has_value()
                ? std::optional<double>(static_cast<double>(*event.counters.retired_instructions) /
                                        static_cast<double>(*event.counters.cpu_cycles))
                : std::nullopt,
        .llc_miss_rate =
            event.counters.status == HardwareCounterStatus::kValid &&
                    event.counters.llc_references.value_or(0) != 0 &&
                    event.counters.llc_misses.has_value()
                ? std::optional<double>(static_cast<double>(*event.counters.llc_misses) /
                                        static_cast<double>(*event.counters.llc_references))
                : std::nullopt,
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
