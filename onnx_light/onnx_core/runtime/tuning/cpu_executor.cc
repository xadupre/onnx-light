// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/cpu_executor.h"

#include "onnx_core/runtime/kernels/parallel_for.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#include <unistd.h>
#if defined(__linux__)
#include <sched.h>
#endif
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

uint64_t CurrentProcessId() noexcept {
#if defined(__linux__) || defined(__APPLE__)
  return static_cast<uint64_t>(getpid());
#elif defined(_WIN32)
  return static_cast<uint64_t>(GetCurrentProcessId());
#else
  return 0;
#endif
}

uint64_t NextCpuExecutorInstanceId() noexcept {
  static std::atomic<uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

void ValidateResolvedPolicy(const ResolvedCpuExecutionPolicy &policy) {
  if (policy.effective_threads == 0) {
    throw std::invalid_argument("CpuExecutor requires at least one effective participant.");
  }
  const size_t expected_workers = static_cast<size_t>(policy.effective_threads - 1);
  if (!policy.worker_processors.empty() && policy.worker_processors.size() != expected_workers) {
    throw std::invalid_argument(
        "CpuExecutor requires either no worker placement or one processor per worker.");
  }
  if (policy.request.affinity_policy == CpuAffinityPolicy::kNone &&
      (policy.caller_processor.has_value() || !policy.worker_processors.empty())) {
    throw std::invalid_argument("CpuExecutor no-affinity policy cannot contain processor "
                                "assignments.");
  }
  if (policy.request.affinity_policy == CpuAffinityPolicy::kExplicit &&
      (!policy.caller_processor.has_value() ||
       policy.worker_processors.size() != expected_workers)) {
    throw std::invalid_argument(
        "CpuExecutor explicit affinity requires one caller and one assignment per worker.");
  }
}

bool PinCurrentThread(CpuLogicalProcessor processor, std::string &error) {
#if defined(__linux__)
  const size_t processor_count = static_cast<size_t>(processor.id) + 1;
  cpu_set_t *affinity = CPU_ALLOC(processor_count);
  if (affinity == nullptr) {
    error = "unable to allocate a CPU affinity mask";
    return false;
  }
  const size_t affinity_size = CPU_ALLOC_SIZE(processor_count);
  CPU_ZERO_S(affinity_size, affinity);
  CPU_SET_S(static_cast<int>(processor.id), affinity_size, affinity);
  const int result = pthread_setaffinity_np(pthread_self(), affinity_size, affinity);
  CPU_FREE(affinity);
  if (result != 0) {
    error = "pthread_setaffinity_np failed for logical processor " + std::to_string(processor.id) +
            ": " + std::strerror(result);
    return false;
  }
  return true;
#elif defined(_WIN32)
  if (processor.id >= sizeof(KAFFINITY) * 8) {
    error = "logical processor identifier exceeds the Windows group mask width";
    return false;
  }
  GROUP_AFFINITY affinity{};
  affinity.Group = processor.group;
  affinity.Mask = static_cast<KAFFINITY>(1) << processor.id;
  if (SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr) == 0) {
    error = "SetThreadGroupAffinity failed for processor group " + std::to_string(processor.group) +
            ", logical processor " + std::to_string(processor.id) + ": error " +
            std::to_string(GetLastError());
    return false;
  }
  return true;
#else
  (void)processor;
  error = "thread affinity is unsupported on this platform";
  return false;
#endif
}

ThreadPoolOptions MakeThreadPoolOptions(const ResolvedCpuExecutionPolicy &policy,
                                        void *worker_context,
                                        ThreadPoolOptions::WorkerStartFn worker_start) {
  ThreadPoolOptions options;
  options.spin_iterations = policy.spin.iterations;
  options.spin_duration_ns = policy.spin.duration_ns;
  if (!policy.worker_processors.empty()) {
    options.worker_start = worker_start;
    options.worker_start_context = worker_context;
  }
  return options;
}

struct ParallelRange {
  void *context = nullptr;
  ParallelRangeFn function = nullptr;
  int64_t base_block_size = 0;
  int64_t extra_blocks = 0;
};

CpuExecutor *&CurrentCpuExecutorSlot() noexcept {
  thread_local CpuExecutor *current = nullptr;
  return current;
}

CpuExecutor *&ActiveCpuExecutorRegionSlot() noexcept {
  thread_local CpuExecutor *active = nullptr;
  return active;
}

class CpuExecutorRegionScope {
public:
  explicit CpuExecutorRegionScope(CpuExecutor *executor) noexcept
      : previous_(ActiveCpuExecutorRegionSlot()) {
    ActiveCpuExecutorRegionSlot() = executor;
  }

  ~CpuExecutorRegionScope() { ActiveCpuExecutorRegionSlot() = previous_; }

private:
  CpuExecutor *previous_;
};

} // namespace

CpuExecutor *CurrentCpuExecutor() noexcept { return CurrentCpuExecutorSlot(); }

CpuExecutorScope::CpuExecutorScope(CpuExecutor *executor) noexcept
    : previous_(CurrentCpuExecutorSlot()) {
  CurrentCpuExecutorSlot() = executor;
}

CpuExecutorScope::~CpuExecutorScope() { CurrentCpuExecutorSlot() = previous_; }

CpuExecutorKey MakeCpuExecutorKey(const ResolvedCpuExecutionPolicy &policy) {
  return CpuExecutorKey{
      policy.effective_threads,
      policy.caller_processor,
      policy.worker_processors,
      policy.spin,
      policy.request.affinity_policy == CpuAffinityPolicy::kNone,
      policy.allow_nested_parallelism,
  };
}

struct CpuExecutor::Impl {
  struct CounterState {
    std::atomic<uint64_t> dispatches{0};
    std::atomic<uint64_t> nested_inline_dispatches{0};
    std::atomic<uint64_t> limited_inline_dispatches{0};
  };

  explicit Impl(ResolvedCpuExecutionPolicy resolved)
      : policy(std::move(resolved)), executor_key(MakeCpuExecutorKey(policy)),
        process_id(CurrentProcessId()), instance_id(NextCpuExecutorInstanceId()) {
    ValidateResolvedPolicy(policy);
    ThreadPoolOptions options = MakeThreadPoolOptions(policy, this, &Impl::StartWorker);
    pool = std::make_unique<ThreadPool>(static_cast<int64_t>(policy.effective_threads) - 1,
                                        std::move(options));
  }

  ~Impl() {
    if (process_id != CurrentProcessId()) {
      (void)pool.release();
    }
  }

  static bool StartWorker(void *context, int64_t worker_index, std::string &error) {
    const auto &self = *static_cast<Impl *>(context);
    return PinCurrentThread(self.policy.worker_processors[static_cast<size_t>(worker_index)],
                            error);
  }

  ResolvedCpuExecutionPolicy policy;
  CpuExecutorKey executor_key;
  uint64_t process_id;
  uint64_t instance_id;
  std::unique_ptr<ThreadPool> pool;
  std::mutex counters_mutex;
  std::unique_ptr<CounterState> counters_storage;
  std::atomic<CounterState *> counters{nullptr};
};

CpuExecutor::CpuExecutor(ResolvedCpuExecutionPolicy policy)
    : impl_(std::make_unique<Impl>(std::move(policy))) {}

CpuExecutor::~CpuExecutor() = default;

uint32_t CpuExecutor::effective_threads() const noexcept { return impl_->policy.effective_threads; }

const ResolvedCpuExecutionPolicy &CpuExecutor::policy() const noexcept { return impl_->policy; }

const CpuExecutorKey &CpuExecutor::key() const noexcept { return impl_->executor_key; }

uint64_t CpuExecutor::instance_id() const noexcept { return impl_->instance_id; }

void CpuExecutor::EnableCounters() {
  if (impl_->counters.load(std::memory_order_acquire) != nullptr) {
    return;
  }
  std::lock_guard lock(impl_->counters_mutex);
  if (impl_->counters_storage == nullptr) {
    impl_->counters_storage = std::make_unique<Impl::CounterState>();
    impl_->counters.store(impl_->counters_storage.get(), std::memory_order_release);
  }
}

bool CpuExecutor::counters_enabled() const noexcept {
  return impl_->counters.load(std::memory_order_relaxed) != nullptr;
}

CpuExecutorCounters CpuExecutor::counters() const noexcept {
  const Impl::CounterState *counters = impl_->counters.load(std::memory_order_acquire);
  if (counters == nullptr) {
    return {};
  }
  return CpuExecutorCounters{
      counters->dispatches.load(std::memory_order_relaxed),
      counters->nested_inline_dispatches.load(std::memory_order_relaxed),
      counters->limited_inline_dispatches.load(std::memory_order_relaxed),
  };
}

void CpuExecutor::ParallelFor(int64_t total, int64_t grain, void *context, ParallelRangeFn function,
                              uint32_t maximum_participants, ParallelRegionCollector *collector,
                              std::string_view label, std::source_location location) {
  if (impl_->process_id != CurrentProcessId()) {
    throw std::runtime_error(
        "CpuExecutor inherited across fork is unusable; acquire a new executor in the child.");
  }
  if (total <= 0) {
    return;
  }
  if (grain <= 0) {
    throw std::invalid_argument("CpuExecutor ParallelFor grain must be positive.");
  }
  if (function == nullptr) {
    throw std::invalid_argument("CpuExecutor ParallelFor function must not be null.");
  }
  std::chrono::steady_clock::time_point start;
  std::optional<uint64_t> process_cpu_start;
  HardwareCounterMeasurement counter_measurement;
  uint64_t region_id = 0;
  uint64_t parent_region_id = 0;
  uint64_t run_id = 0;
  std::thread::id calling_thread_id;
  if (collector != nullptr) {
    start = std::chrono::steady_clock::now();
    process_cpu_start = ReadProcessCpuTimeNs();
    counter_measurement = collector->BeginHardwareCounters();
    region_id = NextParallelRegionId();
    parent_region_id = CurrentParallelRegionId();
    run_id = CurrentParallelRegionRunId();
    calling_thread_id = std::this_thread::get_id();
  }
  Impl::CounterState *counters = impl_->counters.load(std::memory_order_relaxed);
  if (counters != nullptr) {
    counters->dispatches.fetch_add(1, std::memory_order_relaxed);
  }
  const bool nested = ActiveCpuExecutorRegionSlot() == this;
  if (impl_->policy.caller_processor.has_value()) {
    std::string error;
    if (!PinCurrentThread(*impl_->policy.caller_processor, error)) {
      throw std::runtime_error("CpuExecutor caller affinity failed: " + error);
    }
  }
  // Every participant runs with this executor installed so a nested parallel
  // region dispatches here (and therefore runs inline) instead of waking an
  // unrelated process-wide pool.
  CpuExecutorScope caller_scope(this);

  const uint32_t participant_limit =
      maximum_participants == 0 ? impl_->policy.effective_threads
                                : std::min(maximum_participants, impl_->policy.effective_threads);
  const auto record = [&](uint32_t admitted, bool nested_inline) {
    if (collector == nullptr) {
      return;
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const int64_t wall_time = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    const std::optional<uint64_t> wall_time_ns =
        wall_time > 0 ? std::optional<uint64_t>(static_cast<uint64_t>(wall_time)) : std::nullopt;
    const std::optional<uint64_t> process_cpu_end = ReadProcessCpuTimeNs();
    const std::optional<uint64_t> process_cpu_time_ns =
        process_cpu_start.has_value() && process_cpu_end.has_value() &&
                *process_cpu_end >= *process_cpu_start
            ? std::optional<uint64_t>(*process_cpu_end - *process_cpu_start)
            : std::nullopt;
    const HardwareCounterSample hardware_counters =
        collector->EndHardwareCounters(counter_measurement, admitted == 1);
    collector->Record(ParallelRegionEvent{
        .region_id = region_id,
        .parent_region_id = parent_region_id,
        .run_id = run_id,
        .calling_thread_id = calling_thread_id,
        .label = label,
        .location = location,
        .total_iterations = total,
        .grain_size = grain,
        .requested_threads = static_cast<int32_t>(participant_limit),
        .admitted_threads = static_cast<int32_t>(admitted),
        .observed_threads = static_cast<int32_t>(admitted),
        .wall_time_ns = wall_time_ns,
        .process_cpu_time_ns = process_cpu_time_ns,
        .cpu_utilization = ComputeCpuUtilization(process_cpu_time_ns, wall_time_ns,
                                                 static_cast<int32_t>(admitted)),
        .counters = hardware_counters,
        .executor_instance_id = impl_->instance_id,
        .nested_inline = nested_inline,
    });
  };
  if (nested) {
    if (counters != nullptr) {
      counters->nested_inline_dispatches.fetch_add(1, std::memory_order_relaxed);
    }
    CpuExecutorRegionScope region_scope(this);
    if (collector != nullptr) {
      ParallelRegionCollectorScope collector_scope(collector, run_id, region_id);
      function(context, 0, total);
    } else {
      function(context, 0, total);
    }
    record(1, true);
    return;
  }
  if (total < grain || participant_limit <= 1) {
    if (counters != nullptr) {
      counters->limited_inline_dispatches.fetch_add(1, std::memory_order_relaxed);
    }
    CpuExecutorRegionScope region_scope(this);
    if (collector != nullptr) {
      ParallelRegionCollectorScope collector_scope(collector, run_id, region_id);
      function(context, 0, total);
    } else {
      function(context, 0, total);
    }
    record(1, false);
    return;
  }
  const int64_t useful_blocks = total / grain;
  const int64_t num_blocks =
      std::min<int64_t>(static_cast<int64_t>(participant_limit), useful_blocks);
  if (num_blocks <= 1) {
    if (counters != nullptr) {
      counters->limited_inline_dispatches.fetch_add(1, std::memory_order_relaxed);
    }
    CpuExecutorRegionScope region_scope(this);
    if (collector != nullptr) {
      ParallelRegionCollectorScope collector_scope(collector, run_id, region_id);
      function(context, 0, total);
    } else {
      function(context, 0, total);
    }
    record(1, false);
    return;
  }

  ParallelRange range{
      context,
      function,
      total / num_blocks,
      total % num_blocks,
  };
  impl_->pool->Run(num_blocks, [&range, this, collector, run_id, region_id](int64_t block_index) {
    CpuExecutorScope block_scope(this);
    CpuExecutorRegionScope region_scope(this);
    const int64_t begin =
        block_index * range.base_block_size + std::min(block_index, range.extra_blocks);
    const int64_t end = begin + range.base_block_size + (block_index < range.extra_blocks ? 1 : 0);
    if (collector != nullptr) {
      ParallelRegionCollectorScope collector_scope(collector, run_id, region_id);
      range.function(range.context, begin, end);
    } else {
      range.function(range.context, begin, end);
    }
  });
  record(static_cast<uint32_t>(num_blocks), false);
}

CpuExecutorRegistry::CpuExecutorRegistry(size_t capacity)
    : capacity_(capacity), process_id_(CurrentProcessId()) {
  if (capacity == 0) {
    throw std::invalid_argument("CpuExecutorRegistry capacity must be positive.");
  }
  entries_.reserve(capacity);
}

std::shared_ptr<CpuExecutor> CpuExecutorRegistry::Acquire(const CpuExecutionPolicy &request) {
  return Acquire(ResolveCpuExecutionPolicy(request));
}

std::shared_ptr<CpuExecutor>
CpuExecutorRegistry::Acquire(const ResolvedCpuExecutionPolicy &policy) {
  const CpuExecutorKey key = MakeCpuExecutorKey(policy);
  const uint64_t process_id = CurrentProcessId();
  std::lock_guard<std::mutex> lock(mutex_);
  ResetAfterForkLocked(process_id);
  RemoveExpiredLocked();
  auto found = std::find_if(entries_.begin(), entries_.end(),
                            [&key](const Entry &entry) { return entry.key == key; });
  if (found != entries_.end()) {
    if (std::shared_ptr<CpuExecutor> executor = found->executor.lock()) {
      return executor;
    }
    entries_.erase(found);
  }
  if (entries_.size() >= capacity_) {
    throw std::runtime_error(
        "CpuExecutorRegistry capacity exhausted by incompatible live policies.");
  }
  std::shared_ptr<CpuExecutor> executor(new CpuExecutor(policy));
  entries_.push_back(Entry{key, executor});
  return executor;
}

size_t CpuExecutorRegistry::live_pool_count() {
  const uint64_t process_id = CurrentProcessId();
  std::lock_guard<std::mutex> lock(mutex_);
  ResetAfterForkLocked(process_id);
  RemoveExpiredLocked();
  return entries_.size();
}

void CpuExecutorRegistry::ResetAfterForkLocked(uint64_t process_id) {
  if (process_id_ != process_id) {
    entries_.clear();
    process_id_ = process_id;
  }
}

void CpuExecutorRegistry::RemoveExpiredLocked() {
  entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                [](const Entry &entry) { return entry.executor.expired(); }),
                 entries_.end());
}

CpuExecutorRegistry &GlobalCpuExecutorRegistry() {
  static CpuExecutorRegistry registry(kDefaultCpuExecutorRegistryCapacity);
  return registry;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
