// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/parallel_for.h"

#include "onnx_core/runtime/tuning/cpu_executor.h"
#include "onnx_core/runtime/tuning/runtime_parameters.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {

void CpuRelax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ volatile("yield");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

std::chrono::steady_clock::duration SpinDuration(uint64_t duration_ns) noexcept {
  const uint64_t maximum = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  return std::chrono::nanoseconds(static_cast<int64_t>(std::min(duration_ns, maximum)));
}

} // namespace

int64_t ParallelForThreadCount() noexcept {
  if (const CpuExecutor *executor = CurrentCpuExecutor(); executor != nullptr) {
    return static_cast<int64_t>(executor->effective_threads());
  }
  static const int64_t thread_count = RuntimeParameters().EffectiveNumThreads();
  return thread_count;
}

ThreadPool::ThreadPool(int64_t num_workers) : ThreadPool(num_workers, ThreadPoolOptions{}) {}

ThreadPool::ThreadPool(int64_t num_workers, ThreadPoolOptions options)
    : options_(std::move(options)) {
  if (num_workers < 0) {
    num_workers = 0;
  }
  worker_work_.reserve(static_cast<size_t>(num_workers));
  for (int64_t i = 0; i < num_workers; ++i) {
    worker_work_.push_back(std::make_unique<std::condition_variable>());
  }
  workers_.reserve(static_cast<size_t>(num_workers));
  try {
    for (int64_t i = 0; i < num_workers; ++i) {
      workers_.emplace_back([this, i]() { WorkerLoop(i); });
    }
  } catch (const std::system_error &) {
    StopAndJoin();
    throw;
  }
  if (!workers_.empty()) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_started_.wait(
        lock, [this]() { return started_workers_ == static_cast<int64_t>(workers_.size()); });
    if (!startup_error_.empty()) {
      const std::string error = startup_error_;
      lock.unlock();
      StopAndJoin();
      throw std::runtime_error(error);
    }
  }
}

ThreadPool::~ThreadPool() { StopAndJoin(); }

void ThreadPool::StopAndJoin() noexcept {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
  }
  for (const auto &worker : worker_work_) {
    worker->notify_one();
  }
  for (std::thread &worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

bool &ThreadPool::InPoolFlag() noexcept {
  thread_local bool in_pool = false;
  return in_pool;
}

bool ThreadPool::InPool() noexcept { return InPoolFlag(); }

bool ThreadPool::SpinForWork(uint64_t last_generation, int64_t worker_index) const noexcept {
  const auto work_available = [this, last_generation, worker_index]() {
    if (generation_.load(std::memory_order_acquire) == last_generation) {
      return false;
    }
    return worker_index < active_workers_.load(std::memory_order_acquire);
  };
  if (options_.spin_iterations != 0) {
    for (uint64_t spin = 0; spin < options_.spin_iterations; ++spin) {
      if (stop_.load(std::memory_order_acquire) || work_available()) {
        return true;
      }
      if (generation_.load(std::memory_order_acquire) != last_generation) {
        return false;
      }
      CpuRelax();
    }
    return false;
  }
  if (options_.spin_duration_ns == 0) {
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() + SpinDuration(options_.spin_duration_ns);
  while (std::chrono::steady_clock::now() < deadline) {
    if (stop_.load(std::memory_order_acquire) || work_available()) {
      return true;
    }
    if (generation_.load(std::memory_order_acquire) != last_generation) {
      return false;
    }
    CpuRelax();
  }
  return false;
}

bool ThreadPool::SpinForCompletion() const noexcept {
  if (options_.spin_iterations != 0) {
    for (uint64_t spin = 0; spin < options_.spin_iterations; ++spin) {
      if (remaining_.load(std::memory_order_acquire) == 0) {
        return true;
      }
      CpuRelax();
    }
    return false;
  }
  if (options_.spin_duration_ns == 0) {
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() + SpinDuration(options_.spin_duration_ns);
  while (std::chrono::steady_clock::now() < deadline) {
    if (remaining_.load(std::memory_order_acquire) == 0) {
      return true;
    }
    CpuRelax();
  }
  return false;
}

void ThreadPool::RunErased(int64_t num_blocks, void *task_ctx, TaskFn task_fn) {
  if (num_blocks <= 0) {
    return;
  }
  if (workers_.empty() || num_blocks == 1 || InPool()) {
    for (int64_t b = 0; b < num_blocks; ++b) {
      task_fn(task_ctx, b);
    }
    return;
  }
  if (num_blocks > worker_count() + 1) {
    throw std::invalid_argument("ThreadPool num_blocks exceeds its participant capacity.");
  }
  std::lock_guard<std::mutex> region(region_mu_);
  {
    std::lock_guard<std::mutex> lock(mu_);
    task_ctx_ = task_ctx;
    task_fn_ = task_fn;
    num_blocks_ = num_blocks;
    remaining_.store(num_blocks - 1, std::memory_order_relaxed);
    active_workers_.store(num_blocks - 1, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    for (int64_t worker = 0; worker < num_blocks - 1; ++worker) {
      worker_work_[static_cast<size_t>(worker)]->notify_one();
    }
  }

  bool &in_pool = InPoolFlag();
  const bool was_in_pool = in_pool;
  in_pool = true;
  task_fn(task_ctx, static_cast<int64_t>(0));
  in_pool = was_in_pool;

  if (SpinForCompletion()) {
    return;
  }
  std::unique_lock<std::mutex> lock(mu_);
  cv_done_.wait(lock, [this]() { return remaining_.load(std::memory_order_acquire) == 0; });
}

void ThreadPool::WorkerLoop(int64_t worker_index) {
  InPoolFlag() = true;
  std::string startup_error;
  if (options_.worker_start != nullptr &&
      !options_.worker_start(options_.worker_start_context, worker_index, startup_error) &&
      startup_error.empty()) {
    startup_error = "ThreadPool worker startup callback failed.";
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!startup_error.empty() && startup_error_.empty()) {
      startup_error_ = std::move(startup_error);
    }
    ++started_workers_;
  }
  cv_started_.notify_one();

  uint64_t last_generation = 0;
  for (;;) {
    bool work_ready = false;
    if (worker_index < active_workers_.load(std::memory_order_acquire)) {
      work_ready = SpinForWork(last_generation, worker_index);
    }
    void *ctx = nullptr;
    TaskFn fn = nullptr;
    int64_t num_blocks = 0;
    if (stop_.load(std::memory_order_acquire)) {
      return;
    }
    if (work_ready) {
      std::lock_guard<std::mutex> lock(mu_);
      const uint64_t generation = generation_.load(std::memory_order_acquire);
      if (generation == last_generation ||
          worker_index >= active_workers_.load(std::memory_order_acquire)) {
        continue;
      }
      last_generation = generation;
      ctx = task_ctx_;
      fn = task_fn_;
      num_blocks = num_blocks_;
    } else {
      std::unique_lock<std::mutex> lock(mu_);
      worker_work_[static_cast<size_t>(worker_index)]->wait(
          lock, [this, last_generation, worker_index]() {
            return stop_.load(std::memory_order_acquire) ||
                   (generation_.load(std::memory_order_acquire) != last_generation &&
                    worker_index < active_workers_.load(std::memory_order_acquire));
          });
      if (stop_.load(std::memory_order_acquire)) {
        return;
      }
      last_generation = generation_.load(std::memory_order_acquire);
      ctx = task_ctx_;
      fn = task_fn_;
      num_blocks = num_blocks_;
    }
    const int64_t block = worker_index + 1;
    if (block < num_blocks) {
      fn(ctx, block);
      if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(mu_);
        cv_done_.notify_one();
      }
    }
  }
}

ThreadPool &GlobalThreadPool() {
  static ThreadPool pool(ParallelForThreadCount() - 1);
  return pool;
}

namespace {

struct ParallelRange {
  void *task_ctx;
  detail::ParallelRangeFn task_fn;
  int64_t base_block_size;
  int64_t extra_blocks;
};

void RunParallelBlock(ParallelRange &range, int64_t block_index) {
  const int64_t begin =
      block_index * range.base_block_size + std::min(block_index, range.extra_blocks);
  const int64_t end = begin + range.base_block_size + (block_index < range.extra_blocks ? 1 : 0);
  range.task_fn(range.task_ctx, begin, end);
}

} // namespace

namespace detail {

void ParallelForErased(int64_t total, int64_t grain_size, void *task_ctx, ParallelRangeFn task_fn) {
  if (total <= 0) {
    return;
  }
  EXT_ENFORCE_INVALID(grain_size > 0, "ParallelFor grain_size must be positive, got ", grain_size,
                      ".");
  // Inside a session run the leased executor is installed on the calling
  // thread; dispatching through it is what makes the session policy effective.
  // The process-wide pool below only serves standalone callers that run
  // outside any executor scope.
  if (CpuExecutor *executor = CurrentCpuExecutor(); executor != nullptr) {
    executor->ParallelFor(total, grain_size, task_ctx, task_fn);
    return;
  }
  const int64_t max_threads = ParallelForThreadCount();
  if (total < grain_size || max_threads <= 1) {
    task_fn(task_ctx, static_cast<int64_t>(0), total);
    return;
  }

  const int64_t max_useful_blocks = total / grain_size;
  const int64_t num_blocks = std::min(max_threads, max_useful_blocks);
  if (num_blocks <= 1) {
    task_fn(task_ctx, static_cast<int64_t>(0), total);
    return;
  }

  ParallelRange range{
      task_ctx,
      task_fn,
      total / num_blocks,
      total % num_blocks,
  };
  GlobalThreadPool().Run(num_blocks,
                         [&range](int64_t block_index) { RunParallelBlock(range, block_index); });
}

void ParallelForErasedProfiled(int64_t total, int64_t grain_size, void *task_ctx,
                               ParallelRangeFn task_fn, ParallelRegionCollector *collector,
                               std::string_view label, std::source_location location) {
  if (total <= 0) {
    return;
  }
  EXT_ENFORCE_INVALID(grain_size > 0, "ParallelFor grain_size must be positive, got ", grain_size,
                      ".");
  if (CpuExecutor *executor = CurrentCpuExecutor(); executor != nullptr) {
    executor->ParallelFor(total, grain_size, task_ctx, task_fn, 0, collector, label, location);
    return;
  }

  const auto start = std::chrono::steady_clock::now();
  const std::optional<uint64_t> process_cpu_start = ReadProcessCpuTimeNs();
  const HardwareCounterMeasurement counter_measurement = collector->BeginHardwareCounters();
  const uint64_t region_id = NextParallelRegionId();
  const uint64_t parent_region_id = CurrentParallelRegionId();
  const uint64_t run_id = CurrentParallelRegionRunId();
  const std::thread::id calling_thread_id = std::this_thread::get_id();
  const int64_t max_threads = ParallelForThreadCount();
  uint32_t admitted = 1;
  const bool nested_inline = ThreadPool::InParallelRegion();
  if (total < grain_size || max_threads <= 1) {
    ParallelRegionCollectorScope collector_scope(collector, run_id, region_id);
    task_fn(task_ctx, static_cast<int64_t>(0), total);
  } else {
    const int64_t max_useful_blocks = total / grain_size;
    const int64_t num_blocks = std::min(max_threads, max_useful_blocks);
    if (num_blocks <= 1) {
      ParallelRegionCollectorScope collector_scope(collector, run_id, region_id);
      task_fn(task_ctx, static_cast<int64_t>(0), total);
    } else {
      admitted = nested_inline ? 1 : static_cast<uint32_t>(num_blocks);
      ParallelRange range{
          task_ctx,
          task_fn,
          total / num_blocks,
          total % num_blocks,
      };
      GlobalThreadPool().Run(
          num_blocks, [&range, collector, run_id, region_id](int64_t block_index) {
            ParallelRegionCollectorScope collector_scope(collector, run_id, region_id);
            RunParallelBlock(range, block_index);
          });
    }
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
  const HardwareCounterSample counters =
      collector->EndHardwareCounters(counter_measurement, admitted == 1);
  collector->Record(ParallelRegionEvent{
      .region_id = region_id,
      .parent_region_id = parent_region_id,
      .run_id = run_id,
      .calling_thread_id = calling_thread_id,
      .label = label,
      .location = location,
      .total_iterations = total,
      .grain_size = grain_size,
      .requested_threads = static_cast<int32_t>(max_threads),
      .admitted_threads = static_cast<int32_t>(admitted),
      .observed_threads = static_cast<int32_t>(admitted),
      .wall_time_ns = wall_time_ns,
      .process_cpu_time_ns = process_cpu_time_ns,
      .cpu_utilization =
          ComputeCpuUtilization(process_cpu_time_ns, wall_time_ns, static_cast<int32_t>(admitted)),
      .counters = counters,
      .executor_instance_id = 0,
      .nested_inline = nested_inline,
  });
}

} // namespace detail

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
