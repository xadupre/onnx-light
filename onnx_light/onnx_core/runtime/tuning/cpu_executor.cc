// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/tuning/cpu_executor.h"

#include "onnx_core/runtime/kernels/parallel_for.h"

#include <algorithm>
#include <cerrno>
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
  explicit Impl(ResolvedCpuExecutionPolicy resolved)
      : policy(std::move(resolved)), executor_key(MakeCpuExecutorKey(policy)),
        process_id(CurrentProcessId()) {
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
  std::unique_ptr<ThreadPool> pool;
};

CpuExecutor::CpuExecutor(ResolvedCpuExecutionPolicy policy)
    : impl_(std::make_unique<Impl>(std::move(policy))) {}

CpuExecutor::~CpuExecutor() = default;

uint32_t CpuExecutor::effective_threads() const noexcept { return impl_->policy.effective_threads; }

const ResolvedCpuExecutionPolicy &CpuExecutor::policy() const noexcept { return impl_->policy; }

const CpuExecutorKey &CpuExecutor::key() const noexcept { return impl_->executor_key; }

void CpuExecutor::ParallelFor(int64_t total, int64_t grain, void *context, ParallelRangeFn function,
                              uint32_t maximum_participants) {
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
  if (total < grain || participant_limit <= 1) {
    function(context, 0, total);
    return;
  }
  const int64_t useful_blocks = total / grain;
  const int64_t num_blocks =
      std::min<int64_t>(static_cast<int64_t>(participant_limit), useful_blocks);
  if (num_blocks <= 1) {
    function(context, 0, total);
    return;
  }

  ParallelRange range{
      context,
      function,
      total / num_blocks,
      total % num_blocks,
  };
  impl_->pool->Run(num_blocks, [&range, this](int64_t block_index) {
    CpuExecutorScope block_scope(this);
    const int64_t begin =
        block_index * range.base_block_size + std::min(block_index, range.extra_blocks);
    const int64_t end = begin + range.base_block_size + (block_index < range.extra_blocks ? 1 : 0);
    range.function(range.context, begin, end);
  });
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
