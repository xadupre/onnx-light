// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/tuning/parallel_region_collector.h"
#include "onnx_light_helpers.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @file parallel_for.h
 * @brief Persistent thread pool and block-parallel iteration helper for
 *        element-wise kernels.
 */

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/// Iteration count below which :cpp:func:`ParallelFor` runs the whole range
/// inline on the calling thread. Waking worker threads for tiny ranges costs
/// more than the work they save, so small tensors stay single-threaded.
inline constexpr int64_t kParallelForGrainSize = 1 << 15; // 32768 elements

/**
 * Configures worker startup and spin-before-park behavior for a
 * :cpp:class:`ThreadPool`.
 */
struct ThreadPoolOptions {
  /// Type-erased worker-start callback. It returns ``false`` and populates
  /// ``error`` when the worker cannot satisfy its startup policy.
  using WorkerStartFn = bool (*)(void *context, int64_t worker_index, std::string &error);

  /// Number of relax iterations before parking.
  uint64_t spin_iterations = 1000000;
  /// Duration in nanoseconds to spin before parking. Used only when
  /// ``spin_iterations`` is zero.
  uint64_t spin_duration_ns = 0;
  /// Optional callback invoked once by every worker before accepting work.
  WorkerStartFn worker_start = nullptr;
  /// Context passed to :cpp:var:`worker_start`.
  void *worker_start_context = nullptr;
};

/// Returns the number of participating threads :cpp:func:`ParallelFor` may use.
///
/// When a :cpp:class:`CpuExecutor` is installed on the calling thread (see
/// :cpp:class:`CpuExecutorScope`), the executor's effective participant count
/// is returned so the reported value describes the workers that actually run
/// the graph. Outside any executor scope it resolves to one participant per
/// detected physical core, falling back to the
/// detected logical-core count and then ``std::thread::hardware_concurrency()``.
/// The result is always ``>= 1`` and counts the calling thread, which always
/// participates in the work.
///
/// Returns:
///   The effective participant count, always at least ``1``.
int64_t ParallelForThreadCount() noexcept;

/**
 * A persistent pool of worker threads that stay alive between parallel regions.
 *
 * Unlike spawning fresh ``std::thread`` objects per call, the workers are
 * created once and briefly spin before parking on a condition variable. Nearby
 * regions therefore avoid scheduler wakeup latency without busy-waiting
 * indefinitely.
 *
 * The pool exposes a single primitive, :cpp:func:`Run`, that executes a set of
 * indexed blocks with a static assignment: block ``0`` runs on the calling
 * thread and block ``j`` runs on worker ``j - 1``. It deliberately offers no
 * reduction/combine step: callers write disjoint output ranges, so results are
 * independent of how blocks map to threads (bit-exact). The pool handles several
 * scenarios:
 *   - no workers available (single core): every block runs inline on the caller;
 *   - a single block: runs inline without touching the workers;
 *   - nested calls from inside a running block: run inline to avoid deadlock;
 *   - concurrent calls from unrelated threads: serialized so one region runs at
 *     a time, each still internally parallel.
 */
class ThreadPool {
public:
  /// Type-erased block callable: ``fn(context, block_index)``.
  using TaskFn = void (*)(void *, int64_t);

  /// Creates a pool with ``num_workers`` parked worker threads.
  ///
  /// @param num_workers Number of worker threads to spawn. Values ``<= 0``
  ///                    create a pool with no workers, in which case
  ///                    :cpp:func:`Run` executes every block on the caller.
  explicit ThreadPool(int64_t num_workers);

  /// Creates a pool with explicit startup and spin behavior.
  ThreadPool(int64_t num_workers, ThreadPoolOptions options);

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  ~ThreadPool();

  /// Returns the number of worker threads (the calling thread is not counted).
  ///
  /// Returns:
  ///   The worker-thread count, ``>= 0``.
  int64_t worker_count() const noexcept { return static_cast<int64_t>(workers_.size()); }

  /**
   * Runs ``fn(block)`` for every ``block`` in ``[0, num_blocks)``, then blocks
   * until all blocks finish.
   *
   * Block ``0`` runs on the calling thread and block ``j`` runs on worker
   * ``j - 1`` (static assignment). ``fn`` is invoked concurrently and must only
   * touch data disjoint per block; it must not throw. ``num_blocks`` must not
   * exceed ``worker_count() + 1`` when workers are used; :cpp:func:`ParallelFor`
   * enforces this.
   *
   * @param num_blocks Number of blocks to run. Values ``<= 0`` are a no-op.
   * @param fn         Callable invoked as ``fn(int64_t block)``.
   */
  template <typename Fn> void Run(int64_t num_blocks, Fn &&fn) {
    using Callable = std::remove_reference_t<Fn>;
    Callable &callable = fn;
    RunErased(num_blocks, static_cast<void *>(&callable),
              [](void *ctx, int64_t b) { (*static_cast<Callable *>(ctx))(b); });
  }

  /// Returns whether the calling thread is executing a pool region.
  static bool InParallelRegion() noexcept { return InPool(); }

private:
  void RunErased(int64_t num_blocks, void *task_ctx, TaskFn task_fn);
  static bool &InPoolFlag() noexcept;
  static bool InPool() noexcept;
  void StopAndJoin() noexcept;
  bool SpinForWork(uint64_t last_generation) const noexcept;
  bool SpinForCompletion() const noexcept;
  void WorkerLoop(int64_t worker_index);

  ThreadPoolOptions options_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::mutex region_mu_;
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;
  std::condition_variable cv_started_;
  void *task_ctx_ = nullptr;
  TaskFn task_fn_ = nullptr;
  int64_t num_blocks_ = 0;
  int64_t started_workers_ = 0;
  std::string startup_error_;
  std::atomic<int64_t> remaining_{0};
  std::atomic<uint64_t> generation_{0};
  std::atomic<bool> stop_{false};
};

/// Returns the process-wide :cpp:class:`ThreadPool` used by :cpp:func:`ParallelFor`.
///
/// The pool is constructed on first use with ``ParallelForThreadCount() - 1``
/// worker threads (the calling thread makes up the last participant) and lives
/// for the remainder of the process. Threads are therefore created once and
/// reused across every ``ParallelFor`` call.
///
/// Returns:
///   A reference to the shared thread pool.
ThreadPool &GlobalThreadPool();

namespace detail {

using ParallelRangeFn = void (*)(void *, int64_t, int64_t);

void ParallelForErased(int64_t total, int64_t grain_size, void *task_ctx, ParallelRangeFn task_fn);
void ParallelForErasedProfiled(int64_t total, int64_t grain_size, void *task_ctx,
                               ParallelRangeFn task_fn, ParallelRegionCollector *collector,
                               std::string_view label, std::source_location location);

} // namespace detail

/**
 * Splits the half-open range ``[0, total)`` into contiguous blocks and invokes
 * ``fn(begin, end)`` once per block.
 *
 * Blocks are processed on the :cpp:class:`CpuExecutor` installed on the calling
 * thread, or on the shared :cpp:func:`GlobalThreadPool` when the caller runs
 * outside any executor scope (up to :cpp:func:`ParallelForThreadCount`
 * participants, including the calling thread). When ``total`` is below ``grain_size`` or only one
 * thread is available the whole range is processed inline on the calling thread, so
 * ``fn`` must be safe to call once with the full range. The three-argument
 * overload accepts a kernel-specific ``grain_size``; the two-argument overload
 * below uses :cpp:var:`kParallelForGrainSize`. Every
 * block is disjoint and covers the range exactly once, so the observable result
 * is independent of the number of threads: kernels that only map input
 * elements to output elements (no cross-element accumulation) stay bit-exact.
 *
 * ``fn`` is invoked concurrently from several threads and must therefore only
 * touch data disjoint per block (typically writing ``output[begin, end)`` from
 * ``input[begin, end)``). It must not throw.
 *
 * @param total      Number of iterations. Values ``<= 0`` are a no-op.
 * @param grain_size Minimum iterations per parallel block. Must be positive.
 * @param fn         Callable invoked as ``fn(int64_t begin, int64_t end)`` for
 *                   each block, covering ``[begin, end)``.
 */
template <typename Fn>
void ParallelFor(int64_t total, int64_t grain_size, Fn fn, std::string_view label = {},
                 std::source_location location = std::source_location::current()) {
  const auto task_fn = [](void *ctx, int64_t begin, int64_t end) {
    (*static_cast<Fn *>(ctx))(begin, end);
  };
  ParallelRegionCollector *collector = CurrentParallelRegionCollector();
  if (collector == nullptr) {
    detail::ParallelForErased(total, grain_size, static_cast<void *>(&fn), task_fn);
    return;
  }
  detail::ParallelForErasedProfiled(total, grain_size, static_cast<void *>(&fn), task_fn, collector,
                                    label, location);
}

/// Runs ``fn`` over ``[0, total)`` using the default grain size.
template <typename Fn>
void ParallelFor(int64_t total, Fn fn, std::string_view label = {},
                 std::source_location location = std::source_location::current()) {
  ParallelFor(total, kParallelForGrainSize, std::move(fn), label, location);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
