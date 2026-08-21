#pragma once

#include "onnx_light_helpers.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::utils {

/**
 * Persistent fixed-size worker pool.
 *
 * Workers remain available across calls to WaitIdle() and stop only when
 * Shutdown() is called.
 */
class ONNX_LIGHT_PROTO_API WorkerPool {
public:
  WorkerPool();
  explicit WorkerPool(int32_t num_threads);
  ~WorkerPool();

  WorkerPool(const WorkerPool &) = delete;
  WorkerPool &operator=(const WorkerPool &) = delete;

  /** Starts persistent workers. A negative count uses hardware concurrency. */
  void Start(int32_t num_threads);

  /** Enqueues work for execution. */
  void Enqueue(std::function<void()> job);

  /** Waits for all queued work and rethrows the first task exception. */
  void WaitIdle();

  /** Drains pending work and permanently joins the current worker set. */
  void Shutdown();

  size_t GetThreadCount() const;
  bool IsStarted() const;

private:
  void Execute(std::function<void()> &job) noexcept;
  void WorkerLoop();
  void WaitIdleImpl(bool rethrow_exceptions);

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> jobs_;
  mutable std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  bool stop_;
  bool started_;
  size_t pending_jobs_;
  std::exception_ptr first_exception_;
};

} // namespace ONNX_LIGHT_NAMESPACE::utils
