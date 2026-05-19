#pragma once

#include "onnx_light_helpers.h"
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace utils {

/**
 * Thread pool used to parallelize delayed block loading.
 *
 * Manages a fixed set of worker threads that pull jobs from a shared queue.
 * Call Start() to launch workers, SubmitTask() to enqueue work, and Wait() to
 * block until all submitted jobs have completed and the workers have stopped.
 * The pool can be restarted by calling Start() again after Wait() returns.
 * Clear() resets internal state when the pool is idle (not started).
 */
class ThreadPool {
public:
  /** Initializes the thread pool in a stopped, idle state. */
  ThreadPool();

  /**
   * Destroys the thread pool by calling Wait().
   *
   * Blocks until all pending jobs have finished executing, then stops and joins
   * all worker threads. If the pool has already been stopped, this is a no-op.
   */
  ~ThreadPool();

  /**
   * Starts the pool by launching worker threads.
   *
   * @param num_threads Number of worker threads to create. Pass -1 to use the
   *                    value returned by std::thread::hardware_concurrency().
   */
  void Start(int32_t num_threads);

  /**
   * Submits a callable job for asynchronous execution.
   *
   * If no worker threads have been started, the job is queued and will be run
   * inline when Wait() is called.
   *
   * @param job Callable to execute.
   */
  void SubmitTask(std::function<void()> job);

  /**
   * Blocks until all submitted jobs have finished executing, then stops and
   * joins all worker threads.
   *
   * If no workers were started, runs any queued jobs inline on the calling
   * thread. After Wait() returns, the pool is in a stopped state and can be
   * restarted with Start().
   */
  void Wait();

  /** Returns the number of worker threads currently in the pool. */
  inline size_t GetThreadCount() const { return workers_.size(); }

  /** Returns whether the pool has been started and workers are running. */
  inline bool IsStarted() const { return is_started_; }

  /**
   * Resets the pool to an empty, idle state.
   *
   * Clears the worker list and any pending jobs. Must only be called when the
   * pool is not started (i.e., after Wait() has returned or before Start() has
   * been called).
   */
  void Clear();

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> jobs_;

  // Single mutex protecting all shared state (jobs_, pending_jobs_, stop_).
  // Using one mutex eliminates split-brain races between job-queue and
  // done-counter that can cause condition-variable notifications to be
  // missed on Windows.
  mutable std::mutex mutex_;
  std::condition_variable work_cv_; // workers wait here for new jobs
  std::condition_variable done_cv_; // Wait() waits here for all jobs to finish
  bool stop_;
  bool is_started_;

  // Counts jobs that are queued OR currently executing.
  size_t pending_jobs_;

  /** Entry point executed by each worker thread. */
  void worker_thread();
};

} // namespace utils
} // namespace ONNX_LIGHT_NAMESPACE
