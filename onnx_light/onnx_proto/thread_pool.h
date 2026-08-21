#pragma once

#include "onnx_light_helpers.h"
#include "worker_pool.h"
#include <functional>

namespace ONNX_LIGHT_NAMESPACE::utils {

/**
 * Thread pool used to parallelize delayed block loading.
 *
 * Manages a fixed set of worker threads that pull jobs from a shared queue.
 * Call Start() to launch workers, SubmitTask() to enqueue work, and Wait() to
 * block until all submitted jobs have completed and the workers have stopped.
 * The pool can be restarted by calling Start() again after Wait() returns.
 * Clear() resets internal state when the pool is idle (not started).
 */
class ONNX_LIGHT_PROTO_API ThreadPool {
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
   * Starts the pool, deferring worker-thread creation until the first task.
   *
   * The pool is marked started immediately, but the worker threads are only
   * spawned lazily on the first SubmitTask() call. A parse/serialize that never
   * submits a delayed block (typically a small model whose tensor blocks all
   * stay below the parallelization threshold) therefore never pays the cost of
   * creating and joining threads, keeping load/save latency minimal.
   *
   * @param num_threads Number of worker threads to create. Any negative value
   *                    (for example ``-1``) is treated as a request to use the
   *                    value returned by ``std::thread::hardware_concurrency()``.
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
  void SubmitTask(std::function<void()> &&job);
  void SubmitTask(const std::function<void()> &job);

  /**
   * Blocks until all submitted jobs have finished executing, then stops and
   * joins all worker threads.
   *
   * If no workers were started, runs any queued jobs inline on the calling
   * thread. If a submitted job throws, Wait() rethrows the first captured
   * exception on the caller thread after all workers have been joined. After
   * Wait() returns, the pool is in a stopped state and can be restarted with
   * Start().
   */
  void Wait();

  /** Returns the number of worker threads the pool has (or will lazily spawn). */
  inline size_t GetThreadCount() const {
    return workers_.GetThreadCount() == 0 && is_started_ ? static_cast<size_t>(requested_threads_)
                                                         : workers_.GetThreadCount();
  }

  /** Returns whether the pool has been started (workers may start lazily). */
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
  WorkerPool workers_;
  bool is_started_;
  int32_t requested_threads_;
  void WaitImpl(bool rethrow_exceptions);
};

} // namespace ONNX_LIGHT_NAMESPACE::utils
