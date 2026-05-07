#pragma once

#include "onnx_extended_helpers.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace onnx {
namespace utils {

/** Thread pool used to parallelize delayed block loading. */
class ThreadPool {
public:
  ThreadPool();
  ~ThreadPool();
  void Start(int32_t num_threads);
  void SubmitTask(std::function<void()> job);
  void Wait();
  inline size_t GetThreadCount() const { return workers.size(); }
  inline bool IsStarted() const { return is_started; }
  void Clear();

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> jobs;

  std::mutex queue_mutex;
  std::condition_variable condition;
  std::atomic<bool> stop;
  bool is_started;

  // Tracks jobs that are either queued or currently executing.
  // Used to signal callers of Wait() without polling.
  std::atomic<size_t> pending_jobs_;
  std::mutex done_mutex_;
  std::condition_variable done_condition_;

  void worker_thread();
};

} // namespace utils
} // namespace onnx
