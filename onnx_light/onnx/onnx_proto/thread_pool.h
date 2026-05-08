#pragma once

#include "onnx_light_helpers.h"
#include <condition_variable>
#include <functional>
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
  inline size_t GetThreadCount() const { return workers_.size(); }
  inline bool IsStarted() const { return is_started_; }
  void Clear();

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> jobs_;

  // Single mutex protecting all shared state (jobs_, pending_jobs_, stop_).
  // Using one mutex eliminates split-brain races between job-queue and
  // done-counter that can cause condition-variable notifications to be
  // missed on Windows.
  mutable std::mutex mutex_;
  std::condition_variable work_cv_;  // workers wait here for new jobs
  std::condition_variable done_cv_;  // Wait() waits here for all jobs to finish
  bool stop_;
  bool is_started_;

  // Counts jobs that are queued OR currently executing.
  size_t pending_jobs_;

  void worker_thread();
};

} // namespace utils
} // namespace onnx
