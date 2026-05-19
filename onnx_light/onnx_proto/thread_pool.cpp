#include "thread_pool.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace utils {

ThreadPool::ThreadPool() : stop_(false), is_started_(false), pending_jobs_(0) {}

void ThreadPool::Start(int32_t num_threads) {
  EXT_ENFORCE(workers_.size() == 0, "ThreadPool already started");
  if (num_threads == -1)
    num_threads = static_cast<int32_t>(std::thread::hardware_concurrency());

  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = false;
    is_started_ = true;
  }

  for (int32_t i = 0; i < num_threads; ++i) {
    workers_.emplace_back(&ThreadPool::worker_thread, this);
  }
}

void ThreadPool::SubmitTask(std::function<void()> job) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++pending_jobs_;
    jobs_.push(std::move(job));
  }
  if (!workers_.empty())
    work_cv_.notify_one();
}

void ThreadPool::worker_thread() {
  while (true) {
    std::function<void()> job;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      work_cv_.wait(lock, [this]() { return stop_ || !jobs_.empty(); });

      if (stop_ && jobs_.empty())
        return;

      job = std::move(jobs_.front());
      jobs_.pop();
    }

    job();

    {
      std::lock_guard<std::mutex> lock(mutex_);
      --pending_jobs_;
      if (pending_jobs_ == 0)
        done_cv_.notify_all();
    }
  }
}

void ThreadPool::Wait() {
  if (workers_.empty()) {
    // No workers: run jobs inline on the calling thread.
    while (true) {
      std::function<void()> job;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (jobs_.empty())
          break;
        job = std::move(jobs_.front());
        jobs_.pop();
      }
      job();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        --pending_jobs_;
      }
    }
  } else {
    // Block until every submitted job has finished executing.
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this]() { return pending_jobs_ == 0; });
  }

  // Signal workers to stop and join them.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  work_cv_.notify_all();
  for (std::thread &worker : workers_) {
    if (worker.joinable())
      worker.join();
  }
  workers_.clear();
  is_started_ = false;
}

ThreadPool::~ThreadPool() { Wait(); }

void ThreadPool::Clear() {
  EXT_ENFORCE(!IsStarted(), "Cannot clear the pool if threads are still running.");
  std::lock_guard<std::mutex> lock(mutex_);
  workers_.clear();
  while (!jobs_.empty()) {
    jobs_.pop();
  }
  pending_jobs_ = 0;
}

} // namespace utils
} // namespace ONNX_LIGHT_NAMESPACE
