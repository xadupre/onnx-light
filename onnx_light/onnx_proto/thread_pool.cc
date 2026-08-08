#include "thread_pool.h"

namespace ONNX_LIGHT_NAMESPACE::utils {

ThreadPool::ThreadPool()
    : stop_(false), is_started_(false), requested_threads_(0), pending_jobs_(0) {}

void ThreadPool::Start(int32_t num_threads) {
  if (num_threads < 0)
    num_threads = static_cast<int32_t>(std::thread::hardware_concurrency());

  std::lock_guard<std::mutex> lock(mutex_);
  EXT_ENFORCE(!is_started_, "ThreadPool already started");
  stop_ = false;
  is_started_ = true;
  requested_threads_ = num_threads;
  // Worker threads are spawned lazily on the first SubmitTask() (see
  // EnsureWorkersStarted). This keeps load/save of small models fast: when no
  // delayed block is submitted, Wait() runs the empty queue inline and no
  // thread is ever created or joined.
}

void ThreadPool::EnsureWorkersStarted() {
  // Caller must hold mutex_.
  if (!is_started_ || !workers_.empty() || requested_threads_ <= 0)
    return;
  workers_.reserve(static_cast<size_t>(requested_threads_));
  for (int32_t i = 0; i < requested_threads_; ++i) {
    workers_.emplace_back(&ThreadPool::worker_thread, this);
  }
}

void ThreadPool::SubmitTask(std::function<void()> &&job) {
  bool has_workers;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++pending_jobs_;
    jobs_.push(std::move(job));
    EnsureWorkersStarted();
    has_workers = !workers_.empty();
  }
  if (has_workers)
    work_cv_.notify_one();
}

void ThreadPool::SubmitTask(const std::function<void()> &job) {
  bool has_workers;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++pending_jobs_;
    jobs_.push(job);
    EnsureWorkersStarted();
    has_workers = !workers_.empty();
  }
  if (has_workers)
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

    try {
      job();
    } catch (...) {
      // Prevent uncaught exceptions from calling std::terminate().
      // The error is silently swallowed; the caller should validate
      // results after WaitForDelayedBlock() returns.
    }

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

} // namespace ONNX_LIGHT_NAMESPACE::utils
