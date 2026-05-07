#include "thread_pool.h"

namespace onnx {
namespace utils {

ThreadPool::ThreadPool() : stop(false), is_started(false), pending_jobs_(0) {}

void ThreadPool::Start(int32_t num_threads) {
  EXT_ENFORCE(workers.size() == 0, "ThreadPool already started");
  stop = false;
  is_started = true;
  if (num_threads == -1)
    num_threads = std::thread::hardware_concurrency();

  for (size_t i = 0; i < static_cast<size_t>(num_threads); ++i) {
    workers.emplace_back(&ThreadPool::worker_thread, this);
  }
}

void ThreadPool::SubmitTask(std::function<void()> job) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    ++pending_jobs_;
    jobs.push(std::move(job));
  }
  if (!workers.empty())
    condition.notify_one();
}

void ThreadPool::worker_thread() {
  while (true) {
    std::function<void()> job;

    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      condition.wait(lock, [this]() { return stop || !jobs.empty(); });

      if (stop && jobs.empty())
        return;

      job = std::move(jobs.front());
      jobs.pop();
    }

    job();

    {
      std::unique_lock<std::mutex> lock(done_mutex_);
      --pending_jobs_;
    }
    done_condition_.notify_all();
  }
}

void ThreadPool::Wait() {
  if (workers.empty()) {
    // No workers, so run jobs inline on the calling thread.
    while (!jobs.empty()) {
      std::function<void()> job = std::move(jobs.front());
      jobs.pop();
      job();
      {
        std::unique_lock<std::mutex> lock(done_mutex_);
        --pending_jobs_;
      }
    }
  } else {
    // Block until every submitted job has finished executing.
    std::unique_lock<std::mutex> lock(done_mutex_);
    done_condition_.wait(lock, [this]() { return pending_jobs_.load() == 0; });
  }
  stop = true;
  condition.notify_all();
  for (std::thread &worker : workers) {
    if (worker.joinable())
      worker.join();
  }
  workers.clear();
  is_started = false;
}

ThreadPool::~ThreadPool() { Wait(); }

void ThreadPool::Clear() {
  EXT_ENFORCE(!IsStarted(), "Cannot clear the pool if threads are still running.");
  workers.clear();
  while (!jobs.empty()) {
    jobs.pop();
  }
  pending_jobs_.store(0);
}

} // namespace utils
} // namespace onnx
