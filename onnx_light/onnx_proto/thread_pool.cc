#include "thread_pool.h"

namespace ONNX_LIGHT_NAMESPACE::utils {

WorkerPool::WorkerPool()
    : stop_(false), started_(false), pending_jobs_(0), first_exception_(nullptr) {}

WorkerPool::WorkerPool(int32_t num_threads) : WorkerPool() { Start(num_threads); }

void WorkerPool::Start(int32_t num_threads) {
  if (num_threads < 0)
    num_threads = static_cast<int32_t>(std::thread::hardware_concurrency());

  std::lock_guard<std::mutex> lock(mutex_);
  EXT_ENFORCE(!started_, "WorkerPool already started");
  stop_ = false;
  started_ = true;
  first_exception_ = nullptr;
  workers_.reserve(static_cast<size_t>(num_threads));
  for (int32_t i = 0; i < num_threads; ++i) {
    workers_.emplace_back(&WorkerPool::WorkerLoop, this);
  }
}

void WorkerPool::Enqueue(std::function<void()> job) {
  bool has_workers;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    EXT_ENFORCE(started_, "WorkerPool is not started");
    ++pending_jobs_;
    jobs_.push(std::move(job));
    has_workers = !workers_.empty();
  }
  if (has_workers)
    work_cv_.notify_one();
}

void WorkerPool::Execute(std::function<void()> &job) noexcept {
  std::exception_ptr error;
  try {
    job();
  } catch (...) {
    error = std::current_exception();
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error != nullptr && first_exception_ == nullptr)
      first_exception_ = std::move(error);
    --pending_jobs_;
    if (pending_jobs_ == 0)
      done_cv_.notify_all();
  }
}

void WorkerPool::WorkerLoop() {
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

    Execute(job);
  }
}

void WorkerPool::WaitIdleImpl(bool rethrow_exceptions) {
  if (workers_.empty()) {
    while (true) {
      std::function<void()> job;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (jobs_.empty())
          break;
        job = std::move(jobs_.front());
        jobs_.pop();
      }
      Execute(job);
    }
  } else {
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this]() { return pending_jobs_ == 0; });
  }

  std::exception_ptr error;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    error = std::move(first_exception_);
    first_exception_ = nullptr;
  }
  if (rethrow_exceptions && error != nullptr)
    std::rethrow_exception(error);
}

void WorkerPool::WaitIdle() { WaitIdleImpl(true); }

void WorkerPool::Shutdown() {
  if (!IsStarted())
    return;
  WaitIdleImpl(false);
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
  std::lock_guard<std::mutex> lock(mutex_);
  started_ = false;
}

WorkerPool::~WorkerPool() { Shutdown(); }

size_t WorkerPool::GetThreadCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return workers_.size();
}

bool WorkerPool::IsStarted() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return started_;
}

ThreadPool::ThreadPool() : is_started_(false), requested_threads_(0) {}

void ThreadPool::Start(int32_t num_threads) {
  EXT_ENFORCE(!is_started_, "ThreadPool already started");
  if (num_threads < 0)
    num_threads = static_cast<int32_t>(std::thread::hardware_concurrency());
  is_started_ = true;
  requested_threads_ = num_threads;
}

void ThreadPool::SubmitTask(std::function<void()> &&job) {
  if (!workers_.IsStarted())
    workers_.Start(requested_threads_);
  workers_.Enqueue(std::move(job));
}

void ThreadPool::SubmitTask(const std::function<void()> &job) {
  if (!workers_.IsStarted())
    workers_.Start(requested_threads_);
  workers_.Enqueue(job);
}

void ThreadPool::WaitImpl(bool rethrow_exceptions) {
  std::exception_ptr error;
  if (workers_.IsStarted()) {
    try {
      workers_.WaitIdle();
    } catch (...) {
      error = std::current_exception();
    }
    workers_.Shutdown();
  }
  is_started_ = false;
  if (rethrow_exceptions && error != nullptr)
    std::rethrow_exception(error);
}

void ThreadPool::Wait() { WaitImpl(true); }

ThreadPool::~ThreadPool() { WaitImpl(false); }

void ThreadPool::Clear() {
  EXT_ENFORCE(!IsStarted(), "Cannot clear the pool if threads are still running.");
}

} // namespace ONNX_LIGHT_NAMESPACE::utils
