// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/parallel_for.h"

#include <algorithm>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

int64_t ParallelForThreadCount() noexcept {
  static const int64_t thread_count = []() {
    const unsigned int cores = std::thread::hardware_concurrency();
    return cores == 0 ? int64_t{1} : static_cast<int64_t>(cores);
  }();
  return thread_count;
}

ThreadPool::ThreadPool(int64_t num_workers) {
  if (num_workers < 0) {
    num_workers = 0;
  }
  workers_.reserve(static_cast<size_t>(num_workers));
  for (int64_t i = 0; i < num_workers; ++i) {
    workers_.emplace_back([this, i]() { WorkerLoop(i); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
    ++generation_;
  }
  cv_work_.notify_all();
  for (std::thread &worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

bool &ThreadPool::InPoolFlag() noexcept {
  thread_local bool in_pool = false;
  return in_pool;
}

bool ThreadPool::InPool() noexcept { return InPoolFlag(); }

void ThreadPool::RunErased(int64_t num_blocks, void *task_ctx, TaskFn task_fn) {
  if (num_blocks <= 0) {
    return;
  }
  if (workers_.empty() || num_blocks == 1 || InPool()) {
    for (int64_t b = 0; b < num_blocks; ++b) {
      task_fn(task_ctx, b);
    }
    return;
  }

  std::lock_guard<std::mutex> region(region_mu_);
  {
    std::lock_guard<std::mutex> lock(mu_);
    task_ctx_ = task_ctx;
    task_fn_ = task_fn;
    num_blocks_ = num_blocks;
    remaining_.store(num_blocks - 1, std::memory_order_relaxed);
    ++generation_;
  }
  cv_work_.notify_all();

  bool &in_pool = InPoolFlag();
  const bool was_in_pool = in_pool;
  in_pool = true;
  task_fn(task_ctx, static_cast<int64_t>(0));
  in_pool = was_in_pool;

  std::unique_lock<std::mutex> lock(mu_);
  cv_done_.wait(lock, [this]() { return remaining_.load(std::memory_order_acquire) == 0; });
}

void ThreadPool::WorkerLoop(int64_t worker_index) {
  InPoolFlag() = true;
  const int64_t my_block = worker_index + 1;
  uint64_t last_generation = 0;
  for (;;) {
    void *ctx = nullptr;
    TaskFn fn = nullptr;
    int64_t num_blocks = 0;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_work_.wait(lock,
                    [this, last_generation]() { return stop_ || generation_ != last_generation; });
      if (stop_) {
        return;
      }
      last_generation = generation_;
      ctx = task_ctx_;
      fn = task_fn_;
      num_blocks = num_blocks_;
    }
    if (my_block < num_blocks) {
      fn(ctx, my_block);
      if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(mu_);
        cv_done_.notify_one();
      }
    }
  }
}

ThreadPool &GlobalThreadPool() {
  static ThreadPool pool(ParallelForThreadCount() - 1);
  return pool;
}

namespace {

struct ParallelRange {
  void *task_ctx;
  detail::ParallelRangeFn task_fn;
  int64_t block;
  int64_t total;
};

void RunParallelBlock(ParallelRange &range, int64_t block_index) {
  const int64_t begin = block_index * range.block;
  if (begin >= range.total) {
    return;
  }
  const int64_t end = std::min(begin + range.block, range.total);
  range.task_fn(range.task_ctx, begin, end);
}

} // namespace

namespace detail {

void ParallelForErased(int64_t total, int64_t grain_size, void *task_ctx, ParallelRangeFn task_fn) {
  if (total <= 0) {
    return;
  }
  EXT_ENFORCE_INVALID(grain_size > 0, "ParallelFor grain_size must be positive, got ", grain_size,
                      ".");
  const int64_t max_threads = ParallelForThreadCount();
  if (total < grain_size || max_threads <= 1) {
    task_fn(task_ctx, static_cast<int64_t>(0), total);
    return;
  }

  const int64_t max_useful_blocks = total / grain_size;
  const int64_t num_blocks = std::min(max_threads, max_useful_blocks);
  if (num_blocks <= 1) {
    task_fn(task_ctx, static_cast<int64_t>(0), total);
    return;
  }

  ParallelRange range{
      task_ctx,
      task_fn,
      (total + num_blocks - 1) / num_blocks,
      total,
  };
  GlobalThreadPool().Run(num_blocks,
                         [&range](int64_t block_index) { RunParallelBlock(range, block_index); });
}

} // namespace detail

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
