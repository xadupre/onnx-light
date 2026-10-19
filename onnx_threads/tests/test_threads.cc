// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_threads/cpu_descriptor.h"
#include "onnx_threads/cpu_executor.h"
#include "onnx_threads/fork_join.h"
#include "onnx_threads/parallel_for.h"
#include "onnx_threads/task_completion.h"
#include "onnx_threads/thread_pool.h"

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <thread>

namespace runtime = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace utils = ONNX_LIGHT_NAMESPACE::utils;

TEST(StandaloneThreads, BatchOwnsCallbacksAndDrainsBeforeRethrowing) {
  utils::ThreadPool pool;
  std::atomic<int> completed{0};
  auto owner = std::make_shared<int>(7);
  std::weak_ptr<int> observer = owner;
  pool.Start(2);
  pool.SubmitTask([owner, &completed]() { completed.fetch_add(*owner); });
  owner.reset();
  pool.SubmitTask([]() { throw std::runtime_error("failed task"); });
  EXPECT_THROW(pool.Wait(), std::runtime_error);
  EXPECT_EQ(completed.load(), 7);
  EXPECT_TRUE(observer.expired());
  EXPECT_FALSE(pool.IsStarted());
  pool.Start(0);
  pool.SubmitTask([&completed]() { ++completed; });
  pool.Wait();
  EXPECT_EQ(completed.load(), 8);
}

TEST(StandaloneThreads, PersistentWorkersRemainUsableAfterFailure) {
  utils::WorkerPool pool(2);
  pool.Enqueue([]() { throw std::runtime_error("failed task"); });
  EXPECT_THROW(pool.WaitIdle(), std::runtime_error);
  std::atomic<int> completed{0};
  pool.Enqueue([&completed]() { ++completed; });
  pool.WaitIdle();
  EXPECT_EQ(completed.load(), 1);
  EXPECT_EQ(pool.GetThreadCount(), 2u);
  pool.Shutdown();
  EXPECT_FALSE(pool.IsStarted());
}

TEST(StandaloneThreads, ScopedBindingRestoresSerialOverrideAndIsThreadLocal) {
  runtime::CpuExecutorRegistry registry(2);
  runtime::CpuExecutionPolicy request;
  request.num_threads = 2;
  request.spin_policy = runtime::CpuSpinPolicy::kParkImmediately;
  auto parallel = registry.Acquire(request);
  EXPECT_EQ(parallel, registry.Acquire(request));
  request.num_threads = 1;
  auto serial = registry.Acquire(request);
  EXPECT_EQ(runtime::CurrentCpuExecutor(), nullptr);
  {
    runtime::CpuExecutorScope outer(parallel.get());
    EXPECT_EQ(runtime::ParallelForThreadCount(), 2);
    {
      runtime::CpuExecutorScope inner(serial.get());
      EXPECT_EQ(runtime::ParallelForThreadCount(), 1);
      const auto caller = std::this_thread::get_id();
      runtime::ParallelFor(
          100, 1, [caller](int64_t, int64_t) { EXPECT_EQ(std::this_thread::get_id(), caller); });
    }
    EXPECT_EQ(runtime::CurrentCpuExecutor(), parallel.get());
    std::thread other([]() { EXPECT_EQ(runtime::CurrentCpuExecutor(), nullptr); });
    other.join();
  }
  EXPECT_EQ(runtime::CurrentCpuExecutor(), nullptr);
}

TEST(StandaloneThreads, IndependentIoAndComputeMakeProgressConcurrently) {
  runtime::CpuExecutorRegistry registry(1);
  runtime::CpuExecutionPolicy request;
  request.num_threads = 2;
  request.allow_nested_parallelism = true;
  request.spin_policy = runtime::CpuSpinPolicy::kParkImmediately;
  auto executor = registry.Acquire(request);
  std::promise<void> compute_finished;
  auto gate = compute_finished.get_future();
  utils::ThreadPool io;
  io.Start(1);
  std::atomic<bool> io_completed{false};
  io.SubmitTask([&gate, &io_completed]() {
    EXPECT_EQ(gate.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    io_completed = true;
  });
  std::array<std::atomic<int>, 64> hits{};
  {
    runtime::CpuExecutorScope scope(executor.get());
    runtime::ParallelFor(8, 1, [&hits](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        runtime::ParallelFor(8, 1, [&hits, i](int64_t inner_begin, int64_t inner_end) {
          for (int64_t j = inner_begin; j < inner_end; ++j) {
            ++hits[static_cast<size_t>(8 * i + j)];
          }
        });
      }
    });
  }
  compute_finished.set_value();
  io.Wait();
  EXPECT_TRUE(io_completed.load());
  for (const auto &hit : hits) {
    EXPECT_EQ(hit.load(), 1);
  }
}

TEST(StandaloneThreads, CompletionRetainsSharedStateAndCancellation) {
  runtime::TaskCompletion completion(runtime::TaskId{1});
  auto observer = completion;
  std::thread producer([completion]() { completion.Succeed(); });
  observer.Wait();
  producer.join();
  EXPECT_EQ(observer.status(), runtime::TaskStatus::kSucceeded);
  runtime::TaskCompletion cancelled(runtime::TaskId{2});
  cancelled.Cancel("cancelled");
  EXPECT_THROW(cancelled.Succeed(), std::runtime_error);
  EXPECT_EQ(cancelled.status(), runtime::TaskStatus::kCancelled);
}

TEST(StandaloneThreads, ForkJoinCompletesBothBranchesAndOwnsNoContexts) {
  const auto increment = [](void *context) noexcept {
    ++*static_cast<std::atomic<int> *>(context);
  };
  std::atomic<int> count{0};
  onnx_threads::ForkJoin(&count, increment, &count, increment, false);
  EXPECT_EQ(count.load(), 2);
  onnx_threads::ForkJoin(&count, increment, &count, increment, true);
  EXPECT_EQ(count.load(), 4);
}
