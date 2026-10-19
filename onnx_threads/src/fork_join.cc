// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_threads/fork_join.h"

#include <atomic>
#include <future>
#include <new>
#include <system_error>
#include <thread>

namespace onnx_threads {
namespace {

std::atomic<int> &ThreadBudget() {
  static std::atomic<int> budget{[]() {
    const unsigned int cores = std::thread::hardware_concurrency();
    const int count = cores == 0 ? 1 : static_cast<int>(cores);
    return count > 1 ? count - 1 : 0;
  }()};
  return budget;
}

bool TryAcquireThread() {
  std::atomic<int> &budget = ThreadBudget();
  int current = budget.load(std::memory_order_relaxed);
  while (current > 0) {
    if (budget.compare_exchange_weak(current, current - 1, std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

void ReleaseThread() { ThreadBudget().fetch_add(1, std::memory_order_relaxed); }

} // namespace

void ForkJoin(void *left_context, void (*left)(void *) noexcept, void *right_context,
              void (*right)(void *) noexcept, bool parallel) noexcept {
  if (parallel && TryAcquireThread()) {
    std::future<void> worker;
    bool launched = false;
    try {
      worker = std::async(std::launch::async, [right_context, right]() { right(right_context); });
      launched = true;
    } catch (const std::system_error &) {
      launched = false;
    } catch (const std::bad_alloc &) {
      launched = false;
    }
    if (launched) {
      left(left_context);
      worker.get();
      ReleaseThread();
      return;
    }
    ReleaseThread();
  }
  left(left_context);
  right(right_context);
}

} // namespace onnx_threads
