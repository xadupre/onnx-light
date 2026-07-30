// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Multi-threaded join used by the vendored BLAKE3 library.
//
// When the BLAKE3 sources are compiled with ``BLAKE3_USE_TBB`` defined,
// ``blake3_compress_subtree_wide`` delegates the recursion over the two halves
// of a subtree to ``blake3_compress_subtree_wide_join_tbb``. Upstream ships a
// oneTBB-based implementation; onnx-light does not depend on oneTBB, so this
// file provides an equivalent based on ``std::async``.
//
// The two halves are hashed on separate threads only when the combined input is
// large enough to amortise the thread hand-off, and only while a global budget
// (derived from ``std::thread::hardware_concurrency``) still allows spawning a
// worker. This bounds oversubscription from the recursive fork/join while
// keeping the output bit-for-bit identical to the single-threaded digest.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <thread>

extern "C" {
#include "blake3_impl.h"
}

namespace {

// Number of extra worker threads that may run concurrently. The calling thread
// always participates, so the budget is ``hardware_concurrency - 1``.
int InitialThreadBudget() {
  const unsigned int cores = std::thread::hardware_concurrency();
  const int n = cores == 0 ? 1 : static_cast<int>(cores);
  return n > 1 ? n - 1 : 0;
}

std::atomic<int> &ThreadBudget() {
  static std::atomic<int> budget{InitialThreadBudget()};
  return budget;
}

// Minimum combined input size before splitting the work across two threads.
// Below this, waking a worker costs more than the parallel hashing saves.
constexpr std::size_t kParallelThreshold = 256 * 1024;

// Tries to reserve one unit of the thread budget. Returns true on success.
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

extern "C" void blake3_compress_subtree_wide_join_tbb(
    // shared params
    const uint32_t key[8], uint8_t flags, bool use_tbb,
    // left-hand side params
    const uint8_t *l_input, size_t l_input_len, uint64_t l_chunk_counter, uint8_t *l_cvs,
    size_t *l_n,
    // right-hand side params
    const uint8_t *r_input, size_t r_input_len, uint64_t r_chunk_counter, uint8_t *r_cvs,
    size_t *r_n) noexcept {
  if (use_tbb && (l_input_len + r_input_len) >= kParallelThreshold && TryAcquireThread()) {
    // Hash the right half on a worker thread while the left half runs here.
    // std::async can throw (e.g. resource exhaustion) when spawning the thread;
    // in that case fall back to serial hashing so the digest is always produced.
    // Only the launch is guarded: blake3_compress_subtree_wide does not throw,
    // so no exception can escape once the worker is running.
    std::future<void> right;
    bool launched = false;
    try {
      right = std::async(std::launch::async, [&]() {
        *r_n = blake3_compress_subtree_wide(r_input, r_input_len, key, r_chunk_counter, flags,
                                            r_cvs, use_tbb);
      });
      launched = true;
    } catch (const std::exception &) {
      launched = false;
    }
    if (launched) {
      *l_n = blake3_compress_subtree_wide(l_input, l_input_len, key, l_chunk_counter, flags, l_cvs,
                                          use_tbb);
      right.get();
      ReleaseThread();
      return;
    }
    ReleaseThread();
  }
  *l_n = blake3_compress_subtree_wide(l_input, l_input_len, key, l_chunk_counter, flags, l_cvs,
                                      use_tbb);
  *r_n = blake3_compress_subtree_wide(r_input, r_input_len, key, r_chunk_counter, flags, r_cvs,
                                      use_tbb);
}
