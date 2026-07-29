// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_helpers.h"

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

/**
 * @file parallel_for.h
 * @brief Minimal block-parallel iteration helper for element-wise kernels.
 */

namespace ONNX_LIGHT_NAMESPACE {
namespace core {
namespace runtime {

/// Iteration count below which :cpp:func:`ParallelFor` runs the whole range
/// inline on the calling thread. Spawning worker threads for tiny ranges costs
/// more than the work they save, so small tensors stay single-threaded.
inline constexpr int64_t kParallelForGrainSize = 1 << 15; // 32768 elements

/// Returns the number of worker threads :cpp:func:`ParallelFor` may use.
///
/// Resolves to ``std::thread::hardware_concurrency()``, falling back to ``1``
/// when the hardware count is not available. The result is always ``>= 1``.
///
/// Returns:
///   The effective worker-thread count, always at least ``1``.
inline int64_t ParallelForThreadCount() noexcept {
  const unsigned int cores = std::thread::hardware_concurrency();
  return cores == 0 ? 1 : static_cast<int64_t>(cores);
}

/**
 * Splits the half-open range ``[0, total)`` into contiguous blocks and invokes
 * ``fn(begin, end)`` once per block.
 *
 * Blocks are processed on up to :cpp:func:`ParallelForThreadCount` threads. When
 * ``total`` is below :cpp:var:`kParallelForGrainSize` or only one thread is
 * available the whole range is processed inline on the calling thread, so ``fn``
 * must be safe to call once with the full range. Every block is disjoint and
 * covers the range exactly once, so the observable result is independent of the
 * number of threads: kernels that only map input elements to output elements
 * (no cross-element accumulation) stay bit-exact.
 *
 * ``fn`` is invoked concurrently from several threads and must therefore only
 * touch data disjoint per block (typically writing ``output[begin, end)`` from
 * ``input[begin, end)``). It must not throw.
 *
 * @param total Number of iterations. Values ``<= 0`` are a no-op.
 * @param fn    Callable invoked as ``fn(int64_t begin, int64_t end)`` for each
 *              block, covering the half-open sub-range ``[begin, end)``.
 */
template <typename Fn> void ParallelFor(int64_t total, Fn fn) {
  if (total <= 0) {
    return;
  }
  const int64_t max_threads = ParallelForThreadCount();
  if (total < kParallelForGrainSize || max_threads <= 1) {
    fn(static_cast<int64_t>(0), total);
    return;
  }

  // Use as many blocks as threads, but never so many that a block would hold
  // fewer than kParallelForGrainSize iterations.
  const int64_t max_useful_blocks = (total + kParallelForGrainSize - 1) / kParallelForGrainSize;
  const int64_t num_blocks = std::min(max_threads, max_useful_blocks);
  if (num_blocks <= 1) {
    fn(static_cast<int64_t>(0), total);
    return;
  }

  const int64_t block = (total + num_blocks - 1) / num_blocks;
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(num_blocks - 1));
  for (int64_t begin = block; begin < total; begin += block) {
    const int64_t end = std::min(begin + block, total);
    workers.emplace_back([&fn, begin, end]() { fn(begin, end); });
  }
  // Run the first block on the calling thread while the workers run the rest.
  fn(static_cast<int64_t>(0), std::min(block, total));
  for (std::thread &worker : workers) {
    worker.join();
  }
}

} // namespace runtime
} // namespace core
} // namespace ONNX_LIGHT_NAMESPACE
