// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_helpers.h"

#include <cstdint>

/**
 * @file runtime_parameters.h
 * @brief Tunable, model-independent knobs controlling how a graph is
 *        evaluated through :cpp:func:`RunNode` / :cpp:class:`RuntimeSession`.
 */

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Bundles the model-independent execution settings shared across the
 * nodes of a graph evaluated through :cpp:func:`RunNode` /
 * :cpp:class:`RuntimeSession`.
 *
 * Today the only knob is :cpp:var:`num_threads`, the requested degree of
 * parallelism. Grouping it in a dedicated struct keeps the
 * :cpp:class:`RuntimeContext` interface stable as more tuning knobs
 * (for example a thread-pool handle or a device descriptor) are added
 * later without forcing every call site to take an extra argument.
 */
struct RuntimeParameters {
  /** Number of threads used to parallelize the execution of a graph.
   *  - ``0`` (default): use one thread per detected physical CPU core.
   *  - ``1``: no parallelization, everything runs on the calling thread.
   *  - ``> 1``: use exactly this many worker threads.
   *  - ``< 0``: treated the same as ``0`` (use the number of CPU cores). */
  int32_t num_threads = 0;

  RuntimeParameters() = default;
  explicit RuntimeParameters(int32_t num_threads_) : num_threads(num_threads_) {}

  /**
   * Returns the concrete number of threads to use, resolving the special
   * ``num_threads`` values to an actual count.
   *
   * ``0`` and any negative value resolve to the detected physical-core count,
   * falling back to the logical-core count, then
   * ``std::thread::hardware_concurrency()``, and finally ``1``. Every other
   * value is returned unchanged. The result is always ``>= 1``.
   *
   * Returns:
   *   The effective number of threads, always at least ``1``.
   */
  int32_t EffectiveNumThreads() const noexcept;

  /**
   * Returns ``true`` when the graph should be executed with more than one
   * thread, i.e. when :cpp:func:`EffectiveNumThreads` is greater than ``1``.
   *
   * Returns:
   *   ``true`` when parallel execution is enabled, ``false`` otherwise.
   */
  bool is_parallel() const noexcept { return EffectiveNumThreads() > 1; }
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
