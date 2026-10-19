// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "onnx_light_helpers.h"
#include <cstdint>
#include <string>

/**
 * @file onnx_io_policy.h
 * @brief Adaptive external-data I/O policy: lazy worker creation, calibrated block size and
 * worker count, and a lightweight trace of the resolved policy.
 *
 * This is Native PR01 of :ref:`l-next-steps-native-fast-loading-completion`. Parallel
 * reads/writes of external tensor data already create their worker threads lazily (see
 * :cpp:class:`ONNX_LIGHT_NAMESPACE::utils::ThreadPool`, which only spawns OS threads on the
 * first submitted task), so a metadata-only parse (``ParseOptions::skip_raw_data``) or a
 * wholly borrowed/mapped parse (``ParseOptions::no_copy`` or
 * ``TwoFilesStream::use_mmap_weights``) never pays for a worker pool because no delayed block
 * is ever submitted. What was missing is (1) a minimum block size below which a read/write is
 * kept on the calling thread instead of being queued (``ParseOptions::min_parallel_block_size``
 * and ``SerializeOptions::min_parallel_block_size`` were declared but never enforced), and (2)
 * an automatic worker-count/block-size choice that adapts to the storage kind instead of a
 * blind ``std::thread::hardware_concurrency()`` guess. This header adds both.
 */

namespace ONNX_LIGHT_NAMESPACE::utils {

/** Coarse classification of the storage backing an external-data read, used to calibrate the
 *  adaptive I/O policy. */
enum class IOStorageKind : int32_t {
  /** The tensor bytes are borrowed from a memory-mapped file (``no_copy`` or
   *  ``file_load_mode=MMAP``): bytes are resolved by lazy page faults, not by a worker pool. */
  kMmap = 0,
  /** The file is resident (or mostly resident) in the OS page cache: reads are memory-bandwidth
   *  bound rather than I/O-latency bound, so more, smaller-block workers help. */
  kWarmPageCache = 1,
  /** The file's residency could not be determined (unsupported platform, probe failure, or a
   *  stream that is not backed by a real file); a moderate, conservative default is used. */
  kBufferedReads = 2,
  /** The file is not resident in the OS page cache: reads are disk-latency bound, so fewer
   *  workers with larger blocks reduce seek overhead. */
  kColdStorage = 3,
};

/** Resolved worker count and minimum block size for one parallel I/O operation. */
struct IOPolicy {
  /** Number of worker threads to request from the stream's thread pool.
   *  ``0`` means "do not start a thread pool"; every read/write happens on the calling thread. */
  int32_t workers = 1;
  /** Minimum block size (bytes) a single read/write must reach to be submitted to the thread
   *  pool; smaller blocks are processed on the calling thread to avoid thread-pool overhead. */
  int64_t min_block_size = 0;
};

/** Trace of the policy actually applied to one parse or serialize call, plus the observed
 *  byte/fault counters. Left at its default (all zero) when policy tracing was not requested. */
struct IOPolicyTrace {
  /** Storage kind detected (or assumed) when the policy was resolved. */
  IOStorageKind storage_kind = IOStorageKind::kBufferedReads;
  /** Worker count actually used (0 means no thread pool was started). */
  int32_t resolved_workers = 0;
  /** Minimum block size actually enforced. */
  int64_t resolved_min_block_size = 0;
  /** Bytes physically present in the external-data source examined to resolve the policy
   *  (typically the weights file size). */
  int64_t physical_bytes = 0;
  /** Total bytes submitted to the thread pool as delayed blocks and awaited together, i.e. the
   *  peak number of bytes outstanding at once under the current submit-then-wait model. */
  int64_t bytes_in_flight = 0;
  /** Number of memory pages touched by :cpp:var:`ParseOptions::_touch_raw_data_pages`, i.e. an
   *  upper bound on the lazy page faults a wholly mapped load can trigger. Zero when page
   *  touching was not requested. */
  uint64_t page_faults = 0;
};

/**
 * Resolves the worker count and minimum block size to use for one parallel I/O operation.
 *
 * @param kind                     Storage kind backing the operation (see
 * :cpp:enum:`IOStorageKind`).
 * @param total_bytes              Total number of bytes the operation may transfer; used to avoid
 *                                 requesting more workers than there is useful work for.
 * @param requested_num_threads    Caller request, using the same convention as
 *                                 ``ParseOptions::num_threads`` /
 * ``SerializeOptions::num_threads``:
 *                                 ``1`` forces serial execution, ``> 1`` forces exactly that many
 *                                 workers, and ``<= 0`` asks for an automatic, storage-aware
 * choice.
 * @param requested_min_block_size Caller-provided minimum block size, or ``0`` to use the
 *                                 storage-aware default.
 * @returns                        The resolved :cpp:class:`IOPolicy`. ``kind == kMmap`` always
 *                                 resolves to zero workers: mapped bytes are resolved through page
 *                                 faults, not a worker pool.
 */
ONNX_LIGHT_PROTO_API IOPolicy ResolveIOPolicy(IOStorageKind kind, int64_t total_bytes,
                                              int32_t requested_num_threads,
                                              int64_t requested_min_block_size);

/**
 * Best-effort classification of the storage backing *file_path* as warm (resident in the OS
 * page cache) or cold. Samples at most the first 64 MiB of the file with a ``PROT_NONE``
 * mapping and queries per-page residency (``mincore``); only implemented on Linux today.
 *
 * @returns :cpp:enumerator:`IOStorageKind::kWarmPageCache` or
 *          :cpp:enumerator:`IOStorageKind::kColdStorage` when residency could be sampled, or
 *          :cpp:enumerator:`IOStorageKind::kBufferedReads` when the file could not be opened,
 *          probed (non-Linux platform), or is empty.
 */
ONNX_LIGHT_PROTO_API IOStorageKind DetectIOStorageKind(const std::string &file_path);

} // namespace ONNX_LIGHT_NAMESPACE::utils
