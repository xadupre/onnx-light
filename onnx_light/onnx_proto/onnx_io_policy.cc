// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#include "onnx_io_policy.h"

#include <algorithm>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ONNX_LIGHT_NAMESPACE::utils {

namespace {

constexpr int64_t kMiB = 1024 * 1024;

/** Storage-kind-specific ceiling on worker count and default minimum block size. */
struct IOKindDefaults {
  int32_t max_workers;
  int64_t min_block_size;
};

IOKindDefaults DefaultsFor(IOStorageKind kind, int32_t hw) {
  switch (kind) {
  case IOStorageKind::kMmap:
    // Mapped bytes are resolved through page faults; no worker pool is ever useful.
    return {0, 0};
  case IOStorageKind::kWarmPageCache:
    // Memory-bandwidth bound: many workers copying small blocks in parallel help. Use every
    // logical processor, matching the previous blind hardware_concurrency() default so this,
    // the most common case, never regresses.
    return {hw, 1 * kMiB};
  case IOStorageKind::kColdStorage:
    // Disk-latency bound: larger reads reduce seek overhead, but still scale worker count with
    // the machine (half of hw, at least 1) instead of an arbitrary tiny constant.
    return {std::max<int32_t>(1, hw / 2), 16 * kMiB};
  case IOStorageKind::kBufferedReads:
  default:
    // Unknown/mixed (also the only outcome on non-Linux platforms today): keep the previous
    // blind hardware_concurrency() default so unsupported platforms never regress.
    return {hw, 4 * kMiB};
  }
}

} // namespace

IOPolicy ResolveIOPolicy(IOStorageKind kind, int64_t total_bytes, int32_t requested_num_threads,
                         int64_t requested_min_block_size) {
  const int32_t hw =
      std::max<int32_t>(1, static_cast<int32_t>(std::thread::hardware_concurrency()));
  const IOKindDefaults defaults = DefaultsFor(kind, hw);
  IOPolicy policy;
  policy.min_block_size =
      requested_min_block_size > 0 ? requested_min_block_size : defaults.min_block_size;

  if (kind == IOStorageKind::kMmap) {
    policy.workers = 0;
    return policy;
  }
  if (requested_num_threads > 1) {
    policy.workers = requested_num_threads;
    return policy;
  }
  if (requested_num_threads == 1) {
    policy.workers = 1;
    return policy;
  }

  // Automatic policy (requested_num_threads <= 0): derive the worker count from the storage
  // kind's ceiling (already expressed relative to hw by DefaultsFor()) and the amount of work
  // actually available, so a small external-data file never pays for workers it cannot keep
  // busy.
  int32_t workers = defaults.max_workers;
  if (policy.min_block_size > 0 && total_bytes > 0) {
    const int64_t useful_workers = std::max<int64_t>(1, total_bytes / policy.min_block_size);
    workers = static_cast<int32_t>(std::min<int64_t>(workers, useful_workers));
  }
  policy.workers = std::max(1, workers);
  return policy;
}

IOStorageKind DetectIOStorageKind(const std::string &file_path) {
#if defined(__linux__)
  const int fd = ::open(file_path.c_str(), O_RDONLY);
  if (fd < 0)
    return IOStorageKind::kBufferedReads;

  struct stat st{};
  if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
    ::close(fd);
    return IOStorageKind::kBufferedReads;
  }

  const size_t page_size = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  if (page_size == 0) {
    ::close(fd);
    return IOStorageKind::kBufferedReads;
  }
  // Sample at most the first 64 MiB so probing a huge file stays cheap.
  const size_t file_size = static_cast<size_t>(st.st_size);
  const size_t sample_size = std::min(file_size, static_cast<size_t>(64) * 1024 * 1024);

  void *mapped = ::mmap(nullptr, sample_size, PROT_NONE, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (mapped == MAP_FAILED)
    return IOStorageKind::kBufferedReads;

  const size_t n_pages = (sample_size + page_size - 1) / page_size;
  std::vector<unsigned char> resident(n_pages, 0);
  const int rc = ::mincore(mapped, sample_size, resident.data());
  size_t resident_pages = 0;
  if (rc == 0) {
    for (unsigned char v : resident) {
      if (v & 1)
        ++resident_pages;
    }
  }
  ::munmap(mapped, sample_size);
  if (rc != 0 || n_pages == 0)
    return IOStorageKind::kBufferedReads;

  const double resident_fraction =
      static_cast<double>(resident_pages) / static_cast<double>(n_pages);
  return resident_fraction >= 0.5 ? IOStorageKind::kWarmPageCache : IOStorageKind::kColdStorage;
#else
  (void)file_path;
  return IOStorageKind::kBufferedReads;
#endif
}

} // namespace ONNX_LIGHT_NAMESPACE::utils
