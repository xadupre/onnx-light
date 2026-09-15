// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/platform/cpu_descriptor.h"
#include "onnx_core/runtime/tuning/kernel_tuning.h"

#include <filesystem>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/** Controls cache location and execution-descriptor matching. */
struct KernelTuningCacheOptions {
  std::filesystem::path path;
  std::optional<CpuExecutionDescriptor> execution;
  bool read_only = false;
  bool replace_existing = true;
  bool prune_stale_abis = false;
};

/** Stores one calibrated parameter set with its complete execution identity. */
struct CalibratedKernelProfile {
  KernelTuningParameters parameters;
  CpuExecutionDescriptor execution;
};

/** Gives the overall result of atomically updating a tuning cache. */
enum class KernelTuningCacheUpdateStatus {
  kUpdated,
  kReadOnly,
  kUnreadable,
  kMalformed,
  kWriteFailed,
};

/** Reports the merge and atomic replacement performed by a cache update. */
struct KernelTuningCacheUpdateReport {
  KernelTuningCacheUpdateStatus status = KernelTuningCacheUpdateStatus::kUpdated;
  std::vector<KernelTuningKey> updated;
  std::vector<KernelTuningKey> preserved;
  std::vector<KernelTuningKey> pruned;
  std::vector<std::string> diagnostics;
};

/** Gives the overall result of reading a tuning cache. */
enum class KernelTuningCacheLoadStatus {
  kLoaded,
  kNotFound,
  kUnreadable,
  kMalformed,
};

/** Reports every cache classification without throwing for file or data errors. */
struct KernelTuningCacheLoadReport {
  KernelTuningCacheLoadStatus status = KernelTuningCacheLoadStatus::kNotFound;
  uint64_t published_generation = 0;
  std::vector<KernelTuningKey> loaded;
  std::vector<KernelTuningKey> incompatible;
  std::vector<KernelTuningKey> stale;
  std::vector<KernelTuningKey> invalid;
  std::vector<KernelTuningKey> missing;
  std::vector<std::string> diagnostics;
};

/** Reports cache contents without validating or publishing them. */
struct KernelTuningCacheInspectionReport {
  KernelTuningCacheLoadStatus status = KernelTuningCacheLoadStatus::kNotFound;
  std::filesystem::path path;
  std::vector<CalibratedKernelProfile> profiles;
  std::vector<std::string> diagnostics;
};

/** Reports removal of a persisted tuning cache. */
struct KernelTuningCacheRemovalReport {
  std::filesystem::path path;
  bool removed = false;
  std::vector<std::string> diagnostics;
};

/** Controls import of a cache file as read-only deployment profiles. */
struct KernelTuningDeploymentImportOptions {
  std::filesystem::path path;
  platform::CpuSelector processors;
  int priority = 0;
};

/** Reports transactional deployment-profile registration. */
struct KernelTuningDeploymentImportReport {
  KernelTuningCacheLoadStatus status = KernelTuningCacheLoadStatus::kNotFound;
  uint64_t published_generation = 0;
  std::vector<KernelTuningKey> imported;
  std::vector<KernelTuningKey> incompatible;
  std::vector<KernelTuningKey> stale;
  std::vector<KernelTuningKey> invalid;
  std::vector<std::string> diagnostics;
};

/**
 * Returns the platform-specific default tuning cache path.
 *
 * Returns:
 *   A path under the user's cache directory.
 */
std::filesystem::path DefaultKernelTuningCachePath();

/**
 * Reads every persisted profile without changing the tuning registry.
 *
 * Parsed profiles retain their recorded processor and effective thread count.
 * Schema compatibility and value validation are intentionally deferred to
 * :cpp:func:`LoadKernelTuningCache`.
 */
KernelTuningCacheInspectionReport
InspectKernelTuningCache(const KernelTuningCacheOptions &options = {});

/**
 * Removes a persisted tuning cache while holding its inter-process lock.
 *
 * Already published profiles and initialized kernel instances are unchanged.
 */
KernelTuningCacheRemovalReport
RemoveKernelTuningCache(const KernelTuningCacheOptions &options = {});

/**
 * Validates, merges, and atomically persists calibrated profiles.
 *
 * The complete key consists of the tuning key and execution descriptor.
 * Existing valid entries not replaced or pruned are preserved.
 */
KernelTuningCacheUpdateReport
UpdateKernelTuningCache(std::span<const CalibratedKernelProfile> profiles,
                        const KernelTuningCacheOptions &options = {});

/**
 * Updates profiles that all target one execution descriptor.
 *
 * ``options.execution`` selects that descriptor, or the current execution
 * descriptor when it is absent.
 */
KernelTuningCacheUpdateReport
UpdateKernelTuningCache(std::span<const KernelTuningParameters> profiles,
                        const KernelTuningCacheOptions &options = {});

/**
 * Loads compatible cache entries and publishes one immutable registry generation.
 *
 * The versioned text format starts with
 * ``onnx_light_kernel_tuning_cache 1``. Each ``profile``/``end`` block stores
 * one exact key, processor descriptor, effective thread count, and typed
 * ``value`` records. Quoted strings use the C++ ``std::quoted`` representation.
 *
 * Missing and unreadable files, malformed input, incompatible entries, and
 * invalid parameter sets are reported without replacing portable defaults.
 */
KernelTuningCacheLoadReport LoadKernelTuningCache(const KernelCalibrationSelection &selection = {},
                                                  const KernelTuningCacheOptions &options = {});

/**
 * Imports cache entries under one explicit deployment processor selector.
 *
 * All selected profiles validate before one immutable registry generation is
 * published. Cached execution descriptors do not widen the supplied selector.
 */
KernelTuningDeploymentImportReport
ImportKernelTuningDeploymentProfiles(const KernelCalibrationSelection &selection,
                                     const KernelTuningDeploymentImportOptions &options);

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
