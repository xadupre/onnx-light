// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/platform/cpu_descriptor.h"
#include "onnx_core/runtime/kernel_tuning.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/** Describes the stable execution properties used to match a calibrated profile. */
struct CpuExecutionDescriptor {
  platform::CpuDescriptor processor;
  uint32_t effective_threads = 0;

  bool operator==(const CpuExecutionDescriptor &) const = default;
};

/** Filters cache entries. Non-empty fields combine with logical AND. */
struct KernelCalibrationSelection {
  std::optional<std::string> library;
  std::vector<std::string> kernels;
  std::vector<std::string> implementations;
  std::vector<int32_t> element_types;
  std::optional<Device> device;
  bool only_missing = false;

  /** Returns whether an exact key satisfies this selection. */
  bool Matches(const KernelTuningKey &key) const;
};

/** Controls cache location and execution-descriptor matching. */
struct KernelTuningCacheOptions {
  std::filesystem::path path;
  std::optional<CpuExecutionDescriptor> execution;
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

/**
 * Returns the platform-specific default tuning cache path.
 *
 * Returns:
 *   A path under the user's cache directory.
 */
std::filesystem::path DefaultKernelTuningCachePath();

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

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
