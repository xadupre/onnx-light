// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_helpers.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/** Identifies whether a calibration profile is machine-specific or portable. */
enum class CalibrationProfileKind {
  kExact,
  kPortable,
};

/**
 * Identifies a backend calibration profile.
 *
 * Exact profiles use ``model_digest``, ``processor``, and ``thread_count``.
 * Portable profiles instead use ``structural_properties``.
 */
struct CalibrationProfileKey {
  std::string backend;
  std::string operator_name;
  std::string implementation_version;
  CalibrationProfileKind kind = CalibrationProfileKind::kExact;
  std::string model_digest;
  std::string processor;
  uint32_t thread_count = 0;
  std::map<std::string, std::string> structural_properties;

  bool operator==(const CalibrationProfileKey &) const = default;
};

/** Stores one named calibration measurement. */
struct CalibrationMeasurement {
  std::string name;
  double value = 0;
  std::string unit;

  bool operator==(const CalibrationMeasurement &) const = default;
};

/**
 * Stores a serialized backend policy and the measurements used to select it.
 *
 * ``policy`` is opaque to onnx-light. Backends serialize and validate it through
 * callbacks passed to :cpp:class:`CalibrationProfileStore`.
 */
struct CalibrationProfile {
  CalibrationProfileKey key;
  std::string policy;
  std::vector<CalibrationMeasurement> measurements;
  bool user_override = false;

  bool operator==(const CalibrationProfile &) const = default;
};

/** Serializes a backend-owned policy payload. */
using CalibrationPolicySerializer = std::function<std::string()>;

/**
 * Validates or deserializes a serialized backend policy.
 *
 * Returns ``true`` for an accepted payload and writes a rejection explanation
 * to the second argument otherwise.
 */
using CalibrationPolicyValidator = std::function<bool(std::string_view, std::string &)>;

/** Classifies profile loading and mutation results. */
enum class CalibrationProfileStoreStatus {
  kOk,
  kNotFound,
  kDisabled,
  kUnreadable,
  kMalformed,
  kUnsupportedVersion,
  kPolicyRejected,
  kWriteFailed,
};

/** Explains why a stored profile was not selected. */
enum class CalibrationProfileRejectionReason {
  kDifferentBackend,
  kDifferentOperator,
  kOutdatedImplementation,
  kDifferentModel,
  kDifferentProcessor,
  kDifferentThreadCount,
  kDifferentStructure,
  kPolicyRejected,
};

/** Reports one rejected candidate. */
struct CalibrationProfileRejection {
  CalibrationProfileKey key;
  CalibrationProfileRejectionReason reason;
  std::string message;
};

/** Reports loading or mutation of the profile store. */
struct CalibrationProfileStoreReport {
  CalibrationProfileStoreStatus status = CalibrationProfileStoreStatus::kOk;
  size_t affected_profiles = 0;
  std::vector<std::string> diagnostics;
};

/** Controls persistent storage. */
struct CalibrationProfileStoreOptions {
  std::filesystem::path path;
  bool persistence_enabled = true;
};

/** Controls exact/portable profile resolution. */
struct CalibrationProfileLookupOptions {
  CalibrationProfileKey exact_key;
  std::map<std::string, std::string> structural_properties;
  bool force_portable = false;
};

/** Reports profile selection and every rejected candidate. */
struct CalibrationProfileLookupReport {
  CalibrationProfileStoreStatus status = CalibrationProfileStoreStatus::kNotFound;
  std::optional<CalibrationProfile> profile;
  std::vector<CalibrationProfileRejection> rejections;
  std::vector<std::string> diagnostics;
};

/**
 * Stores backend calibration profiles in memory and, optionally, on disk.
 *
 * Mutations are synchronized between threads and processes. Persistent updates
 * merge the latest disk contents and atomically replace the store file.
 */
class CalibrationProfileStore {
public:
  explicit CalibrationProfileStore(CalibrationProfileStoreOptions options = {});

  /** Returns the platform-specific default calibration profile path. */
  static std::filesystem::path DefaultPath();

  /**
   * Reloads profiles from persistent storage.
   *
   * Existing in-memory profiles remain unchanged when the file is malformed,
   * incompatible, or unreadable.
   */
  CalibrationProfileStoreReport Reload();

  /**
   * Stores calibration results after backend serialization and validation.
   *
   * The serializer is invoked exactly once. A rejected payload is not stored.
   */
  CalibrationProfileStoreReport Store(const CalibrationProfileKey &key,
                                      std::vector<CalibrationMeasurement> measurements,
                                      const CalibrationPolicySerializer &serialize,
                                      const CalibrationPolicyValidator &validate = {});

  /** Resolves an override, exact profile, or portable profile in that order. */
  CalibrationProfileLookupReport Lookup(const CalibrationProfileLookupOptions &options,
                                        const CalibrationPolicyValidator &validate = {}) const;

  /** Installs and persists an explicit user override. */
  CalibrationProfileStoreReport InstallOverride(const CalibrationProfileKey &key,
                                                const CalibrationPolicySerializer &serialize,
                                                const CalibrationPolicyValidator &validate = {});

  /** Clears matching user overrides. */
  CalibrationProfileStoreReport ClearOverride(const CalibrationProfileKey &key);

  /**
   * Invalidates profiles for a backend and, optionally, one implementation version.
   *
   * An absent version invalidates every version owned by the backend.
   */
  CalibrationProfileStoreReport
  Invalidate(std::string_view backend,
             std::optional<std::string_view> implementation_version = std::nullopt);

  /** Returns a snapshot of every in-memory profile. */
  std::vector<CalibrationProfile> Inspect() const;

  /** Returns whether persistent reads and writes are enabled. */
  bool persistence_enabled() const noexcept { return options_.persistence_enabled; }

  /** Returns the configured or default persistent path. */
  const std::filesystem::path &path() const noexcept { return path_; }

private:
  CalibrationProfileStoreReport
  Mutate(const std::function<size_t(std::vector<CalibrationProfile> &)> &);

  CalibrationProfileStoreOptions options_;
  std::filesystem::path path_;
  mutable std::mutex mutex_;
  std::vector<CalibrationProfile> profiles_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
