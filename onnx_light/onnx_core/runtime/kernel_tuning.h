// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/platform/cpu_descriptor.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_light_helpers.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

using symbolic::Device;

/**
 * Identifies one tunable kernel implementation and element type.
 *
 * ``tuning_abi`` must change whenever persisted parameters for the
 * implementation become incompatible.
 */
struct KernelTuningKey {
  std::string library;
  std::string kernel;
  std::string implementation;
  int32_t element_type = 0;
  Device device = Device::kUndefined;
  uint32_t tuning_abi = 0;

  bool operator==(const KernelTuningKey &) const = default;
};

/** Hashes every field of a :cpp:class:`KernelTuningKey`. */
struct KernelTuningKeyHash {
  size_t operator()(const KernelTuningKey &key) const noexcept;
};

/** Describes the stable execution properties used to resolve a processor profile. */
struct CpuExecutionDescriptor {
  platform::CpuDescriptor processor;
  uint32_t effective_threads = 0;

  bool operator==(const CpuExecutionDescriptor &) const = default;
};

/** Stores one portable scalar tuning value. */
using TuningValue = std::variant<int64_t, double, bool, std::string>;

/**
 * Returns the stable type name of a tuning value.
 *
 * Returns:
 *   One of ``"int64"``, ``"double"``, ``"bool"``, or ``"string"``.
 */
std::string_view TuningValueTypeName(const TuningValue &value) noexcept;

/** Stores the named values associated with an exact tuning key. */
struct KernelTuningParameters {
  KernelTuningKey key;
  std::unordered_map<std::string, TuningValue> values;

  /** Returns whether a named value is present. */
  bool Contains(std::string_view name) const;

  /**
   * Returns a named value with its exact scalar type.
   *
   * @throws std::invalid_argument if the name is absent or has another type.
   *
   * Returns:
   *   The requested value.
   */
  template <typename T> const T &Get(std::string_view name) const {
    static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, double> ||
                      std::is_same_v<T, bool> || std::is_same_v<T, std::string>,
                  "T must be one of the TuningValue alternatives.");
    auto found = values.find(std::string(name));
    if (found == values.end()) {
      ThrowMissingValue(name);
    }
    const T *value = std::get_if<T>(&found->second);
    if (value == nullptr) {
      ThrowWrongType(name, TuningTypeName<T>(), TuningValueTypeName(found->second));
    }
    return *value;
  }

private:
  template <typename T> static constexpr std::string_view TuningTypeName() {
    if constexpr (std::is_same_v<T, int64_t>) {
      return "int64";
    } else if constexpr (std::is_same_v<T, double>) {
      return "double";
    } else if constexpr (std::is_same_v<T, bool>) {
      return "bool";
    } else {
      return "string";
    }
  }

  [[noreturn]] static void ThrowMissingValue(std::string_view name);
  [[noreturn]] static void ThrowWrongType(std::string_view name, std::string_view expected,
                                          std::string_view actual);
};

/** Filters tuning keys. Non-empty fields combine with logical AND. */
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

/** Bounds one explicit calibration request. Zero means callback-defined. */
struct CalibrationOptions {
  std::optional<CpuExecutionDescriptor> execution;
  uint64_t maximum_duration_ms = 0;
  uint64_t maximum_memory_bytes = 0;
  std::optional<uint32_t> maximum_threads;
};

/** Collects diagnostics emitted by one calibration callback. */
class CalibrationReporter {
public:
  /** Appends one diagnostic message. */
  void AddDiagnostic(std::string message);

  /** Returns callback diagnostics in emission order. */
  const std::vector<std::string> &diagnostics() const noexcept { return diagnostics_; }

private:
  std::vector<std::string> diagnostics_;
};

/** Calibrates one exact, registered kernel tuning key. */
using KernelCalibrationFunction =
    std::function<KernelTuningParameters(const KernelTuningKey &, const CpuExecutionDescriptor &,
                                         const CalibrationOptions &, CalibrationReporter &)>;

/** Associates one calibration diagnostic with its exact key. */
struct KernelCalibrationDiagnostic {
  KernelTuningKey key;
  std::string message;
};

/** Reports one explicit batch calibration and its atomic publication. */
struct CalibrationBatchReport {
  uint64_t published_generation = 0;
  std::vector<KernelTuningParameters> calibrated;
  std::vector<KernelTuningKey> skipped;
  std::vector<KernelTuningKey> unsupported;
  std::vector<KernelTuningKey> failed;
  std::vector<KernelCalibrationDiagnostic> diagnostics;

  /** Returns successfully validated profiles. */
  std::span<const KernelTuningParameters> successful_profiles() const noexcept {
    return calibrated;
  }
};

/**
 * Validates kernel-specific value ranges and relationships.
 *
 * A hook throws ``std::invalid_argument`` when the complete parameter set is
 * invalid. Generic name, key, presence, and type checks run before the hook.
 */
using KernelTuningValidationHook = std::function<void(const KernelTuningParameters &)>;

/**
 * Defines one kernel's portable defaults and validation contract.
 *
 * Construction validates the key, names, and portable defaults immediately.
 * Every subsequently validated parameter set must contain exactly the same
 * names and scalar types, so no tunable value can exist without a compiled
 * portable fallback.
 */
class KernelTuningSchema {
public:
  explicit KernelTuningSchema(KernelTuningParameters portable_defaults,
                              KernelTuningValidationHook validation_hook = {});

  /** Returns the exact kernel key described by this schema. */
  const KernelTuningKey &key() const noexcept { return portable_defaults_.key; }

  /** Returns the validated, hard-coded portable parameter set. */
  const KernelTuningParameters &portable_defaults() const noexcept { return portable_defaults_; }

  /**
   * Validates a complete parameter set against this schema.
   *
   * @throws std::invalid_argument for a mismatched key, unknown or missing
   * name, wrong scalar type, or a kernel-specific validation failure.
   */
  void Validate(const KernelTuningParameters &parameters) const;

private:
  void ValidateValues(const KernelTuningParameters &parameters) const;

  KernelTuningParameters portable_defaults_;
  KernelTuningValidationHook validation_hook_;
};

class KernelTuningRegistry;

/**
 * Holds one immutable generation of resolved kernel tuning parameters.
 *
 * A snapshot remains valid after later registrations or cache loads publish a
 * newer generation.
 */
class KernelTuningRegistrySnapshot {
public:
  /** Returns the registry generation captured by this snapshot. */
  uint64_t generation() const noexcept;

  /**
   * Finds the resolved parameters for an exact tuning key.
   *
   * This lookup does not select an execution-specific calibrated profile; use
   * :cpp:func:`Resolve` when an execution descriptor is available.
   *
   * Returns:
   *   The parameters, or ``nullptr`` when the key is not registered.
   */
  const KernelTuningParameters *Find(const KernelTuningKey &key) const noexcept;

  /** Returns whether a compatible cached, calibrated, or override profile exists. */
  bool HasPublishedProfile(const KernelTuningKey &key,
                           const CpuExecutionDescriptor &execution) const noexcept;

  /**
   * Resolves the highest-precedence profile matching an execution descriptor.
   *
   * Explicitly published parameters take precedence over registered processor
   * profiles, which in turn take precedence over portable defaults.
   *
   * Returns:
   *   The resolved parameters, or ``nullptr`` when the key is not registered.
   */
  const KernelTuningParameters *Resolve(const KernelTuningKey &key,
                                        const CpuExecutionDescriptor &execution) const noexcept;

private:
  struct State;
  explicit KernelTuningRegistrySnapshot(std::shared_ptr<const State> state)
      : state_(std::move(state)) {}

  std::shared_ptr<const State> state_;

  friend class KernelTuningRegistry;
};

/**
 * Registers tuning schemas and atomically publishes immutable parameter sets.
 *
 * Publication validates the complete batch before making any value visible.
 */
class KernelTuningRegistry {
public:
  KernelTuningRegistry();
  ~KernelTuningRegistry();
  KernelTuningRegistry(const KernelTuningRegistry &) = delete;
  KernelTuningRegistry &operator=(const KernelTuningRegistry &) = delete;

  /**
   * Registers one schema and its portable defaults.
   *
   * @throws std::invalid_argument if its key is already registered.
   */
  void RegisterSchema(KernelTuningSchema schema);

  /**
   * Registers one validated processor-specific profile.
   *
   * Exact vendor/family/model selectors outrank processor-list and
   * microarchitecture selectors, which outrank instruction-set selectors.
   * Priority only distinguishes profiles at the same specificity.
   *
   * @throws std::invalid_argument for an invalid selector, parameters that do
   * not match the schema, or a selector ambiguous with an existing profile.
   */
  void RegisterProfile(const KernelTuningKey &key, platform::CpuSelector processors,
                       KernelTuningParameters parameters, int priority = 0);

  /**
   * Registers one trusted native calibration callback.
   *
   * @throws std::invalid_argument when the key has no schema, the callback is
   * empty, or another callback is already registered for the key.
   */
  void RegisterCalibrationFunction(const KernelTuningKey &key, KernelCalibrationFunction function);

  /** Returns the current immutable registry generation. */
  KernelTuningRegistrySnapshot Snapshot() const noexcept;

  /**
   * Publishes a validated batch and resets selected keys to portable defaults.
   *
   * Every replacement and reset key must be registered. Validation failure
   * leaves the current generation unchanged.
   */
  void PublishProfiles(std::span<const KernelTuningParameters> profiles,
                       std::span<const KernelTuningKey> reset_keys = {});

  /**
   * Publishes profiles scoped to one exact execution descriptor.
   *
   * Other execution descriptors and universal overrides remain unchanged.
   */
  void PublishCalibratedProfiles(std::span<const KernelTuningParameters> profiles,
                                 const CpuExecutionDescriptor &execution,
                                 std::span<const KernelTuningKey> reset_keys = {});

  /** Returns the registered keys. */
  std::vector<KernelTuningKey> RegisteredKeys() const;

  /** Returns the schema for a registered key, or ``nullptr``. */
  std::shared_ptr<const KernelTuningSchema> FindSchema(const KernelTuningKey &key) const;

  /** Returns the callback registered for a key, or an empty function. */
  KernelCalibrationFunction FindCalibrationFunction(const KernelTuningKey &key) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/** Returns the process-wide tuning registry. */
KernelTuningRegistry &GetKernelTuningRegistry();

/** Registers a schema in the process-wide tuning registry. */
void RegisterKernelTuningSchema(KernelTuningSchema schema);

/** Registers a processor-specific profile in the process-wide tuning registry. */
void RegisterKernelTuningProfile(const KernelTuningKey &key, platform::CpuSelector processors,
                                 KernelTuningParameters parameters, int priority = 0);

/** Registers a trusted calibration callback in the process-wide registry. */
void RegisterKernelCalibrationFunction(const KernelTuningKey &key,
                                       KernelCalibrationFunction function);

/**
 * Runs selected registered callbacks and atomically publishes successful profiles.
 *
 * Unsupported keys have a tuning schema but no callback. ``only_missing``
 * skips keys with an explicitly published profile. Callback or validation
 * failures are reported without discarding successful keys.
 */
CalibrationBatchReport CalibrateRegisteredKernels(const KernelCalibrationSelection &selection = {},
                                                  const CalibrationOptions &options = {});

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
