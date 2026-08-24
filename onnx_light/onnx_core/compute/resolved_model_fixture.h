// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/prepared_task.h"
#include "onnx_light_helpers.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

enum class PayloadResolution {
  kRequired,
  kDormantFallback,
  kDead,
  kSuperseded,
  kPreparedCacheReplaced,
};

struct PayloadManifestEntry {
  std::string id;
  std::filesystem::path location;
  uint64_t offset = 0;
  uint64_t length = 0;
  PayloadResolution resolution = PayloadResolution::kRequired;
};

struct PayloadAvoidanceReport {
  uint64_t dead_bytes = 0;
  uint64_t superseded_bytes = 0;
  uint64_t cache_replaced_bytes = 0;
  uint64_t total_avoided_bytes = 0;

  uint64_t total_bytes() const noexcept { return total_avoided_bytes; }
};

/**
 * Owns the immutable payload requirements selected before materialization.
 */
class ONNX_LIGHT_CORE_API RequiredPayloadManifest {
public:
  RequiredPayloadManifest(RequiredPayloadManifest &&) = default;
  RequiredPayloadManifest &operator=(RequiredPayloadManifest &&) = delete;
  RequiredPayloadManifest(const RequiredPayloadManifest &) = delete;
  RequiredPayloadManifest &operator=(const RequiredPayloadManifest &) = delete;

  static RequiredPayloadManifest
  Freeze(std::vector<PayloadManifestEntry> entries,
         std::optional<std::string> eager_fallback_diagnostic = std::nullopt);

  const std::vector<PayloadManifestEntry> &entries() const noexcept { return entries_; }
  const PayloadAvoidanceReport &avoided() const noexcept { return avoided_; }
  bool ContainsActive(const std::string &id) const noexcept;
  bool uses_eager_fallback() const noexcept { return eager_fallback_diagnostic_.has_value(); }
  const std::optional<std::string> &eager_fallback_diagnostic() const noexcept {
    return eager_fallback_diagnostic_;
  }

private:
  RequiredPayloadManifest() = default;

  std::vector<PayloadManifestEntry> entries_;
  std::unordered_map<std::string, size_t> active_payloads_;
  PayloadAvoidanceReport avoided_;
  std::optional<std::string> eager_fallback_diagnostic_;

  friend class ResolvedModelFixture;
};

/**
 * Provides the frozen resolved-model contract used by prepared-execution tests.
 *
 * This fixture is intentionally smaller than the production resolver planned
 * by the model-resolution roadmap. Reads are addressed only through its frozen
 * active payload manifest.
 */
class ONNX_LIGHT_CORE_API ResolvedModelFixture {
public:
  ResolvedModelFixture(std::filesystem::path model_path, RequiredPayloadManifest payload_manifest);

  const std::filesystem::path &model_path() const noexcept { return model_path_; }
  const std::vector<PayloadManifestEntry> &payload_manifest() const noexcept {
    return payload_manifest_.entries();
  }
  const RequiredPayloadManifest &required_payloads() const noexcept { return payload_manifest_; }

  /** Reads one complete active payload and rejects IDs outside the manifest. */
  std::vector<uint8_t> ReadPayload(const std::string &id) const;
  /** Reads the payload named by a manifest-backed read task. */
  std::vector<uint8_t> ReadPayload(const TaskDescriptor &task) const;

private:
  std::filesystem::path model_path_;
  RequiredPayloadManifest payload_manifest_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
