// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_helpers.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

struct PayloadManifestEntry {
  std::string id;
  std::filesystem::path location;
  uint64_t offset = 0;
  uint64_t length = 0;
  bool active = true;
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
  ResolvedModelFixture(std::filesystem::path model_path,
                       std::vector<PayloadManifestEntry> payload_manifest);

  const std::filesystem::path &model_path() const noexcept { return model_path_; }
  const std::vector<PayloadManifestEntry> &payload_manifest() const noexcept {
    return payload_manifest_;
  }

  /** Reads one complete active payload and rejects IDs outside the manifest. */
  std::vector<uint8_t> ReadPayload(const std::string &id) const;

private:
  std::filesystem::path model_path_;
  std::vector<PayloadManifestEntry> payload_manifest_;
  std::unordered_map<std::string, size_t> active_payloads_;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
