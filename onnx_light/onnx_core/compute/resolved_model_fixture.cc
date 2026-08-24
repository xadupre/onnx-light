// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "resolved_model_fixture.h"

#include <fstream>
#include <limits>
#include <unordered_set>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

namespace {

void AddAvoidedBytes(uint64_t &category, PayloadAvoidanceReport &report, uint64_t bytes) {
  EXT_ENFORCE(category <= std::numeric_limits<uint64_t>::max() - bytes,
              "Payload avoidance category byte count overflows uint64_t.");
  EXT_ENFORCE(report.total_avoided_bytes <= std::numeric_limits<uint64_t>::max() - bytes,
              "Total payload avoidance byte count overflows uint64_t.");
  category += bytes;
  report.total_avoided_bytes += bytes;
}

} // namespace

RequiredPayloadManifest
RequiredPayloadManifest::Freeze(std::vector<PayloadManifestEntry> entries,
                                std::optional<std::string> eager_fallback_diagnostic) {
  RequiredPayloadManifest manifest;
  manifest.entries_ = std::move(entries);
  manifest.eager_fallback_diagnostic_ = std::move(eager_fallback_diagnostic);
  if (manifest.eager_fallback_diagnostic_) {
    EXT_ENFORCE(!manifest.eager_fallback_diagnostic_->empty(),
                "An eager-loading fallback must include a diagnostic.");
  }

  std::unordered_set<std::string> payload_ids;
  for (size_t i = 0; i < manifest.entries_.size(); ++i) {
    PayloadManifestEntry &entry = manifest.entries_[i];
    EXT_ENFORCE(!entry.id.empty(), "Payload manifest IDs must not be empty.");
    EXT_ENFORCE(payload_ids.insert(entry.id).second, "Duplicate payload manifest ID '", entry.id,
                "'.");
    if (manifest.uses_eager_fallback()) {
      entry.resolution = PayloadResolution::kRequired;
      manifest.active_payloads_.emplace(entry.id, i);
      continue;
    }
    if (entry.resolution == PayloadResolution::kRequired) {
      manifest.active_payloads_.emplace(entry.id, i);
      continue;
    }
    switch (entry.resolution) {
    case PayloadResolution::kDead:
      AddAvoidedBytes(manifest.avoided_.dead_bytes, manifest.avoided_, entry.length);
      break;
    case PayloadResolution::kSuperseded:
      AddAvoidedBytes(manifest.avoided_.superseded_bytes, manifest.avoided_, entry.length);
      break;
    case PayloadResolution::kPreparedCacheReplaced:
      AddAvoidedBytes(manifest.avoided_.cache_replaced_bytes, manifest.avoided_, entry.length);
      break;
    case PayloadResolution::kRequired:
    case PayloadResolution::kDormantFallback:
      break;
    }
  }
  return manifest;
}

bool RequiredPayloadManifest::ContainsActive(const std::string &id) const noexcept {
  return active_payloads_.count(id) != 0;
}

ResolvedModelFixture::ResolvedModelFixture(std::filesystem::path model_path,
                                           RequiredPayloadManifest payload_manifest)
    : model_path_(std::move(model_path)), payload_manifest_(std::move(payload_manifest)) {}

std::vector<uint8_t> ResolvedModelFixture::ReadPayload(const std::string &id) const {
  const auto found = payload_manifest_.active_payloads_.find(id);
  EXT_ENFORCE(found != payload_manifest_.active_payloads_.end(), "Payload '", id,
              "' is not present in the active payload manifest.");
  const PayloadManifestEntry &entry = payload_manifest_.entries_[found->second];
  EXT_ENFORCE(entry.length <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
              "Payload '", id, "' is too large for this process.");
  EXT_ENFORCE(entry.offset <= static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()),
              "Payload '", id, "' offset is too large for this process.");
  EXT_ENFORCE(entry.length <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()),
              "Payload '", id, "' read length is too large for this process.");

  std::ifstream stream(entry.location, std::ios::binary);
  EXT_ENFORCE(stream.is_open(), "Unable to open payload '", id, "' at ", entry.location.string(),
              ".");
  stream.seekg(static_cast<std::streamoff>(entry.offset));
  EXT_ENFORCE(stream.good(), "Unable to seek to payload '", id, "'.");

  std::vector<uint8_t> bytes(static_cast<size_t>(entry.length));
  stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  EXT_ENFORCE(static_cast<size_t>(stream.gcount()) == bytes.size(), "Payload '", id,
              "' is shorter than its manifest entry.");
  return bytes;
}

std::vector<uint8_t> ResolvedModelFixture::ReadPayload(const TaskDescriptor &task) const {
  EXT_ENFORCE(task.kind == TaskKind::kReadPayload, "Task ", task.id.value,
              " is not a payload-read task.");
  EXT_ENFORCE(!task.payload_id.empty(), "Payload-read task ", task.id.value,
              " does not name a frozen manifest entry.");
  return ReadPayload(task.payload_id);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
