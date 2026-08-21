// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "resolved_model_fixture.h"

#include <fstream>
#include <limits>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

ResolvedModelFixture::ResolvedModelFixture(std::filesystem::path model_path,
                                           std::vector<PayloadManifestEntry> payload_manifest)
    : model_path_(std::move(model_path)), payload_manifest_(std::move(payload_manifest)) {
  for (size_t i = 0; i < payload_manifest_.size(); ++i) {
    const PayloadManifestEntry &entry = payload_manifest_[i];
    EXT_ENFORCE(!entry.id.empty(), "Payload manifest IDs must not be empty.");
    if (entry.active) {
      EXT_ENFORCE(active_payloads_.emplace(entry.id, i).second,
                  "Duplicate active payload manifest ID '", entry.id, "'.");
    }
  }
}

std::vector<uint8_t> ResolvedModelFixture::ReadPayload(const std::string &id) const {
  auto found = active_payloads_.find(id);
  EXT_ENFORCE(found != active_payloads_.end(), "Payload '", id,
              "' is not present in the active payload manifest.");
  const PayloadManifestEntry &entry = payload_manifest_[found->second];
  EXT_ENFORCE(entry.length <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
              "Payload '", id, "' is too large for this process.");

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

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
