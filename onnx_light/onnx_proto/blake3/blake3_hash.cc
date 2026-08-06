// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "blake3_hash.h"

namespace ONNX_LIGHT_NAMESPACE::utils {

Blake3Hasher::Blake3Hasher() { blake3_hasher_init(&state_); }

void Blake3Hasher::Update(const void *data, std::size_t size) {
  // blake3_hasher_update_tbb hashes large subtrees in parallel via the
  // std::async-based join in blake3_join.cc; it falls back to the serial
  // update for small inputs.
  blake3_hasher_update_tbb(&state_, data, size);
}

int64_t Blake3Hasher::Finalize64() const {
  uint8_t digest[BLAKE3_OUT_LEN];
  blake3_hasher_finalize(&state_, digest, sizeof(digest));
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(digest[i]) << (8 * i);
  }
  return static_cast<int64_t>(value);
}

} // namespace ONNX_LIGHT_NAMESPACE::utils
