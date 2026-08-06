// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_light_helpers.h"

#include <cstddef>
#include <cstdint>

#include "blake3.h"

namespace ONNX_LIGHT_NAMESPACE::utils {

/**
 * Incremental BLAKE3 hasher used to hash tensor payloads.
 *
 * Wraps the vendored BLAKE3 C library. Update() hashes large buffers in
 * parallel using the BLAKE3 tree structure (see ``blake3/blake3_join.cc``); the
 * digest is identical regardless of how many threads participate.
 */
class Blake3Hasher {
public:
  /** Initializes an empty hasher. */
  Blake3Hasher();

  /** Absorbs ``size`` bytes at ``data`` into the running hash. */
  void Update(const void *data, std::size_t size);

  /**
   * Reduces the 256-bit BLAKE3 digest to a 64-bit value.
   *
   * Returns:
   *   The first eight digest bytes interpreted as a little-endian integer.
   */
  int64_t Finalize64() const;

private:
  blake3_hasher state_;
};

} // namespace ONNX_LIGHT_NAMESPACE::utils
