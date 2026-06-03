// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "onnx_light_helpers.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace nnef {

/** NNEF binary tensor magic bytes (``0x4E 0xEF``). */
constexpr uint8_t kNnefMagic0 = 0x4E;
constexpr uint8_t kNnefMagic1 = 0xEF;
/** NNEF binary tensor format major version implemented here. */
constexpr int kNnefVersionMajor = 1;
/** NNEF binary tensor format minor version implemented here. */
constexpr int kNnefVersionMinor = 0;
/** Size of the fixed NNEF binary tensor header. */
constexpr int kNnefHeaderSize = 128;
/** Maximum rank that fits in the fixed-size ``extents`` field. */
constexpr int kNnefMaxRank = 8;

/** NNEF item type codes. */
constexpr int kItemTypeFloat = 0;
constexpr int kItemTypeQuant = 1;
constexpr int kItemTypeSigned = 2;
constexpr int kItemTypeUnsigned = 3;
constexpr int kItemTypeBool = 4;

/**
 * In-memory representation of a NNEF binary tensor: shape, item type
 * code (0=float, 2=signed, 3=unsigned, 4=bool), bits per item and
 * little-endian raw payload.
 */
struct NNEFTensor {
  std::vector<int64_t> shape;
  int item_type = kItemTypeFloat;
  int bits = 32;
  std::vector<uint8_t> data;
};

/** Writes a NNEF tensor to ``path`` (128-byte header + payload). */
void WriteNNEFTensor(const std::string &path, const NNEFTensor &tensor);

/** Reads a NNEF tensor from ``path``. */
NNEFTensor ReadNNEFTensor(const std::string &path);

} // namespace nnef
} // namespace ONNX_LIGHT_NAMESPACE
