// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "nnef/tensor_io.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace nnef {

namespace {

inline void WriteU32LE(uint8_t *p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v & 0xff);
  p[1] = static_cast<uint8_t>((v >> 8) & 0xff);
  p[2] = static_cast<uint8_t>((v >> 16) & 0xff);
  p[3] = static_cast<uint8_t>((v >> 24) & 0xff);
}

inline void WriteU16LE(uint8_t *p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v & 0xff);
  p[1] = static_cast<uint8_t>((v >> 8) & 0xff);
}

inline uint32_t ReadU32LE(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline uint16_t ReadU16LE(const uint8_t *p) {
  return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8));
}

std::vector<uint8_t> PackHeader(const std::vector<int64_t> &shape, int item_type, int bits,
                                uint32_t data_length) {
  const int rank = static_cast<int>(shape.size());
  if (rank > kNnefMaxRank) {
    throw std::invalid_argument("NNEF binary tensor format supports rank <= " +
                                std::to_string(kNnefMaxRank) + ", got " + std::to_string(rank));
  }
  std::vector<uint8_t> header(kNnefHeaderSize, 0);
  header[0] = kNnefMagic0;
  header[1] = kNnefMagic1;
  header[2] = static_cast<uint8_t>(kNnefVersionMajor);
  header[3] = static_cast<uint8_t>(kNnefVersionMinor);
  WriteU32LE(header.data() + 4, data_length);
  WriteU32LE(header.data() + 8, static_cast<uint32_t>(rank));
  for (int i = 0; i < kNnefMaxRank; ++i) {
    uint32_t v = i < rank ? static_cast<uint32_t>(shape[i]) : 0u;
    WriteU32LE(header.data() + 12 + 4 * i, v);
  }
  WriteU32LE(header.data() + 44, static_cast<uint32_t>(bits));
  WriteU16LE(header.data() + 48, static_cast<uint16_t>(item_type));
  WriteU16LE(header.data() + 50, 0); // item_type_data_length
  return header;
}

} // namespace

void WriteNNEFTensor(const std::string &path, const NNEFTensor &tensor) {
  const uint32_t data_length = static_cast<uint32_t>(tensor.data.size());
  auto header = PackHeader(tensor.shape, tensor.item_type, tensor.bits, data_length);
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("WriteNNEFTensor: cannot open '" + path + "' for writing");
  }
  out.write(reinterpret_cast<const char *>(header.data()),
            static_cast<std::streamsize>(header.size()));
  if (!tensor.data.empty()) {
    out.write(reinterpret_cast<const char *>(tensor.data.data()),
              static_cast<std::streamsize>(tensor.data.size()));
  }
}

NNEFTensor ReadNNEFTensor(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("ReadNNEFTensor: cannot open '" + path + "'");
  }
  std::vector<uint8_t> header(kNnefHeaderSize);
  in.read(reinterpret_cast<char *>(header.data()), kNnefHeaderSize);
  if (in.gcount() != kNnefHeaderSize || header[0] != kNnefMagic0 || header[1] != kNnefMagic1) {
    throw std::runtime_error("Not a NNEF binary tensor file (bad magic): " + path);
  }
  uint32_t data_length = ReadU32LE(header.data() + 4);
  uint32_t rank = ReadU32LE(header.data() + 8);
  if (rank > static_cast<uint32_t>(kNnefMaxRank)) {
    throw std::runtime_error("NNEF tensor has invalid rank");
  }
  NNEFTensor t;
  t.shape.resize(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    t.shape[i] = static_cast<int64_t>(ReadU32LE(header.data() + 12 + 4 * i));
  }
  t.bits = static_cast<int>(ReadU32LE(header.data() + 44));
  t.item_type = static_cast<int>(ReadU16LE(header.data() + 48));
  t.data.resize(data_length);
  if (data_length > 0) {
    in.read(reinterpret_cast<char *>(t.data.data()), static_cast<std::streamsize>(data_length));
    if (static_cast<uint32_t>(in.gcount()) != data_length) {
      throw std::runtime_error("Truncated NNEF tensor: " + path);
    }
  }
  return t;
}

} // namespace nnef
} // namespace ONNX_LIGHT_NAMESPACE
