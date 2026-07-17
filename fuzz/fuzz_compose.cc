// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
//
// libFuzzer harness for ``onnx_light`` model composition helpers.
//
// Exercises the C++ port of ``onnx.compose`` (``MergeModels``,
// ``MergeGraphs``, ``AddPrefix``, ``CheckOverlappingNames`` and
// ``ExpandOutDim``). The single fuzzer buffer is split into two
// serialized ``ModelProto`` blobs using a little-endian uint32 length
// prefix so a single input drives both operands of the binary merge
// helpers.

#include "onnx_manipulations/compose.h"
#include "onnx_proto/onnx.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using ONNX_LIGHT_NAMESPACE::GraphProto;
using ONNX_LIGHT_NAMESPACE::ModelProto;
namespace compose = ONNX_LIGHT_NAMESPACE;

namespace {

// Splits the fuzzer buffer into two byte ranges. The first four bytes
// encode (little-endian) how many of the remaining bytes belong to the
// first model; the rest belong to the second.
std::pair<std::string, std::string> split_input(const uint8_t *data, size_t size) {
  if (size < 4) {
    return {std::string(), std::string()};
  }
  uint32_t first_len = static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
                       (static_cast<uint32_t>(data[2]) << 16) |
                       (static_cast<uint32_t>(data[3]) << 24);
  const size_t body = size - 4;
  if (first_len > body) {
    first_len = static_cast<uint32_t>(body);
  }
  const char *body_ptr = reinterpret_cast<const char *>(data + 4);
  std::string first(body_ptr, first_len);
  std::string second(body_ptr + first_len, body - first_len);
  return {std::move(first), std::move(second)};
}

// Builds an io_map by pairing outputs of g1 with inputs of g2 in
// order, giving the merge helpers a realistic set of connections to
// resolve. Returns an empty map when either side has no candidates.
std::vector<std::pair<std::string, std::string>> build_io_map(const GraphProto &g1,
                                                              const GraphProto &g2) {
  std::vector<std::pair<std::string, std::string>> io_map;
  const auto &outputs = g1.ref_output();
  const auto &inputs = g2.ref_input();
  const size_t count = outputs.size() < inputs.size() ? outputs.size() : inputs.size();
  io_map.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    io_map.emplace_back(std::string(outputs[i].name()), std::string(inputs[i].name()));
  }
  return io_map;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  auto [first_bytes, second_bytes] = split_input(data, size);

  ModelProto m1;
  ModelProto m2;
  try {
    m1.ParseFromString(first_bytes);
    m2.ParseFromString(second_bytes);
  } catch (...) {
    return 0;
  }

  // Single-model helpers: prefixing and output-dimension expansion.
  try {
    compose::AddPrefix(m1, "p1/");
  } catch (...) {
    // Expected on malformed inputs.
  }
  try {
    compose::ExpandOutDim(m1, 0);
  } catch (...) {
    // Expected on malformed inputs.
  }

  if (!m1.has_graph() || !m2.has_graph()) {
    return 0;
  }
  const GraphProto &g1 = m1.ref_graph();
  const GraphProto &g2 = m2.ref_graph();

  try {
    compose::CheckOverlappingNames(g1, g2);
  } catch (...) {
    // Expected on malformed inputs.
  }

  const auto io_map = build_io_map(g1, g2);

  try {
    compose::MergeGraphs(g1, g2, io_map);
  } catch (...) {
    // Mismatched names / shapes are expected on random inputs.
  }
  try {
    compose::MergeModels(m1, m2, io_map);
  } catch (...) {
    // IR / opset mismatches are expected on random inputs.
  }
  return 0;
}
