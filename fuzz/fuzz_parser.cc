// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
//
// libFuzzer harness for ``OnnxParser::Parse<ModelProto>``.
//
// Feeds NUL-terminated text to the ONNX textual-format parser. Mirrors
// the former ``onnx_light/fuzz/fuzz_parser.py``.

#include "onnx_lib/defs/parser.h"
#include "onnx_proto/onnx.h"

#include <cstddef>
#include <cstdint>
#include <string>

using ONNX_LIGHT_NAMESPACE::ModelProto;
using ONNX_LIGHT_NAMESPACE::OnnxParser;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // OnnxParser reads a C string, so guard against embedded NULs by
  // constructing a std::string (which adds the trailing NUL).
  std::string text(reinterpret_cast<const char *>(data), size);
  ModelProto model;
  try {
    (void)OnnxParser::Parse(model, text.c_str());
  } catch (...) {
    // Parse errors are expected for random / malformed inputs.
  }
  return 0;
}
