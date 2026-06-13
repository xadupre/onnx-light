// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
//
// libFuzzer harness for the ORT flatbuffer parser path of
// ``ModelProto::ParseFromString``.
//
// Feeds random bytes to ``ModelProto::ParseFromString`` with
// ``ParseOptions::format`` set to ``SerializeFormat::kOrtFlatbuffers``,
// exercising the onnxruntime flatbuffer (``.ort``) reader. The reader is
// not implemented yet and currently raises ``RuntimeError`` on every
// input; catching it keeps the harness green. Once the flatbuffer reader
// lands, this target will already feed it fuzzed buffers so OSS-Fuzz can
// surface crashes, hangs, or sanitizer reports in the new parser without
// any additional wiring.

#include "onnx_proto/onnx.h"
#include "onnx_proto/stream_class.h"

#include <cstddef>
#include <cstdint>
#include <string>

using ONNX_LIGHT_NAMESPACE::ModelProto;
using ONNX_LIGHT_NAMESPACE::ParseOptions;
using ONNX_LIGHT_NAMESPACE::SerializeFormat;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  const std::string buffer(reinterpret_cast<const char *>(data), size);

  // Exercise both the sequential and parallel reader code paths.
  for (int num_threads : {1, 2}) {
    ParseOptions options;
    options.format = SerializeFormat::kOrtFlatbuffers;
    options.num_threads = num_threads;
    ModelProto model;
    try {
      model.ParseFromString(buffer, options);
    } catch (...) {
      // Malformed flatbuffers and the current "not implemented yet"
      // RuntimeError are expected. Real bugs surface as crashes, hangs,
      // or sanitizer reports.
    }
  }
  return 0;
}
