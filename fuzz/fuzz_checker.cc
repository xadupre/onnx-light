// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
//
// libFuzzer harness for ``onnx_light::checker::check_model``.
//
// Feeds random bytes to ``ModelProto::ParseFromString`` and, on a
// successful parse, runs the structural checker on the resulting model.
// Mirrors the behaviour of the former ``onnx_light/fuzz/fuzz_checker.py``
// atheris target, but uses libFuzzer's ``LLVMFuzzerTestOneInput`` ABI
// directly so the harness can run under OSS-Fuzz's standard C++
// pipeline (libFuzzer + ASan/UBSan/MSan).

#include "onnx_lib/checker.h"
#include "onnx_proto/onnx.h"

#include <cstddef>
#include <cstdint>
#include <string>

using ONNX_LIGHT_NAMESPACE::ModelProto;
namespace checker = ONNX_LIGHT_NAMESPACE::checker;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  ModelProto model;
  try {
    model.ParseFromString(std::string(reinterpret_cast<const char *>(data), size));
  } catch (...) {
    // Malformed protobuf input: expected for random bytes.
    return 0;
  }
  try {
    checker::check_model(model);
  } catch (...) {
    // ValidationError and friends are expected on random / malformed
    // inputs. Real bugs surface as crashes, hangs, or sanitizer reports.
  }
  return 0;
}
