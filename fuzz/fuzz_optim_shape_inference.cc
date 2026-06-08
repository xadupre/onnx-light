// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
//
// libFuzzer harness for ``onnx_light::onnx_optim::shapes::InferShapesModel``.
//
// Exercises the onnx-light optim shape-inference pipeline (distinct
// from ``onnx_light::shape_inference::InferShapes`` which mirrors
// upstream ONNX). Mirrors the former
// ``onnx_light/fuzz/fuzz_optim_shape_inference.py``.

#include "onnx_optim/shapes/shape_inference.h"
#include "onnx_proto/onnx.h"

#include <cstddef>
#include <cstdint>
#include <string>

using ONNX_LIGHT_NAMESPACE::ModelProto;
namespace optim_shapes = ONNX_LIGHT_NAMESPACE::onnx_optim::shapes;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  ModelProto model;
  try {
    model.ParseFromString(std::string(reinterpret_cast<const char *>(data), size));
  } catch (...) {
    return 0;
  }
  try {
    optim_shapes::InferShapesModel(model);
  } catch (...) {
    // Expected on malformed inputs.
  }
  return 0;
}
