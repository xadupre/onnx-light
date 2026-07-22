// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
//
// libFuzzer harness for ``onnx_light::core::shapes::InferShapesModel``.
//
// Exercises the onnx-light optim shape-inference pipeline (distinct
// from ``onnx_light::shape_inference::InferShapes`` which mirrors
// upstream ONNX). Mirrors the former
// ``onnx_light/fuzz/fuzz_optim_shape_inference.py``.

#include "onnx_core/shapes/shape_inference.h"
#include "onnx_extensions/shapes/dispatch_table.h"
#include "onnx_proto/onnx.h"

#include <cstddef>
#include <cstdint>
#include <string>

using ONNX_LIGHT_NAMESPACE::ModelProto;
namespace core_shapes = ONNX_LIGHT_NAMESPACE::core::shapes;
namespace onnx_shapes = ONNX_LIGHT_NAMESPACE::onnx_shapes;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // `core::shapes::DispatchTable()` starts out empty; register the built-in
  // `onnx_shapes` shape functions once so `InferShapesModel` can resolve ops.
  onnx_shapes::RegisterShapeFunctions();
  ModelProto model;
  try {
    model.ParseFromString(std::string(reinterpret_cast<const char *>(data), size));
  } catch (...) {
    return 0;
  }
  try {
    core_shapes::InferShapesModel(model);
  } catch (...) {
    // Expected on malformed inputs.
  }
  return 0;
}
