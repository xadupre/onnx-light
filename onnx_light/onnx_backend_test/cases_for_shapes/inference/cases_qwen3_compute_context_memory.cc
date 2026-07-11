// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_optim/annotations/inplace_reuse.h"
#include "onnx_optim/annotations/value_tags.h"
#include "onnx_optim/shapes/shapes_context.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

std::string ReadQwen3ModelBytes() {
  const std::filesystem::path source_file(__FILE__);
  const std::filesystem::path model_path =
      source_file.parent_path() / "data" / "bench_qwen3_compute_context_memory.onnx";
  std::ifstream stream(model_path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Unable to open model file: " + model_path.string());
  }
  return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

} // namespace

void RegisterQwen3ComputeContextMemoryShapeInferenceCases(std::vector<TestCase> &registry) {
  const std::string name = "test_cc_shape_inference_qwen3_compute_context_memory";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-5;

  if (!tc.model.ParseFromString(ReadQwen3ModelBytes())) {
    throw std::runtime_error("Unable to parse backend model for case: " + name);
  }
  if (!tc.model.has_graph()) {
    throw std::runtime_error("Backend model has no graph for case: " + name);
  }
  tc.model.mutable_graph()->set_name(name);

  onnx_optim::shapes::ShapesContext ctx;
  ctx.ComputeShapeModel(tc.model);
  ctx.ApplyInferredShapesToModel(tc.model);

  onnx_optim::annotations::WriteValueAndNodeTagsToMetadata(tc.model);
  const auto value_and_node_tags =
      onnx_optim::annotations::InferValueAndNodeTags(tc.model.ref_graph());
  onnx_optim::annotations::WriteInPlaceReuseToMetadata(*tc.model.mutable_graph(), ctx,
                                                       value_and_node_tags.first);

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
