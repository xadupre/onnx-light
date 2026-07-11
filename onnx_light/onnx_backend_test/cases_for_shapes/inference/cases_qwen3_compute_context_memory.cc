// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_optim/annotations/inplace_reuse.h"
#include "onnx_optim/annotations/value_tags.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

constexpr int64_t kQwen3ModelIrVersion = 8;

} // namespace

// Registers a Qwen3-inspired compute-context-memory model as a backend.
// shape-inference case, then pre-embeds expected shape, value-tag,
// release-after and in-place-reuse metadata computed by onnx-light.
void RegisterQwen3ComputeContextMemoryShapeInferenceCase(std::vector<TestCase> &registry) {
  const std::string name = "test_cc_shape_inference_qwen3_compute_context_memory";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-5;

  ModelProto &model = tc.model;
  InitModel(model, kQwen3ModelIrVersion, {DefaultOpset(18)});
  GraphProto *graph = model.mutable_graph();
  graph->set_name(name);

  // The small chain mirrors the compute-context-memory pattern used in the
  // benchmark: projection, concat growth, shape extraction, elementwise op and
  // shape-driven reshape.
  AddNode(*graph, "MatMul", {"X", "W"}, {"M"});
  NodeProto &concat_with_axis_node = AddNode(*graph, "Concat", {"M", "X"}, {"C"});
  AddAttribute<int64_t>(concat_with_axis_node, "axis", 0);
  AddNode(*graph, "Shape", {"C"}, {"S"});
  AddNode(*graph, "Abs", {"C"}, {"A"});
  AddNode(*graph, "Reshape", {"A", "S"}, {"Z"});

  AddInitializer<float>(*graph, "W", {4, 4},
                        {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 1.0f});

  const auto float_type = static_cast<int32_t>(DataType::FLOAT);
  AppendValueInfo(*graph->add_input(), "X", float_type, {"N", 4});
  AppendValueInfo(*graph->add_output(), "Z", float_type, {"2N", 4});

  onnx_optim::shapes::ShapesContext ctx;
  ctx.ComputeShapeModel(model);
  ctx.ApplyInferredShapesToModel(model);

  onnx_optim::annotations::WriteValueAndNodeTagsToMetadata(model);
  const auto value_and_node_tags =
      onnx_optim::annotations::InferValueAndNodeTags(model.ref_graph());
  onnx_optim::annotations::WriteInPlaceReuseToMetadata(*model.mutable_graph(), ctx,
                                                       value_and_node_tags.first);

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
