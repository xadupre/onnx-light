// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/shape_tag/include_shape_tag_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_optim/annotations/value_tags.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

constexpr int64_t kDefaultIrVersion = 10;

} // namespace

// ---------------------------------------------------------------------------
// ``Shape → Reshape`` — the simplest pattern that exercises the shape-tag
// annotation path. ``Shape(X)`` produces an INT64 tensor whose *value*
// represents the shape of its input, so ``WriteValueAndNodeTagsToMetadata``
// must emit:
//
//   * ``onnx_light.node_tag = "shape"`` on the ``Shape`` node.
//   * ``onnx_light.node_tag = "weight"`` on the ``Reshape`` node (inherited
//     from its first input X which is a rank-2 float tensor).
//   * ``onnx_light.value_tags = {"S":"shape","X":"weight","Y":"weight"}`` on
//     the graph (X is a rank-2 float tensor → "weight"; Y inherits from X).
//
// The expected metadata is pre-embedded into the model so consumers can
// verify that ``WriteValueAndNodeTagsToMetadata`` reproduces it exactly.
// ---------------------------------------------------------------------------
void RegisterShapeTagCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};

  const std::string name = "test_cc_shape_tag_shape_reshape";

  TestCase tc(name, name, "model", "shape_tag");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  AddNode(*graph, "Shape", {"X"}, {"S"});
  AddNode(*graph, "Reshape", {"X", "S"}, {"Y"});

  // Concrete input shape [2, 3] so the model is executable end-to-end.
  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {DimSpec(2), DimSpec(3)});
  AppendValueInfo(*graph->add_value_info(), "S", DataType::INT64, {DimSpec(2)});
  AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {DimSpec(2), DimSpec(3)});

  // Pre-embed the expected shape-tag metadata so tests can verify that
  // WriteValueAndNodeTagsToMetadata produces identical results.
  // X is a rank-2 float tensor so CollectGraphSeedTags tags it as "weight";
  // Y inherits "weight" from X through Reshape; S is tagged "shape" by Shape.
  // DumpValueTagsAsJson sorts keys, so the canonical order is S < X < Y.
  graph->add_metadata(onnx_optim::annotations::kValueTagsMetadataKey,
                      "{\"S\":\"shape\",\"X\":\"weight\",\"Y\":\"weight\"}");
  (*graph->mutable_node())[0].add_metadata(onnx_optim::annotations::kNodeTagMetadataKey, "shape");
  // Reshape (node[1]) inherits "weight" from its first input X.
  (*graph->mutable_node())[1].add_metadata(onnx_optim::annotations::kNodeTagMetadataKey, "weight");
  // S (value_info[0]) also receives onnx_light.value_tag = "shape".
  {
    StringStringEntryProto *entry = graph->mutable_value_info(0)->add_metadata_props();
    entry->set_key(onnx_optim::annotations::kValueTagMetadataKey);
    entry->set_value("shape");
  }
  // X (input[0]) receives onnx_light.value_tag = "weight".
  {
    StringStringEntryProto *entry = graph->mutable_input(0)->add_metadata_props();
    entry->set_key(onnx_optim::annotations::kValueTagMetadataKey);
    entry->set_value("weight");
  }
  // Y (output[0]) receives onnx_light.value_tag = "weight".
  {
    StringStringEntryProto *entry = graph->mutable_output(0)->add_metadata_props();
    entry->set_key(onnx_optim::annotations::kValueTagMetadataKey);
    entry->set_value("weight");
  }

  // Build the reference DataSet so the case is executable end-to-end.
  const Tensor x = Tensor::FromFloat("X", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor s = kernel::Shape(ctx)(x, kernel::Shape::Attributes{});
  s.name = "S";
  Tensor y = kernel::Reshape(ctx)(x, s);
  y.name = "Y";

  AppendDataSet(tc, {x}, {y});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
