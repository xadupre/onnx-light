// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/cases_for_shape_inference/include_shape_inference_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built models below.
constexpr int64_t kDefaultIrVersion = 10;

} // namespace

// ---------------------------------------------------------------------------
// ``Add → Concat(axis=2) → Reshape(shape=[0, 0, -1])`` — mirrors the
// "Add + Concat + Reshape" model from the ``plot_computed_shapes`` gallery
// page (https://xadupre.github.io/docs/yet-another-onnx-builder/
// auto_examples_core/plot_computed_shapes.html). The ``reshape_shape``
// initializer is ``[0, 0, -1]`` exactly as on the page. The concrete
// per-tensor shapes registered in the graph (inputs, ``value_info`` and
// output) are the literal shapes printed on the page for
// ``context = dict(batch=2, seq=5, d_model=8)``:
//
//   X            (2, 5, 8)
//   Y            (2, 5, 8)
//   added        (2, 5, 8)
//   concat_out   (2, 5, 16)
//   Z            (2, 5, 16)
//
// The generic shape-inference tests substitute symbolic ``dim_params`` on
// top to also exercise the symbolic propagation path.
// ---------------------------------------------------------------------------
void RegisterAddConcatReshapeShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};

  // Concrete shape from the gallery page for the inputs / ``added``
  // (``batch=2, seq=5, d_model=8``). ``concat_shape`` is declared further
  // down, next to where it is used.
  const std::vector<int64_t> input_shape = {2, 5, 8}; // X, Y, added

  Tensor reshape_shape = Tensor::FromInt64("reshape_shape", {3}, {0, 0, -1});

  const std::string name = "test_cc_shape_inference_add_concat_reshape";

  TestCase tc;
  tc.name = name;
  tc.model_name = name;
  tc.kind = "model";
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  AddNode(*graph, "Add", {"X", "Y"}, {"added"});
  NodeProto &concat_node = AddNode(*graph, "Concat", {"added", "X"}, {"concat_out"});
  AddAxisAttribute(concat_node, /*axis=*/2);
  AddNode(*graph, "Reshape", {"concat_out", "reshape_shape"}, {"Z"});

  // Helper to declare a tensor-typed ValueInfo with a literal float shape.
  const auto add_float_value_info = [](ValueInfoProto &vi, const std::string &vi_name,
                                       const std::vector<int64_t> &shape) {
    Tensor t;
    t.name = vi_name;
    t.data_type = static_cast<int32_t>(DataType::FLOAT);
    t.shape = shape;
    FillValueInfo(t, vi);
  };

  // Graph inputs: X, Y and the shape tensor — shapes from the page.
  add_float_value_info(*graph->add_input(), "X", input_shape);
  add_float_value_info(*graph->add_input(), "Y", input_shape);
  FillValueInfo(reshape_shape, *graph->add_input());

  // Intermediate value_info entries with the literal shapes from the page.
  // ``concat_out`` / ``Z`` use ``[2, 5, 16]`` (last dim doubled by Concat).
  const std::vector<int64_t> concat_shape = {2, 5, 16}; // concat_out, Z
  add_float_value_info(*graph->add_value_info(), "added", input_shape);
  add_float_value_info(*graph->add_value_info(), "concat_out", concat_shape);

  // Graph output Z — literal shape from the page.
  add_float_value_info(*graph->add_output(), "Z", concat_shape);

  // Build the reference DataSet right next to its consumers: simple,
  // fully-populated input tensors, then run the kernels to materialise Z.
  const int64_t input_size = input_shape[0] * input_shape[1] * input_shape[2];
  std::vector<float> x_values(static_cast<size_t>(input_size));
  std::vector<float> y_values(static_cast<size_t>(input_size));
  for (size_t i = 0; i < x_values.size(); ++i) {
    x_values[i] = static_cast<float>(i) * 0.1f;
    y_values[i] = static_cast<float>(i) * 0.01f + 1.0f;
  }
  Tensor x = Tensor::FromFloat("X", input_shape, x_values);
  Tensor y = Tensor::FromFloat("Y", input_shape, y_values);
  Tensor z = kernel::Reshape(ctx)(kernel::Concat(ctx)({kernel::Add(ctx)(x, y), x}, /*axis=*/2),
                                  reshape_shape);
  z.name = "Z";

  DataSet ds;
  ds.inputs.push_back(x);
  ds.inputs.push_back(y);
  ds.inputs.push_back(reshape_shape);
  ds.outputs.push_back(z);
  tc.data_sets.emplace_back(std::move(ds));

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
