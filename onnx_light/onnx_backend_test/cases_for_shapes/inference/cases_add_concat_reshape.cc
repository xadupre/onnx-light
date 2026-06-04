// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
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
// initializer is ``[0, 0, -1]`` exactly as on the page. The per-tensor
// shapes registered in the graph (inputs, ``value_info`` and output) use
// the symbolic dim names from the page (``batch``/``seq``/``d_model``):
//
//   X            (batch, seq, d_model)
//   Y            (batch, seq, d_model)
//   added        (batch, seq, d_model)
//   concat_out   (batch, seq, two_d_model)  // last dim is ``2 * d_model``
//   Z            (batch, seq, two_d_model)  // recovered by Reshape([0, 0, -1])
//
// The reference DataSet still uses concrete sizes (``batch=2, seq=5,
// d_model=8``) for the actual tensor values, so the case is executable.
// ---------------------------------------------------------------------------
void RegisterAddConcatReshapeShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};

  Tensor reshape_shape = Tensor::FromInt64("reshape_shape", {3}, {0, 0, -1});

  const std::string name = "test_cc_shape_inference_add_concat_reshape";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  AddNode(*graph, "Add", {"X", "Y"}, {"added"});
  NodeProto &concat_node = AddNode(*graph, "Concat", {"added", "X"}, {"concat_out"});
  AddAxisAttribute(concat_node, 2);
  AddNode(*graph, "Reshape", {"concat_out", "reshape_shape"}, {"Z"});

  // Graph inputs: X, Y use the symbolic shape from the page; reshape_shape
  // is a concrete-shape initializer carried by ``FillValueInfo``.
  const int32_t kFloat = static_cast<int32_t>(DataType::FLOAT);
  const std::vector<DimSpec> input_shape = {"batch", "seq", "d_model"};
  AppendValueInfo(*graph->add_input(), "X", kFloat, input_shape);
  AppendValueInfo(*graph->add_input(), "Y", kFloat, input_shape);
  FillValueInfo(reshape_shape, *graph->add_input());

  // Intermediate value_info entries with symbolic dim names. The last dim
  // of ``concat_out``/``Z`` is ``2 * d_model`` on the page; ONNX shape
  // inference cannot represent that symbolic product, so we give it a
  // dedicated symbolic name (``two_d_model``) instead of leaving the dim
  // unannotated. The reference DataSet uses ``2 * d_model = 16``.
  const std::vector<DimSpec> concat_shape = {"batch", "seq", "two_d_model"};
  AppendValueInfo(*graph->add_value_info(), "added", kFloat, input_shape);
  AppendValueInfo(*graph->add_value_info(), "concat_out", kFloat, concat_shape);

  // Graph output Z — same symbolic dims as concat_out.
  AppendValueInfo(*graph->add_output(), "Z", kFloat, concat_shape);

  // Build the reference DataSet right next to its consumers: concrete
  // ``batch=2, seq=5, d_model=8`` tensors, then run the kernels to
  // materialise Z.
  const std::vector<int64_t> data_shape = {2, 5, 8}; // batch=2, seq=5, d_model=8
  const int64_t input_size = data_shape[0] * data_shape[1] * data_shape[2];
  std::vector<float> x_values(static_cast<size_t>(input_size));
  std::vector<float> y_values(static_cast<size_t>(input_size));
  for (size_t i = 0; i < x_values.size(); ++i) {
    x_values[i] = static_cast<float>(i) * 0.1f;
    y_values[i] = static_cast<float>(i) * 0.01f + 1.0f;
  }
  Tensor x = Tensor::FromFloat("X", data_shape, x_values);
  Tensor y = Tensor::FromFloat("Y", data_shape, y_values);
  Tensor z = kernel::Reshape(ctx)(kernel::Concat(ctx)({kernel::Add(ctx)(x, y), x}, /*axis=*/2),
                                  reshape_shape);
  z.name = "Z";

  AppendDataSet(tc, {x, y, reshape_shape}, {z});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
