// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
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
// ``Shape → Identity → Unsqueeze`` — exercises shape-data propagation through
// ``Shape`` (which produces an INT64 1-D tensor whose values are the input's
// shape), ``Identity`` (which must forward both the type and the propagated
// shape data) and finally ``Unsqueeze``, whose ``axes`` input is provided via
// a graph **initializer** (INT64). Shape inference should be able to recover
// the fully-known output shape ``[1, 1, ..., 1, rank]`` even when the only
// statically known piece of data is the ``axes`` initializer.
//
// This mirrors the upstream onnxruntime regression test added in
// https://github.com/microsoft/onnxruntime/pull/28778 which exposed a bug in
// the in-memory-external INT64 initializer path inside
// ``Graph::SaveShapeValuesFromDataPropagation``: the destination
// ``input_values`` vector was not resized before the ``memcpy``, leading to a
// buffer overrun. The model topology here is deliberately the same so that
// any equivalent regression in onnx-light's data propagation would be caught
// by the generic ``BackendTestCaseShapeInference`` tests.
// ---------------------------------------------------------------------------
void RegisterShapeIdentityUnsqueezeShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};

  // Use rank 15 (one less than the upstream regression model, which used
  // ``axis_count = 16``). The upstream test exercised a memcpy buffer
  // overrun in ``Graph::SaveShapeValuesFromDataPropagation`` that triggers
  // for any rank > 0; the exact rank is not load-bearing for the
  // regression. We cap at 15 here so the ``Unsqueeze`` output (rank
  // ``kAxisCount + 1 = 16``) still fits inside ``onnx_optim``'s
  // :cpp:var:`onnx_optim::kMaxOptimRank` (``= 16``) bound. With
  // ``axis_count = 16`` the output rank would be 17, which would cause
  // :cpp:func:`onnx_optim::shapes::InferShapesModel` to throw
  // ``std::length_error("OptimShape exceeds maximum rank")`` and fail
  // optim shape inference on this case.
  constexpr int64_t kAxisCount = 15;
  const std::vector<int64_t> input_shape(static_cast<size_t>(kAxisCount), 1);

  std::vector<int64_t> axes_values(static_cast<size_t>(kAxisCount));
  for (int64_t i = 0; i < kAxisCount; ++i) {
    axes_values[static_cast<size_t>(i)] = i;
  }
  const Tensor axes_tensor = Tensor::FromInt64("unsq_axes", {kAxisCount}, axes_values);

  const std::string name = "test_cc_shape_inference_shape_identity_unsqueeze";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  AddNode(*graph, "Shape", {"input"}, {"shape_out"});
  AddNode(*graph, "Identity", {"shape_out"}, {"identity_out"});
  AddNode(*graph, "Unsqueeze", {"identity_out", "unsq_axes"}, {"output"});

  // Register ``unsq_axes`` as an actual initializer (not just a graph input)
  // so shape inference's data-propagation pass sees the concrete INT64 axes
  // values; this is the path the upstream regression model exercises.
  TensorProto *init = graph->add_initializer();
  init->set_name("unsq_axes");
  init->set_data_type(static_cast<DataType>(axes_tensor.data_type));
  for (int64_t d : axes_tensor.shape) {
    init->add_dims(static_cast<uint64_t>(d));
  }
  init->set_raw_data(utils::ByteSpan(axes_tensor.data));

  // Graph input: ``input`` with all-ones static shape (rank ``kAxisCount``).
  const int32_t kFloat = static_cast<int32_t>(DataType::FLOAT);
  const int32_t kInt64 = static_cast<int32_t>(DataType::INT64);
  std::vector<DimSpec> input_dims;
  input_dims.reserve(static_cast<size_t>(kAxisCount));
  for (int64_t i = 0; i < kAxisCount; ++i) {
    input_dims.emplace_back(static_cast<int64_t>(1));
  }
  AppendValueInfo(*graph->add_input(), "input", kFloat, input_dims);

  // Expected intermediate value_info entries: ``shape_out`` and
  // ``identity_out`` are both 1-D INT64 tensors of length ``kAxisCount``.
  AppendValueInfo(*graph->add_value_info(), "shape_out", kInt64,
                  {DimSpec(static_cast<int64_t>(kAxisCount))});
  AppendValueInfo(*graph->add_value_info(), "identity_out", kInt64,
                  {DimSpec(static_cast<int64_t>(kAxisCount))});

  // Expected output: rank ``kAxisCount + 1`` (16 leading 1s + trailing
  // ``kAxisCount``). Shape inference must recover this fully from the
  // ``axes`` initializer and the (data-propagated) shape of ``input``.
  std::vector<DimSpec> output_dims;
  output_dims.reserve(static_cast<size_t>(kAxisCount) + 1);
  for (int64_t i = 0; i < kAxisCount; ++i) {
    output_dims.emplace_back(static_cast<int64_t>(1));
  }
  output_dims.emplace_back(static_cast<int64_t>(kAxisCount));
  AppendValueInfo(*graph->add_output(), "output", kInt64, output_dims);

  // Build the reference DataSet so the case is executable end-to-end.
  std::vector<float> input_values(static_cast<size_t>(1));
  input_values[0] = 0.0f;
  Tensor input_tensor = Tensor::FromFloat("input", input_shape, input_values);

  // Shape(input) -> [1, 1, ..., 1] (kAxisCount INT64 entries)
  Tensor shape_out = kernel::Shape(ctx)(input_tensor, kernel::Shape::Attributes{});
  // Identity is a no-op other than renaming the output.
  Tensor identity_out = shape_out;
  identity_out.name = "identity_out";
  // Unsqueeze along all axes => shape becomes ``[1] * kAxisCount + [kAxisCount]``.
  Tensor output_tensor = kernel::Unsqueeze(ctx)(identity_out, axes_values);
  output_tensor.name = "output";

  AppendDataSet(tc, {input_tensor}, {output_tensor});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
