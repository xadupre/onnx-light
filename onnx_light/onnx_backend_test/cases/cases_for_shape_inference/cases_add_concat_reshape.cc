// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/cases_for_shape_inference/include_shape_inference_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built models below.
constexpr int64_t kDefaultIrVersion = 10;

// Builds the 1-D INT64 ``shape`` tensor consumed by the ``Reshape`` node.
Tensor MakeShapeTensor(const std::string &name, const std::vector<int64_t> &dims) {
  const std::vector<int64_t> shape_shape = {static_cast<int64_t>(dims.size())};
  std::vector<uint8_t> data(dims.size() * sizeof(int64_t));
  if (!dims.empty()) {
    std::memcpy(data.data(), dims.data(), data.size());
  }
  return Tensor(name, static_cast<int32_t>(DataType::INT64), shape_shape, std::move(data));
}

} // namespace

// ---------------------------------------------------------------------------
// ``Add → Concat(axis=2) → Reshape(shape=[0, 0, -1])`` — mirrors the
// "Add + Concat + Reshape" model from the ``plot_computed_shapes`` gallery
// page. The Reshape uses ``[0, 0, -1]`` so the output retains the leading
// ``(batch, seq)`` dimensions and the last dimension is recovered as
// ``2 * d_model``. The model is registered with concrete dims so the
// reference kernels can produce expected outputs; the generic shape-inference
// tests substitute symbolic ``dim_params`` to also exercise the symbolic
// propagation path.
// ---------------------------------------------------------------------------
void RegisterAddConcatReshapeShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};

  // Concrete dimensions used to materialise reference outputs.
  constexpr int64_t kBatch = 2;
  constexpr int64_t kSeq = 5;
  constexpr int64_t kDModel = 8;
  const std::vector<int64_t> input_shape = {kBatch, kSeq, kDModel};

  // Simple, fully-populated input tensors.
  std::vector<float> x_values(kBatch * kSeq * kDModel);
  std::vector<float> y_values(kBatch * kSeq * kDModel);
  for (size_t i = 0; i < x_values.size(); ++i) {
    x_values[i] = static_cast<float>(i) * 0.1f;
    y_values[i] = static_cast<float>(i) * 0.01f + 1.0f;
  }
  Tensor x = Tensor::FromFloat("X", input_shape, x_values);
  Tensor y = Tensor::FromFloat("Y", input_shape, y_values);
  Tensor reshape_shape = MakeShapeTensor("reshape_shape", {0, 0, -1});

  // Compute expected intermediate/output tensors with the reference kernels.
  Tensor added = kernel::Add(ctx)(x, y);
  Tensor concat_out = kernel::Concat(ctx)({added, x}, /*axis=*/2);
  Tensor z = kernel::Reshape(ctx)(concat_out, reshape_shape);
  z.name = "Z";

  const std::string name = "test_cc_shape_inference_add_concat_reshape";

  TestCase tc;
  tc.name = name;
  tc.model_name = name;
  tc.kind = "node";
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  model.set_ir_version(kDefaultIrVersion);
  model.set_producer_name("backend-test");
  OperatorSetIdProto proto;
  proto.set_domain(opset.domain);
  proto.set_version(opset.version);
  model.add_opset_import(proto);

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  NodeProto *add_node = graph->add_node();
  add_node->set_op_type("Add");
  add_node->add_input("X");
  add_node->add_input("Y");
  add_node->add_output("added");

  NodeProto *concat_node = graph->add_node();
  concat_node->set_op_type("Concat");
  concat_node->add_input("added");
  concat_node->add_input("X");
  concat_node->add_output("concat_out");
  AttributeProto *axis_attr = concat_node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(AttributeProto::AttributeType::INT);
  axis_attr->set_i(2);

  NodeProto *reshape_node = graph->add_node();
  reshape_node->set_op_type("Reshape");
  reshape_node->add_input("concat_out");
  reshape_node->add_input("reshape_shape");
  reshape_node->add_output("Z");

  // Graph inputs: X, Y and the shape tensor.
  FillValueInfo(x, *graph->add_input());
  FillValueInfo(y, *graph->add_input());
  FillValueInfo(reshape_shape, *graph->add_input());

  // Graph output Z carries the fully resolved shape ``[kBatch, kSeq, 2 *
  // kDModel]``.
  FillValueInfo(z, *graph->add_output());

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
