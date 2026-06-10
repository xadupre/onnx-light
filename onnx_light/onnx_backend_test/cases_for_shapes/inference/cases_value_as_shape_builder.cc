// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// IR version used by the manually-built model below.
constexpr int64_t kDefaultIrVersion = 10;

} // namespace

// ---------------------------------------------------------------------------
// ``Shape → Concat → 3 × MatMul → 3 × Reshape → 3 × Transpose`` — mirrors the
// ``test_value_as_shape`` example from yet-another-onnx-builder
// (https://github.com/xadupre/yet-another-onnx-builder/blob/main/
// unittests/xshape/test_shape_builder.py). Exercises value-as-shape
// propagation: ``new_shape`` is built at graph-runtime by concatenating the
// first two dims of ``ids_weight`` with the constant ``[32, 8]`` initializer,
// and is then consumed by ``Reshape`` to produce ``(batch, seq, 32, 8)``
// tensors.
//
//   shape     = Shape(ids_weight, start=0, end=2)     # int64[2] = [batch, seq]
//   new_shape = Concat([shape, init328=[32, 8]], 0)   # int64[4] = [batch, seq, 32, 8]
//   A1 / B1 / C1 = MatMul(ids_weight, A/B/C)          # (batch, seq, 256)
//   Areshaped/Breshaped/Creshaped = Reshape(_, new_shape) # (batch, seq, 32, 8)
//   At / Bt / Ct = Transpose(_, perm=[0, 2, 1, 3])    # (batch, 32, seq, 8)
//
// Input:
//   ids_weight : float[batch, seq, 256]
// Outputs:
//   At, Bt, Ct : float[batch, 32, seq, 8]
//
// The reference DataSet uses concrete sizes ``batch=2, seq=5`` so the case
// is executable. The ``A``/``B``/``C`` matrices are part of the graph as
// FLOAT[256, 256] initializers.
// ---------------------------------------------------------------------------
void RegisterValueAsShapeBuilderShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};

  const std::string name = "test_cc_shape_inference_value_as_shape_builder";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-5;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  NodeProto &shape_node = AddNode(*graph, "Shape", {"ids_weight"}, {"shape"});
  AddAttribute<int64_t>(shape_node, "start", 0);
  AddAttribute<int64_t>(shape_node, "end", 2);

  NodeProto &concat_node = AddNode(*graph, "Concat", {"shape", "init328"}, {"new_shape"});
  AddAxisAttribute(concat_node, 0);

  AddNode(*graph, "MatMul", {"ids_weight", "A"}, {"A1"});
  AddNode(*graph, "MatMul", {"ids_weight", "B"}, {"B1"});
  AddNode(*graph, "MatMul", {"ids_weight", "C"}, {"C1"});

  AddNode(*graph, "Reshape", {"A1", "new_shape"}, {"Areshaped"});
  AddNode(*graph, "Reshape", {"B1", "new_shape"}, {"Breshaped"});
  AddNode(*graph, "Reshape", {"C1", "new_shape"}, {"Creshaped"});

  NodeProto &at_node = AddNode(*graph, "Transpose", {"Areshaped"}, {"At"});
  AddAttribute<std::vector<int64_t>>(at_node, "perm", {0, 2, 1, 3});
  NodeProto &bt_node = AddNode(*graph, "Transpose", {"Breshaped"}, {"Bt"});
  AddAttribute<std::vector<int64_t>>(bt_node, "perm", {0, 2, 1, 3});
  NodeProto &ct_node = AddNode(*graph, "Transpose", {"Creshaped"}, {"Ct"});
  AddAttribute<std::vector<int64_t>>(ct_node, "perm", {0, 2, 1, 3});

  // Constant initializers: init328 = [32, 8] and the three 256×256 matrices.
  AddInitializer<int64_t>(*graph, "init328", {2}, {32, 8});

  constexpr int64_t kK = 256;
  std::vector<float> a_values(static_cast<size_t>(kK * kK));
  std::vector<float> b_values(static_cast<size_t>(kK * kK));
  std::vector<float> c_values(static_cast<size_t>(kK * kK));
  for (size_t i = 0; i < a_values.size(); ++i) {
    a_values[i] = static_cast<float>(i % 7) * 0.001f;
    b_values[i] = static_cast<float>(i % 11) * 0.002f;
    c_values[i] = static_cast<float>(i % 13) * 0.003f;
  }
  AddInitializer<float>(*graph, "A", {kK, kK}, a_values);
  AddInitializer<float>(*graph, "B", {kK, kK}, b_values);
  AddInitializer<float>(*graph, "C", {kK, kK}, c_values);

  // Graph input: ids_weight uses symbolic batch / seq dims.
  AppendValueInfo(*graph->add_input(), "ids_weight", DataType::FLOAT,
                  {"batch", "seq", DimSpec(int64_t{256})});

  // Intermediate value_info entries with the shapes that shape inference
  // should recover (modulo symbolic dim renaming).
  AppendValueInfo(*graph->add_value_info(), "shape", DataType::INT64, {DimSpec(int64_t{2})});
  AppendValueInfo(*graph->add_value_info(), "new_shape", DataType::INT64, {DimSpec(int64_t{4})});
  AppendValueInfo(*graph->add_value_info(), "A1", DataType::FLOAT,
                  {"batch", "seq", DimSpec(int64_t{256})});
  AppendValueInfo(*graph->add_value_info(), "B1", DataType::FLOAT,
                  {"batch", "seq", DimSpec(int64_t{256})});
  AppendValueInfo(*graph->add_value_info(), "C1", DataType::FLOAT,
                  {"batch", "seq", DimSpec(int64_t{256})});
  AppendValueInfo(*graph->add_value_info(), "Areshaped", DataType::FLOAT,
                  {"batch", "seq", DimSpec(int64_t{32}), DimSpec(int64_t{8})});
  AppendValueInfo(*graph->add_value_info(), "Breshaped", DataType::FLOAT,
                  {"batch", "seq", DimSpec(int64_t{32}), DimSpec(int64_t{8})});
  AppendValueInfo(*graph->add_value_info(), "Creshaped", DataType::FLOAT,
                  {"batch", "seq", DimSpec(int64_t{32}), DimSpec(int64_t{8})});

  // Graph outputs At/Bt/Ct — float[batch, 32, seq, 8].
  AppendValueInfo(*graph->add_output(), "At", DataType::FLOAT,
                  {"batch", DimSpec(int64_t{32}), "seq", DimSpec(int64_t{8})});
  AppendValueInfo(*graph->add_output(), "Bt", DataType::FLOAT,
                  {"batch", DimSpec(int64_t{32}), "seq", DimSpec(int64_t{8})});
  AppendValueInfo(*graph->add_output(), "Ct", DataType::FLOAT,
                  {"batch", DimSpec(int64_t{32}), "seq", DimSpec(int64_t{8})});

  // Build the reference DataSet — concrete batch=2, seq=5 tensors and the
  // kernels chained to materialise At/Bt/Ct.
  constexpr int64_t kBatch = 2;
  constexpr int64_t kSeq = 5;
  std::vector<float> ids_values(static_cast<size_t>(kBatch * kSeq * kK));
  for (size_t i = 0; i < ids_values.size(); ++i) {
    ids_values[i] = static_cast<float>(i % 17) * 0.01f;
  }
  Tensor ids_weight = Tensor::FromFloat("ids_weight", {kBatch, kSeq, kK}, ids_values);
  Tensor a = Tensor::FromFloat("", {kK, kK}, a_values);
  Tensor b = Tensor::FromFloat("", {kK, kK}, b_values);
  Tensor c = Tensor::FromFloat("", {kK, kK}, c_values);

  const Tensor new_shape = Tensor::FromInt64("", {4}, {kBatch, kSeq, 32, 8});
  auto build_branch = [&](const Tensor &mat, const std::string &out_name) {
    Tensor m1 = kernel::MatMul(ctx)(ids_weight, mat);
    Tensor reshaped = kernel::Reshape(ctx)(m1, new_shape);
    Tensor t = kernel::Transpose(ctx)(reshaped, /*perm=*/{0, 2, 1, 3});
    t.name = out_name;
    return t;
  };
  Tensor at = build_branch(a, "At");
  Tensor bt = build_branch(b, "Bt");
  Tensor ct = build_branch(c, "Ct");

  AppendDataSet(tc, {std::move(ids_weight)}, {std::move(at), std::move(bt), std::move(ct)});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
