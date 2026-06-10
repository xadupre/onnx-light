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
// ``Unsqueeze → Unsqueeze → Reshape → Reshape → Cast → MatMul → Reshape`` —
// mirrors the ``test_check_shape`` example from yet-another-onnx-builder
// (https://github.com/xadupre/yet-another-onnx-builder/blob/main/
// unittests/xshape/test_shape_builder.py). Exercises shape inference through
// rank-changing ``Unsqueeze``/``Reshape`` and through ``MatMul`` of two
// 3-D tensors with the leading dim broadcast.
//
//   xu1  = Unsqueeze(X,   zero)          # axes=[0]  → (1, D32, D128)
//   xu2  = Unsqueeze(xu1, un)            # axes=[1]  → (1, 1, D32, D128)
//   xm1  = Reshape(xu2,   shape1=[1,32,128])      # (1, 32, 128)
//   xm2c = Reshape(Y,     shape2=[15,128,64])     # (15, 128, 64)
//   xm2  = Cast(xm2c, to=FLOAT)                   # (15, 128, 64)
//   xm   = MatMul(xm1,  xm2)                      # (15, 32, 64)
//   Z    = Reshape(xm,  shape3=[3,5,32,64])       # (3, 5, 32, 64)
//
// Inputs:
//   X : float[D32, D128]
//   Y : float[batch, channel, D128, D64]
// Output:
//   Z : float[batch, channel, D32, 64]   (declared symbolic; inferred concrete)
//
// The reference DataSet uses concrete sizes ``D32=32, D128=128, batch=3,
// channel=5, D64=64`` so the case is executable.
// ---------------------------------------------------------------------------
void RegisterCheckShapeShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};

  const std::string name = "test_cc_shape_inference_check_shape";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  AddNode(*graph, "Unsqueeze", {"X", "zero"}, {"xu1"});
  AddNode(*graph, "Unsqueeze", {"xu1", "un"}, {"xu2"});
  AddNode(*graph, "Reshape", {"xu2", "shape1"}, {"xm1"});
  AddNode(*graph, "Reshape", {"Y", "shape2"}, {"xm2c"});
  NodeProto &cast_node = AddNode(*graph, "Cast", {"xm2c"}, {"xm2"});
  AddAttribute<int64_t>(cast_node, "to", static_cast<int64_t>(DataType::FLOAT));
  AddNode(*graph, "MatMul", {"xm1", "xm2"}, {"xm"});
  AddNode(*graph, "Reshape", {"xm", "shape3"}, {"Z"});

  AddInitializer<int64_t>(*graph, "zero", {1}, {0});
  AddInitializer<int64_t>(*graph, "un", {1}, {1});
  AddInitializer<int64_t>(*graph, "shape1", {3}, {1, 32, 128});
  AddInitializer<int64_t>(*graph, "shape2", {3}, {15, 128, 64});
  AddInitializer<int64_t>(*graph, "shape3", {4}, {3, 5, 32, 64});

  // Graph inputs: X uses symbolic dims (D32, D128); Y uses a mix of symbolic
  // (batch, channel, D128, D64) dims that resolve to concrete sizes in the
  // reference DataSet.
  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {"D32", "D128"});
  AppendValueInfo(*graph->add_input(), "Y", DataType::FLOAT, {"batch", "channel", "D128", "D64"});

  // Intermediate value_info entries with the shapes that shape inference
  // should recover. These are stripped by ``SnapshotAndStripValueInfo`` in
  // the ``AllCollectedCasesInferOutputShapes`` test and used as the ground
  // truth.
  AppendValueInfo(*graph->add_value_info(), "xu1", DataType::FLOAT,
                  {DimSpec(int64_t{1}), "D32", "D128"});
  AppendValueInfo(*graph->add_value_info(), "xu2", DataType::FLOAT,
                  {DimSpec(int64_t{1}), DimSpec(int64_t{1}), "D32", "D128"});
  AppendValueInfo(*graph->add_value_info(), "xm1", DataType::FLOAT,
                  {DimSpec(int64_t{1}), DimSpec(int64_t{32}), DimSpec(int64_t{128})});
  AppendValueInfo(*graph->add_value_info(), "xm2c", DataType::FLOAT,
                  {DimSpec(int64_t{15}), DimSpec(int64_t{128}), DimSpec(int64_t{64})});
  AppendValueInfo(*graph->add_value_info(), "xm2", DataType::FLOAT,
                  {DimSpec(int64_t{15}), DimSpec(int64_t{128}), DimSpec(int64_t{64})});
  AppendValueInfo(*graph->add_value_info(), "xm", DataType::FLOAT,
                  {DimSpec(int64_t{15}), DimSpec(int64_t{32}), DimSpec(int64_t{64})});

  // Graph output Z — concrete dims recovered from the final Reshape.
  AppendValueInfo(
      *graph->add_output(), "Z", DataType::FLOAT,
      {DimSpec(int64_t{3}), DimSpec(int64_t{5}), DimSpec(int64_t{32}), DimSpec(int64_t{64})});

  // Build the reference DataSet — concrete D32=32, D128=128, batch=3,
  // channel=5, D64=64 tensors, then run the kernels to materialise Z.
  constexpr int64_t kD32 = 32;
  constexpr int64_t kD128 = 128;
  constexpr int64_t kBatch = 3;
  constexpr int64_t kChannel = 5;
  constexpr int64_t kD64 = 64;
  std::vector<float> x_values(static_cast<size_t>(kD32 * kD128));
  for (size_t i = 0; i < x_values.size(); ++i) {
    x_values[i] = static_cast<float>(i) * 0.001f;
  }
  std::vector<float> y_values(static_cast<size_t>(kBatch * kChannel * kD128 * kD64));
  for (size_t i = 0; i < y_values.size(); ++i) {
    y_values[i] = static_cast<float>(i) * 0.0001f + 1.0f;
  }
  Tensor x = Tensor::FromFloat("X", {kD32, kD128}, x_values);
  Tensor y = Tensor::FromFloat("Y", {kBatch, kChannel, kD128, kD64}, y_values);

  const Tensor shape1 = Tensor::FromInt64("", {3}, {1, 32, 128});
  const Tensor shape2 = Tensor::FromInt64("", {3}, {15, 128, 64});
  const Tensor shape3 = Tensor::FromInt64("", {4}, {3, 5, 32, 64});
  Tensor xu1 = kernel::Unsqueeze(ctx)(x, /*axes=*/{0});
  Tensor xu2 = kernel::Unsqueeze(ctx)(xu1, /*axes=*/{1});
  Tensor xm1 = kernel::Reshape(ctx)(xu2, shape1);
  Tensor xm2c = kernel::Reshape(ctx)(y, shape2);
  Tensor xm2 = kernel::Cast(ctx)(xm2c, static_cast<int32_t>(DataType::FLOAT));
  Tensor xm = kernel::MatMul(ctx)(xm1, xm2);
  Tensor z = kernel::Reshape(ctx)(xm, shape3);
  z.name = "Z";

  AppendDataSet(tc, {std::move(x), std::move(y)}, {std::move(z)});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
