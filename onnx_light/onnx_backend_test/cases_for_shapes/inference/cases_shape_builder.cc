// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Shape-inference backend test cases translated from the
// ``yet-another-onnx-builder``
// (https://github.com/xadupre/yet-another-onnx-builder/blob/main/
// unittests/xshape/test_shape_builder.py) examples. Each
// ``Register*ShapeInferenceCases`` helper below mirrors one ``test_*`` entry
// from that file:
//
//   * ``RegisterCheckShapeShapeInferenceCases``         — ``test_check_shape``
//   * ``RegisterReshapeReshapeShapeInferenceCases``     — ``test_reshape_reshape``
//   * ``RegisterValueAsShapeBuilderShapeInferenceCases``— ``test_value_as_shape``
//   * ``RegisterConcatSplitShapeInferenceCases``        — ``test_concat_split``

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

// IR version used by the manually-built models below.
constexpr int64_t kDefaultIrVersion = 10;

} // namespace

// ---------------------------------------------------------------------------
// ``Unsqueeze → Unsqueeze → Reshape → Reshape → Cast → MatMul → Reshape`` —
// mirrors the ``test_check_shape`` example from yet-another-onnx-builder.
// Exercises shape inference through rank-changing ``Unsqueeze``/``Reshape``
// and through ``MatMul`` of two 3-D tensors with the leading dim broadcast.
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

  AddInitializer<int64_t>(*graph, "zero", {1}, {0});
  AddInitializer<int64_t>(*graph, "un", {1}, {1});
  AddInitializer<int64_t>(*graph, "shape1", {3}, {0, -1, 128});
  AddInitializer<int64_t>(*graph, "shape2", {3}, {-1, 128, 64});
  AddInitializer<int64_t>(*graph, "shape3", {4}, {3, 5, 32, 64});

  AddNode(*graph, "Unsqueeze", {"X", "zero"}, {"xu1"});
  AddNode(*graph, "Unsqueeze", {"xu1", "un"}, {"xu2"});
  AddNode(*graph, "Reshape", {"xu2", "shape1"}, {"xm1"});
  AddNode(*graph, "Reshape", {"Y", "shape2"}, {"xm2c"});
  NodeProto &cast_node = AddNode(*graph, "Cast", {"xm2c"}, {"xm2"});
  AddAttribute<int64_t>(cast_node, "to", static_cast<int64_t>(DataType::FLOAT));
  AddNode(*graph, "MatMul", {"xm1", "xm2"}, {"xm"});
  AddNode(*graph, "Reshape", {"xm", "shape3"}, {"Z"});


  // Graph inputs: X uses symbolic dims (D32, D128); Y uses a mix of symbolic
  // (batch, channel, D128, D64) dims that resolve to concrete sizes in the
  // reference DataSet.
  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {"D32", "D128"});
  AppendValueInfo(*graph->add_input(), "Y", DataType::FLOAT, {"batch", "channel", "D128", "D64"});

  // Intermediate value_info entries with the shapes that shape inference
  // should recover. These are stripped by ``SnapshotAndStripValueInfo`` in
  // the ``AllCollectedCasesInferOutputShapes`` test and used as the ground
  // truth.
  AppendValueInfo(*graph->add_value_info(), "xu1", DataType::FLOAT, {DimSpec(1), "D32", "D128"});
  AppendValueInfo(*graph->add_value_info(), "xu2", DataType::FLOAT, {DimSpec(1), DimSpec(1), "D32", "D128"});
  AppendValueInfo(*graph->add_value_info(), "xm1", DataType::FLOAT, {DimSpec(1), "D32", "D128"});
  AppendValueInfo(*graph->add_value_info(), "xm2c", DataType::FLOAT, {"batch*channel", "D128", "D64"});
  AppendValueInfo(*graph->add_value_info(), "xm2", DataType::FLOAT, {"batch*channel", "D128", "D64"});
  AppendValueInfo(*graph->add_value_info(), "xm", DataType::FLOAT, {"batch*channel", "D128", "D64"});

  // Graph output Z — concrete dims recovered from the final Reshape.
  AppendValueInfo(*graph->add_output(), "Z", DataType::FLOAT, {"batch", "channel", "D32", "D64");

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

// ---------------------------------------------------------------------------
// ``Reshape → Reshape → Add`` — mirrors the ``test_reshape_reshape`` example
// from yet-another-onnx-builder. The ``[0, 0, 2, -1]`` reshape pattern carries
// the leading dims through and splits the last dim by 2; a second
// ``[0, 0, -1]`` reshape collapses the trailing dims back to ``(a, b, c)``.
//
//   xr  = Reshape(X,  shape1=[0, 0, 2, -1])    # (a, b, 2, c//2)
//   xrr = Reshape(xr, shape2=[0, 0, -1])       # (a, b, c)
//   Y   = Add(xrr, one)                        # (a, b, c)
//
// Input:
//   X : float[a, b, c]
// Output:
//   Y : float[a, b, c]
//
// The reference DataSet uses concrete sizes ``a=2, b=3, c=4`` so the case is
// executable (``c`` is even and divisible by 2 so the inferred ``c//2``
// resolves to a concrete int64).
// ---------------------------------------------------------------------------
void RegisterReshapeReshapeShapeInferenceCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};

  const std::string name = "test_cc_shape_inference_reshape_reshape";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  AddNode(*graph, "Reshape", {"X", "shape1"}, {"xr"});
  AddNode(*graph, "Reshape", {"xr", "shape2"}, {"xrr"});
  AddNode(*graph, "Add", {"xrr", "one"}, {"Y"});

  AddInitializer<int64_t>(*graph, "shape1", {4}, {0, 0, 2, -1});
  AddInitializer<int64_t>(*graph, "shape2", {3}, {0, 0, -1});
  AddInitializer<float>(*graph, "one", {1}, {1.0f});

  // Graph input: X uses symbolic dims (a, b, c).
  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {"a", "b", "c"});

  // Intermediate value_info entries with the shapes that shape inference
  // should recover. The Python ``BasicShapeBuilder`` records
  // ``xr = (a, b, 2, c//2)``; in C++ we only declare ranks/elem types and
  // the symbolic leading dims here so the test framework's ground-truth
  // check is independent of how shape inference renders the symbolic
  // division.
  AppendValueInfo(*graph->add_value_info(), "xr", DataType::FLOAT,
                  {"a", "b", DimSpec(2), "c//2"});
  AppendValueInfo(*graph->add_value_info(), "xrr", DataType::FLOAT, {"a", "b", "c"});

  // Graph output Y — same symbolic dims as X.
  AppendValueInfo(*graph->add_output(), "Y", DataType::FLOAT, {"a", "b", "c"});

  // Build the reference DataSet — concrete a=2, b=3, c=4 tensors and the
  // kernels chained to materialise Y.
  constexpr int64_t kA = 2;
  constexpr int64_t kB = 3;
  constexpr int64_t kC = 4;
  std::vector<float> x_values(static_cast<size_t>(kA * kB * kC));
  for (size_t i = 0; i < x_values.size(); ++i) {
    x_values[i] = static_cast<float>(i) * 0.1f;
  }
  Tensor x = Tensor::FromFloat("X", {kA, kB, kC}, x_values);

  const Tensor shape1 = Tensor::FromInt64("", {4}, {0, 0, 2, -1});
  const Tensor shape2 = Tensor::FromInt64("", {3}, {0, 0, -1});
  const Tensor one = Tensor::FromFloat("", {1}, {1.0f});
  Tensor xr = kernel::Reshape(ctx)(x, shape1);
  Tensor xrr = kernel::Reshape(ctx)(xr, shape2);
  Tensor y = kernel::Add(ctx)(xrr, one);
  y.name = "Y";

  AppendDataSet(tc, {std::move(x)}, {std::move(y)});

  registry.emplace_back(std::move(tc));
}

// ---------------------------------------------------------------------------
// ``Shape → Concat → 3 × MatMul → 3 × Reshape → 3 × Transpose`` — mirrors the
// ``test_value_as_shape`` example from yet-another-onnx-builder. Exercises
// value-as-shape propagation: ``new_shape`` is built at graph-runtime by
// concatenating the first two dims of ``ids_weight`` with the constant
// ``[32, 8]`` initializer, and is then consumed by ``Reshape`` to produce
// ``(batch, seq, 32, 8)`` tensors.
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

// ---------------------------------------------------------------------------
// ``Concat → Split → Concat → Relu`` — mirrors the ``test_concat_split``
// example from yet-another-onnx-builder. Exercises Concat / Split shape
// propagation when the concat axis dims are symbolic (``b + c``) and the
// Split divides the dim by 2.
//
//   xy = Concat([X, Y],  axis=1)            # (a, b+c)
//   S1, S2 = Split(xy,   axis=1, num_outputs=2)  # (a, (b+c+1)//2), (a, b+c - (b+c+1)//2)
//   zs = Concat([S2, S1], axis=1)           # (a, b+c)
//   Z  = Relu(zs)                           # (a, b+c)
//
// Inputs:
//   X : float[a, b]   Y : float[a, c]
// Output:
//   Z : float[a, e]   (symbolic; the runtime shape is (a, b+c))
//
// The reference DataSet uses concrete sizes ``a=3, b=4, c=6`` so the case
// is executable (``b + c = 10`` is even and divisible by 2 so Split produces
// two equal halves).
// ---------------------------------------------------------------------------
void RegisterConcatSplitShapeInferenceCases(std::vector<TestCase> &registry) {
  // Opset 18 is required for Split's ``num_outputs`` attribute; the older
  // IR version 9 used in the original split-out file remains compatible.
  constexpr int64_t kConcatSplitIrVersion = 9;
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};

  const std::string name = "test_cc_shape_inference_concat_split";

  TestCase tc(name, name, "model", "inference");
  tc.rtol = 1e-3;
  tc.atol = 1e-7;

  ModelProto &model = tc.model;
  InitModel(model, kConcatSplitIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  NodeProto &xy_node = AddNode(*graph, "Concat", {"X", "Y"}, {"xy"});
  AddAxisAttribute(xy_node, 1);

  NodeProto &split_node = AddNode(*graph, "Split", {"xy"}, {"S1", "S2"});
  AddAxisAttribute(split_node, 1);
  AddAttribute<int64_t>(split_node, "num_outputs", 2);

  NodeProto &zs_node = AddNode(*graph, "Concat", {"S2", "S1"}, {"zs"});
  AddAxisAttribute(zs_node, 1);

  AddNode(*graph, "Relu", {"zs"}, {"Z"});

  // Graph inputs: X = float[a, b], Y = float[a, c].
  AppendValueInfo(*graph->add_input(), "X", DataType::FLOAT, {"a", "b"});
  AppendValueInfo(*graph->add_input(), "Y", DataType::FLOAT, {"a", "c"});

  // Intermediate value_info — leave the concat axis dim unannotated; shape
  // inference renders it as a fresh symbolic dim (e.g. ``Concat_axis1``).
  AppendValueInfo(*graph->add_value_info(), "xy", DataType::FLOAT, {"a", "b+c"});
  AppendValueInfo(*graph->add_value_info(), "S1", DataType::FLOAT, {"a", "(b+c)//2"});
  AppendValueInfo(*graph->add_value_info(), "S2", DataType::FLOAT, {"a", "(b+c)//2"});
  AppendValueInfo(*graph->add_value_info(), "zs", DataType::FLOAT, {"a", "b+c"});

  // Graph output Z — same shape as zs.
  AppendValueInfo(*graph->add_output(), "Z", DataType::FLOAT, {"a", "b+c"});

  // Build the reference DataSet — concrete a=3, b=4, c=6 tensors so that
  // b + c = 10 is divisible by 2 and Split produces two (3, 5) halves.
  constexpr int64_t kA = 3;
  constexpr int64_t kB = 4;
  constexpr int64_t kC = 6;
  std::vector<float> x_values(static_cast<size_t>(kA * kB));
  for (size_t i = 0; i < x_values.size(); ++i) {
    x_values[i] = static_cast<float>(i) * 0.1f - 1.0f;
  }
  std::vector<float> y_values(static_cast<size_t>(kA * kC));
  for (size_t i = 0; i < y_values.size(); ++i) {
    y_values[i] = static_cast<float>(i) * 0.2f - 0.5f;
  }
  Tensor x = Tensor::FromFloat("X", {kA, kB}, x_values);
  Tensor y = Tensor::FromFloat("Y", {kA, kC}, y_values);

  Tensor xy = kernel::Concat(ctx)({x, y}, /*axis=*/1);
  std::vector<Tensor> splits = kernel::Split(ctx)(xy, /*axis=*/1, /*split=*/{}, /*num_outputs=*/2);
  Tensor zs = kernel::Concat(ctx)({splits[1], splits[0]}, /*axis=*/1);
  Tensor z = kernel::Relu(ctx)(zs);
  z.name = "Z";

  AppendDataSet(tc, {std::move(x), std::move(y)}, {std::move(z)});

  registry.emplace_back(std::move(tc));
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
