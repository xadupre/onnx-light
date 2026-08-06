// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_extensions/backend_test/cases_for_shapes/inference/include_inference_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// IR version used by the manually-built models below.
constexpr int64_t kDefaultIrVersion = 10;

} // namespace

// ---------------------------------------------------------------------------
// ``Shape → Gather → Unsqueeze → Concat → Reshape`` — exercises
// *value-as-shape* (VAS) propagation through ``Unsqueeze``, which was
// previously dropped, breaking the common pattern where a ``Reshape`` target
// shape is assembled from individual symbolic dims via
// ``Concat(Unsqueeze(Gather(Shape(y), 1), 0), ...)``.
//
//   x : float[D1, D2]
//   y : float[M, D1]   ← dim 1 of y gives D1
//   z : float[K, D2]   ← dim 1 of z gives D2
//   initializer idx    : int64 scalar = 1
//   initializer axes0  : int64[1] = [0]
//
//   shape_y   = Shape(y)                        # int64[2]  VAS=[M, D1]
//   shape_z   = Shape(z)                        # int64[2]  VAS=[K, D2]
//   d1        = Gather(shape_y, idx, axis=0)    # int64 scalar  VAS=[D1]
//   d2        = Gather(shape_z, idx, axis=0)    # int64 scalar  VAS=[D2]
//   u1        = Unsqueeze(d1, axes0)            # int64[1]  VAS=[D1]  ← was dropped
//   u2        = Unsqueeze(d2, axes0)            # int64[1]  VAS=[D2]  ← was dropped
//   new_shape = Concat(u1, u2, axis=0)          # int64[2]  VAS=[D1, D2]
//   reshaped  = Reshape(x, new_shape)           # float[D1, D2]
//   out       = Abs(reshaped)                   # float[D1, D2]
//
// Without VAS forwarding in ``ComputeShapeUnsqueeze``, ``Concat`` never sees
// the per-element symbolic values and ``Reshape`` falls back to inventing
// undefined placeholder names (``Reshape_dim0``, ``Reshape_dim1``) instead of
// the real dims ``D1``/``D2``. With the fix, shape inference correctly
// recovers ``float[D1, D2]``.
//
// The reference DataSet uses concrete sizes (M=2, D1=3, K=4, D2=5) so the
// case is executable end-to-end by ``BackendTestCaseRunModel``.
// ---------------------------------------------------------------------------
void RegisterUnsqueezeVasReshapeShapeInferenceCases(std::vector<TestCase> &registry,
                                                    TestMode mode) {
  const OpsetId opset = DefaultOpset(18);
  const onnx_kernels::kernel::KernelContext ctx{opset};

  const std::string name = "test_cc_shape_inference_unsqueeze_vas_reshape";

  TestCase tc(name, name, "model", "inference", 1e-7, 1e-3);

  ModelProto &model = tc.emplace_model();
  InitModel(model, kDefaultIrVersion, {opset});

  GraphProto *graph = model.add_graph();
  graph->set_name(name);

  // shape_y = Shape(y)
  AddNode(*graph, "Shape", {"y"}, {"shape_y"});

  // shape_z = Shape(z)
  AddNode(*graph, "Shape", {"z"}, {"shape_z"});

  // d1 = Gather(shape_y, idx, axis=0)
  NodeProto &d1_node = AddNode(*graph, "Gather", {"shape_y", "idx"}, {"d1"});
  AddAxisAttribute(d1_node, 0);

  // d2 = Gather(shape_z, idx, axis=0)
  NodeProto &d2_node = AddNode(*graph, "Gather", {"shape_z", "idx"}, {"d2"});
  AddAxisAttribute(d2_node, 0);

  // u1 = Unsqueeze(d1, axes0)
  AddNode(*graph, "Unsqueeze", {"d1", "axes0"}, {"u1"});

  // u2 = Unsqueeze(d2, axes0)
  AddNode(*graph, "Unsqueeze", {"d2", "axes0"}, {"u2"});

  // new_shape = Concat([u1, u2], axis=0)
  NodeProto &concat_node = AddNode(*graph, "Concat", {"u1", "u2"}, {"new_shape"});
  AddAxisAttribute(concat_node, 0);

  // reshaped = Reshape(x, new_shape)
  AddNode(*graph, "Reshape", {"x", "new_shape"}, {"reshaped"});

  // out = Abs(reshaped)
  AddNode(*graph, "Abs", {"reshaped"}, {"out"});

  // Initializers: scalar idx = 1 and axes0 = int64[1] = [0].
  AddInitializer<int64_t>(*graph, "idx", {}, {int64_t{1}});
  AddInitializerShape(*graph, "axes0", {int64_t{0}});

  // Graph inputs: x, y, z with symbolic dim names.
  const int32_t kFloat = static_cast<int32_t>(DataType::FLOAT);
  const int32_t kInt64 = static_cast<int32_t>(DataType::INT64);
  AppendValueInfo(*graph->add_input(), "x", kFloat, {DimSpec("D1"), DimSpec("D2")});
  AppendValueInfo(*graph->add_input(), "y", kFloat, {DimSpec("M"), DimSpec("D1")});
  AppendValueInfo(*graph->add_input(), "z", kFloat, {DimSpec("K"), DimSpec("D2")});

  // Intermediate value_info entries with expected shapes. These are stripped
  // by the shape-inference test harness, which then verifies inference
  // re-populates them with compatible shapes.
  AppendValueInfo(*graph->add_value_info(), "shape_y", kInt64, {DimSpec(int64_t{2})});
  AppendValueInfo(*graph->add_value_info(), "shape_z", kInt64, {DimSpec(int64_t{2})});
  AppendValueInfo(*graph->add_value_info(), "d1", kInt64, std::vector<DimSpec>{});
  AppendValueInfo(*graph->add_value_info(), "d2", kInt64, std::vector<DimSpec>{});
  AppendValueInfo(*graph->add_value_info(), "u1", kInt64, {DimSpec(int64_t{1})});
  AppendValueInfo(*graph->add_value_info(), "u2", kInt64, {DimSpec(int64_t{1})});
  AppendValueInfo(*graph->add_value_info(), "new_shape", kInt64, {DimSpec(int64_t{2})});
  AppendValueInfo(*graph->add_value_info(), "reshaped", kFloat, {DimSpec("D1"), DimSpec("D2")});

  // Graph output: out : float[D1, D2]. Shape inference must recover the
  // symbolic dim names D1/D2 from the VAS annotation on new_shape.
  AppendValueInfo(*graph->add_output(), "out", kFloat, {DimSpec("D1"), DimSpec("D2")});

  // Build the reference DataSet — concrete M=2, D1=3, K=4, D2=5 tensors.
  constexpr int64_t kM = 2, kD1 = 3, kK = 4, kD2 = 5;

  std::vector<float> x_values(static_cast<size_t>(kD1 * kD2));
  for (size_t i = 0; i < x_values.size(); ++i) {
    x_values[i] = static_cast<float>(i) * 0.1f + 1.0f;
  }
  Tensor x = Tensor::FromFloat("x", {kD1, kD2}, x_values);

  std::vector<float> y_values(static_cast<size_t>(kM * kD1));
  for (size_t i = 0; i < y_values.size(); ++i) {
    y_values[i] = static_cast<float>(i) * 0.2f + 0.5f;
  }
  Tensor y = Tensor::FromFloat("y", {kM, kD1}, y_values);

  std::vector<float> z_values(static_cast<size_t>(kK * kD2));
  for (size_t i = 0; i < z_values.size(); ++i) {
    z_values[i] = static_cast<float>(i) * 0.05f + 2.0f;
  }
  Tensor z = Tensor::FromFloat("z", {kK, kD2}, z_values);

  // Concrete execution.
  const Tensor idx_t = Tensor::FromInt64("idx", {}, {int64_t{1}});
  const std::vector<int64_t> axes0 = {0};

  Tensor shape_y = onnx_kernels::kernel::Shape(ctx)(y, onnx_kernels::kernel::Shape::Attributes{});
  shape_y.name = "shape_y";

  Tensor shape_z = onnx_kernels::kernel::Shape(ctx)(z, onnx_kernels::kernel::Shape::Attributes{});
  shape_z.name = "shape_z";

  Tensor d1 = onnx_kernels::kernel::Gather(ctx)(shape_y, idx_t, 0);
  d1.name = "d1";

  Tensor d2 = onnx_kernels::kernel::Gather(ctx)(shape_z, idx_t, 0);
  d2.name = "d2";

  Tensor u1 = onnx_kernels::kernel::Unsqueeze(ctx)(d1, axes0);
  u1.name = "u1";

  Tensor u2 = onnx_kernels::kernel::Unsqueeze(ctx)(d2, axes0);
  u2.name = "u2";

  Tensor new_shape_t = onnx_kernels::kernel::Concat(ctx)({u1, u2}, 0);
  new_shape_t.name = "new_shape";

  Tensor reshaped = onnx_kernels::kernel::Reshape(ctx)(x, new_shape_t);
  reshaped.name = "reshaped";

  Tensor out = onnx_kernels::kernel::Abs(ctx)(reshaped);
  out.name = "out";

  AppendDataSet(tc, {std::move(x), std::move(y), std::move(z)}, {std::move(out)});

  registry.emplace_back(std::move(tc));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
