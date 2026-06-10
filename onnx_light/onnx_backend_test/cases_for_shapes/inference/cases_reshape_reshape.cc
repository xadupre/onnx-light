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
// ``Reshape → Reshape → Add`` — mirrors the ``test_reshape_reshape`` example
// from yet-another-onnx-builder (https://github.com/xadupre/
// yet-another-onnx-builder/blob/main/unittests/xshape/test_shape_builder.py).
// The ``[0, 0, 2, -1]`` reshape pattern carries the leading dims through and
// splits the last dim by 2; a second ``[0, 0, -1]`` reshape collapses the
// trailing dims back to ``(a, b, c)``.
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
                  {"a", "b", DimSpec(int64_t{2}), DimSpec()});
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

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
